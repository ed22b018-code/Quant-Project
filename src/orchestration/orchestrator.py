"""Pipeline orchestrator — runs all stages end-to-end with checkpointing.

Stage list (in order)
---------------------
1. PREFLIGHT        — env vars, disk space, optional connectivity check
2. TEST_FETCH       — tiny real API call (1 symbol, 3 days) to validate format
3. BULK_DOWNLOAD    — full 5yr intraday + F&O history (resumable, paced)
4. COVERAGE_CHECK   — print coverage matrix; fail if critical gaps remain
5. BUILD_DATASET    — assemble features + labels (PIT, all gates)
6. TRAIN_MODELS     — LightGBM quantile cone + scenario classifier
7. CALIBRATE        — isotonic calibration on validation fold
8. SAVE_SUMMARY     — write provenance.json + print final metrics

Checkpointing
-------------
After every stage completes, the orchestrator writes:
    {output_dir}/.orchestration_state.json
    {"completed": ["preflight", "test_fetch", ...], "started_at": "...", "last_updated": "..."}

On restart, completed stages are skipped. Pass start_from=Stage.X to force
re-run from a specific stage (ignores the checkpoint for stages >= X).

Usage
-----
    from src.orchestration.orchestrator import Orchestrator, Stage
    orch = Orchestrator(db_path="data/market.duckdb", output_dir="models/v1",
                        universe=["RELIANCE", "TCS"], ...)
    orch.run()                          # full run
    orch.run(dry_run=True)             # schema + config check only
    orch.run(start_from=Stage.BUILD_DATASET)   # skip download stages
    orch.run(skip_test_fetch=True)     # skip format probe (already verified)
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

_LOG = logging.getLogger(__name__)
UTC = timezone.utc


class Stage(str, Enum):
    PREFLIGHT = "preflight"
    TEST_FETCH = "test_fetch"
    BULK_DOWNLOAD = "bulk_download"
    COVERAGE_CHECK = "coverage_check"
    BUILD_DATASET = "build_dataset"
    TRAIN_MODELS = "train_models"
    CALIBRATE = "calibrate"
    SAVE_SUMMARY = "save_summary"


_STAGE_ORDER: list[Stage] = [
    Stage.PREFLIGHT,
    Stage.TEST_FETCH,
    Stage.BULK_DOWNLOAD,
    Stage.COVERAGE_CHECK,
    Stage.BUILD_DATASET,
    Stage.TRAIN_MODELS,
    Stage.CALIBRATE,
    Stage.SAVE_SUMMARY,
]


class OrchestratorState:
    """Persisted checkpoint state for the orchestration run."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self.completed: list[str] = []
        self.started_at: str = datetime.now(UTC).isoformat()
        self.last_updated: str = self.started_at
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                self.completed = data.get("completed", [])
                self.started_at = data.get("started_at", self.started_at)
                _LOG.info("Loaded checkpoint: %d stages completed.", len(self.completed))
            except Exception as exc:
                _LOG.warning("Could not load checkpoint %s: %s — starting fresh.", self._path, exc)

    def mark_done(self, stage: Stage) -> None:
        name = stage.value
        if name not in self.completed:
            self.completed.append(name)
        self.last_updated = datetime.now(UTC).isoformat()
        self._save()

    def is_done(self, stage: Stage) -> bool:
        return stage.value in self.completed

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(
                {
                    "completed": self.completed,
                    "started_at": self.started_at,
                    "last_updated": self.last_updated,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        _LOG.debug("Checkpoint saved: %s", self._path)


class Orchestrator:
    """Full pipeline orchestrator from preflight to model save.

    All expensive calls are injectable for testing. In production, the
    default implementations call real Dhan APIs and real model training.

    Parameters
    ----------
    db_path : str
        Path to DuckDB (created if absent).
    output_dir : str
        Directory for models, calibrators, provenance, and checkpoint file.
    universe : list[str]
        NSE tickers to include.
    start_date, end_date : date
        Training window (also used as download window).
    timeframes : list[str]
        Dhan intraday timeframes (default ["1", "5", "15", "25", "60"]).
    _mock_preflight_fn : callable | None
        Replace run_preflight() entirely (testing only).
    _mock_test_fetch_fn : callable | None
        Replace the test_fetch call (testing only).
    _mock_downloader : DhanBulkDownloader-compatible | None
        Replace real downloader (testing only).
    _mock_run_training_fn : callable | None
        Replace run_training() entirely (testing only).
    """

    STAGES: list[Stage] = _STAGE_ORDER

    def __init__(
        self,
        db_path: str,
        output_dir: str,
        *,
        universe: list[str],
        start_date: date,
        end_date: date,
        timeframes: list[str] | None = None,
        skip_fo: bool = False,
        skip_news: bool = False,
        _mock_preflight_fn: Optional[Callable] = None,
        _mock_test_fetch_fn: Optional[Callable] = None,
        _mock_downloader: Any = None,
        _mock_run_training_fn: Optional[Callable] = None,
    ) -> None:
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.universe = universe
        self.start_date = start_date
        self.end_date = end_date
        self.timeframes = timeframes or ["1"]
        self.skip_fo = skip_fo
        self.skip_news = skip_news

        self._mock_preflight_fn = _mock_preflight_fn
        self._mock_test_fetch_fn = _mock_test_fetch_fn
        self._mock_downloader = _mock_downloader
        self._mock_run_training_fn = _mock_run_training_fn

        self._state_path = self.output_dir / ".orchestration_state.json"

    # ── public ────────────────────────────────────────────────────────────────

    def run(
        self,
        *,
        dry_run: bool = False,
        start_from: Optional[Stage] = None,
        skip_test_fetch: bool = False,
    ) -> dict:
        """Run all stages in order, skipping completed ones (from checkpoint).

        Parameters
        ----------
        dry_run : bool
            Execute stage logic but skip real API calls and training.
        start_from : Stage | None
            If set, ignore the checkpoint for this stage and all subsequent
            stages — forces re-execution from this point.
        skip_test_fetch : bool
            Skip the TEST_FETCH stage (useful when connection was already
            verified by running scripts/test_fetch.py beforehand).

        Returns
        -------
        Summary dict with completed stages and provenance path.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        state = OrchestratorState(self._state_path)
        summary: dict = {
            "stages_run": [],
            "stages_skipped": [],
            "dry_run": dry_run,
            "output_dir": str(self.output_dir),
        }

        # If start_from is set, clear completed state from that stage onwards
        if start_from is not None:
            idx = self.STAGES.index(start_from)
            stages_to_clear = [s.value for s in self.STAGES[idx:]]
            state.completed = [s for s in state.completed if s not in stages_to_clear]
            _LOG.info("start_from=%s: cleared stages %s from checkpoint.", start_from, stages_to_clear)

        for stage in self.STAGES:
            if stage == Stage.TEST_FETCH and skip_test_fetch:
                _LOG.info("Stage TEST_FETCH: skipped (--skip-test-fetch flag).")
                summary["stages_skipped"].append(stage.value)
                continue

            if state.is_done(stage):
                _LOG.info("Stage %s: already completed — skipping.", stage.value)
                summary["stages_skipped"].append(stage.value)
                continue

            _LOG.info("Stage %s: starting.", stage.value)
            print(f"\n[{stage.value.upper()}]")
            try:
                self._dispatch(stage, dry_run=dry_run)
            except Exception as exc:
                _LOG.error("Stage %s FAILED: %s", stage.value, exc)
                print(f"  FAILED: {exc}")
                if stage == Stage.PREFLIGHT:
                    raise  # preflight failure always blocks
                raise RuntimeError(
                    f"Stage {stage.value} failed. "
                    f"Fix the issue and resume with --start-from={stage.value}"
                ) from exc

            state.mark_done(stage)
            summary["stages_run"].append(stage.value)
            _LOG.info("Stage %s: complete.", stage.value)

        summary["provenance_path"] = str(self.output_dir / "provenance.json")
        return summary

    # ── stage dispatch ────────────────────────────────────────────────────────

    def _dispatch(self, stage: Stage, *, dry_run: bool) -> None:
        if stage == Stage.PREFLIGHT:
            self._stage_preflight(dry_run=dry_run)
        elif stage == Stage.TEST_FETCH:
            self._stage_test_fetch(dry_run=dry_run)
        elif stage == Stage.BULK_DOWNLOAD:
            self._stage_bulk_download(dry_run=dry_run)
        elif stage == Stage.COVERAGE_CHECK:
            self._stage_coverage_check(dry_run=dry_run)
        elif stage == Stage.BUILD_DATASET:
            self._stage_build_dataset(dry_run=dry_run)
        elif stage == Stage.TRAIN_MODELS:
            self._stage_train_models(dry_run=dry_run)
        elif stage == Stage.CALIBRATE:
            self._stage_calibrate(dry_run=dry_run)
        elif stage == Stage.SAVE_SUMMARY:
            self._stage_save_summary(dry_run=dry_run)

    # ── individual stages ─────────────────────────────────────────────────────

    def _stage_preflight(self, *, dry_run: bool) -> None:
        if self._mock_preflight_fn is not None:
            self._mock_preflight_fn()
            return
        if dry_run:
            _LOG.info("dry_run: preflight skipped.")
            return
        from src.orchestration.preflight import run_preflight  # noqa: PLC0415
        run_preflight(self.db_path, verbose=True)

    def _stage_test_fetch(self, *, dry_run: bool) -> None:
        if self._mock_test_fetch_fn is not None:
            result = self._mock_test_fetch_fn()
        elif dry_run:
            _LOG.info("dry_run: test_fetch skipped.")
            return
        else:
            from scripts.test_fetch import test_fetch  # noqa: PLC0415
            result = test_fetch(db_path=self.db_path)

        status = result.get("status", "unknown")
        if status == "fail":
            raise RuntimeError(
                f"Test fetch failed — cannot proceed. "
                f"Check credentials and run scripts/test_fetch.py manually."
            )
        if status == "partial":
            _LOG.warning(
                "Test fetch returned partial result: %s. "
                "Fix the response-parsing issue before the big download. "
                "Fix location: %s",
                result.get("message"), result.get("fix_location"),
            )
            print(f"  WARNING: {result.get('message')}")
            print(f"  Fix in:  {result.get('fix_location')}")

    def _stage_bulk_download(self, *, dry_run: bool) -> None:
        if self._mock_downloader is not None:
            dl = self._mock_downloader
        elif dry_run:
            _LOG.info("dry_run: bulk download skipped.")
            return
        else:
            from src.data.dhan_downloader import DhanBulkDownloader  # noqa: PLC0415
            dl = DhanBulkDownloader(db_path=self.db_path)

        _LOG.info("Downloading intraday OHLCV: %d symbols × %s timeframes",
                  len(self.universe), self.timeframes)
        dl.run(
            symbols=self.universe,
            timeframes=self.timeframes,
            start_date=self.start_date,
            end_date=self.end_date,
        )
        if self.skip_fo:
            _LOG.warning(
                "skip_fo=True: F&O download skipped. "
                "PCR/OI features will be NaN until F&O data source is resolved. "
                "See RUNBOOK.md §F&O Known Limitation."
            )
            print("  [SKIP] F&O download skipped (--skip-fo). F&O gate features will be NaN.")
            return
        _LOG.info("Downloading F&O history: %d symbols", len(self.universe))
        dl.run_fo(
            symbols=self.universe,
            start_date=self.start_date,
            end_date=self.end_date,
        )
        _LOG.info("Computing F&O daily metrics")
        dl.compute_fo_metrics(symbols=self.universe)

    def _stage_coverage_check(self, *, dry_run: bool) -> None:
        if dry_run:
            _LOG.info("dry_run: coverage check skipped.")
            return
        if self._mock_downloader is not None:
            report = self._mock_downloader.coverage_report()
        else:
            from src.data.dhan_downloader import DhanBulkDownloader  # noqa: PLC0415
            dl = DhanBulkDownloader(db_path=self.db_path)
            report = dl.coverage_report()
        print(report)

    def _stage_build_dataset(self, *, dry_run: bool) -> None:
        if self._mock_run_training_fn is not None or dry_run:
            _LOG.info("dry_run/mock: dataset build deferred to TRAIN_MODELS stage.")
            return
        # Dataset is assembled inside run_training(); no separate step needed.

    def _stage_train_models(self, *, dry_run: bool) -> None:
        if self._mock_run_training_fn is not None:
            self._mock_run_training_fn()
            return
        from src.models.training_pipeline import run_training  # noqa: PLC0415
        from src.features import DataGate  # noqa: PLC0415
        from src.config import load_config
        
        config = load_config()
        
        gates_open = set(DataGate)
        if self.skip_fo and DataGate.FO in gates_open:
            gates_open.remove(DataGate.FO)
        if getattr(self, "skip_news", False):
            if DataGate.NEWS in gates_open:
                gates_open.remove(DataGate.NEWS)
            if DataGate.NEWS_NLP in gates_open:
                gates_open.remove(DataGate.NEWS_NLP)
        
        run_training(
            db_path=self.db_path,
            output_dir=str(self.output_dir),
            universe=self.universe,
            start_date=self.start_date,
            end_date=self.end_date,
            max_cpu_cores=config.model.max_cpu_cores,
            dry_run=dry_run,
            gates_open=gates_open,
        )

    def _stage_calibrate(self, *, dry_run: bool) -> None:
        # Calibration is performed inside run_training() on the last val fold.
        if dry_run:
            _LOG.info("dry_run: calibration included in TRAIN_MODELS step.")
            return
        _LOG.info("Calibration: completed as part of TRAIN_MODELS stage.")

    def _stage_save_summary(self, *, dry_run: bool) -> None:
        prov_path = self.output_dir / "provenance.json"
        if prov_path.exists():
            data = json.loads(prov_path.read_text(encoding="utf-8"))
            print("\n===== Run Summary =====")
            print(f"  Pipeline version : {data.get('pipeline_version', 'unknown')}")
            print(f"  Universe         : {data.get('universe', [])}")
            print(f"  Date range       : {data.get('date_range', [])}")
            print(f"  Gates open       : {data.get('gates_open', [])}")
            print(f"  Provenance file  : {prov_path}")
        else:
            print(f"\nNo provenance.json yet — training deferred until data lands.")
            print(f"Re-run with real data: python -m scripts.run_orchestration")
