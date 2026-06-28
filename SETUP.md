# SETUP.md — New-Device Onboarding & Environment Rebuild

## §1. Prerequisites

| Requirement | Minimum | Notes |
|-------------|---------|-------|
| Python | 3.11+ | Tested on 3.14.5. `tomllib` is stdlib from 3.11. |
| OS | Windows 10/11, macOS, Linux | Path examples are Windows; adjust separators for others. |
| Git | any | To clone / verify the repo |
| GPU | None required | Phase 1–4 run on CPU. Phase 2 FinBERT is slower without GPU but works. Phase 5 LLM narrator uses a template fallback when Ollama/GPU is unavailable. |

---

## §2. Clone / verify the repository

```powershell
# If cloning fresh:
git clone <repo-url> "Quant Fin Project"
cd "Quant Fin Project"

# If already on disk (post-move), just verify key files arrived:
Get-ChildItem -Name DESIGN.md, TASKS.md, CHANGELOG.md, requirements.txt, pyproject.toml
```

All five files must be present before proceeding.

---

## §3. Confirm source files are intact

```powershell
# Should list 12 .py files across src/ subdirectories:
Get-ChildItem src -Recurse -Filter *.py | Select-Object -ExpandProperty FullName

# Should list 7 test files:
Get-ChildItem tests -Filter *.py | Select-Object -ExpandProperty Name
```

Expected src files:
```
src/__init__.py
src/config.py
src/log.py
src/data/__init__.py
src/data/base.py
src/data/historical.py
src/db/__init__.py
src/db/schema.py
src/db/duckdb_client.py
src/db/sqlite_client.py
src/experiment/__init__.py
src/experiment/tracker.py
```

Expected test files:
```
tests/__init__.py  tests/conftest.py
tests/test_config.py            tests/test_schema.py
tests/test_pit_enforcement.py   tests/test_data_interface.py
tests/test_experiment.py
```

---

## §4. Secrets / environment variables

```powershell
# Create your local .env from the template (never commit .env):
Copy-Item .env.example .env
# Then open .env in an editor and fill in real Upstox keys when needed (Phase 1).
```

`.env` is listed in `.gitignore` — it will not be committed.  
`.env.example` stays committed as documentation of which keys are needed.

---

## §5. Rebuild the virtual environment

```powershell
# From the project root:
python -m venv .venv

# Activate (Windows PowerShell):
.\.venv\Scripts\Activate.ps1
# Or activate (Windows cmd):
.\.venv\Scripts\activate.bat

# Install all Phase 0 pinned dependencies:
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Expected output:** `Successfully installed ...` with 19 packages (no errors).  
If a wheel is not available for your Python version, note the error and raise it — do not manually downgrade packages.

### Phase 2+ dependencies (install only when reaching that phase)

```powershell
# Phase 1 — broker + feeds:
pip install requests feedparser websocket-client upstox-python-api

# Phase 2 — ML models:
pip install lightgbm scikit-learn shap transformers tokenizers

# Phase 2 — PyTorch (CPU-only, no GPU required):
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Phase 2 — PyTorch (GPU, CUDA 12.1, RTX 30/40 series):
# pip install torch --index-url https://download.pytorch.org/whl/cu121

# Phase 5 — dashboard:
pip install streamlit
```

---

## §6. Select the interpreter in VS Code

1. Press **Ctrl + Shift + P** → `Python: Select Interpreter`
2. Choose `.\.venv\Scripts\python.exe` (the one under this project folder)
3. Restart the terminal panel if it was open

---

## §7. Verify the environment — run the test suite

```powershell
python -m pytest tests/ -v
```

**Expected result: 73 passed** — this confirms:
- All Phase 0 scaffolding is intact
- DuckDB and SQLite schemas create and are idempotent
- The point-in-time leakage guard works (boundary, timezone-mixing-raises, news `first_knowable_at` vs `published_at`)
- Config loads, validates, and hashes correctly
- Experiment tracker round-trips run records

If anything other than 73 passed appears, treat it as an **environment problem** and diagnose before touching project code.

---

## §8. Rebuild after deleting `.venv`

`.venv` is in `.gitignore` and is never committed. To recreate it on any machine:

```powershell
python -m venv .venv
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pytest tests/ -v   # must be 73 passed
```

---

## §9. Project layout (quick reference)

```
Quant Fin Project/
├── DESIGN.md          Source of truth — read before all else
├── TASKS.md           Phase/subtask checklist
├── CHANGELOG.md       What changed at each step
├── SETUP.md           This file
├── requirements.txt   Pinned deps — rebuild from this alone
├── pyproject.toml     Package discovery, pytest config, ruff/black
├── .env.example       Secrets template — copy to .env, never commit
├── .gitignore         Excludes .env, .venv/, data files, logs
├── config/
│   └── config.toml    Runtime config (universe, horizons, pool size, guardrails)
├── src/
│   ├── config.py      Pydantic v2 config loader (tomllib)
│   ├── log.py         Logging setup
│   ├── data/
│   │   ├── base.py    Abstract DataSource + _enforce_pit (PIT leakage guard)
│   │   └── historical.py  StubHistoricalSource (Phase 0 placeholder)
│   ├── db/
│   │   ├── schema.py       DuckDB + SQLite DDL
│   │   ├── duckdb_client.py  DuckDB context-manager wrapper
│   │   └── sqlite_client.py  SQLite context-manager wrapper
│   └── experiment/
│       └── tracker.py  SQLite run-log (config hash, git commit, metrics)
├── backtester/        Phase 1+ (stub __init__.py only)
├── tests/             73 tests, all green after Phase 0
├── data/              Git-ignored except .gitkeep
└── logs/              Git-ignored except .gitkeep
```
