"""Logging configuration.  Call configure_logging() once at application startup."""
from __future__ import annotations

import logging
import logging.config
import logging.handlers
from pathlib import Path
from typing import Any


def configure_logging(level: str = "INFO", log_dir: Path | None = None) -> None:
    """Set up root logger with a console handler and an optional rotating file handler."""
    handlers: dict[str, Any] = {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        }
    }
    active_handlers = ["console"]

    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        handlers["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "standard",
            "filename": str(Path(log_dir) / "quant_fin.log"),
            "maxBytes": 10 * 1024 * 1024,  # 10 MB per file
            "backupCount": 5,
            "encoding": "utf-8",
        }
        active_handlers.append("file")

    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s %(levelname)-8s %(name)s — %(message)s",
                "datefmt": "%Y-%m-%dT%H:%M:%S",
            }
        },
        "handlers": handlers,
        "root": {
            "level": level.upper(),
            "handlers": active_handlers,
        },
    })
