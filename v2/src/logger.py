"""
Logging setup. One root logger ("stt") with two handlers:
- console: filtered by `console_level` (default WARNING — keeps the live UI clean)
- file: always DEBUG, rotated, written to logs/<date>.log

Modules call get_logger(__name__) to get a namespaced child logger.
"""

import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

_configured = False


def setup_logging(
    log_dir: Path | str | None = "logs",
    file_level: str = "DEBUG",
    console_level: str = "WARNING",
    file_enabled: bool = True,
    console_enabled: bool = True,
) -> logging.Logger:
    """
    Configure the 'stt' logger. Idempotent — calling twice is a no-op.
    """
    global _configured
    root = logging.getLogger("stt")
    if _configured:
        return root

    root.setLevel(logging.DEBUG)
    root.propagate = False
    root.handlers.clear()

    file_lv = _LEVELS.get(file_level.upper(), logging.DEBUG)
    console_lv = _LEVELS.get(console_level.upper(), logging.WARNING)

    if console_enabled:
        ch = logging.StreamHandler(sys.stderr)
        ch.setLevel(console_lv)
        ch.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                              datefmt="%H:%M:%S")
        )
        root.addHandler(ch)

    if file_enabled and log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"stt_{datetime.now().strftime('%Y%m%d')}.log"
        fh = RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5,
            encoding="utf-8",
        )
        fh.setLevel(file_lv)
        fh.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root.addHandler(fh)
        root.debug(f"File logging initialized at {log_file}")

    _configured = True
    return root


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the 'stt' namespace.

    Pass `__name__` from a module — e.g. 'src.transcriber' becomes 'stt.transcriber'.
    """
    short = name.split(".")[-1] if name else "root"
    return logging.getLogger(f"stt.{short}")
