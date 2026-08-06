"""Central logging configuration for RepoBrain."""
from __future__ import annotations

import logging
import sys

from repobrain.config import LoggingConfig


def configure_logging(cfg: LoggingConfig) -> None:
    level = getattr(logging, cfg.level.upper(), logging.INFO)
    root = logging.getLogger("repobrain")
    root.setLevel(level)
    root.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if cfg.file:
        file_handler = logging.FileHandler(cfg.file)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"repobrain.{name}")
