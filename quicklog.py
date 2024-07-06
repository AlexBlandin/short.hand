"""
Quick setup for logging, call init() and all good.

Copyright 2020 Alex Blandin
"""

import logging
import logging.config
import logging.handlers
from collections.abc import Callable
from pathlib import Path


def filter_maker(level: str) -> Callable[..., bool]:
  """Creates filters for logging to ignore, say, WARNING but not ERROR."""
  lvl = getattr(logging, level)

  def fltr(record: logging.LogRecord) -> bool:
    return record.levelno <= lvl

  return fltr


LOCAL = Path(__file__).parent

LOG_CONFIG = {
  "version": 1,
  "disable_existing_loggers": False,
  "formatters": {
    "simple": {"format": "{levelname:<8s} :: {message}", "style": "{"},
    "precise": {"format": "{asctime} {levelname:8s} :: {message}", "style": "{"},
  },
  "filters": {"warnings_and_below": {"()": "quicklog.filter_maker", "level": "WARNING"}},
  "handlers": {
    "stdout": {
      "class": "logging.StreamHandler",
      "level": "INFO",
      "formatter": "simple",
      "stream": "ext://sys.stdout",
      "filters": ["warnings_and_below"],
    },
    "stderr": {"class": "logging.StreamHandler", "level": "ERROR", "formatter": "simple", "stream": "ext://sys.stderr"},
    "file": {
      "class": "logging.handlers.RotatingFileHandler",
      "formatter": "precise",
      "filename": LOCAL / "debug.log",
      "level": "DEBUG",
      "maxBytes": 10**6,
      "backupCount": 5,
      "encoding": "utf-8",
    },
  },
  "root": {"level": "DEBUG", "handlers": ["stderr", "stdout", "file"]},
}


def init() -> None:
  """Just call this once and logging should be configured."""
  logging.Formatter.default_time_format = "%Y-%m-%d-%H-%M-%S"
  logging.config.dictConfig(LOG_CONFIG)
  logging.debug("Logging initialised")
