"""
my quicklog shorthand.

Use:
```py
from quicklog import get_logger

logger = get_logger(__name__)
```

And then regular `logger.info()` is ready!

Copyright 2020 Alex Blandin
"""

import logging
import logging.config
import logging.handlers
import os
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path


def filter_maker(level: str) -> Callable[..., bool]:
  """Creates filters for logging to ignore, say, WARNING but not ERROR."""
  lvl = getattr(logging, level)

  def fltr(record: logging.LogRecord) -> bool:
    return record.levelno <= lvl

  return fltr


LOCAL = Path(__file__).parent

# Track if init() has been called to make it idempotent
_initialized = False

# Generate unique log file suffix once per process (timestamp_PID)
# Format: 2025-11-10-21-55-36_12345
_log_suffix = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{os.getpid()}"


def _get_log_config() -> dict:
  """Generate log configuration with unique timestamped filenames.

  Each process gets unique log files: dashboard_2025-11-10-21-55-36_12345.log
  Format: {type}_{timestamp}_{pid}.log
  This eliminates file locking issues and enables parallel execution.
  """
  return {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
      "simple": {"format": "{levelname:<8s} :: {message}", "style": "{"},
      "precise": {"format": "{asctime} {levelname:<8s} :: {name} :: {message}", "style": "{"},
    },
    "filters": {"warnings_and_below": {"()": "logs.quicklog.filter_maker", "level": "WARNING"}},
    "handlers": {
      "stdout": {
        "class": "logging.StreamHandler",
        "level": "INFO",
        "formatter": "simple",
        "stream": "ext://sys.stdout",
        "filters": ["warnings_and_below"],
      },
      "stderr": {
        "class": "logging.StreamHandler",
        "level": "ERROR",
        "formatter": "simple",
        "stream": "ext://sys.stderr",
      },
      "dashboard_file": {
        "class": "logging.FileHandler",
        "formatter": "precise",
        "filename": str(LOCAL / f"dashboard_{_log_suffix}.log"),
        "level": "INFO",
        "mode": "w",
        "encoding": "utf-8",
      },
      "debug_file": {
        "class": "logging.FileHandler",
        "formatter": "precise",
        "filename": str(LOCAL / f"debug_{_log_suffix}.log"),
        "level": "DEBUG",
        "mode": "w",
        "encoding": "utf-8",
      },
      "error_file": {
        "class": "logging.FileHandler",
        "formatter": "precise",
        "filename": str(LOCAL / f"error_{_log_suffix}.log"),
        "level": "ERROR",
        "mode": "w",
        "encoding": "utf-8",
      },
    },
    "root": {"level": "DEBUG", "handlers": ["stderr", "stdout", "dashboard_file", "debug_file", "error_file"]},
  }


def handle_uncaught_exception(exc_type, exc_value, exc_traceback):  # noqa: ANN001
  """Log uncaught exceptions to the logging system.

  This function is installed as sys.excepthook to ensure all uncaught exceptions
  are captured in log files (error.log, debug.log) instead of only appearing in
  terminal output.

  Args:
    exc_type: The type of the exception
    exc_value: The exception instance
    exc_traceback: The traceback object
  """
  # Don't log KeyboardInterrupt (Ctrl+C) - let it terminate cleanly
  if issubclass(exc_type, KeyboardInterrupt):
    sys.__excepthook__(exc_type, exc_value, exc_traceback)
    return

  # Log the exception with full traceback
  logger = logging.getLogger(__name__)
  logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


def init() -> None:
  """Initialize logging with unique timestamped log files.

  Idempotent - safe to call multiple times. Only initializes once per program lifetime.

  Each process creates unique log files with format: {type}_{timestamp}_{pid}.log
    - dashboard_2025-11-10-21-55-36_12345.log
    - debug_2025-11-10-21-55-36_12345.log
    - error_2025-11-10-21-55-36_12345.log

  This eliminates file locking issues and enables parallel execution.
  No rotation needed - each run gets fresh files with inherent recency and PID.

  This also installs a sys.excepthook handler to ensure uncaught exceptions are
  logged to error.log and debug.log, not just printed to terminal stderr.
  """
  global _initialized

  # Return early if already initialized (idempotent)
  if _initialized:
    return

  # Configure logging with unique timestamped filenames
  logging.Formatter.default_time_format = "%Y-%m-%d-%H-%M-%S"
  logging.config.dictConfig(_get_log_config())

  # Install exception hook to capture uncaught exceptions in log files
  sys.excepthook = handle_uncaught_exception

  # Mark as initialized
  _initialized = True

  logging.debug(f"Logging initialised: {_log_suffix}")


def get_logger(name: str = "main") -> logging.Logger:
  """
  Get a logger instance for the given module.

  Use:
  ```py
  from quicklog import get_logger
  logger = get_logger(__name__)
  ```

  Args:
      name: Logger name (typically __name__ from calling module)

  Returns:
      Logger instance configured with quicklog settings
  """
  init()
  return logging.getLogger("main" if name == "__main__" else name)
