"""
logger.py — Structured Logging for the Offline RAG Framework
=============================================================
Provides a single `get_logger()` factory function used by every module
in the project to obtain a consistently configured logger.

Features:
    - Coloured console output (DEBUG=cyan, INFO=green, WARNING=yellow,
      ERROR=red, CRITICAL=bold red) via colorlog
    - Rotating file handler — prevents log files from growing unbounded
    - Structured format that includes timestamp, level, module name,
      and line number for easy debugging
    - Single configuration point — changing log level in config.py
      propagates to every module automatically
    - Thread-safe by design (Python's logging module is thread-safe)
    - No duplicate handlers — calling get_logger() multiple times for
      the same name is safe

Usage (in any module):
    from logger import get_logger
    log = get_logger(__name__)

    log.info("Database loaded successfully.")
    log.warning("Similarity score below threshold: %.3f", score)
    log.error("Ollama server unreachable at %s", host)
    log.debug("Retrieved %d documents in %.2fs", count, elapsed)
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# colorlog adds ANSI colour codes to console output.
# Falls back gracefully to plain logging if not installed.
try:
    import colorlog
    _COLORLOG_AVAILABLE = True
except ImportError:
    _COLORLOG_AVAILABLE = False

# ---------------------------------------------------------------------------
# Import config lazily to avoid circular imports at module load time.
# logger.py is imported very early (before config is fully initialised in
# some test scenarios), so we defer the import inside the factory function.
# ---------------------------------------------------------------------------


def _get_log_level(level_str: str) -> int:
    """
    Convert a log level string to its corresponding logging integer constant.

    Args:
        level_str: One of "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".
                   Case-insensitive.

    Returns:
        The corresponding logging level integer (e.g. logging.INFO = 20).

    Raises:
        ValueError: If an unrecognised level string is provided.
    """
    level_map: dict[str, int] = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    normalised = level_str.strip().upper()
    if normalised not in level_map:
        raise ValueError(
            f"Invalid log level '{level_str}'. "
            f"Valid options: {list(level_map.keys())}"
        )
    return level_map[normalised]


def _build_console_handler(level: int) -> logging.Handler:
    """
    Build a StreamHandler that writes to stdout with colour support.

    Coloured format (when colorlog is available):
        2024-01-15 10:32:01 | INFO     | retriever:42 | Database loaded

    Plain format (fallback):
        2024-01-15 10:32:01 | INFO     | retriever:42 | Database loaded

    Args:
        level: The minimum log level this handler will emit.

    Returns:
        A configured StreamHandler instance.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    if _COLORLOG_AVAILABLE:
        # Colour mapping per log level
        log_colors = {
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        }
        fmt = colorlog.ColoredFormatter(
            fmt=(
                "%(log_color)s%(asctime)s%(reset)s"
                " | %(log_color)s%(levelname)-8s%(reset)s"
                " | %(cyan)s%(name)s:%(lineno)d%(reset)s"
                " | %(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors=log_colors,
            reset=True,
            style="%",
        )
    else:
        # Plain formatter as fallback (identical layout, no colour codes)
        fmt = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    handler.setFormatter(fmt)
    return handler


def _build_file_handler(logs_dir: Path, level: int) -> logging.Handler:
    """
    Build a RotatingFileHandler that writes structured plain-text logs to disk.

    Rotation policy:
        - Maximum file size: 5 MB per file
        - Backup count: 5 files (i.e. up to 25 MB of log history retained)
        - When a file exceeds 5 MB, it is renamed to rag_framework.log.1,
          the previous .1 becomes .2, etc.

    Plain-text (no colour codes) so log files can be read by any tool
    (grep, tail, log aggregators, etc.)

    Args:
        logs_dir: Path object pointing to the logs/ directory.
        level: The minimum log level this handler will emit.

    Returns:
        A configured RotatingFileHandler instance.
    """
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "rag_framework.log"

    handler = RotatingFileHandler(
        filename=str(log_file),
        maxBytes=5 * 1024 * 1024,   # 5 MB
        backupCount=5,
        encoding="utf-8",
        delay=True,                  # File created only when first log is written
    )
    handler.setLevel(level)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(fmt)
    return handler


def get_logger(name: str) -> logging.Logger:
    """
    Factory function — returns a named logger configured for this framework.

    This function is idempotent: calling it multiple times with the same
    `name` returns the same logger without adding duplicate handlers.
    This is important because Python's logging module stores loggers in a
    global registry — if handlers were added unconditionally, every import
    would double the number of handlers, causing duplicate log lines.

    Args:
        name: Logger name, typically passed as `__name__` from the calling
              module. This produces names like:
                  "ingest", "retriever", "llm", "app"
              which appear in the log output for easy source tracing.

    Returns:
        A fully configured logging.Logger instance.

    Example:
        from logger import get_logger
        log = get_logger(__name__)
        log.info("Ingestion pipeline started.")
    """
    # Deferred import to avoid circular dependency at module load time
    from config import config

    log_level: int = _get_log_level(config.LOG_LEVEL)

    logger = logging.getLogger(name)

    # Guard: if this logger already has handlers, it's already configured.
    # Return it immediately to prevent duplicate log lines.
    if logger.handlers:
        return logger

    # Set the logger's own level to the lowest possible so that all handlers
    # can independently filter at their own level if needed.
    logger.setLevel(logging.DEBUG)

    # Prevent log records from bubbling up to the root logger.
    # This avoids duplicate output if the root logger also has handlers
    # (common in Streamlit's runtime environment).
    logger.propagate = False

    # Attach handlers
    logger.addHandler(_build_console_handler(log_level))
    logger.addHandler(_build_file_handler(config.LOGS_DIR, log_level))

    return logger


# ---------------------------------------------------------------------------
# Convenience helpers — used throughout the project for structured events
# ---------------------------------------------------------------------------

def log_section_header(logger: logging.Logger, title: str) -> None:
    """
    Log a visually distinct section header.

    Useful at the start of major pipeline stages (ingestion, retrieval, etc.)
    to make logs easy to scan.

    Args:
        logger: The logger instance to write to.
        title:  The section title string.

    Example output:
        2024-01-15 10:32:01 | INFO     | ingest:88 | ══════════════════════
        2024-01-15 10:32:01 | INFO     | ingest:88 |   INGESTION PIPELINE
        2024-01-15 10:32:01 | INFO     | ingest:88 | ══════════════════════
    """
    border = "═" * 50
    logger.info(border)
    logger.info("  %s", title.upper())
    logger.info(border)


def log_elapsed(logger: logging.Logger, stage: str, elapsed_seconds: float) -> None:
    """
    Log a standardised timing measurement for a pipeline stage.

    Args:
        logger:          The logger instance to write to.
        stage:           Human-readable name of the stage (e.g. "Embedding").
        elapsed_seconds: Wall-clock time in seconds.

    Example output:
        2024-01-15 10:32:05 | INFO | ingest:112 | ⏱  Embedding completed in 3.42s
    """
    logger.info("⏱  %s completed in %.2fs", stage, elapsed_seconds)


def log_retrieval_result(
    logger: logging.Logger,
    query: str,
    num_results: int,
    elapsed_seconds: float,
) -> None:
    """
    Log a standardised retrieval event with query preview and timing.

    Args:
        logger:          The logger instance to write to.
        query:           The user's question (truncated for readability).
        num_results:     Number of documents retrieved from ChromaDB.
        elapsed_seconds: Wall-clock time for the retrieval step.

    Example output:
        2024-01-15 10:33:01 | INFO | retriever:88 | 🔍 Query: "What is rice prod..." → 5 docs in 0.08s
    """
    preview = query[:60] + "..." if len(query) > 60 else query
    logger.info(
        '🔍 Query: "%s" → %d doc(s) in %.3fs',
        preview,
        num_results,
        elapsed_seconds,
    )


def log_llm_inference(
    logger: logging.Logger,
    model: str,
    elapsed_seconds: float,
    token_estimate: int | None = None,
) -> None:
    """
    Log a standardised LLM inference completion event.

    Args:
        logger:          The logger instance to write to.
        model:           Name of the Ollama model used.
        elapsed_seconds: Wall-clock time for the LLM inference step.
        token_estimate:  Approximate token count of the response (optional).

    Example output:
        2024-01-15 10:33:04 | INFO | llm:55 | 🤖 qwen2.5:3b → response in 2.91s (~312 tokens)
    """
    token_str = f" (~{token_estimate} tokens)" if token_estimate else ""
    logger.info(
        "🤖 %s → response in %.2fs%s",
        model,
        elapsed_seconds,
        token_str,
    )
