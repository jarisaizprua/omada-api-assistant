"""Logging: readable in the terminal, structured (JSON lines) for servers.

``LOG_FORMAT=console`` (default) prints coloured, human-friendly lines.
``LOG_FORMAT=json`` prints one JSON object per line, ready for any log collector (journald, Loki,
CloudWatch, ELK...). Both include a request id so every line of one question can be found together.
``LOG_FILE`` additionally writes a rotating file (10 MB x 5).
"""

from __future__ import annotations

import contextvars
import json
import logging
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.logging import RichHandler

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

_STANDARD = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime", "taskName"}


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):  # a caller may pass it explicitly (e.g. after a stream ended)
            record.request_id = request_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD and key != "request_id" and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def log_event(logger: logging.Logger, event: str, message: str, level: int = logging.INFO, **fields) -> None:
    """One structured log line: readable message in the console, ``event`` + fields in JSON."""
    logger.log(level, message, extra={"event": event, **fields})


def setup_logging(level: str = "INFO", fmt: str = "console", log_file: str | Path | None = None) -> None:
    handlers: list[logging.Handler] = []
    if fmt.lower() == "json":
        stream = logging.StreamHandler()
        stream.setFormatter(JsonFormatter())
        handlers.append(stream)
    else:
        handlers.append(RichHandler(rich_tracebacks=True, show_path=False, markup=False))
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(path, maxBytes=10_000_000, backupCount=5, encoding="utf-8")
        file_handler.setFormatter(JsonFormatter())
        handlers.append(file_handler)

    for handler in handlers:
        handler.addFilter(RequestIdFilter())
    logging.basicConfig(
        level=level.upper(),
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
        force=True,
    )
    # Third-party libraries are chatty at INFO.
    for noisy in ("httpx", "httpcore", "chromadb", "openai", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
