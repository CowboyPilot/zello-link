"""Logging configuration with unconditional secret redaction.

Redaction happens in the *formatter*, not a filter, because the formatter is
the single choke point that sees the fully rendered record -- message, args,
and the formatted traceback of an uncaught exception. A filter that only
inspects ``record.msg`` would miss a token that appears in a stack frame's
repr, which is exactly how secrets escape in practice.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
import threading
from pathlib import Path
from typing import Any, Iterable

from pydantic import SecretStr

__all__ = [
    "REDACTED", "SecretRegistry", "RedactingFormatter", "setup_logging", "bind",
    "StatusLine",
]

REDACTED = "***REDACTED***"

# Below this length a "secret" is too likely to occur as an ordinary substring
# (a password of "1234" would blank out timestamps and stream IDs).
_MIN_REDACTABLE_LEN = 4


class SecretRegistry:
    """Process-wide set of literal strings that must never reach a log sink."""

    def __init__(self) -> None:
        self._secrets: set[str] = set()

    def add(self, value: str | SecretStr | None) -> None:
        if value is None:
            return
        if isinstance(value, SecretStr):
            value = value.get_secret_value()
        if not isinstance(value, str):
            return
        value = value.strip()
        if len(value) >= _MIN_REDACTABLE_LEN:
            self._secrets.add(value)

    def add_all(self, values: Iterable[Any]) -> None:
        for v in values:
            self.add(v)

    def scrub(self, text: str) -> str:
        if not self._secrets or not text:
            return text
        # Longest first, so a token that contains a shorter secret as a
        # substring is replaced whole rather than leaving a fragment behind.
        for secret in sorted(self._secrets, key=len, reverse=True):
            if secret in text:
                text = text.replace(secret, REDACTED)
        return text

    def __len__(self) -> int:
        return len(self._secrets)


#: The registry consulted by every formatter this module installs.
SECRETS = SecretRegistry()


def register_config_secrets(cfg: Any) -> None:
    """Harvest every secret-bearing field from a validated config."""
    z = cfg.zello
    SECRETS.add(z.password)
    SECRETS.add(z.auth_token)
    # The refresh token itself is registered by the auth layer once read or
    # issued -- it is not present in the config file.


class RedactingFormatter(logging.Formatter):
    """Formatter that scrubs registered secrets from the final rendered text."""

    def __init__(self, fmt: str, datefmt: str | None = None, registry: SecretRegistry | None = None):
        super().__init__(fmt=fmt, datefmt=datefmt)
        self._registry = registry if registry is not None else SECRETS

    def format(self, record: logging.LogRecord) -> str:
        return self._registry.scrub(super().format(record))


class _ContextFilter(logging.Filter):
    """Guarantee the structured fields referenced by the format string exist."""

    def __init__(self, instance: str) -> None:
        super().__init__()
        self._instance = instance

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "instance"):
            record.instance = self._instance
        return True


class StatusLine:
    """A single line pinned to the bottom of the terminal.

    Log output and the status line share stderr, so writing a log record
    while the line is displayed would smear the two together. Every write
    goes through here: the line is erased, the record is printed, and the
    line is redrawn underneath it. Scrollback ends up clean, with the meter
    always sitting on the last row.

    Disabled automatically when stderr is not a TTY, so piping to a file or
    running under systemd produces ordinary logs with no escape codes.
    """

    def __init__(self, stream: Any = None) -> None:
        self._stream = stream if stream is not None else sys.stderr
        try:
            self.enabled = bool(self._stream.isatty())
        except Exception:
            self.enabled = False
        self._text = ""
        self._visible = False
        self._lock = threading.RLock()

    def set(self, text: str) -> None:
        """Update and redraw the pinned line."""
        with self._lock:
            self._text = text
            if self.enabled:
                self._write(f"\r\033[2K{text}")
                self._visible = True

    def clear(self) -> None:
        """Erase the line so other output can be written cleanly."""
        with self._lock:
            if self.enabled and self._visible:
                self._write("\r\033[2K")
                self._visible = False

    def redraw(self) -> None:
        with self._lock:
            if self.enabled and self._text and not self._visible:
                self._write(f"\r\033[2K{self._text}")
                self._visible = True

    def finish(self) -> None:
        """Erase the line and drop to a fresh row on shutdown."""
        with self._lock:
            if self.enabled and self._visible:
                self._write("\r\033[2K")
                self._visible = False
            self._text = ""

    def _write(self, s: str) -> None:
        try:
            self._stream.write(s)
            self._stream.flush()
        except Exception:
            self.enabled = False

    @property
    def lock(self) -> threading.RLock:
        return self._lock


class _StatusAwareHandler(logging.StreamHandler):
    """Console handler that yields the bottom row to the status line."""

    def __init__(self, stream: Any, status: StatusLine) -> None:
        super().__init__(stream)
        self._status = status

    def emit(self, record: logging.LogRecord) -> None:
        with self._status.lock:
            self._status.clear()
            super().emit(record)
            self._status.redraw()


_CONSOLE_FMT = "%(asctime)s %(levelname)-5s %(instance)s %(message)s"
_FILE_FMT = "%(asctime)s %(levelname)-5s %(instance)s [%(name)s] %(message)s"
_DATEFMT = "%Y-%m-%dT%H:%M:%S"


def setup_logging(
    cfg: Any, *, console_only: bool = False, status: "StatusLine | None" = None
) -> logging.Logger:
    """Install console/file handlers per config. Idempotent.

    A file handler that cannot be opened degrades to console-only with a
    loud warning rather than aborting. Losing the log file is bad; refusing
    to bridge radio traffic because a log directory is unwritable is worse,
    and the operator is told either way.

    ``console_only`` skips the file handler entirely -- diagnostic modes
    should not need write access to the production log directory.
    """
    register_config_secrets(cfg)

    root = logging.getLogger("zello_dmr_bridge")
    root.setLevel(cfg.instance.log_level)
    root.propagate = False

    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()

    ctx = _ContextFilter(cfg.instance.name)

    if cfg.logging.console:
        ch = (
            _StatusAwareHandler(sys.stderr, status)
            if status is not None
            else logging.StreamHandler(sys.stderr)
        )
        ch.setFormatter(RedactingFormatter(_CONSOLE_FMT, _DATEFMT))
        ch.addFilter(ctx)
        root.addHandler(ch)

    if cfg.logging.file is not None and not console_only:
        path = Path(cfg.logging.file)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.handlers.RotatingFileHandler(
                path,
                maxBytes=cfg.logging.max_bytes,
                backupCount=cfg.logging.backup_count,
                encoding="utf-8",
            )
        except OSError as e:
            root.warning(
                "cannot open log file %s (%s); continuing with console logging only", path, e
            )
        else:
            fh.setFormatter(RedactingFormatter(_FILE_FMT, _DATEFMT))
            fh.addFilter(ctx)
            root.addHandler(fh)

    if not root.handlers:
        root.addHandler(logging.NullHandler())

    return root


def bind(name: str, **context: Any) -> logging.LoggerAdapter:
    """Return a logger carrying fixed structured context.

    Context is rendered as trailing ``key=value`` pairs so operator-facing
    INFO lines stay readable in a terminal while remaining greppable.
    """
    logger = logging.getLogger(f"zello_dmr_bridge.{name}")
    return _ContextAdapter(logger, context)


class _ContextAdapter(logging.LoggerAdapter):
    def process(self, msg: Any, kwargs: Any) -> tuple[Any, Any]:
        extra = {**self.extra, **kwargs.pop("extra", {})}
        # Pull out fields the formatter consumes directly.
        instance = extra.pop("instance", None)
        pairs = " ".join(f"{k}={_fmt(v)}" for k, v in extra.items() if v is not None)
        if pairs:
            msg = f"{msg} {pairs}"
        kwargs["extra"] = {"instance": instance} if instance else {}
        return msg, kwargs


def _fmt(v: Any) -> str:
    s = str(v)
    return f'"{s}"' if " " in s else s
