"""Credential selection and refresh-token persistence.

The refresh token is written atomically (temp file in the same directory,
fsync, then ``os.replace``). A crash or power loss partway through a write
would otherwise leave a truncated token, which fails on the next start and
forces a full re-authentication -- exactly when the bridge is trying to come
back up unattended.

The token is registered with the log scrubber the moment it is read or
issued, so it cannot appear in a log line or a traceback.
"""

from __future__ import annotations

import logging
import os
import stat
import tempfile
from pathlib import Path
from typing import Any

from ..logging_setup import SECRETS

__all__ = ["TokenStore", "AuthError", "build_logon_credentials"]

log = logging.getLogger(__name__)

#: Owner read/write only.
_SECRET_MODE = stat.S_IRUSR | stat.S_IWUSR


class AuthError(Exception):
    """Authentication could not be performed with the configured credentials."""


class TokenStore:
    """Reads and atomically writes the Zello refresh token."""

    def __init__(self, path: str | os.PathLike[str] | None) -> None:
        self.path = Path(path) if path is not None else None
        self._token: str | None = None

    @property
    def token(self) -> str | None:
        return self._token

    def load(self) -> str | None:
        """Read a previously persisted refresh token, if any."""
        if self.path is None or not self.path.exists():
            return None
        try:
            value = self.path.read_text(encoding="utf-8").strip()
        except OSError as e:
            log.warning("cannot read refresh token file %s: %s", self.path, e)
            return None

        if not value:
            return None

        self._token = value
        SECRETS.add(value)
        log.info("loaded refresh token from %s", self.path)
        return value

    def save(self, token: str) -> None:
        """Persist a refresh token atomically with owner-only permissions."""
        if not token:
            return

        self._token = token
        SECRETS.add(token)

        if self.path is None:
            return

        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            log.warning("cannot create directory for %s: %s", self.path, e)
            return

        # mkstemp must be inside the guard: a read-only or wrongly-owned state
        # directory would otherwise raise straight out of a reconnect, which
        # is exactly when the bridge is trying to recover unattended. Losing
        # token persistence degrades to a full re-auth; crashing does not.
        try:
            fd, tmp_name = tempfile.mkstemp(
                dir=str(self.path.parent), prefix=f".{self.path.name}.", suffix=".tmp"
            )
        except OSError as e:
            log.warning("cannot create temp file for %s: %s", self.path, e)
            return

        tmp = Path(tmp_name)
        try:
            os.fchmod(fd, _SECRET_MODE)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(token)
                f.flush()
                os.fsync(f.fileno())

            os.replace(tmp, self.path)

            # fsync the directory so the rename itself is durable.
            try:
                dir_fd = os.open(str(self.path.parent), os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except OSError:
                pass

            log.debug("persisted refresh token to %s", self.path)
        except OSError as e:
            log.warning("cannot write refresh token to %s: %s", self.path, e)
            tmp.unlink(missing_ok=True)

    def clear(self) -> None:
        """Forget the token after the server rejects it."""
        self._token = None
        if self.path is not None:
            try:
                self.path.unlink(missing_ok=True)
            except OSError as e:
                log.warning("cannot remove %s: %s", self.path, e)


def build_logon_credentials(cfg: Any, store: TokenStore, *, use_refresh: bool) -> dict[str, Any]:
    """Choose which credentials to present on this logon attempt.

    ``refresh_token`` substitutes for ``auth_token`` -- the *application*
    credential -- and for nothing else. ``username``/``password`` are the
    *user* credential and must be sent either way.

    Getting this wrong is not subtle at the server: sending a refresh token
    alone is rejected with "invalid username", and sending it with a username
    but no password is rejected with "no permission". Both were observed live
    before this was corrected.
    """
    creds: dict[str, Any] = {}

    if use_refresh and store.token:
        creds["refresh_token"] = store.token
    elif cfg.zello.auth_token is not None:
        creds["auth_token"] = cfg.zello.auth_token.get_secret_value()

    if cfg.zello.username:
        creds["username"] = cfg.zello.username
    if cfg.zello.password is not None:
        creds["password"] = cfg.zello.password.get_secret_value()

    if "refresh_token" not in creds and "auth_token" not in creds:
        raise AuthError(
            "no application credential: set zello.auth_token (or have a stored "
            "refresh token) -- username/password alone is not sufficient"
        )
    return creds
