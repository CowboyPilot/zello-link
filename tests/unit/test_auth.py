"""Refresh-token persistence: atomicity, permissions, and redaction."""

from __future__ import annotations

import os
import stat

import pytest

from zello_link.zello.auth import AuthError, TokenStore, build_logon_credentials


class TestTokenStore:
    def test_save_then_load(self, tmp_path):
        path = tmp_path / "west.refresh"
        TokenStore(path).save("refresh-token-value-1")
        assert TokenStore(path).load() == "refresh-token-value-1"

    def test_load_missing_file(self, tmp_path):
        assert TokenStore(tmp_path / "absent").load() is None

    def test_load_empty_file(self, tmp_path):
        path = tmp_path / "empty"
        path.write_text("")
        assert TokenStore(path).load() is None

    def test_load_strips_whitespace(self, tmp_path):
        path = tmp_path / "t"
        path.write_text("  token-with-space-value \n")
        assert TokenStore(path).load() == "token-with-space-value"

    def test_save_creates_parent_directory(self, tmp_path):
        path = tmp_path / "nested" / "deep" / "t.refresh"
        TokenStore(path).save("token-value-abc")
        assert path.read_text() == "token-value-abc"

    def test_overwrite_replaces_cleanly(self, tmp_path):
        path = tmp_path / "t"
        store = TokenStore(path)
        store.save("first-token-value")
        store.save("second-token-value")
        assert path.read_text() == "second-token-value"

    def test_no_temp_files_left_behind(self, tmp_path):
        """The atomic write must not litter the state directory."""
        path = tmp_path / "t"
        store = TokenStore(path)
        for i in range(5):
            store.save(f"token-value-{i}")
        assert sorted(p.name for p in tmp_path.iterdir()) == ["t"]

    def test_file_is_owner_only(self, tmp_path):
        path = tmp_path / "t"
        TokenStore(path).save("token-value-abc")
        mode = stat.S_IMODE(os.stat(path).st_mode)
        assert mode == 0o600, f"token file is {oct(mode)}, expected 0600"

    def test_none_path_is_memory_only(self):
        store = TokenStore(None)
        store.save("token-value-abc")
        assert store.token == "token-value-abc"
        assert store.load() is None

    def test_empty_token_is_ignored(self, tmp_path):
        path = tmp_path / "t"
        TokenStore(path).save("")
        assert not path.exists()

    def test_clear_removes_the_file(self, tmp_path):
        path = tmp_path / "t"
        store = TokenStore(path)
        store.save("token-value-abc")
        store.clear()
        assert not path.exists()
        assert store.token is None

    def test_clear_when_absent_is_safe(self, tmp_path):
        TokenStore(tmp_path / "absent").clear()

    def test_unwritable_directory_does_not_raise(self, tmp_path):
        """A read-only state dir must not crash a reconnect."""
        target = tmp_path / "ro"
        target.mkdir()
        path = target / "t"
        os.chmod(target, 0o500)
        try:
            TokenStore(path).save("token-value-abc")   # must not raise
        finally:
            os.chmod(target, 0o700)

    def test_saved_token_is_registered_for_redaction(self, tmp_path):
        from zello_link.logging_setup import REDACTED, SECRETS

        TokenStore(tmp_path / "t").save("very-secret-refresh-token")
        assert SECRETS.scrub("rt=very-secret-refresh-token") == f"rt={REDACTED}"

    def test_loaded_token_is_registered_for_redaction(self, tmp_path):
        from zello_link.logging_setup import REDACTED, SECRETS

        path = tmp_path / "t"
        path.write_text("loaded-secret-refresh-token")
        TokenStore(path).load()
        assert SECRETS.scrub("rt=loaded-secret-refresh-token") == f"rt={REDACTED}"


class FakeZelloCfg:
    class zello:
        auth_token = None
        password = None
        username = "bridge-account"


def make_cfg(*, token=None, password=None, username="bridge-account"):
    from pydantic import SecretStr

    class Cfg:
        class zello:
            pass

    Cfg.zello.auth_token = SecretStr(token) if token else None
    Cfg.zello.password = SecretStr(password) if password else None
    Cfg.zello.username = username
    return Cfg


class TestCredentialSelection:
    """refresh_token replaces auth_token (the APP credential) and nothing else.

    Verified live: refresh_token alone -> "invalid username";
    refresh_token + username, no password -> "no permission";
    refresh_token + username + password -> logon succeeds.
    """

    def test_refresh_token_replaces_auth_token(self):
        store = TokenStore(None)
        store.save("stored-refresh-token")
        creds = build_logon_credentials(make_cfg(token="auth-tok"), store, use_refresh=True)
        assert creds["refresh_token"] == "stored-refresh-token"
        assert "auth_token" not in creds, "refresh_token replaces the app credential"

    def test_refresh_token_still_sends_username_and_password(self):
        """The bug: sending refresh_token alone is rejected outright."""
        store = TokenStore(None)
        store.save("stored-refresh-token")
        creds = build_logon_credentials(
            make_cfg(token="auth-tok", password="pw-value"), store, use_refresh=True
        )
        assert creds["username"] == "bridge-account"
        assert creds["password"] == "pw-value"

    def test_falls_back_to_auth_token(self):
        store = TokenStore(None)
        creds = build_logon_credentials(make_cfg(token="auth-tok"), store, use_refresh=True)
        assert creds["auth_token"] == "auth-tok"
        assert "refresh_token" not in creds

    def test_explicit_no_refresh(self):
        store = TokenStore(None)
        store.save("stored-refresh-token")
        creds = build_logon_credentials(make_cfg(token="auth-tok"), store, use_refresh=False)
        assert "refresh_token" not in creds
        assert creds["auth_token"] == "auth-tok"

    def test_includes_username_and_password(self):
        creds = build_logon_credentials(
            make_cfg(token="auth-tok", password="pw-value"), TokenStore(None),
            use_refresh=False,
        )
        assert creds["username"] == "bridge-account"
        assert creds["password"] == "pw-value"

    def test_no_application_credential_raises(self):
        with pytest.raises(AuthError, match="no application credential"):
            build_logon_credentials(
                make_cfg(password="pw-value"), TokenStore(None), use_refresh=False
            )

    def test_secrets_are_unwrapped_for_the_wire(self):
        """SecretStr must be unwrapped here and nowhere else."""
        creds = build_logon_credentials(
            make_cfg(token="auth-tok-value"), TokenStore(None), use_refresh=False
        )
        assert creds["auth_token"] == "auth-tok-value"
        assert not hasattr(creds["auth_token"], "get_secret_value")
