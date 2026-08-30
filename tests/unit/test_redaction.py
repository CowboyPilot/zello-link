"""AT-10: no secret may reach a log sink, including via a traceback."""

from __future__ import annotations

import io
import logging

import pytest
from pydantic import SecretStr

from zello_dmr_bridge.logging_setup import (
    REDACTED,
    RedactingFormatter,
    SecretRegistry,
)


@pytest.fixture
def registry():
    r = SecretRegistry()
    r.add("super-secret-password")
    r.add("refresh-token-abc123")
    return r


@pytest.fixture
def capture(registry):
    """A logger writing to a StringIO through the redacting formatter."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(RedactingFormatter("%(levelname)s %(message)s", registry=registry))

    logger = logging.getLogger(f"redaction_test_{id(stream)}")
    logger.handlers = [handler]
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger, stream


class TestRegistry:
    def test_scrubs_a_known_secret(self, registry):
        assert registry.scrub("token=super-secret-password") == f"token={REDACTED}"

    def test_leaves_other_text_alone(self, registry):
        assert registry.scrub("nothing sensitive") == "nothing sensitive"

    def test_accepts_secretstr(self):
        r = SecretRegistry()
        r.add(SecretStr("wrapped-secret-value"))
        assert "wrapped-secret-value" not in r.scrub("x wrapped-secret-value y")

    def test_ignores_none(self):
        r = SecretRegistry()
        r.add(None)
        assert len(r) == 0

    def test_ignores_dangerously_short_values(self):
        """A 3-char password would blank out unrelated substrings everywhere."""
        r = SecretRegistry()
        r.add("abc")
        assert len(r) == 0
        assert r.scrub("abcdef") == "abcdef"

    def test_longest_secret_wins(self):
        """A shorter secret nested in a longer one must not leave a fragment."""
        r = SecretRegistry()
        r.add("token-abc")
        r.add("token-abc-extended-value")
        out = r.scrub("t=token-abc-extended-value")
        assert "token-abc" not in out
        assert out == f"t={REDACTED}"

    def test_multiple_occurrences(self, registry):
        out = registry.scrub("a super-secret-password b super-secret-password c")
        assert "super-secret-password" not in out
        assert out.count(REDACTED) == 2

    def test_empty_registry_is_a_passthrough(self):
        assert SecretRegistry().scrub("anything") == "anything"


class TestLogRedaction:
    def test_secret_in_message(self, capture):
        logger, stream = capture
        logger.info("logging on with super-secret-password")
        assert "super-secret-password" not in stream.getvalue()
        assert REDACTED in stream.getvalue()

    def test_secret_in_lazy_format_args(self, capture):
        """%-style args are rendered by the formatter, so they must be scrubbed too."""
        logger, stream = capture
        logger.info("token=%s", "refresh-token-abc123")
        assert "refresh-token-abc123" not in stream.getvalue()

    def test_secret_in_dict_arg(self, capture):
        logger, stream = capture
        logger.info("payload=%s", {"password": "super-secret-password"})
        assert "super-secret-password" not in stream.getvalue()

    def test_secret_in_traceback(self, capture):
        """The reason redaction lives in the formatter, not a filter."""
        logger, stream = capture
        try:
            raise ValueError("auth failed for token refresh-token-abc123")
        except ValueError:
            logger.exception("logon failed")
        out = stream.getvalue()
        assert "refresh-token-abc123" not in out
        assert "Traceback" in out
        assert REDACTED in out

    def test_secret_in_nested_exception_context(self, capture):
        logger, stream = capture
        try:
            try:
                raise ValueError("inner super-secret-password")
            except ValueError as e:
                raise RuntimeError("outer") from e
        except RuntimeError:
            logger.exception("failed")
        assert "super-secret-password" not in stream.getvalue()

    def test_debug_level_also_redacted(self, capture):
        logger, stream = capture
        logger.debug("dumping config: password=super-secret-password")
        assert "super-secret-password" not in stream.getvalue()

    def test_non_secret_text_survives(self, capture):
        logger, stream = capture
        logger.info("Zello connected channel=\"Event Security\"")
        assert "Event Security" in stream.getvalue()


class TestConfigIntegration:
    def test_config_secrets_are_registered(self, tmp_path, monkeypatch):
        import yaml

        from zello_dmr_bridge.config import load_config
        from zello_dmr_bridge.logging_setup import SECRETS, register_config_secrets

        monkeypatch.setenv("ZP", "config-file-password-value")
        cfg_path = tmp_path / "b.yaml"
        cfg_path.write_text(
            yaml.safe_dump(
                {
                    "config_version": 1,
                    "instance": {"name": "t"},
                    "zello": {
                        "channel": "c",
                        "username": "u",
                        "password": "${ZP}",
                        "auth_token": "auth-token-value-xyz",
                    },
                    "sound": {"input_device": "d", "output_device": "d"},
                    "ptt": {"mode": "none"},
                }
            )
        )
        cfg = load_config(cfg_path)
        register_config_secrets(cfg)

        assert SECRETS.scrub("p=config-file-password-value") == f"p={REDACTED}"
        assert SECRETS.scrub("a=auth-token-value-xyz") == f"a={REDACTED}"
