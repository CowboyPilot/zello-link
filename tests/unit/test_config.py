"""Config schema, environment expansion, and cross-section validation."""

from __future__ import annotations

import copy
import textwrap
from typing import Any

import pytest
import yaml

from zello_link.config import (
    CONFIG_VERSION,
    ConfigError,
    expand_env,
    load_config,
)

MINIMAL = """
config_version: 2
instance:
  name: test-bridge
zello:
  channel: "Test Channel"
  username: "bridge"
  auth_token: "tok-abcdef"
sound:
  input_device: "AIOC Audio"
  output_device: "AIOC Audio"
ptt:
  mode: "serial"
  tty_device: "/dev/serial/by-id/usb-AIOC"
"""

BASE: dict[str, Any] = yaml.safe_load(MINIMAL)


def write(tmp_path, text, name="bridge.yaml"):
    p = tmp_path / name
    p.write_text(textwrap.dedent(text))
    return p


def _merge(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def write_cfg(tmp_path, overrides: dict | None = None, *, drop: list[str] | None = None):
    """Write the minimal config with a deep-merged override tree.

    Deep-merging matters: appending a second ``sound:`` block to the YAML
    text would be a duplicate key, which the loader rejects outright.
    """
    data = copy.deepcopy(BASE)
    if overrides:
        _merge(data, overrides)
    for path in drop or []:
        section, _, key = path.partition(".")
        if key:
            data.get(section, {}).pop(key, None)
        else:
            data.pop(section, None)
    p = tmp_path / "bridge.yaml"
    p.write_text(yaml.safe_dump(data))
    return p


class TestLoading:
    def test_minimal_config_loads(self, tmp_path):
        cfg = load_config(write(tmp_path, MINIMAL))
        assert cfg.instance.name == "test-bridge"
        assert cfg.zello.channel == "Test Channel"
        assert cfg.ptt.tty_device == "/dev/serial/by-id/usb-AIOC"

    def test_defaults_are_applied(self, tmp_path):
        cfg = load_config(write(tmp_path, MINIMAL))
        assert cfg.opus.sample_rate == 16000
        assert cfg.opus.frame_ms == 20
        assert cfg.cos.mode == "internal_audio"
        assert cfg.bridge.arbitration == "first_talker_wins"
        assert cfg.bridge.queue_incoming_zello is False

    def test_missing_file(self, tmp_path):
        with pytest.raises(ConfigError, match="cannot read"):
            load_config(tmp_path / "nope.yaml")

    def test_malformed_yaml(self, tmp_path):
        with pytest.raises(ConfigError, match="cannot parse"):
            load_config(write(tmp_path, "instance: {unclosed"))

    def test_non_mapping(self, tmp_path):
        with pytest.raises(ConfigError, match="mapping"):
            load_config(write(tmp_path, "- a\n- b\n"))

    def test_unknown_key_is_rejected(self, tmp_path):
        """A typo must fail at startup, not silently do nothing."""
        with pytest.raises(ConfigError):
            load_config(write_cfg(tmp_path, {"cos": {"hang_tyme": 400}}))


class TestDuplicateKeys:
    """A repeated section must not silently override the earlier one."""

    def test_duplicate_section_rejected(self, tmp_path):
        text = MINIMAL + '\nsound:\n  input_device: "Other"\n'
        with pytest.raises(ConfigError, match="duplicate key"):
            load_config(write(tmp_path, text))

    def test_duplicate_scalar_rejected(self, tmp_path):
        text = MINIMAL.replace(
            '  name: test-bridge', '  name: test-bridge\n  name: other-bridge'
        )
        with pytest.raises(ConfigError, match="duplicate key"):
            load_config(write(tmp_path, text))

    def test_error_names_the_key_and_line(self, tmp_path):
        text = MINIMAL + '\nsound:\n  input_device: "Other"\n'
        with pytest.raises(ConfigError, match="'sound'"):
            load_config(write(tmp_path, text))


class TestConfigVersion:
    def test_current_version_accepted(self, tmp_path):
        assert load_config(write(tmp_path, MINIMAL)).config_version == CONFIG_VERSION

    def test_future_version_rejected(self, tmp_path):
        with pytest.raises(ConfigError, match="not supported"):
            load_config(write_cfg(tmp_path, {"config_version": 99}))


class TestEnvExpansion:
    def test_expands_variable(self, monkeypatch):
        monkeypatch.setenv("MY_SECRET", "hunter2000")
        assert expand_env("${MY_SECRET}") == "hunter2000"

    def test_expands_nested(self, monkeypatch):
        monkeypatch.setenv("A", "1")
        assert expand_env({"x": ["${A}", {"y": "${A}"}]}) == {"x": ["1", {"y": "1"}]}

    def test_unset_variable_is_an_error(self):
        """Refusing to start beats authenticating with an empty password."""
        with pytest.raises(ConfigError, match="not set"):
            expand_env("${DEFINITELY_NOT_SET_12345}")

    def test_leaves_plain_strings(self):
        assert expand_env("no vars here") == "no vars here"

    def test_partial_interpolation(self, monkeypatch):
        monkeypatch.setenv("HOST", "zello.io")
        assert expand_env("wss://${HOST}/ws") == "wss://zello.io/ws"

    def test_password_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ZP", "s3cr3t-value")
        p = write_cfg(tmp_path, {"zello": {"password": "${ZP}"}})
        assert load_config(p).zello.password.get_secret_value() == "s3cr3t-value"


class TestSecretsAreOpaque:
    def test_secret_not_in_repr(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ZP", "super-secret-password")
        cfg = load_config(write_cfg(tmp_path, {"zello": {"password": "${ZP}"}}))
        assert "super-secret-password" not in repr(cfg)
        assert "super-secret-password" not in str(cfg)

    def test_secret_not_in_model_dump(self, tmp_path):
        cfg = load_config(write(tmp_path, MINIMAL))
        assert "tok-abcdef" not in str(cfg.model_dump())


class TestValidationRanges:
    @pytest.mark.parametrize("value", [-97.0, 0.1, 5.0])
    def test_threshold_out_of_range(self, tmp_path, value):
        with pytest.raises(ConfigError):
            load_config(write_cfg(tmp_path, {"cos": {"threshold_dbfs": value}}))

    @pytest.mark.parametrize("value", [-96.0, -38.0, 0.0])
    def test_threshold_in_range(self, tmp_path, value):
        load_config(write_cfg(tmp_path, {"cos": {"threshold_dbfs": value}}))

    def test_hang_ms_over_5000_rejected(self, tmp_path):
        with pytest.raises(ConfigError):
            load_config(write_cfg(tmp_path, {"cos": {"hang_ms": 5001}}))

    def test_hang_ms_at_limit_accepted(self, tmp_path):
        load_config(write_cfg(tmp_path, {"cos": {"hang_ms": 5000}}))

    def test_max_tx_must_be_positive(self, tmp_path):
        with pytest.raises(ConfigError):
            load_config(write_cfg(tmp_path, {"ptt": {"max_tx_s": 0}}))

    def test_plaintext_websocket_rejected(self, tmp_path):
        with pytest.raises(ConfigError, match="wss"):
            load_config(write_cfg(tmp_path, {"zello": {"server": "ws://zello.io/ws"}}))

    def test_invalid_opus_rate_rejected(self, tmp_path):
        with pytest.raises(ConfigError, match="sample_rate"):
            load_config(write_cfg(tmp_path, {"opus": {"sample_rate": 44100}}))

    def test_frames_per_packet_limited_to_api_values(self, tmp_path):
        """API.md documents 1 or 2 only."""
        with pytest.raises(ConfigError):
            load_config(write_cfg(tmp_path, {"opus": {"frames_per_packet": 3}}))


class TestRequiredCombinations:
    def test_credential_required(self, tmp_path):
        with pytest.raises(ConfigError, match="auth_token or password"):
            load_config(write_cfg(tmp_path, drop=["zello.auth_token"]))

    def test_consumer_network_requires_auth_token(self, tmp_path):
        """Verified live: password-only logon returns 'not enough params'."""
        with pytest.raises(ConfigError, match="auth_token is required"):
            load_config(
                write_cfg(tmp_path, {"zello": {"password": "pw-value"}},
                          drop=["zello.auth_token"])
            )

    def test_auth_token_error_points_at_the_developer_portal(self, tmp_path):
        with pytest.raises(ConfigError, match="developers.zello.com"):
            load_config(
                write_cfg(tmp_path, {"zello": {"password": "pw-value"}},
                          drop=["zello.auth_token"])
            )

    def test_zello_work_may_use_password_only(self, tmp_path):
        """Zello Work authenticates differently; the rule is consumer-only."""
        load_config(
            write_cfg(
                tmp_path,
                {"zello": {"server": "wss://acme.zellowork.com/ws", "password": "pw-value"}},
                drop=["zello.auth_token"],
            )
        )

    def test_serial_ptt_needs_tty(self, tmp_path):
        with pytest.raises(ConfigError, match="tty_device"):
            load_config(write_cfg(tmp_path, drop=["ptt.tty_device"]))

    def test_hid_ptt_needs_hid_device(self, tmp_path):
        with pytest.raises(ConfigError, match="hid_device"):
            load_config(write_cfg(tmp_path, {"ptt": {"mode": "cm108_hid"}}))

    def test_aioc_cos_needs_hid_device(self, tmp_path):
        with pytest.raises(ConfigError, match="hid_device"):
            load_config(write_cfg(tmp_path, {"cos": {"mode": "aioc_virtual"}}))

    def test_rf_to_zello_with_cos_disabled_rejected(self, tmp_path):
        with pytest.raises(ConfigError, match="no way to detect"):
            load_config(
                write_cfg(tmp_path, {"cos": {"mode": "disabled"}, "bridge": {"rf_to_zello": True}})
            )

    def test_cos_disabled_is_ok_when_rf_to_zello_off(self, tmp_path):
        load_config(
            write_cfg(tmp_path, {"cos": {"mode": "disabled"}, "bridge": {"rf_to_zello": False}})
        )

    def test_block_must_match_opus_frame(self, tmp_path):
        with pytest.raises(ConfigError, match="block_ms"):
            load_config(write_cfg(tmp_path, {"opus": {"frame_ms": 20}, "sound": {"block_ms": 40}}))

    def test_no_default_sound_card(self, tmp_path):
        """The bridge must never guess a sound device."""
        with pytest.raises(ConfigError):
            load_config(write_cfg(tmp_path, drop=["sound.input_device", "sound.output_device"]))

    def test_zello_to_rf_needs_output_device(self, tmp_path):
        with pytest.raises(ConfigError, match="output_device"):
            load_config(write_cfg(tmp_path, drop=["sound.output_device"]))

    def test_internal_cos_needs_input_device(self, tmp_path):
        with pytest.raises(ConfigError, match="input_device"):
            load_config(
                write_cfg(tmp_path, {"bridge": {"rf_to_zello": True}}, drop=["sound.input_device"])
            )

    def test_both_directions_disabled_rejected(self, tmp_path):
        with pytest.raises(ConfigError, match="at least one"):
            load_config(
                write_cfg(tmp_path, {"bridge": {"rf_to_zello": False, "zello_to_rf": False}})
            )

    def test_jitter_max_must_exceed_target(self, tmp_path):
        with pytest.raises(ConfigError, match="jitter_max_ms"):
            load_config(write_cfg(tmp_path, {"sound": {"jitter_ms": 500, "jitter_max_ms": 100}}))

    def test_reconnect_max_must_exceed_initial(self, tmp_path):
        with pytest.raises(ConfigError, match="reconnect_max_s"):
            load_config(
                write_cfg(tmp_path, {"zello": {"reconnect_initial_s": 30, "reconnect_max_s": 5}})
            )


class TestGainNaming:
    def test_gains_are_radio_centric(self, tmp_path):
        """rx_gain_db = radio receive audio in; tx_gain_db = audio out to the radio."""
        cfg = load_config(write_cfg(tmp_path, {"sound": {"rx_gain_db": 6.0, "tx_gain_db": -3.0}}))
        assert cfg.sound.rx_gain_db == 6.0
        assert cfg.sound.tx_gain_db == -3.0

    def test_gain_defaults_to_unity(self, tmp_path):
        cfg = load_config(write(tmp_path, MINIMAL))
        assert cfg.sound.rx_gain_db == 0.0
        assert cfg.sound.tx_gain_db == 0.0

    def test_sound_card_centric_names_are_rejected(self, tmp_path):
        """input_gain_db/output_gain_db were ambiguous about RF direction."""
        with pytest.raises(ConfigError):
            load_config(write_cfg(tmp_path, {"sound": {"input_gain_db": 6.0}}))

    def test_absurd_gain_rejected(self, tmp_path):
        with pytest.raises(ConfigError):
            load_config(write_cfg(tmp_path, {"sound": {"rx_gain_db": 60.0}}))


class TestWarnings:
    def test_threshold_below_floor_warns(self, tmp_path):
        cfg = load_config(write_cfg(tmp_path, {"cos": {"threshold_dbfs": -95.0}}))
        assert any("never be crossed" in w for w in cfg.warnings())

    def test_resample_warns(self, tmp_path):
        cfg = load_config(write_cfg(tmp_path, {"sound": {"sample_rate": 48000}}))
        assert any("resampler" in w for w in cfg.warnings())

    def test_latency_budget_exceeded_warns(self, tmp_path):
        cfg = load_config(
            write_cfg(
                tmp_path,
                {
                    "sound": {"jitter_ms": 400, "jitter_max_ms": 800},
                    "bridge": {"latency_budget_ms": 200},
                },
            )
        )
        assert any("latency" in w for w in cfg.warnings())

    def test_fractional_block_warns(self, tmp_path):
        """20 ms at 11025 Hz is 220.5 samples, so blocks get truncated."""
        cfg = load_config(write_cfg(tmp_path, {"sound": {"sample_rate": 11025}}))
        assert any("not a whole number" in w for w in cfg.warnings())

    @pytest.mark.parametrize("rate", [8000, 12000, 16000, 24000, 32000, 44100, 48000])
    def test_whole_block_rates_do_not_warn(self, tmp_path, rate):
        cfg = load_config(write_cfg(tmp_path, {"sound": {"sample_rate": rate}}))
        assert not any("not a whole number" in w for w in cfg.warnings())

    def test_ptt_none_warns(self, tmp_path):
        cfg = load_config(write_cfg(tmp_path, {"ptt": {"mode": "none"}}))
        assert any("never be keyed" in w for w in cfg.warnings())

    def test_password_without_refresh_file_warns(self, tmp_path):
        cfg = load_config(write_cfg(tmp_path, {"zello": {"password": "pw-value"}}))
        assert any("refresh_token_file" in w for w in cfg.warnings())

    def test_clean_config_has_no_warnings(self, tmp_path):
        cfg = load_config(
            write_cfg(tmp_path, {"zello": {"refresh_token_file": "/tmp/x.refresh"}})
        )
        assert cfg.warnings() == []


class TestDeviceIndexWarning:
    """Numeric indices are not stable across a USB unplug.

    Observed live: with the AIOC gone, index 0/1 resolved to the host's
    built-in speakers and microphone. Automatic hardware recovery makes this
    reachable, so it must be flagged.
    """

    @pytest.mark.parametrize("selector", [0, 1, "2"])
    def test_numeric_index_warns(self, tmp_path, selector):
        cfg = load_config(write_cfg(tmp_path, {"sound": {"input_device": selector}}))
        assert any("not stable" in w.lower() for w in cfg.warnings())

    def test_warning_names_the_field(self, tmp_path):
        cfg = load_config(write_cfg(tmp_path, {"sound": {"output_device": 3}}))
        assert any("sound.output_device" in w for w in cfg.warnings())

    def test_warning_explains_the_consequence(self, tmp_path):
        cfg = load_config(write_cfg(tmp_path, {"sound": {"input_device": 1}}))
        text = " ".join(cfg.warnings())
        assert "built-in" in text and "recovery" in text

    def test_device_names_do_not_warn(self, tmp_path):
        cfg = load_config(write_cfg(tmp_path, {"sound": {"input_device": "AIOC Audio"}}))
        assert not any("not stable" in w.lower() for w in cfg.warnings())

    def test_both_devices_warn_independently(self, tmp_path):
        cfg = load_config(
            write_cfg(tmp_path, {"sound": {"input_device": 1, "output_device": 0}})
        )
        warns = [w for w in cfg.warnings() if "not stable" in w.lower()]
        assert len(warns) == 2


class TestMinTxMs:
    def test_defaults_to_disabled(self, tmp_path):
        assert load_config(write(tmp_path, MINIMAL)).cos.min_tx_ms == 0

    def test_accepted_when_above_hang(self, tmp_path):
        cfg = load_config(
            write_cfg(tmp_path, {"cos": {"hang_ms": 200, "min_tx_ms": 800}})
        )
        assert cfg.cos.min_tx_ms == 800

    def test_rejected_when_not_above_hang(self, tmp_path):
        """A floor at or below the hang can never be the binding constraint."""
        with pytest.raises(ConfigError, match="must exceed cos.hang_ms"):
            load_config(write_cfg(tmp_path, {"cos": {"hang_ms": 450, "min_tx_ms": 300}}))

    def test_rejected_when_equal_to_hang(self, tmp_path):
        with pytest.raises(ConfigError, match="must exceed cos.hang_ms"):
            load_config(write_cfg(tmp_path, {"cos": {"hang_ms": 400, "min_tx_ms": 400}}))

    def test_zero_is_always_allowed(self, tmp_path):
        load_config(write_cfg(tmp_path, {"cos": {"hang_ms": 450, "min_tx_ms": 0}}))

    def test_upper_bound_enforced(self, tmp_path):
        with pytest.raises(ConfigError):
            load_config(write_cfg(tmp_path, {"cos": {"min_tx_ms": 10001}}))
