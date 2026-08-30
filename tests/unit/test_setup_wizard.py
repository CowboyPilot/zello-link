"""Interactive device setup: the config edit must be surgical.

The whole value of the example config is its comments -- the calibration
notes, the warnings about PTT being live. A yaml.safe_dump round trip would
silently delete every one of them, so edits replace exactly one line and
leave the file otherwise byte-identical.
"""

from __future__ import annotations

import pytest

from zello_link.audio.devices import AudioDevice
from zello_link.diagnostics.setup_wizard import (
    _device_value,
    detect_cos_choices,
    set_config_value,
    yaml_scalar,
)

SAMPLE = '''# leading comment
config_version: 1

sound:
  # which card the radio is plugged into
  input_device: "Old Input"     # trailing note
  output_device: 3
  sample_rate: 16000

ptt:
  mode: "serial"
  tty_device: "/dev/ttyACM0"

cos:
  mode: "internal_audio"
  threshold_dbfs: -50.0
'''


class TestSurgicalEdit:
    def test_replaces_the_value(self):
        out = set_config_value(SAMPLE, "sound", "input_device", '"New Input"')
        assert 'input_device: "New Input"' in out
        assert "Old Input" not in out

    def test_preserves_the_trailing_comment(self):
        out = set_config_value(SAMPLE, "sound", "input_device", '"New Input"')
        assert "# trailing note" in out

    def test_preserves_every_other_line(self):
        out = set_config_value(SAMPLE, "sound", "output_device", "7")
        before = [ln for ln in SAMPLE.splitlines() if "output_device" not in ln]
        after = [ln for ln in out.splitlines() if "output_device" not in ln]
        assert before == after, "an unrelated line changed"

    def test_preserves_comments_and_blank_lines(self):
        out = set_config_value(SAMPLE, "cos", "mode", '"disabled"')
        assert "# leading comment" in out
        assert "# which card the radio is plugged into" in out
        assert out.count("\n\n") == SAMPLE.count("\n\n")

    def test_section_aware_for_duplicate_keys(self):
        """`mode:` exists under both ptt and cos -- edit only the right one."""
        out = set_config_value(SAMPLE, "cos", "mode", '"aioc_virtual"')
        assert 'mode: "aioc_virtual"' in out
        assert 'mode: "serial"' in out, "ptt.mode must not be touched"

    def test_editing_ptt_mode_leaves_cos_alone(self):
        out = set_config_value(SAMPLE, "ptt", "mode", '"none"')
        assert 'mode: "none"' in out
        assert 'mode: "internal_audio"' in out

    def test_missing_key_raises(self):
        """Appending blindly could land the setting in the wrong block."""
        with pytest.raises(KeyError, match="sound.nonexistent"):
            set_config_value(SAMPLE, "sound", "nonexistent", "1")

    def test_missing_section_raises(self):
        with pytest.raises(KeyError):
            set_config_value(SAMPLE, "nosuch", "mode", "1")

    def test_hash_inside_a_quoted_value_is_not_a_comment(self):
        src = 'sound:\n  input_device: "USB #2 Audio"\n'
        out = set_config_value(src, "sound", "input_device", '"Other"')
        assert out == 'sound:\n  input_device: "Other"\n'

    def test_result_still_parses(self):
        import yaml

        out = set_config_value(SAMPLE, "sound", "input_device", '"New Input"')
        assert yaml.safe_load(out)["sound"]["input_device"] == "New Input"

    def test_indentation_is_preserved(self):
        out = set_config_value(SAMPLE, "sound", "input_device", '"X"')
        line = next(ln for ln in out.splitlines() if "input_device" in ln)
        assert line.startswith("  input_device")

    def test_only_the_first_match_is_edited(self):
        out = set_config_value(SAMPLE, "sound", "sample_rate", "48000")
        assert out.count("sample_rate: 48000") == 1


class TestScalarRendering:
    def test_string_is_quoted(self):
        assert yaml_scalar("AIOC Audio") == '"AIOC Audio"'

    def test_int_is_bare(self):
        assert yaml_scalar(3) == "3"

    def test_bool_is_lowercase(self):
        assert yaml_scalar(True) == "true"
        assert yaml_scalar(False) == "false"

    def test_embedded_quote_is_escaped(self):
        assert yaml_scalar('a "b" c') == '"a \\"b\\" c"'


def dev(index, name, ins=0, outs=0):
    return AudioDevice(index, name, ins, outs, 48000.0, "CoreAudio")


class TestDeviceReference:
    """Prefer the name; fall back to the index only when ambiguous."""

    def test_unique_name_is_used(self):
        devices = [dev(1, "AIOC Audio", ins=1), dev(2, "Built-in Mic", ins=1)]
        assert _device_value(devices[0], devices) == "AIOC Audio"

    def test_duplicate_name_falls_back_to_index(self):
        """Two identical interfaces would otherwise produce an unusable config."""
        devices = [dev(1, "AIOC Audio", ins=1), dev(2, "AIOC Audio", ins=1)]
        assert _device_value(devices[0], devices) == 1

    def test_same_name_in_the_other_direction_does_not_count(self):
        """The AIOC appears twice overall but once per direction."""
        inputs = [dev(1, "AIOC Audio", ins=1)]
        assert _device_value(inputs[0], inputs) == "AIOC Audio"


class TestCosChoices:
    def test_always_offers_software_and_disabled(self):
        values = [c.value for c in detect_cos_choices()]
        assert "internal_audio" in values
        assert "disabled" in values

    def test_software_is_listed_first(self):
        assert detect_cos_choices()[0].value == "internal_audio"

    def test_every_choice_is_labelled(self):
        for c in detect_cos_choices():
            assert c.label

    def test_aioc_options_depend_on_detection(self, monkeypatch):
        import zello_link.hardware.aioc_hid as hid

        monkeypatch.setattr(hid, "find_aioc_hid_path", lambda **kw: ["/dev/hidraw0"])
        values = [c.value for c in detect_cos_choices()]
        assert "aioc_virtual" in values
        assert "aioc_hardware" in values

    def test_aioc_marked_unavailable_without_hardware(self, monkeypatch):
        import zello_link.hardware.aioc_hid as hid

        monkeypatch.setattr(hid, "find_aioc_hid_path", lambda **kw: [])
        labels = " ".join(c.label for c in detect_cos_choices())
        assert "unavailable" in labels

    def test_missing_hidapi_does_not_crash(self, monkeypatch):
        import zello_link.hardware.aioc_hid as hid

        def boom(**kw):
            raise RuntimeError("hidapi is required")

        monkeypatch.setattr(hid, "find_aioc_hid_path", boom)
        choices = detect_cos_choices()
        assert any(c.value == "internal_audio" for c in choices)


class TestInsertMissingKey:
    """Selecting an AIOC COS mode needs cos.hid_device, which many configs lack.

    Found live: picking aioc_virtual produced
    "cos.hid_device is required when cos.mode='aioc_virtual'" and the whole
    write was refused, because the key was not in the file to edit.
    """

    COS_ONLY = 'cos:\n  mode: "internal_audio"\n  hang_ms: 200\nbridge:\n  rf_to_zello: true\n'

    def test_inserts_under_the_right_section(self):
        out = set_config_value(
            self.COS_ONLY, "cos", "hid_device", '"/dev/hidraw0"', insert_if_missing=True
        )
        cos_block = out[out.index("cos:") : out.index("bridge:")]
        assert 'hid_device: "/dev/hidraw0"' in cos_block

    def test_does_not_leak_into_the_next_section(self):
        out = set_config_value(
            self.COS_ONLY, "cos", "hid_device", '"/dev/hidraw0"', insert_if_missing=True
        )
        bridge_block = out[out.index("bridge:") :]
        assert "hid_device" not in bridge_block

    def test_matches_the_section_indentation(self):
        src = 'cos:\n    mode: "internal_audio"\n'
        out = set_config_value(src, "cos", "hid_device", '"x"', insert_if_missing=True)
        line = next(ln for ln in out.splitlines() if "hid_device" in ln)
        assert line.startswith("    hid_device")

    def test_result_parses_and_keeps_siblings(self):
        import yaml

        out = set_config_value(
            self.COS_ONLY, "cos", "hid_device", '"/dev/hidraw0"', insert_if_missing=True
        )
        cos = yaml.safe_load(out)["cos"]
        assert cos["hid_device"] == "/dev/hidraw0"
        assert cos["mode"] == "internal_audio"
        assert cos["hang_ms"] == 200

    def test_still_raises_without_the_flag(self):
        with pytest.raises(KeyError):
            set_config_value(self.COS_ONLY, "cos", "hid_device", '"x"')

    def test_unknown_section_raises_even_with_the_flag(self):
        with pytest.raises(KeyError, match="nosuch"):
            set_config_value(
                self.COS_ONLY, "nosuch", "hid_device", '"x"', insert_if_missing=True
            )

    def test_existing_key_is_replaced_not_duplicated(self):
        src = 'cos:\n  hid_device: "old"\n  mode: "internal_audio"\n'
        out = set_config_value(src, "cos", "hid_device", '"new"', insert_if_missing=True)
        assert out.count("hid_device") == 1
        assert '"new"' in out and '"old"' not in out


class TestAiocCosWarnings:
    """Choosing an AIOC COS mode must say that the HID layout is unverified."""

    def _warnings(self, tmp_path, mode):
        import yaml

        from zello_link.config import load_config

        data = {
            "config_version": 1,
            "instance": {"name": "t"},
            "zello": {"channel": "c", "username": "u", "auth_token": "tok-abcdef"},
            "sound": {"input_device": "d", "output_device": "d"},
            "ptt": {"mode": "none"},
            "cos": {"mode": mode, "hid_device": "/dev/hidraw0"},
        }
        p = tmp_path / "b.yaml"
        p.write_text(yaml.safe_dump(data))
        return load_config(p).warnings()

    def test_virtual_warns_about_the_unverified_layout(self, tmp_path):
        assert any("not published" in w for w in self._warnings(tmp_path, "aioc_virtual"))

    def test_hardware_warns_about_the_unverified_layout(self, tmp_path):
        assert any("not published" in w for w in self._warnings(tmp_path, "aioc_hardware"))

    def test_hardware_also_warns_about_2_pin_connectors(self, tmp_path):
        assert any("2-pin" in w for w in self._warnings(tmp_path, "aioc_hardware"))

    def test_internal_audio_gets_no_such_warning(self, tmp_path):
        import yaml

        from zello_link.config import load_config

        data = {
            "config_version": 1,
            "instance": {"name": "t"},
            "zello": {"channel": "c", "username": "u", "auth_token": "tok-abcdef"},
            "sound": {"input_device": "d", "output_device": "d"},
            "ptt": {"mode": "none"},
            "cos": {"mode": "internal_audio"},
        }
        p = tmp_path / "b.yaml"
        p.write_text(yaml.safe_dump(data))
        assert not any("not published" in w for w in load_config(p).warnings())
