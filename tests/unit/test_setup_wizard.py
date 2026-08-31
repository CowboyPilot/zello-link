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
config_version: 2

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
            "config_version": 2,
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
            "config_version": 2,
            "instance": {"name": "t"},
            "zello": {"channel": "c", "username": "u", "auth_token": "tok-abcdef"},
            "sound": {"input_device": "d", "output_device": "d"},
            "ptt": {"mode": "none"},
            "cos": {"mode": "internal_audio"},
        }
        p = tmp_path / "b.yaml"
        p.write_text(yaml.safe_dump(data))
        assert not any("not published" in w for w in load_config(p).warnings())


class TestSectionCreation:
    """A config written for AIOC has no usrp: block to insert into."""

    def test_creates_a_missing_section(self):
        from zello_link.diagnostics.setup_wizard import ensure_section

        out = ensure_section('bridge:\n  backend: "aioc"\n', "usrp")
        assert "\nusrp:\n" in out

    def test_is_idempotent(self):
        from zello_link.diagnostics.setup_wizard import ensure_section

        src = 'usrp:\n  bind_port: 34001\n'
        assert ensure_section(src, "usrp") == src

    def test_does_not_disturb_existing_content(self):
        from zello_link.diagnostics.setup_wizard import ensure_section

        src = 'bridge:\n  backend: "aioc"   # note\n'
        out = ensure_section(src, "usrp")
        assert src in out
        assert "# note" in out

    def test_created_section_accepts_inserts(self):
        import yaml

        from zello_link.diagnostics.setup_wizard import ensure_section, set_config_value

        out = ensure_section('bridge:\n  backend: "aioc"\n', "usrp")
        out = set_config_value(out, "usrp", "bind_port", "34001", insert_if_missing=True)
        assert yaml.safe_load(out)["usrp"]["bind_port"] == 34001


class TestLocalAddresses:
    """A VPN-reached bridge binds the tunnel address, which a default-route
    probe misses -- so all interfaces are enumerated."""

    def test_always_offers_loopback_last(self):
        from zello_link.diagnostics.setup_wizard import local_addresses

        addrs = local_addresses()
        assert addrs[-1] == "127.0.0.1"

    def test_no_duplicates(self):
        from zello_link.diagnostics.setup_wizard import local_addresses

        addrs = local_addresses()
        assert len(addrs) == len(set(addrs))

    def test_excludes_other_loopback_addresses(self):
        from zello_link.diagnostics.setup_wizard import local_addresses

        assert [a for a in local_addresses() if a.startswith("127.")] == ["127.0.0.1"]


class TestAslInstructions:
    """The generated rpt.conf guidance is the payoff: the operator should not
    have to work out the rxchannel field order themselves."""

    def _text(self, **kw):
        from zello_link.diagnostics.setup_wizard import render_asl_instructions

        kw = {"bind_host": "10.0.0.5", "bind_port": 34001, "asl_port": 32001, **kw}
        return render_asl_instructions(**kw)

    def test_rxchannel_field_order_is_hisip_hisport_myport(self):
        assert "USRP/10.0.0.5:34001:32001" in self._text()

    def test_uses_the_configured_values(self):
        t = self._text(bind_host="192.168.1.9", bind_port=44001, asl_port=42001)
        assert "USRP/192.168.1.9:44001:42001" in t

    def test_defaults_to_node_1998(self):
        t = self._text()
        assert "[1998]" in t
        assert "1998 = radio@127.0.0.1/1998,NONE" in t

    def test_node_number_is_overridable(self):
        t = self._text(node=1234)
        assert "[1234]" in t and "*31234" in t

    def test_covers_modules_conf(self):
        t = self._text()
        assert "modules.conf" in t
        assert "load => chan_usrp.so" in t
        assert "autoload=no" in t

    def test_suppresses_telemetry_in_the_stanza(self):
        """Courtesy tones would reach Zello as spurious transmissions."""
        t = self._text()
        for setting in ("duplex = 0", "nounkeyct = 1", "telemdefault = 0",
                        "idtime = 0", "unlinkedct = |", "linkunkeyct = |"):
            assert setting in t, f"missing {setting}"

    def test_warns_about_one_rxchannel_per_node(self):
        t = self._text()
        assert "ONE rxchannel" in t

    def test_says_a_restart_is_needed_not_a_reload(self):
        t = self._text()
        assert "systemctl restart asterisk" in t
        assert "not a" in t and "reload" in t

    def test_includes_verification_commands(self):
        t = self._text()
        assert "module show like chan_usrp" in t
        assert "core show channels" in t

    def test_includes_the_startup_macro(self):
        assert "startup_macro = *31998" in self._text()


REAL_RPT = """\
[general]
;rxchannel = USRP/127.0.0.1:34001:32001    ;GNU Radio interface USRP

[nodes]
531133 = radio@127.0.0.1/531133,NONE
675 = radio@127.0.0.1/675,NONE

[node-main](!)
duplex = 1

[531133](node-main)
rxchannel = SimpleUSB/531133      ; SimpleUSB
;rxchannel = USRP/100.81.118.119:34001:32001
duplex = 1

[675]
;rxchannel = SimpleUSB/675
rxchannel = USRP/127.0.0.1:34001:32001   ; zello-link bridge
duplex = 0

[functions]
[telemetry]
"""


class TestRptConfParsing:
    """Most USRP bridges run on the ASL box, so an existing node turns a
    four-question interview into one keystroke."""

    def _nodes(self, text=REAL_RPT):
        from zello_link.diagnostics.setup_wizard import parse_rpt_conf

        return parse_rpt_conf(text)

    def test_finds_usrp_nodes(self):
        assert {n.node for n in self._nodes()} == {"531133", "675"}

    def test_ignores_commented_sample_outside_a_node_stanza(self):
        """The shipped rpt.conf documents USRP under [general]; that is not
        a node and must not be offered."""
        assert all(n.node.isdigit() for n in self._nodes())

    def test_ignores_non_usrp_rxchannels(self):
        assert all("USRP" in n.rxchannel for n in self._nodes())

    def test_reads_the_template_suffix_off_the_section_name(self):
        """[531133](node-main) is node 531133, not '531133](node-main'."""
        assert any(n.node == "531133" for n in self._nodes())

    def test_flags_a_commented_rxchannel(self):
        by_node = {n.node: n for n in self._nodes()}
        assert by_node["531133"].enabled is False
        assert by_node["675"].enabled is True

    def test_extracts_all_three_fields(self):
        n = {x.node: x for x in self._nodes()}["675"]
        assert (n.bind_host, n.bind_port, n.asl_port) == ("127.0.0.1", 34001, 32001)

    def test_remote_bind_host_is_preserved(self):
        n = {x.node: x for x in self._nodes()}["531133"]
        assert n.bind_host == "100.81.118.119"

    def test_myport_defaults_when_omitted(self):
        """rxchannel = USRP/host:port is legal; MYPORT falls back to 32001."""
        nodes = self._nodes("[1998]\nrxchannel = USRP/127.0.0.1:34001\n")
        assert nodes[0].asl_port == 32001

    def test_rxchannel_round_trips(self):
        n = {x.node: x for x in self._nodes()}["675"]
        assert n.rxchannel == "USRP/127.0.0.1:34001:32001"

    def test_trailing_comment_does_not_break_parsing(self):
        n = {x.node: x for x in self._nodes()}["675"]
        assert n.bind_port == 34001

    def test_empty_config(self):
        assert self._nodes("") == []

    def test_config_with_no_usrp(self):
        assert self._nodes("[675]\nrxchannel = SimpleUSB/675\n") == []


class TestLocalAslDetection:
    def test_absent_when_no_config_exists(self, tmp_path):
        from zello_link.diagnostics.setup_wizard import detect_local_asl

        present, nodes = detect_local_asl(paths=[str(tmp_path / "nope.conf")])
        assert nodes == []

    def test_finds_nodes_in_a_real_config(self, tmp_path):
        from zello_link.diagnostics.setup_wizard import detect_local_asl

        f = tmp_path / "rpt.conf"
        f.write_text(REAL_RPT)
        present, nodes = detect_local_asl(paths=[str(f)])
        assert present is True
        assert len(nodes) == 2

    def test_records_where_it_was_found(self, tmp_path):
        from zello_link.diagnostics.setup_wizard import detect_local_asl

        f = tmp_path / "rpt.conf"
        f.write_text(REAL_RPT)
        _, nodes = detect_local_asl(paths=[str(f)])
        assert all(n.source == str(f) for n in nodes)


class TestFlowStyleRefusal:
    """Line-based editing cannot safely add a key to an inline mapping."""

    def test_refuses_rather_than_producing_invalid_yaml(self):
        from zello_link.diagnostics.setup_wizard import set_config_value

        with pytest.raises(KeyError, match="flow style"):
            set_config_value(
                'bridge: {backend: "aioc"}\n', "bridge", "backend", '"usrp"',
                insert_if_missing=True,
            )

    def test_ensure_section_also_refuses(self):
        from zello_link.diagnostics.setup_wizard import ensure_section

        with pytest.raises(KeyError, match="flow style"):
            ensure_section('usrp: {bind_port: 1}\n', "usrp")

    def test_error_shows_how_to_fix_it(self):
        from zello_link.diagnostics.setup_wizard import ensure_section

        with pytest.raises(KeyError) as exc:
            ensure_section('usrp: {bind_port: 1}\n', "usrp")
        assert "block style" in str(exc.value)

    def test_block_style_is_unaffected(self):
        import yaml

        from zello_link.diagnostics.setup_wizard import set_config_value

        out = set_config_value(
            'bridge:\n  backend: "aioc"\n', "bridge", "backend", '"usrp"'
        )
        assert yaml.safe_load(out)["bridge"]["backend"] == "usrp"
