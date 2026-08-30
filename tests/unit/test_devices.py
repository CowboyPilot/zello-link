"""Device selection: exact, partial, ambiguous, and missing.

Ambiguity handling is the point. On a host with two AIOCs, silently picking
one would cross-wire two bridge instances onto the same radio.
"""

from __future__ import annotations

import pytest

from zello_dmr_bridge.audio.devices import (
    AudioDevice,
    DeviceError,
    format_device_table,
    resolve_device,
)


def dev(index, name, ins=0, outs=0, rate=48000.0, api="ALSA"):
    return AudioDevice(index, name, ins, outs, rate, api)


# Mirrors a real AIOC: one input-only node and one output-only node, both
# reporting the same name.
AIOC = [
    dev(0, "AIOC Audio", ins=0, outs=1),
    dev(1, "AIOC Audio", ins=1, outs=0),
    dev(2, "Built-in Microphone", ins=1, outs=0),
    dev(3, "Built-in Output", ins=0, outs=2),
]

TWO_AIOCS = [
    dev(0, "AIOC Audio", ins=1, outs=1),
    dev(1, "AIOC Audio", ins=1, outs=1),
    dev(2, "USB Audio CODEC", ins=1, outs=1),
]


class TestDirectionDisambiguation:
    """Same-named input/output nodes must resolve by direction alone."""

    def test_input_resolves_to_the_capture_node(self):
        assert resolve_device("AIOC Audio", "input", devices=AIOC).index == 1

    def test_output_resolves_to_the_playback_node(self):
        assert resolve_device("AIOC Audio", "output", devices=AIOC).index == 0

    def test_partial_name_still_disambiguates(self):
        assert resolve_device("AIOC", "input", devices=AIOC).index == 1
        assert resolve_device("AIOC", "output", devices=AIOC).index == 0


class TestAmbiguity:
    def test_two_identical_devices_is_an_error(self):
        with pytest.raises(DeviceError, match="are named"):
            resolve_device("AIOC Audio", "input", devices=TWO_AIOCS)

    def test_ambiguous_partial_match_is_an_error(self):
        devices = [dev(0, "USB Audio A", ins=1), dev(1, "USB Audio B", ins=1)]
        with pytest.raises(DeviceError, match="ambiguous"):
            resolve_device("USB Audio", "input", devices=devices)

    def test_ambiguity_error_lists_the_candidates(self):
        with pytest.raises(DeviceError) as exc:
            resolve_device("AIOC Audio", "input", devices=TWO_AIOCS)
        assert "AIOC Audio" in str(exc.value)
        assert "index" in str(exc.value)

    def test_index_resolves_an_otherwise_ambiguous_name(self):
        assert resolve_device(1, "input", devices=TWO_AIOCS).index == 1


class TestIndexSelection:
    def test_numeric_index(self):
        assert resolve_device(2, "input", devices=AIOC).index == 2

    def test_numeric_string_index(self):
        assert resolve_device("2", "input", devices=AIOC).index == 2

    def test_index_without_the_right_direction(self):
        with pytest.raises(DeviceError, match="no input channels"):
            resolve_device(0, "input", devices=AIOC)

    def test_unknown_index(self):
        with pytest.raises(DeviceError, match="index 99"):
            resolve_device(99, "input", devices=AIOC)


class TestMissingAndInvalid:
    def test_no_device_configured(self):
        with pytest.raises(DeviceError, match="no input device configured"):
            resolve_device(None, "input", devices=AIOC)

    def test_unknown_name_lists_alternatives(self):
        with pytest.raises(DeviceError) as exc:
            resolve_device("Nonexistent Card", "input", devices=AIOC)
        assert "Built-in Microphone" in str(exc.value)

    def test_empty_selector(self):
        with pytest.raises(DeviceError, match="empty"):
            resolve_device("   ", "input", devices=AIOC)

    def test_no_devices_for_direction(self):
        with pytest.raises(DeviceError, match="no audio devices with output"):
            resolve_device("x", "output", devices=[dev(0, "Mic Only", ins=1)])

    def test_matching_is_case_insensitive(self):
        assert resolve_device("aioc audio", "input", devices=AIOC).index == 1


class TestFormatting:
    def test_table_includes_every_device(self):
        table = format_device_table(AIOC)
        for d in AIOC:
            assert d.name in table

    def test_table_shows_channel_counts(self):
        assert "in:1 out:0" in format_device_table(AIOC)

    def test_empty_table(self):
        assert "no audio devices" in format_device_table([])

    def test_supports_predicate(self):
        assert dev(0, "x", ins=1, outs=0).supports("input")
        assert not dev(0, "x", ins=1, outs=0).supports("output")
