"""CLI argument handling and the safety guarantees of the diagnostic modes."""

from __future__ import annotations

import os

import yaml

import pytest

from zello_link.cli import EXIT_CONFIG, EXIT_DEVICE, EXIT_OK, build_parser, main

CONFIG = {
    "config_version": 2,
    "instance": {"name": "cli-test"},
    "zello": {"channel": "C", "username": "u", "auth_token": "tok-abcdef"},
    "sound": {"input_device": "fake", "output_device": "fake"},
    "ptt": {"mode": "none"},
    "logging": {"console": False, "file": None},
}


@pytest.fixture
def cfg_path(tmp_path):
    p = tmp_path / "bridge.yaml"
    p.write_text(yaml.safe_dump(CONFIG))
    return str(p)


class TestParser:
    def test_config_is_required(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_diagnostic_modes_are_mutually_exclusive(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--config", "x", "--validate", "--cos-monitor"])

    def test_cos_monitor_is_a_real_flag(self):
        """The spec referenced --cos-monitor without ever defining it."""
        args = build_parser().parse_args(["--config", "x", "--cos-monitor"])
        assert args.cos_monitor is True

    def test_ptt_test_is_a_real_flag(self):
        args = build_parser().parse_args(["--config", "x", "--diagnose-aioc", "--ptt-test"])
        assert args.ptt_test is True

    def test_duration_is_parsed(self):
        args = build_parser().parse_args(["--config", "x", "--cos-monitor", "--duration", "5"])
        assert args.duration == 5.0


class TestConfigErrors:
    def test_missing_file(self, tmp_path):
        assert main(["--config", str(tmp_path / "nope.yaml")]) == EXIT_CONFIG

    def test_invalid_config(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("instance: {name: x}\n")     # missing required sections
        assert main(["--config", str(p)]) == EXIT_CONFIG

    def test_config_error_does_not_traceback(self, tmp_path, capsys):
        main(["--config", str(tmp_path / "nope.yaml")])
        assert "Traceback" not in capsys.readouterr().err


class TestPttTestGuard:
    def test_ptt_test_requires_diagnose_aioc(self, cfg_path, capsys):
        """--ptt-test transmits; it must not be usable by itself."""
        assert main(["--config", cfg_path, "--ptt-test"]) == EXIT_CONFIG
        assert "only valid with --diagnose-aioc" in capsys.readouterr().err


class TestValidate:
    def test_validate_reports_and_exits(self, cfg_path, capsys):
        code = main(["--config", cfg_path, "--validate"])
        out = capsys.readouterr().out
        assert "cli-test" in out
        assert code in (EXIT_OK, EXIT_DEVICE)   # device presence varies by host

    def test_validate_never_prints_secrets(self, cfg_path, capsys):
        """AT-10: --validate output must be redacted too."""
        main(["--config", cfg_path, "--validate"])
        captured = capsys.readouterr()
        assert "tok-abcdef" not in captured.out
        assert "tok-abcdef" not in captured.err

    def test_validate_shows_radio_centric_gain_labels(self, cfg_path, capsys):
        main(["--config", cfg_path, "--validate"])
        out = capsys.readouterr().out
        assert "radio receive -> Zello" in out
        assert "Zello -> radio transmit" in out

    def test_validate_shows_the_latency_estimate(self, cfg_path, capsys):
        main(["--config", cfg_path, "--validate"])
        assert "latency estimate" in capsys.readouterr().out

    def test_validate_does_not_key_ptt(self, cfg_path, monkeypatch):
        """AT-02: config validation must never touch the transmitter."""
        import zello_link.hardware.ptt as ptt_module

        keyed = []
        original = ptt_module.NullPtt.key

        def spy(self):
            keyed.append(True)
            return original(self)

        monkeypatch.setattr(ptt_module.NullPtt, "key", spy)
        main(["--config", cfg_path, "--validate"])
        assert keyed == [], "validation keyed the transmitter"


class TestRenderMeter:
    """The level meter used for setting a radio's volume knob."""

    def setup_method(self):
        from zello_link.diagnostics import status

        self.status = status

    def test_meter_and_carets_are_the_same_width(self):
        meter, carets = self.status.render_meter(-30.0, -20.0, color=False)
        assert len(meter) == len(carets) == self.status.METER_WIDTH

    def test_fill_uses_equals_and_dashes(self):
        meter, _ = self.status.render_meter(-30.0, -30.0, color=False)
        assert "=" in meter and "-" in meter

    def test_louder_signal_fills_more(self):
        quiet, _ = self.status.render_meter(-55.0, -55.0, color=False)
        loud, _ = self.status.render_meter(-10.0, -10.0, color=False)
        assert loud.count("=") > quiet.count("=")

    def test_peak_marker_is_placed(self):
        meter, _ = self.status.render_meter(-40.0, -10.0, color=False)
        assert meter.count("|") >= 1

    def test_peak_is_right_of_level_when_higher(self):
        meter, _ = self.status.render_meter(-40.0, -10.0, color=False)
        assert meter.index("|") > meter.index("=")

    def test_threshold_marker_is_placed(self):
        meter, _ = self.status.render_meter(-30.0, -20.0, threshold=-50.0, color=False)
        assert meter.count("|") == 2, "expected both threshold and peak markers"

    def test_threshold_is_red_when_coloured(self):
        meter, _ = self.status.render_meter(-30.0, -20.0, threshold=-50.0, color=True)
        assert "\033[31m|" in meter

    def test_carets_bound_the_target_window(self):
        _, carets = self.status.render_meter(-30.0, -20.0, color=False)
        assert carets.count("^") == 2
        lo, hi = carets.index("^"), carets.rindex("^")
        assert lo < hi

    def test_peak_in_window_lands_between_carets(self):
        mid = (self.status.TARGET_PEAK_LOW + self.status.TARGET_PEAK_HIGH) / 2
        meter, carets = self.status.render_meter(-40.0, mid, color=False)
        lo, hi = carets.index("^"), carets.rindex("^")
        assert lo <= meter.index("|") <= hi

    def test_clamps_at_the_extremes(self):
        for level in (-200.0, 50.0):
            meter, _ = self.status.render_meter(level, level, color=False)
            assert len(meter) == self.status.METER_WIDTH


class TestRollingPeak:
    def test_holds_the_peak_within_the_window(self):
        from zello_link.diagnostics.status import RollingPeak

        rp = RollingPeak(window_s=5.0)
        rp.add(-40.0, 100.0)
        rp.add(-6.0, 101.0)
        assert rp.add(-40.0, 102.0) == -6.0

    def test_peak_expires_after_the_window(self):
        from zello_link.diagnostics.status import RollingPeak

        rp = RollingPeak(window_s=5.0)
        rp.add(-6.0, 100.0)
        assert rp.add(-40.0, 106.0) == -40.0

    def test_empty_reads_at_the_floor(self):
        from zello_link.diagnostics.status import METER_LO, RollingPeak

        assert RollingPeak().value == METER_LO

    def test_does_not_grow_without_bound(self):
        from zello_link.diagnostics.status import RollingPeak

        rp = RollingPeak(window_s=1.0)
        for i in range(1000):
            rp.add(-30.0, 100.0 + i * 0.02)
        assert len(rp._samples) < 60


class TestVerdict:
    def test_says_turn_down_when_hot(self):
        from zello_link.diagnostics.status import _verdict

        assert "DOWN" in _verdict(-2.0, 0, False)

    def test_says_turn_up_when_quiet(self):
        from zello_link.diagnostics.status import _verdict

        assert "UP" in _verdict(-40.0, 0, False)

    def test_says_ok_in_window(self):
        from zello_link.diagnostics.status import _verdict

        assert "OK" in _verdict(-9.0, 0, False)

    def test_clipping_takes_priority(self):
        from zello_link.diagnostics.status import _verdict

        assert "CLIPPING" in _verdict(-9.0, 5, False)


class TestStatusLine:
    """The pinned meter must never corrupt log output (--showmonitor)."""

    def _stream(self):
        import io

        class TTY(io.StringIO):
            def isatty(self):
                return True

        return TTY()

    def test_disabled_when_not_a_tty(self):
        import io

        from zello_link.logging_setup import StatusLine

        assert StatusLine(io.StringIO()).enabled is False

    def test_non_tty_writes_nothing(self):
        import io

        from zello_link.logging_setup import StatusLine

        s = io.StringIO()
        StatusLine(s).set("meter")
        assert s.getvalue() == "", "escape codes leaked into a non-TTY stream"

    def test_enabled_on_a_tty(self):
        from zello_link.logging_setup import StatusLine

        assert StatusLine(self._stream()).enabled is True

    def test_set_erases_before_drawing(self):
        from zello_link.logging_setup import StatusLine

        stream = self._stream()
        StatusLine(stream).set("hello")
        out = stream.getvalue()
        assert "\033[2K" in out and out.endswith("hello")

    def test_clear_then_redraw_restores_the_line(self):
        from zello_link.logging_setup import StatusLine

        stream = self._stream()
        sl = StatusLine(stream)
        sl.set("meter")
        sl.clear()
        sl.redraw()
        assert stream.getvalue().endswith("meter")

    def test_redraw_is_idempotent_while_visible(self):
        from zello_link.logging_setup import StatusLine

        stream = self._stream()
        sl = StatusLine(stream)
        sl.set("meter")
        before = stream.getvalue()
        sl.redraw()
        assert stream.getvalue() == before

    def test_finish_leaves_no_text(self):
        from zello_link.logging_setup import StatusLine

        stream = self._stream()
        sl = StatusLine(stream)
        sl.set("meter")
        sl.finish()
        assert stream.getvalue().endswith("\033[2K")

    def test_log_record_does_not_smear_the_line(self):
        """The whole point: a log line must not land on top of the meter."""
        import logging

        from zello_link.logging_setup import StatusLine, _StatusAwareHandler

        stream = self._stream()
        sl = StatusLine(stream)
        sl.set("METER")

        handler = _StatusAwareHandler(stream, sl)
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler.emit(
            logging.LogRecord("t", logging.INFO, "f", 1, "a log line", None, None)
        )

        out = stream.getvalue()
        assert "a log line" in out
        # The meter is redrawn after the record, so it ends up last.
        assert out.rindex("METER") > out.rindex("a log line")


class TestStatusLineRendering:
    def _render(self, **kw):
        from zello_link.diagnostics.status import render_status_line

        kw.setdefault("color", False)
        return render_status_line(**kw)

    def test_is_a_single_line(self):
        assert "\n" not in self._render(rms_dbfs=-30.0, peak_dbfs=-10.0)

    def test_target_window_is_bracketed_inline(self):
        line = self._render(rms_dbfs=-40.0, peak_dbfs=-30.0)
        assert "[" in line and "]" in line
        assert line.index("[") < line.index("]")

    def test_peak_inside_brackets_when_in_window(self):
        from zello_link.diagnostics.status import (
            TARGET_PEAK_HIGH,
            TARGET_PEAK_LOW,
        )

        mid = (TARGET_PEAK_LOW + TARGET_PEAK_HIGH) / 2
        line = self._render(rms_dbfs=-40.0, peak_dbfs=mid)
        assert line.index("[") < line.index("|", line.index("[")) < line.index("]")

    def test_shows_tx_when_keyed(self):
        assert "TX" in self._render(rms_dbfs=-30.0, peak_dbfs=-10.0, keyed=True)

    def test_shows_rx_when_cos_active(self):
        assert "RX" in self._render(rms_dbfs=-30.0, peak_dbfs=-10.0, cos_active=True)

    def test_clip_takes_priority_over_tx(self):
        line = self._render(rms_dbfs=-3.0, peak_dbfs=-0.5, keyed=True, clipped=9)
        assert "CLIP" in line

    def test_threshold_marker_present(self):
        line = self._render(rms_dbfs=-30.0, peak_dbfs=-10.0, threshold=-50.0)
        assert "|" in line

    def test_verdict_included(self):
        assert "turn DOWN" in self._render(rms_dbfs=-5.0, peak_dbfs=-1.0)


class TestStatusLineFitsTheTerminal:
    """The line must never wrap.

    It is redrawn with a bare carriage return, and CR returns to the start of
    the last *physical* line. A line wider than the terminal wraps, every
    redraw leaves the wrapped remnant behind, and the meter appears to scroll
    instead of holding still. Observed at 107 visible chars on an 80-column
    terminal.
    """

    def _line(self, total_width, **kw):
        from zello_link.diagnostics.status import render_status_line

        kw.setdefault("color", False)
        kw.setdefault("threshold", -50.0)
        return render_status_line(-30.0, -9.0, total_width=total_width, **kw)

    @pytest.mark.parametrize("width", [60, 72, 80, 100, 120, 200])
    def test_fits_with_a_spare_column(self, width):
        assert len(self._line(width)) <= width - 1

    @pytest.mark.parametrize("width", [60, 80, 100])
    @pytest.mark.parametrize(
        "kw",
        [{}, {"cos_active": True}, {"keyed": True}, {"clipped": 99999}],
    )
    def test_fits_in_every_state(self, width, kw):
        """The verdict and state text vary in length; the worst case must fit."""
        assert len(self._line(width, **kw)) <= width - 1

    @pytest.mark.parametrize("peak", [-60.0, -51.0, -20.0, -9.0, -0.1, 0.0])
    def test_fits_at_every_level(self, peak):
        from zello_link.diagnostics.status import render_status_line

        line = render_status_line(-30.0, peak, threshold=-50.0,
                                  total_width=80, color=False)
        assert len(line) <= 79

    def test_narrow_terminal_still_shows_a_usable_bar(self):
        from zello_link.diagnostics.status import _MIN_BAR

        line = self._line(60)
        bar = line[line.index("=") : line.rindex("-") + 1]
        assert len(bar) >= _MIN_BAR

    def test_wide_terminal_does_not_produce_an_absurd_bar(self):
        from zello_link.diagnostics.status import _MAX_BAR

        assert len(self._line(300)) < _MAX_BAR + 60

    def test_verdict_is_shortened_before_the_bar_is_starved(self):
        narrow = self._line(60, clipped=5)
        assert "CLIPPING" in narrow
        assert "turn DOWN" not in narrow, "long verdict should be trimmed first"

    def test_detects_terminal_width_when_not_given(self, monkeypatch):
        import shutil

        from zello_link.diagnostics.status import render_status_line

        monkeypatch.setattr(
            shutil, "get_terminal_size", lambda fallback=(80, 24): os.terminal_size((64, 24))
        )
        line = render_status_line(-30.0, -9.0, threshold=-50.0, color=False)
        assert len(line) <= 63


class TestRunWrapper:
    """./run-bridge.sh must not silently enable the meter."""

    def _script(self):
        import pathlib

        return pathlib.Path("run-bridge.sh").read_text()

    def test_does_not_inject_showmonitor(self):
        assert '"${@:---showmonitor}"' not in self._script(), (
            "the wrapper must not default to --showmonitor; the flag is opt-in"
        )

    def test_passes_arguments_through(self):
        assert '"$@"' in self._script()

class TestValidateReportsKeying:
    """Which line keys the radio is the setting most likely to be wrong on a
    new interface, and it fails silently when it is -- the radio simply never
    transmits. It belongs in --validate output."""

    def _text(self, tmp_path, **ptt):
        import yaml

        from zello_link.config import load_config
        from zello_link.diagnostics.status import render_validation

        data = {
            "config_version": 2,
            "instance": {"name": "v"},
            "zello": {"channel": "C", "username": "u", "auth_token": "tok-abcdef"},
            "sound": {"input_device": "in", "output_device": "out"},
            "ptt": {"mode": "serial", "tty_device": "/dev/ttyACM0", **ptt},
            "logging": {"console": False, "file": None},
        }
        p = tmp_path / "b.yaml"
        p.write_text(yaml.safe_dump(data))
        return render_validation(load_config(p))

    def test_dtr_is_reported(self, tmp_path):
        t = self._text(tmp_path)
        assert "DTR asserts" in t and "RTS held low" in t

    def test_rts_is_reported(self, tmp_path):
        t = self._text(tmp_path, serial_signal="rts")
        assert "RTS asserts" in t and "DTR held low" in t

    def test_gpio_pin_is_reported(self, tmp_path):
        t = self._text(
            tmp_path, mode="cm108_hid", hid_device="/dev/hidraw0",
            tty_device=None, gpio_pin=4,
        )
        assert "CM108 GPIO 4" in t
