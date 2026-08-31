"""Serial and CM108 HID PTT backends.

Neither had any test coverage, and these are the AT-02 safety paths: opening
an interface must never key the transmitter, whatever state the OS left the
lines in.

They are also where interface support lives. The AIOC keys on DTR and the
Digirig Mobile on RTS; the AIOC and Digirig Lite both use CM108 GPIO 3, while
some interfaces use 4. Those are parameters, not separate drivers, so these
tests pin the parameterisation down.
"""

from __future__ import annotations

import sys
import types

import pytest

from zello_link.hardware.aioc_hid import Cm108HidPtt, Cm108Report
from zello_link.hardware.aioc_serial import SerialPtt
from zello_link.hardware.ptt import PttError


class FakePort:
    """A serial port that starts with DTR asserted, as some platforms do."""

    def __init__(self, *a, **kw):
        self.dtr = True
        self.rts = True
        self.closed = False
        self.kwargs = kw

    def close(self):
        self.closed = True


@pytest.fixture
def fake_serial(monkeypatch):
    made: list[FakePort] = []

    def factory(*a, **kw):
        p = FakePort(*a, **kw)
        made.append(p)
        return p

    mod = types.ModuleType("serial")
    mod.Serial = factory
    monkeypatch.setitem(sys.modules, "serial", mod)
    return made


class TestSerialSafety:
    def test_open_drops_both_lines(self, fake_serial):
        """AT-02: opening must not key, even if the OS asserted DTR."""
        ptt = SerialPtt("/dev/ttyACM0")
        ptt.open()
        port = fake_serial[0]
        assert port.dtr is False and port.rts is False
        assert not ptt.is_keyed()

    def test_open_does_not_let_pyserial_raise_dtr(self, fake_serial):
        SerialPtt("/dev/ttyACM0").open()
        assert fake_serial[0].kwargs["dsrdtr"] is False

    def test_close_drops_the_lines(self, fake_serial):
        ptt = SerialPtt("/dev/ttyACM0")
        ptt.open()
        ptt.key()
        ptt.close()
        assert fake_serial[0].dtr is False and fake_serial[0].rts is False
        assert fake_serial[0].closed

    def test_key_before_open_is_refused(self):
        with pytest.raises(PttError, match="before open"):
            SerialPtt("/dev/ttyACM0").key()

    def test_device_path_is_required(self):
        with pytest.raises(PttError, match="tty device"):
            SerialPtt("")


class TestSerialSignalSelection:
    def test_dtr_is_the_default_and_keys_on_dtr(self, fake_serial):
        """The AIOC convention: DTR asserts, RTS stays low."""
        ptt = SerialPtt("/dev/ttyACM0")
        ptt.open()
        ptt.key()
        port = fake_serial[0]
        assert port.dtr is True
        assert port.rts is False, "the idle line must never assert"
        assert ptt.is_keyed()

    def test_rts_mode_keys_on_rts(self, fake_serial):
        """The Digirig Mobile convention: RTS asserts, DTR stays low."""
        ptt = SerialPtt("/dev/ttyACM0", signal="rts")
        ptt.open()
        ptt.key()
        port = fake_serial[0]
        assert port.rts is True
        assert port.dtr is False, "keying RTS must not also raise DTR"

    def test_rts_mode_releases_rts(self, fake_serial):
        ptt = SerialPtt("/dev/ttyACM0", signal="rts")
        ptt.open()
        ptt.key()
        ptt.unkey()
        assert fake_serial[0].rts is False
        assert not ptt.is_keyed()

    def test_unkey_releases_only_the_asserting_line(self, fake_serial):
        ptt = SerialPtt("/dev/ttyACM0")
        ptt.open()
        ptt.key()
        ptt.unkey()
        assert fake_serial[0].dtr is False
        assert not ptt.is_keyed()

    def test_unknown_signal_is_refused_at_construction(self):
        """Fail before the radio is connected, not on the first key."""
        with pytest.raises(PttError, match="serial_signal"):
            SerialPtt("/dev/ttyACM0", signal="cts")

    def test_idle_line_is_the_other_one(self):
        assert SerialPtt("/dev/x").idle_signal == "rts"
        assert SerialPtt("/dev/x", signal="rts").idle_signal == "dtr"


class FakeHid:
    def __init__(self):
        self.writes: list[bytes] = []

    def write(self, payload):
        self.writes.append(bytes(payload))
        return len(payload)


class TestCm108GpioPin:
    def test_default_pin_is_three(self):
        assert Cm108HidPtt(None).report.gpio_pin == 3

    def test_pin_is_configurable(self):
        assert Cm108HidPtt(None, gpio_pin=4).report.gpio_pin == 4

    def test_report_and_pin_together_are_refused(self):
        """Ambiguous: silently ignoring one would key the wrong pin."""
        with pytest.raises(PttError, match="not both"):
            Cm108HidPtt(None, report=Cm108Report(), gpio_pin=4)

    def test_out_of_range_pin_is_refused(self):
        with pytest.raises(ValueError, match="1-8"):
            Cm108HidPtt(None, gpio_pin=9)

    @pytest.mark.parametrize("pin", [1, 3, 4, 8])
    def test_key_drives_the_selected_pin_only(self, pin):
        ptt = Cm108HidPtt(None, gpio_pin=pin)
        ptt._dev = FakeHid()
        ptt.key()
        payload = ptt._dev.writes[-1]
        bit = 1 << (pin - 1)
        assert payload[2] & bit, "selected GPIO must be driven high"
        assert payload[2] == bit, "no other GPIO may be driven high"
        assert payload[3] & bit, "the mask must claim the pin"

    def test_unkey_clears_the_pin_but_keeps_the_mask(self):
        """Releasing the mask would float the line rather than hold it low."""
        ptt = Cm108HidPtt(None, gpio_pin=3)
        ptt._dev = FakeHid()
        ptt.unkey()
        payload = ptt._dev.writes[-1]
        assert payload[2] & 0b100 == 0
        assert payload[3] & 0b100, "mask must still drive the pin low"

    def test_use_before_open_is_refused(self):
        with pytest.raises(PttError, match="before open"):
            Cm108HidPtt(None).key()


class TestCm108InputReport:
    """COS arrives as a BUTTON press, not a GPIO level.

    The AIOC maps VCOS onto one of the four CM108 buttons in its own "CM108
    Button Sources" panel, and the interrupt-IN report carries those four
    states in byte 0. An earlier implementation read byte 2 -- the OUTPUT
    report's GPIO data byte -- and treated the bit as a 1-8 GPIO pin. Both
    wrong, both silent: the bridge simply never heard the radio.
    """

    @pytest.mark.parametrize("button,bit", [(1, 0x01), (2, 0x02), (3, 0x04), (4, 0x08)])
    def test_each_button_maps_to_its_bit_in_byte_zero(self, button, bit):
        r = Cm108Report()
        assert r.read_button(bytes([bit, 0, 0, 0]), button=button) is True
        assert r.read_button(bytes([0, 0, 0, 0]), button=button) is False

    def test_other_buttons_do_not_register(self):
        """VCOS on button 2 must not be tripped by button 1 or 3."""
        r = Cm108Report()
        report = bytes([0x01 | 0x04, 0, 0, 0])
        assert r.read_button(report, button=2) is False

    def test_byte_two_is_not_consulted(self):
        """Regression: byte 2 is the output report's GPIO byte, not input."""
        r = Cm108Report()
        assert r.read_button(bytes([0x00, 0, 0xFF, 0]), button=2) is False

    def test_button_out_of_range_is_refused(self):
        with pytest.raises(ValueError, match="1-4"):
            Cm108Report().read_button(bytes([0, 0, 0, 0]), button=5)

    def test_short_report_is_refused(self):
        with pytest.raises(PttError, match="input report"):
            Cm108Report().read_button(b"", button=2)


class TestHidCosWiring:
    def test_button_defaults_to_two(self):
        """The AIOC's usual VCOS routing is VOL DOWN."""
        from zello_link.hardware.aioc_hid import HidCos

        assert HidCos(None).button == 2

    def test_button_is_configurable(self):
        from zello_link.hardware.aioc_hid import HidCos

        assert HidCos(None, button=3).button == 3

    def test_config_reaches_the_backend(self, tmp_path):
        """hid_button was unreachable from YAML, like cos_pin before it."""
        import yaml

        from zello_link.config import load_config

        data = {
            "config_version": 2,
            "instance": {"name": "c"},
            "zello": {"channel": "C", "username": "u", "auth_token": "tok-abcdef"},
            "sound": {"input_device": "in", "output_device": "out"},
            "ptt": {"mode": "none"},
            "cos": {"mode": "aioc_virtual", "hid_device": "/dev/hidraw0",
                    "hid_button": 3},
            "logging": {"console": False, "file": None},
        }
        p = tmp_path / "b.yaml"
        p.write_text(yaml.safe_dump(data))
        assert load_config(p).cos.hid_button == 3
