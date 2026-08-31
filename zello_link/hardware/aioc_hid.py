"""AIOC CM108-compatible HID: PTT output and COS input.

BENCH VERIFICATION REQUIRED (spec section 22).

The AIOC README documents that firmware >= 1.2.0 exposes "a CM108 compatible
HID endpoint for CM108-style PTT", but it does not publish the report byte
layout. What is implemented here is the conventional CM108 GPIO output report
used by the established ham packet tools -- which is what "CM108 compatible"
means in practice -- expressed as a small, swappable descriptor rather than
magic numbers scattered through the backend.

Consequences of that, by design:
  * The layout lives in ``Cm108Report``, is pure, and is unit tested. If the
    bench shows a different byte order or GPIO pin, one class changes.
  * ``gpio_pin`` is configurable, because which GPIO drives PTT is a wiring
    decision, not a protocol constant.
  * Nothing silently falls back. If the device is missing or the report is
    rejected, startup fails with the actual error, per section 10.2.

Verify on the bench with ``--diagnose-aioc``, and only then ``--ptt-test``.
"""

from __future__ import annotations

import logging
import platform
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .ptt import PttBackend, PttError

__all__ = [
    "AIOC_VID",
    "AIOC_PID",
    "CM108_DEVICE_IDS",
    "Cm108Report",
    "Cm108HidPtt",
    "HidCos",
    "find_aioc_hid_path",
    "find_cm108_hid_devices",
]

log = logging.getLogger(__name__)

#: AIOC USB identifiers, from the project README (dfu-util -d 1209:7388).
#: How long to wait before concluding that input reports are never coming.
_SILENT_HID_WARN_S = 10.0

#: hidapi prefixes every write with a report ID. CM108-class devices declare
#: no report IDs, so it is always zero.
_HIDAPI_REPORT_ID = b"\x00"

AIOC_VID = 0x1209
AIOC_PID = 0x7388

#: CM108-class interfaces this project knows by sight. They all speak the
#: same 4-byte GPIO output report, so supporting one is supporting all of
#: them -- what differs is only the USB id and which GPIO is wired to PTT.
#:
#: An id missing from this table is not unsupported: set ptt.hid_device
#: explicitly and it will work. The table exists so the common cases need no
#: configuration at all.
CM108_DEVICE_IDS: dict[tuple[int, int], str] = {
    (0x1209, 0x7388): "AIOC All-In-One-Cable",
    # Shared by the Digirig Lite and the sound half of the Digirig Mobile,
    # among many generic adapters -- the id cannot tell them apart.
    (0x0D8C, 0x0012): "C-Media CM108 (Digirig Lite/Mobile, generic USB sound)",
    (0x0D8C, 0x000C): "C-Media CM108",
    (0x0D8C, 0x000E): "C-Media CM108",
    (0x0D8C, 0x013C): "C-Media CM108AH",
    (0x0D8C, 0x0013): "C-Media CM119",
    (0x0D8C, 0x0014): "C-Media CM119A",
}


@dataclass(frozen=True)
class Cm108Report:
    """Byte layout of the CM108 GPIO output report.

    Four bytes, verified against AllStarLink's chan_simpleusb.c, which is
    the reference implementation for these interfaces::

        byte 0  unused      (0x00)
        byte 1  GPIO data       bit N-1 set = GPIO N driven high
        byte 2  GPIO direction  bit N-1 set = GPIO N is an output
        byte 3  unused      (0x00)

    ASL sets ``hid_gpio_loc = 1`` and ``hid_gpio_ctl_loc = 2``, with
    ``hid_io_ptt = 4`` (GPIO 3) and ``hid_gpio_ctl = 0x04`` to make that pin
    an output.

    This was previously written with data at byte 2 and a "mask" at byte 3 --
    off by one, so every value went into the direction register and the data
    register stayed zero. It appeared to work exactly once, on an AIOC, via a
    4-byte raw hidraw write: hidraw consumes the first byte as the report
    number, which shifted the remaining bytes into the right places by
    accident. Through hidapi, with the report id supplied properly, the same
    struct asserted nothing at all -- a Digirig Lite accepted every write and
    never lit its PTT LED.

    ``gpio_pin`` is 1-based to match the silkscreen and the datasheet.
    """

    gpio_pin: int = 3
    report_id: int = 0x00
    #: GPIO data register.
    data_index: int = 1
    #: GPIO direction register. Named "mask" historically; it selects which
    #: pins are outputs, which is why it is set for both key and unkey --
    #: releasing PTT must drive the line low, not float it.
    mask_index: int = 2
    length: int = 4

    #: Input reports are NOT the output layout. The CM108 interrupt-IN report
    #: carries the four button states in byte 0, one bit each:
    #:   bit 0 VOL UP, bit 1 VOL DOWN, bit 2 PLAYBACK MUTE, bit 3 RECORD MUTE
    #: The AIOC signals COS by mapping VCOS onto one of those buttons in its
    #: "CM108 Button Sources" configuration, so COS arrives as a button press
    #: and not as a GPIO level.
    button_index: int = 0

    def __post_init__(self) -> None:
        if not 1 <= self.gpio_pin <= 8:
            raise ValueError(f"gpio_pin must be 1-8, got {self.gpio_pin}")
        if self.length < 4:
            raise ValueError("CM108 report must be at least 4 bytes")
        for idx in (self.data_index, self.mask_index):
            if not 0 <= idx < self.length:
                raise ValueError(f"report index {idx} outside a {self.length}-byte report")
        if self.data_index == self.mask_index:
            raise ValueError("data and mask cannot share a byte")

    @property
    def bit(self) -> int:
        return 1 << (self.gpio_pin - 1)

    def build(self, asserted: bool) -> bytes:
        """Render the output report for the requested PTT state.

        The mask bit is always set: the GPIO is driven both directions rather
        than floated, so releasing PTT actively pulls the line rather than
        leaving the radio's state ambiguous.
        """
        buf = bytearray(self.length)
        buf[0] = self.report_id
        buf[self.data_index] = self.bit if asserted else 0x00
        buf[self.mask_index] = self.bit
        return bytes(buf)

    def read_button(self, report: bytes, *, button: int) -> bool:
        """True if the given CM108 button (1-4) is pressed in an input report.

        Replaces an earlier read_cos() that indexed ``data_index`` -- byte 2,
        the OUTPUT report's GPIO data byte -- against an INPUT report, and
        treated the bit as a 1-8 GPIO pin. Both were wrong, and both failed
        silently: the bridge simply never heard the radio.
        """
        if not 1 <= button <= 4:
            raise ValueError(f"button must be 1-4, got {button}")
        if len(report) <= self.button_index:
            raise PttError(
                f"HID input report is {len(report)} bytes; expected more than "
                f"{self.button_index}"
            )
        return bool(report[self.button_index] & (1 << (button - 1)))


def _import_hid() -> Any:
    try:
        import hid
    except ImportError as e:
        raise PttError(
            "hidapi is required for CM108/HID PTT or AIOC COS. Install with:\n"
            "  pip install 'zello-link[hid]'\n"
            "On Debian/Raspberry Pi OS also: sudo apt install libhidapi-hidraw0"
        ) from e
    return hid


def find_cm108_hid_devices() -> list[dict[str, Any]]:
    """Every attached interface this project recognises as CM108-class.

    Each entry is ``{path, vid, pid, name, product}``. ``path`` is whatever
    form hidapi wants back: the hidraw backend gives "/dev/hidrawN", the
    libusb backend gives a bus id like "1-1.4:1.3", and macOS gives
    "DevSrvsID:...". Auto-detection therefore works on any backend, while a
    hand-written path only works on the one it was copied from.
    """
    hid = _import_hid()
    found: list[dict[str, Any]] = []
    try:
        for d in hid.enumerate():
            key = (d.get("vendor_id", 0), d.get("product_id", 0))
            if key not in CM108_DEVICE_IDS:
                continue
            path = d["path"]
            found.append({
                "path": path.decode(errors="replace") if isinstance(path, bytes) else path,
                "vid": key[0],
                "pid": key[1],
                "name": CM108_DEVICE_IDS[key],
                "product": d.get("product_string") or "",
            })
    except Exception as e:
        raise PttError(f"cannot enumerate HID devices: {e}") from e
    return found


def find_aioc_hid_path(*, vid: int = AIOC_VID, pid: int = AIOC_PID) -> list[str]:
    """Enumerate HID paths for one specific USB id.

    Returns every match so a multi-interface host can be told to disambiguate
    rather than being handed an arbitrary one.
    """
    hid = _import_hid()
    try:
        return [
            d["path"].decode(errors="replace") if isinstance(d["path"], bytes) else d["path"]
            for d in hid.enumerate(vid, pid)
        ]
    except Exception as e:
        raise PttError(f"cannot enumerate HID devices: {e}") from e


def _check_device_node(path: Any) -> None:
    """Refuse a filesystem path that is not a character device.

    Only checks paths that look like filesystem paths: hidapi's libusb
    backend reports bus-style ids such as "1-1:1.3", which are not files.

    This exists because of a real incident. The AIOC dropped off the USB bus
    mid-test (RF ingress browning out the port), taking /dev/hidraw0 with it.
    A later write to that path CREATED a regular file there and reported
    success, having touched no hardware at all -- and a stale regular file at
    a /dev path also stops udev recreating the real node when the device
    returns. Failing loudly beats silently writing into the void.
    """
    if not isinstance(path, str) or not path.startswith("/"):
        return
    p = Path(path)
    if not p.exists():
        raise PttError(
            f"{path} does not exist. The interface may have been unplugged or "
            "dropped off the USB bus; check `dmesg` and replug it."
        )
    if not stat.S_ISCHR(p.stat().st_mode):
        raise PttError(
            f"{path} exists but is not a character device. Something has "
            "replaced the device node with a regular file; remove it and "
            "replug the interface so udev can recreate it."
        )


class _HidDevice:
    """Shared open/close for the HID-backed PTT and COS backends."""

    def __init__(self, path: str | None) -> None:
        self._path = path
        self._dev: Any = None

    def _open_device(self) -> Any:
        hid = _import_hid()
        path = self._path

        if not path:
            found = find_cm108_hid_devices()
            if not found:
                raise PttError(
                    "no CM108-class HID interface found. Known ids: "
                    + ", ".join(f"{v:04x}:{p:04x}" for v, p in CM108_DEVICE_IDS)
                    + ". If your interface is not listed it is still supported: "
                    "set ptt.hid_device (or cos.hid_device) explicitly."
                )
            if len(found) > 1:
                raise PttError(
                    f"{len(found)} CM108-class interfaces present; set an explicit "
                    "path. Found: "
                    + ", ".join(f"{d['path']} ({d['name']})" for d in found)
                )
            path = found[0]["path"]
            log.info(
                "auto-selected %s at %s", found[0]["name"], path
            )

        _check_device_node(path)

        try:
            dev = hid.device()
            dev.open_path(path.encode() if isinstance(path, str) else path)
            dev.set_nonblocking(True)
        except Exception as e:
            raise PttError(f"cannot open HID device {path}: {e}") from e
        return dev

    def _close_device(self) -> None:
        if self._dev is None:
            return
        try:
            self._dev.close()
        except Exception:
            log.error("error closing HID device", exc_info=True)
        finally:
            self._dev = None


class Cm108HidPtt(_HidDevice, PttBackend):
    """PTT via a CM108-style HID GPIO output report."""

    name = "cm108_hid"

    def __init__(
        self,
        device: str | None,
        *,
        report: Cm108Report | None = None,
        gpio_pin: int | None = None,
    ) -> None:
        super().__init__(device)
        if report is not None and gpio_pin is not None:
            raise PttError("pass report or gpio_pin, not both")
        if report is None:
            report = Cm108Report() if gpio_pin is None else Cm108Report(gpio_pin=gpio_pin)
        self.report = report
        self._keyed = False

    def open(self) -> None:
        self._dev = self._open_device()
        # Drive the GPIO low before anything else can key it (AT-02).
        self.unkey()
        log.info("PTT HID opened gpio=%d", self.report.gpio_pin)

    def _write(self, asserted: bool) -> None:
        if self._dev is None:
            raise PttError("Cm108HidPtt used before open()")

        # hidapi takes the report ID as the FIRST byte of write(), ahead of
        # the report itself. These devices declare no report IDs, so that
        # byte is 0x00 and the 4-byte GPIO report follows -- five bytes on
        # the wire to hidapi, four bytes of actual report.
        #
        # Without the prefix hidapi rejects the write with -1. Measured on a
        # Digirig Lite (C-Media 0d8c:0012): a 4-byte write returns -1, the
        # 5-byte form returns 5.
        payload = _HIDAPI_REPORT_ID + self.report.build(asserted)
        try:
            written = self._dev.write(payload)
        except Exception as e:
            raise PttError(f"HID write failed: {e}") from e
        if written < 0:
            raise PttError(
                f"HID write returned {written} for a {len(payload)}-byte report. "
                "The interface rejected it; check the device is not claimed by "
                "another process (Asterisk's chan_simpleusb holds it)."
            )

    def key(self) -> None:
        self._write(True)
        self._keyed = True

    def unkey(self) -> None:
        try:
            self._write(False)
        finally:
            self._keyed = False

    def is_keyed(self) -> bool:
        return self._keyed

    def close(self) -> None:
        if self._dev is not None:
            try:
                self._write(False)
            except Exception:
                log.error("could not drop HID PTT during close", exc_info=True)
        self._close_device()
        self._keyed = False


class HidCos(_HidDevice):
    """Reads COS state from the AIOC HID input report.

    Polled rather than interrupt-driven: hidapi's read is non-blocking here
    and the audio loop already ticks every block, so COS is sampled at block
    rate with no extra thread.
    """

    name = "aioc_hid"

    def __init__(
        self,
        device: str | None,
        *,
        button: int = 2,
        report: Cm108Report | None = None,
    ) -> None:
        super().__init__(device)
        #: Which CM108 button the AIOC maps VCOS onto. Must match its
        #: "CM108 Button Sources" panel; there is no way to discover it.
        self.button = button
        self.report = report or Cm108Report()
        self._state = False
        self._reads = 0
        self._read_errors = 0
        self._opened_at = 0.0
        self._warned_silent = False

    def open(self) -> None:
        self._dev = self._open_device()
        self._state = False
        self._opened_at = time.monotonic()
        self._warned_silent = False
        log.info("COS HID opened button=%d", self.button)

    def poll(self) -> bool:
        """Sample COS. Returns the last known state if no report is pending."""
        if self._dev is None:
            raise PttError("HidCos used before open()")
        try:
            data = self._dev.read(self.report.length, timeout_ms=0)
        except Exception as e:
            self._read_errors += 1
            raise PttError(f"HID read failed: {e}") from e

        if data:
            self._reads += 1
            self._state = self.report.read_button(bytes(data), button=self.button)
        else:
            self._warn_if_silent()
        return self._state

    def _warn_if_silent(self) -> None:
        """Say so when no input report has EVER arrived.

        Otherwise this fails invisibly: poll() keeps returning the last known
        state, so a bridge that can never hear the radio looks healthy and
        simply never opens a stream. Reading these reports is not universally
        permitted -- on macOS the AIOC is a Consumer Control device and the
        system gates input behind Input Monitoring, while PTT output reports
        are unrestricted.
        """
        if self._reads or self._warned_silent or not self._opened_at:
            return
        if time.monotonic() - self._opened_at < _SILENT_HID_WARN_S:
            return
        self._warned_silent = True
        extra = (
            " On macOS grant your terminal Privacy & Security > Input "
            "Monitoring, or use cos.mode='internal_audio'."
            if platform.system() == "Darwin" else
            " Check VCOS is enabled and mapped to a CM108 button, and that the"
            " button matches cos.hid_button."
        )
        log.warning(
            "no HID input reports in %.0fs: COS will never trigger and the "
            "bridge cannot hear the radio.%s", _SILENT_HID_WARN_S, extra,
        )

    @property
    def reads(self) -> int:
        """Input reports received since open. Zero means COS cannot work."""
        return self._reads

    @property
    def state(self) -> bool:
        return self._state

    def close(self) -> None:
        self._close_device()
        self._state = False
