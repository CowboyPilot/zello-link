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
from dataclasses import dataclass
from typing import Any

from .ptt import PttBackend, PttError

__all__ = ["AIOC_VID", "AIOC_PID", "Cm108Report", "Cm108HidPtt", "HidCos", "find_aioc_hid_path"]

log = logging.getLogger(__name__)

#: AIOC USB identifiers, from the project README (dfu-util -d 1209:7388).
AIOC_VID = 0x1209
AIOC_PID = 0x7388


@dataclass(frozen=True)
class Cm108Report:
    """Byte layout of the CM108 GPIO output report.

    Conventional 4-byte report::

        byte 0  report id   (0x00)
        byte 1  unused      (0x00)
        byte 2  GPIO data   bit N-1 set = GPIO N high
        byte 3  GPIO mask   bit N-1 set = GPIO N is being driven

    ``gpio_pin`` is 1-based to match the silkscreen and the datasheet.
    """

    gpio_pin: int = 3
    report_id: int = 0x00
    data_index: int = 2
    mask_index: int = 3
    length: int = 4

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

    def read_cos(self, report: bytes, *, cos_pin: int) -> bool:
        """Extract a COS bit from an HID input report."""
        if not 1 <= cos_pin <= 8:
            raise ValueError(f"cos_pin must be 1-8, got {cos_pin}")
        if len(report) <= self.data_index:
            raise PttError(
                f"HID input report is {len(report)} bytes; expected more than {self.data_index}"
            )
        return bool(report[self.data_index] & (1 << (cos_pin - 1)))


def _import_hid() -> Any:
    try:
        import hid
    except ImportError as e:
        raise PttError(
            "hidapi is required for CM108/HID PTT or AIOC COS. Install with:\n"
            "  pip install 'zello-dmr-bridge[hid]'\n"
            "On Debian/Raspberry Pi OS also: sudo apt install libhidapi-hidraw0"
        ) from e
    return hid


def find_aioc_hid_path(*, vid: int = AIOC_VID, pid: int = AIOC_PID) -> list[str]:
    """Enumerate AIOC HID device paths.

    Returns every match so a multi-interface host can be told to disambiguate
    rather than being handed an arbitrary one.
    """
    hid = _import_hid()
    try:
        return [d["path"].decode() for d in hid.enumerate(vid, pid)]
    except Exception as e:
        raise PttError(f"cannot enumerate HID devices: {e}") from e


class _HidDevice:
    """Shared open/close for the HID-backed PTT and COS backends."""

    def __init__(self, path: str | None) -> None:
        self._path = path
        self._dev: Any = None

    def _open_device(self) -> Any:
        hid = _import_hid()
        path = self._path

        if not path:
            found = find_aioc_hid_path()
            if not found:
                raise PttError(
                    f"no AIOC HID device found (VID:PID {AIOC_VID:04x}:{AIOC_PID:04x}); "
                    "set the device path explicitly in the config"
                )
            if len(found) > 1:
                raise PttError(
                    f"{len(found)} AIOC HID devices present; set an explicit path. Found: "
                    + ", ".join(found)
                )
            path = found[0]
            log.info("auto-selected AIOC HID device path=%s", path)

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

    def __init__(self, device: str | None, *, report: Cm108Report | None = None) -> None:
        super().__init__(device)
        self.report = report or Cm108Report()
        self._keyed = False

    def open(self) -> None:
        self._dev = self._open_device()
        # Drive the GPIO low before anything else can key it (AT-02).
        self.unkey()
        log.info("PTT HID opened gpio=%d", self.report.gpio_pin)

    def _write(self, asserted: bool) -> None:
        if self._dev is None:
            raise PttError("Cm108HidPtt used before open()")
        payload = self.report.build(asserted)
        try:
            written = self._dev.write(payload)
        except Exception as e:
            raise PttError(f"HID write failed: {e}") from e
        if written < 0:
            raise PttError(f"HID write returned {written}")

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
        cos_pin: int = 4,
        report: Cm108Report | None = None,
    ) -> None:
        super().__init__(device)
        self.cos_pin = cos_pin
        self.report = report or Cm108Report()
        self._state = False
        self._reads = 0
        self._read_errors = 0

    def open(self) -> None:
        self._dev = self._open_device()
        self._state = False
        log.info("COS HID opened pin=%d", self.cos_pin)

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
            self._state = self.report.read_cos(bytes(data), cos_pin=self.cos_pin)
        return self._state

    @property
    def state(self) -> bool:
        return self._state

    def close(self) -> None:
        self._close_device()
        self._state = False
