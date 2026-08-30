"""AIOC serial PTT over the CDC-ACM (ttyACM) interface.

AIOC firmware >= 1.2.0 asserts PTT on DTR=1 with RTS=0. Both lines are driven
explicitly on open: pyserial and the underlying tty driver each have their own
notion of default line state, and on some platforms opening the port asserts
DTR -- which would key the transmitter at startup. AT-02 forbids that.
"""

from __future__ import annotations

import logging
from typing import Any

from .ptt import PttBackend, PttError

__all__ = ["SerialPtt"]

log = logging.getLogger(__name__)


class SerialPtt(PttBackend):
    """PTT via DTR/RTS on an AIOC serial port."""

    name = "serial"

    def __init__(self, device: str, *, baudrate: int = 115200) -> None:
        if not device:
            raise PttError("SerialPtt requires a tty device path")
        self.device = device
        self.baudrate = baudrate
        self._port: Any = None
        self._keyed = False

    def open(self) -> None:
        try:
            import serial
        except ImportError as e:
            raise PttError(
                "pyserial is required for ptt.mode='serial'. Install with:\n"
                "  pip install 'zello-link[serial]'"
            ) from e

        try:
            # dsrdtr=False stops pyserial from asserting DTR as it opens; the
            # explicit unkey below then establishes the known-safe state.
            self._port = serial.Serial(
                self.device,
                baudrate=self.baudrate,
                timeout=0,
                write_timeout=0,
                dsrdtr=False,
                rtscts=False,
                exclusive=True,
            )
        except Exception as e:
            raise PttError(f"cannot open PTT serial device {self.device}: {e}") from e

        self._safe_lines()
        log.info("PTT serial opened device=%s", self.device)

    def _safe_lines(self) -> None:
        """Establish the documented non-keying state: DTR=0, RTS=0."""
        try:
            self._port.dtr = False
            self._port.rts = False
        except Exception as e:
            raise PttError(f"cannot set safe line state on {self.device}: {e}") from e
        self._keyed = False

    def key(self) -> None:
        if self._port is None:
            raise PttError("SerialPtt.key() before open()")
        try:
            # RTS first, then DTR: DTR is the asserting line, so it is set
            # last and cleared first.
            self._port.rts = False
            self._port.dtr = True
        except Exception as e:
            self._keyed = False
            raise PttError(f"cannot assert PTT on {self.device}: {e}") from e
        self._keyed = True

    def unkey(self) -> None:
        if self._port is None:
            self._keyed = False
            return
        try:
            self._port.dtr = False
        except Exception as e:
            raise PttError(f"cannot release PTT on {self.device}: {e}") from e
        finally:
            self._keyed = False

    def is_keyed(self) -> bool:
        return self._keyed

    def close(self) -> None:
        if self._port is None:
            return
        try:
            self._port.dtr = False
            self._port.rts = False
        except Exception:
            log.error("could not drop PTT lines on %s during close", self.device, exc_info=True)
        try:
            self._port.close()
        except Exception:
            log.error("error closing %s", self.device, exc_info=True)
        finally:
            self._port = None
            self._keyed = False
