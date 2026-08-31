"""PTT backend abstraction and the transmit safety watchdog.

Design rule from the spec: PTT code exists in exactly one place. Backends do
nothing but perform the hardware action they are told to perform;
``BridgeController`` decides direction. Nothing else in the package may call
``key()``.

The safety contract every backend must satisfy:
  * PTT is OFF after ``open()``. The OS/driver's idea of default line states
    is never trusted.
  * PTT is driven OFF on ``close()``, on any exception, and from a ``finally``
    during shutdown -- best effort, never raising past the fail-safe.
  * A keyed period is bounded by ``max_tx_s`` regardless of application state.
"""

from __future__ import annotations

import abc
import asyncio
import logging
import time
from typing import Any, Callable

__all__ = ["PttBackend", "NullPtt", "SafePtt", "PttError", "create_ptt_backend"]

log = logging.getLogger(__name__)


class PttError(Exception):
    """PTT hardware fault. Treated as fail-safe: never keep audio flowing."""


class PttBackend(abc.ABC):
    """One hardware method of asserting push-to-talk."""

    name = "abstract"

    @abc.abstractmethod
    def open(self) -> None:
        """Acquire the device and establish a known un-keyed state."""

    @abc.abstractmethod
    def key(self) -> None:
        """Assert PTT."""

    @abc.abstractmethod
    def unkey(self) -> None:
        """Release PTT."""

    @abc.abstractmethod
    def is_keyed(self) -> bool:
        """Last commanded state. Not a readback of the radio."""

    @abc.abstractmethod
    def close(self) -> None:
        """Release PTT and drop the device. Must tolerate being called twice."""

    def __enter__(self) -> "PttBackend":
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class NullPtt(PttBackend):
    """Backend that keys nothing. For bench runs, tests, and ptt.mode='none'."""

    name = "none"

    def __init__(self) -> None:
        self._keyed = False
        self._open = False
        self.key_count = 0

    def open(self) -> None:
        self._open = True
        self._keyed = False

    def key(self) -> None:
        if not self._open:
            raise PttError("NullPtt.key() before open()")
        self._keyed = True
        self.key_count += 1

    def unkey(self) -> None:
        self._keyed = False

    def is_keyed(self) -> bool:
        return self._keyed

    def close(self) -> None:
        self._keyed = False
        self._open = False


class SafePtt:
    """Async wrapper enforcing the transmit-time ceiling and fail-safe.

    Everything above the hardware layer talks to this, never to a backend
    directly. It guarantees two things the backends cannot:

    * A keyed period never outlives ``max_tx_s``. The watchdog is an
      independent asyncio task, so a lost Zello ``on_stream_stop``, a wedged
      coroutine, or a stalled audio device cannot leave the radio
      transmitting. This is AT-07.
    * ``fail_safe()`` always attempts an unkey and never raises, so it is
      safe to call from an exception handler or a ``finally``.
    """

    def __init__(
        self,
        backend: PttBackend,
        *,
        max_tx_s: float,
        on_timeout: Callable[[float], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._backend = backend
        self._max_tx_s = max_tx_s
        self._on_timeout = on_timeout
        self._clock = clock

        self._watchdog: asyncio.Task[None] | None = None
        self._keyed_at: float | None = None
        self._lock = asyncio.Lock()

        self.timeouts = 0
        self.total_keyed_s = 0.0
        self.key_cycles = 0

    def set_timeout_callback(self, callback: Callable[[float], None] | None) -> None:
        """Attach the controller's watchdog-timeout handler after construction."""
        self._on_timeout = callback

    @property
    def backend_name(self) -> str:
        return self._backend.name

    def is_keyed(self) -> bool:
        return self._backend.is_keyed()

    @property
    def keyed_for_s(self) -> float:
        return 0.0 if self._keyed_at is None else self._clock() - self._keyed_at

    def open(self) -> None:
        self._backend.open()
        # Belt and braces: the backend already establishes an un-keyed state,
        # but AT-02 says the radio must never key during startup.
        self._backend.unkey()

    async def key(self) -> None:
        """Assert PTT and arm the transmit watchdog."""
        async with self._lock:
            if self._backend.is_keyed():
                return
            self._backend.key()
            self._keyed_at = self._clock()
            self.key_cycles += 1
            self._arm_watchdog()

    async def unkey(self) -> None:
        """Release PTT and disarm the watchdog."""
        async with self._lock:
            self._disarm_watchdog()
            if self._keyed_at is not None:
                self.total_keyed_s += self._clock() - self._keyed_at
                self._keyed_at = None
            if self._backend.is_keyed():
                self._backend.unkey()

    def _arm_watchdog(self) -> None:
        self._disarm_watchdog()
        self._watchdog = asyncio.create_task(self._watch(), name="ptt-watchdog")

    def _disarm_watchdog(self) -> None:
        if self._watchdog is not None and not self._watchdog.done():
            self._watchdog.cancel()
        self._watchdog = None

    async def _watch(self) -> None:
        try:
            await asyncio.sleep(self._max_tx_s)
        except asyncio.CancelledError:
            return

        # Deliberately does not take the lock: the whole point is to unkey
        # even if something else is wedged holding it.
        held = self.keyed_for_s
        self.timeouts += 1
        try:
            self._backend.unkey()
        except Exception:
            log.critical("PTT watchdog: unkey FAILED after %.1fs", held, exc_info=True)
        else:
            log.critical(
                "PTT watchdog fired after %.1fs (max_tx_s=%.1f); PTT forced OFF",
                held,
                self._max_tx_s,
            )
        self._keyed_at = None
        if self._on_timeout is not None:
            try:
                self._on_timeout(held)
            except Exception:
                log.exception("PTT timeout callback raised")

    def fail_safe(self) -> None:
        """Force PTT off. Never raises. Safe from any exception boundary."""
        self._disarm_watchdog()
        self._keyed_at = None
        try:
            self._backend.unkey()
        except Exception:
            log.critical("fail-safe unkey failed on %s", self._backend.name, exc_info=True)

    def close(self) -> None:
        """Fail-safe, then release the device. Never raises."""
        self.fail_safe()
        try:
            self._backend.close()
        except Exception:
            log.error("error closing PTT backend %s", self._backend.name, exc_info=True)


def create_ptt_backend(cfg: Any) -> PttBackend:
    """Build the backend named by ``ptt.mode``.

    Backend modules are imported lazily so a host without pyserial or hidapi
    can still run --validate and the unit suite.
    """
    mode = cfg.ptt.mode

    if mode == "none":
        return NullPtt()

    if mode == "serial":
        from .aioc_serial import SerialPtt

        return SerialPtt(cfg.ptt.tty_device, signal=cfg.ptt.serial_signal)

    if mode == "cm108_hid":
        from .aioc_hid import Cm108HidPtt

        return Cm108HidPtt(cfg.ptt.hid_device, gpio_pin=cfg.ptt.gpio_pin)

    raise PttError(f"unknown ptt.mode {mode!r}")
