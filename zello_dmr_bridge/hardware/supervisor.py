"""Hardware supervision: fail safe on device loss, then recover.

Written in response to a real failure. Mid-test the AIOC dropped off the USB
bus entirely -- ``/dev/cu.usbmodem*`` disappeared and every PTT call raised
``OSError: [Errno 6] Device not configured``. The fail-safe path held (it
logged CRITICAL and never raised, so shutdown completed cleanly), but the
bridge then *exited*. AT-09 requires the other half: recovering safely when
the device returns.

Both the sound card and the serial/HID PTT live on the same USB device, so
they are supervised together: losing one means losing both, and a partial
reopen would leave the bridge playing audio into a radio whose transmit state
it cannot control.

The ordering rule on every recovery cycle is fixed and not negotiable:

    fail safe (PTT off)  ->  close everything  ->  wait  ->  reopen

PTT is released before anything else is touched, because a device that is
misbehaving is exactly when the transmitter must not be left keyed.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any, Callable

from ..logging_setup import bind

__all__ = ["HardwareSupervisor"]

log = logging.getLogger(__name__)


class HardwareSupervisor:
    """Runs the capture loop and rebuilds the hardware when it disappears."""

    def __init__(
        self,
        cfg: Any,
        *,
        engine: Any,
        ptt: Any,
        controller: Any,
        on_block: Callable[[Any], Any] | None = None,
        sleep: Callable[[float], Any] = asyncio.sleep,
    ) -> None:
        self.cfg = cfg
        self.engine = engine
        self.ptt = ptt
        self.controller = controller
        self._on_block = on_block or controller.on_capture_block
        self._sleep = sleep

        self.log = bind("hardware", instance=cfg.instance.name)

        self.device_losses = 0
        self.recoveries = 0
        self.failed_attempts = 0
        self.healthy = True

    async def run(self, shutdown: asyncio.Event) -> None:
        """Serve capture blocks until shutdown, recovering across faults."""
        while not shutdown.is_set():
            try:
                await self._serve(shutdown)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if shutdown.is_set():
                    return
                self.device_losses += 1
                self.healthy = False
                self.log.error("audio/PTT device lost: %s", e)

            if shutdown.is_set():
                return

            if not await self._recover(shutdown):
                self.log.critical("hardware did not recover; giving up")
                shutdown.set()
                return

    async def _serve(self, shutdown: asyncio.Event) -> None:
        """Feed capture blocks to the controller until the device fails."""
        async for block in self.engine.capture_blocks():
            if shutdown.is_set():
                return
            try:
                await self._on_block(block)
            except Exception:
                # A controller-level error is not a device loss: fail safe and
                # keep serving rather than tearing down working hardware.
                self.log.exception("capture block handling failed")
                self.controller.fail_safe("capture_loop")

    async def _recover(self, shutdown: asyncio.Event) -> bool:
        """Fail safe, close, and retry the hardware with bounded backoff."""
        # PTT first, always. Best effort -- the device may already be gone.
        with contextlib.suppress(Exception):
            self.controller.fail_safe("hardware_lost")
        with contextlib.suppress(Exception):
            self.ptt.close()
        with contextlib.suppress(Exception):
            self.engine.close()

        delay = self.cfg.hardware.retry_initial_s
        attempt = 0
        limit = self.cfg.hardware.max_attempts

        while not shutdown.is_set():
            attempt += 1
            if limit and attempt > limit:
                return False

            self.log.info("hardware retry %d in %.1fs", attempt, delay)
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(shutdown.wait(), timeout=delay)
            if shutdown.is_set():
                return False

            try:
                # PortAudio caches its device list at init, so a replugged
                # device stays unopenable (paInternalError -9986) until the
                # library itself is restarted. Do this before every reopen.
                reset = getattr(self.engine, "reset_backend", None)
                if reset is not None:
                    reset()

                self.engine.open()
                self.ptt.open()
            except Exception as e:
                self.failed_attempts += 1
                self.log.warning("hardware still unavailable: %s", e)
                # Never leave a half-open interface behind.
                with contextlib.suppress(Exception):
                    self.ptt.close()
                with contextlib.suppress(Exception):
                    self.engine.close()
                delay = min(delay * 2, self.cfg.hardware.retry_max_s)
                continue

            self.recoveries += 1
            self.healthy = True
            self.log.info("hardware recovered after %d attempt(s)", attempt)
            return True

        return False

    def stats(self) -> dict[str, int | bool]:
        return {
            "hw_device_losses": self.device_losses,
            "hw_recoveries": self.recoveries,
            "hw_failed_attempts": self.failed_attempts,
            "hw_healthy": self.healthy,
        }
