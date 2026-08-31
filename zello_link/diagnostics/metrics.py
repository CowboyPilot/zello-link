"""Periodic metrics as a structured log line.

Deliberately not an HTTP endpoint: the bridge stays free of a web server and
its attack surface, and a log line is greppable on an appliance that may have
no operator network. It also gives the 24-hour soak test (AT-11) something
concrete to assert against -- queue depths, overflow counters, and PTT
totals are exactly what "no unbounded growth, no stuck PTT" means.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

from ..logging_setup import bind

__all__ = ["MetricsReporter", "collect"]

log = logging.getLogger(__name__)


def collect(*, controller: Any, audio: Any = None, zello: Any = None, ptt: Any = None) -> dict:
    """Assemble one flat snapshot from every subsystem."""
    snap: dict[str, Any] = {"state": controller.state.value}
    snap.update(controller.stats.as_dict())

    # The radio side reports its own counters, whatever it is: USB device
    # errors for a sound card, packet/sequence stats for chan_usrp.
    radio = getattr(controller, "backend", None)
    if radio is not None and hasattr(radio, "stats"):
        with contextlib.suppress(Exception):
            snap.update(radio.stats())

    if audio is not None and hasattr(audio, "stats"):
        snap.update(audio.stats())
        with contextlib.suppress(Exception):
            snap["playback_queued_ms"] = int(audio.drain_pending_s() * 1000)

    if zello is not None and hasattr(zello, "stats"):
        snap.update(zello.stats())

    if ptt is not None:
        snap["ptt_key_cycles"] = getattr(ptt, "key_cycles", 0)
        snap["ptt_total_keyed_s"] = round(getattr(ptt, "total_keyed_s", 0.0), 1)
        snap["ptt_timeouts"] = getattr(ptt, "timeouts", 0)
        snap["ptt_keyed_now"] = ptt.is_keyed()

    cos = getattr(controller, "cos", None)
    meter = getattr(cos, "meter", None) if cos is not None else None
    if meter is not None and meter.blocks:
        snap["rx_mean_dbfs"] = round(meter.mean_dbfs, 1)
        snap["rx_peak_dbfs"] = round(meter.peak_hold_dbfs, 1)
        snap["rx_clipped_samples"] = meter.clipped_samples

    return snap


class MetricsReporter:
    """Emits a metrics line every ``metrics.interval_s``."""

    def __init__(self, cfg: Any, *, controller: Any, audio: Any = None,
                 zello: Any = None, ptt: Any = None, supervisor: Any = None) -> None:
        self.cfg = cfg
        self.controller = controller
        self.audio = audio
        self.zello = zello
        self.ptt = ptt
        self.supervisor = supervisor
        self.log = bind("metrics", instance=cfg.instance.name)
        self._task: asyncio.Task[None] | None = None

    def snapshot(self) -> dict:
        snap = collect(
            controller=self.controller, audio=self.audio, zello=self.zello, ptt=self.ptt
        )
        if self.supervisor is not None:
            snap.update(self.supervisor.stats())
        return snap

    async def _run(self) -> None:
        try:
            while True:
                await asyncio.sleep(self.cfg.metrics.interval_s)
                snap = self.snapshot()
                self.log.info(
                    "metrics " + " ".join(f"{k}={v}" for k, v in sorted(snap.items()))
                )
                # Peak-hold is per-interval so a single transient does not
                # mask later clipping.
                cos = getattr(self.controller, "cos", None)
                meter = getattr(cos, "meter", None) if cos is not None else None
                if meter is not None:
                    meter.reset_peak()
        except asyncio.CancelledError:
            raise

    def start(self) -> None:
        if self.cfg.metrics.enabled and self._task is None:
            self._task = asyncio.create_task(self._run(), name="metrics")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
