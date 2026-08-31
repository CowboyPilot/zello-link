"""Radio backend over a CM108-class USB interface (AIOC, Digirig, ...).

Wraps the audio engine, the PTT line and the COS detector behind the same
contract the USRP backend implements, so the controller stops caring which
one it is driving.

The transmit timing lives here rather than in the controller because it is
radio-specific: a transmitter needs an attack time before audio is safe to
release, and a tail after the last sample before PTT drops. chan_usrp needs
neither. Putting it in the controller would have meant the controller
knowing which backend it had.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np

from .base import RadioBackend

__all__ = ["AiocBackend"]

log = logging.getLogger(__name__)


class AiocBackend(RadioBackend):
    name = "aioc"

    def __init__(
        self,
        cfg: Any,
        *,
        engine: Any = None,
        ptt: Any = None,
        cos: Any = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.sample_rate = cfg.sound.sample_rate

        # Injected by the CLI, which owns the hardware supervisor; constructed
        # here only for direct/test use.
        self.engine = engine
        self.ptt = ptt
        self.cos = cos

    # -- lifecycle --------------------------------------------------------
    async def start(self) -> None:
        # The supervisor opens the hardware for the AIOC path, so nothing to
        # do here beyond confirming we were wired up.
        if self.ptt is None or self.engine is None:
            raise RuntimeError("AiocBackend needs an engine and a PTT backend")

    async def stop(self) -> None:
        self.fail_safe()

    # -- core -> radio ----------------------------------------------------
    async def key(self) -> None:
        """Assert PTT and wait out the transmitter's attack time.

        Returning before the radio is actually transmitting would clip the
        first syllable of every over.
        """
        await self.ptt.key()
        if self.cfg.ptt.pre_key_ms:
            await asyncio.sleep(self.cfg.ptt.pre_key_ms / 1000.0)

    async def write_audio(self, pcm: np.ndarray) -> None:
        self.engine.play(pcm)

    async def unkey(self) -> None:
        """Let queued audio reach the radio, then drop PTT.

        The drain is bounded: a stuck sound device must not be able to hold
        the transmitter up indefinitely.
        """
        pending = min(
            self.engine.drain_pending_s(), self.cfg.sound.jitter_max_ms / 1000.0
        )
        if pending > 0:
            await asyncio.sleep(pending)
        if self.cfg.ptt.post_audio_ms:
            await asyncio.sleep(self.cfg.ptt.post_audio_ms / 1000.0)
        await self.ptt.unkey()

    # -- loopback suppression --------------------------------------------
    def suppress_rx(self, on: bool) -> Any:
        """Stop our own transmit audio, bleeding back through the radio's
        receive path, from opening a Zello stream."""
        if self.cos is None:
            return None
        return self.cos.suppress(on)

    # -- safety -----------------------------------------------------------
    def fail_safe(self) -> None:
        """PTT off first, then discard queued audio. Never raises."""
        try:
            if self.ptt is not None:
                self.ptt.fail_safe()
        except Exception:
            log.critical("AIOC fail-safe: PTT off failed", exc_info=True)
        try:
            if self.engine is not None:
                self.engine.flush()
        except Exception:
            log.error("AIOC fail-safe: could not flush playback", exc_info=True)
        try:
            if self.cos is not None:
                self.cos.suppress(False)
        except Exception:
            pass

    @property
    def keyed(self) -> bool:
        return bool(self.ptt is not None and self.ptt.is_keyed())

    def stats(self) -> dict[str, Any]:
        out: dict[str, Any] = {"backend": self.name}
        if self.engine is not None and hasattr(self.engine, "stats"):
            out.update(self.engine.stats())
        if self.ptt is not None:
            out["ptt_key_cycles"] = getattr(self.ptt, "key_cycles", 0)
            out["ptt_keyed_now"] = self.ptt.is_keyed()
        return out
