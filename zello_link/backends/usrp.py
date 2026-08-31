"""AllStarLink backend: chan_usrp over UDP.

No sound card, no serial line, no audio-level COS. Key state is carried in
the protocol itself, which is strictly better than inferring it from audio
levels -- all the threshold, attack, hang and min_tx tuning the AIOC path
needs simply does not apply here.

Audio crosses the backend boundary at ``sample_rate`` (8 kHz, fixed by
chan_usrp). The resampling between that and the Opus rate happens here, using
the same streaming polyphase resampler as the AIOC path, so a block boundary
is never audible.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np

from ..audio.resample import Resampler
from ..usrp.protocol import FRAME_MS, SAMPLE_RATE, VOICE_SAMPLES
from ..usrp.transport import UsrpEvents, UsrpTransport
from .base import RadioBackend

__all__ = ["UsrpBackend"]

log = logging.getLogger(__name__)


class UsrpBackend(RadioBackend):
    name = "usrp"
    sample_rate = SAMPLE_RATE

    def __init__(self, cfg: Any, *, transport: UsrpTransport | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self._loop: asyncio.AbstractEventLoop | None = None

        self._transport = transport or UsrpTransport(
            bind_host=cfg.usrp.bind_host,
            bind_port=cfg.usrp.bind_port,
            asl_host=cfg.usrp.asl_host,
            asl_port=cfg.usrp.asl_port,
            strict_source=cfg.usrp.strict_source,
            rx_unkey_timeout_ms=cfg.usrp.rx_unkey_timeout_ms,
        )

        # chan_usrp wants exactly 160 samples per datagram. Whatever block
        # size the core hands us gets re-cut to that, so a mismatch upstream
        # cannot produce a short frame on the wire.
        self._tx_buf = np.zeros(0, dtype=np.int16)
        self._keyed = False

    # -- lifecycle --------------------------------------------------------
    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._transport.events = UsrpEvents(
            on_key=self._rx_key,
            on_audio=self._rx_audio,
            on_unkey=self._rx_unkey,
        )
        await self._transport.start()
        log.info(
            "usrp backend ready: %s:%d -> %s:%d",
            self.cfg.usrp.bind_host, self.cfg.usrp.bind_port,
            self.cfg.usrp.asl_host, self.cfg.usrp.asl_port,
        )

    async def stop(self) -> None:
        try:
            await self._transport.stop()      # sends an unkey if we were keyed
        except Exception:
            log.error("error stopping the USRP transport", exc_info=True)
        self._keyed = False

    # -- core -> ASL ------------------------------------------------------
    async def key(self) -> None:
        """No transmitter attack time to wait for; ASL keys on the packet."""
        self._tx_buf = np.zeros(0, dtype=np.int16)
        self._transport.send_key()
        self._keyed = True

    async def write_audio(self, pcm: np.ndarray) -> None:
        if pcm.dtype != np.int16:
            pcm = pcm.astype(np.int16)
        self._tx_buf = (
            np.concatenate((self._tx_buf, pcm)) if self._tx_buf.size else pcm
        )
        while self._tx_buf.size >= VOICE_SAMPLES:
            frame = self._tx_buf[:VOICE_SAMPLES]
            self._tx_buf = self._tx_buf[VOICE_SAMPLES:]
            self._transport.send_voice(frame.tobytes())
            self._keyed = True

    async def unkey(self) -> None:
        """Drop the partial tail, then signal end of transmission explicitly.

        A sub-frame remainder is under 20 ms; padding it to a full frame
        would append audible silence to every over. Going quiet without the
        unkey packet is not enough -- ASL would sit on its receive timeout.
        """
        if self.cfg.usrp.tx_hang_ms:
            await asyncio.sleep(self.cfg.usrp.tx_hang_ms / 1000.0)
        self._tx_buf = np.zeros(0, dtype=np.int16)
        self._transport.send_unkey()
        self._keyed = False

    # -- ASL -> core ------------------------------------------------------
    # The transport calls these from the datagram callback, which is not a
    # coroutine, so each hands off to the loop rather than blocking it.
    def _dispatch(self, coro: Any) -> None:
        if self._loop is None:
            return
        self._loop.create_task(coro)

    def _rx_key(self) -> None:
        if self.events.on_rx_key:
            self._dispatch(self.events.on_rx_key())

    def _rx_audio(self, payload: bytes) -> None:
        if not self.events.on_rx_audio:
            return
        pcm = np.frombuffer(payload, dtype="<i2")
        self._dispatch(self.events.on_rx_audio(pcm))

    def _rx_unkey(self) -> None:
        if self.events.on_rx_unkey:
            self._dispatch(self.events.on_rx_unkey())

    # -- safety -----------------------------------------------------------
    def fail_safe(self) -> None:
        """Tell ASL to stop, best effort, without ever raising."""
        try:
            if self._keyed:
                self._transport.send_unkey()
        except Exception:
            log.error("USRP fail-safe unkey failed", exc_info=True)
        finally:
            self._keyed = False
            self._tx_buf = np.zeros(0, dtype=np.int16)

    @property
    def keyed(self) -> bool:
        return self._keyed

    def stats(self) -> dict[str, Any]:
        s = self._transport.stats.as_dict()
        s["backend"] = self.name
        s["usrp_tx_keyed"] = self._keyed
        s["usrp_rx_keyed"] = self._transport.rx_keyed
        return s

    def make_resampler(self, opus_rate: int) -> Resampler:
        """Converter between the Opus rate and chan_usrp's fixed 8 kHz."""
        return Resampler(opus_rate, self.sample_rate)

    @property
    def frame_ms(self) -> int:
        return FRAME_MS
