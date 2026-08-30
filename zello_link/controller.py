"""BridgeController: the single authority for transmission direction.

Nothing else in the package decides when to key the radio or when to open a
Zello stream. Audio callbacks hand blocks to a queue; the Zello client hands
events in; this class decides what happens.

Half-duplex rules implemented here (spec section 12):
  * First talker wins. Whichever direction starts first holds the bridge.
  * While transmitting Zello->RF, local COS is suppressed. This is also what
    stops the gateway's own transmit audio, bleeding back through the radio's
    receive path, from opening a Zello stream and looping.
  * While transmitting RF->Zello, an incoming Zello stream does not key the
    radio. It is logged as a collision and discarded -- v0.1 never replays a
    competing call late, because stale traffic on a security net is worse
    than an explicitly logged collision.
  * After either direction ends, a guard interval must elapse before the
    opposite direction is accepted.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .audio.levels import CosEvent, apply_gain_db
from .audio.resample import Resampler
from .logging_setup import bind
from .state import Direction, State, direction_of, is_legal
from .zello.opus import OpusEncoder, StreamDecoder
from .zello.protocol import CodecHeader, pack_codec_header

__all__ = ["BridgeController", "StreamMeta", "BridgeStats"]

log = logging.getLogger(__name__)

#: How many capture blocks to hold while waiting for the server's
#: start_stream response. Without this the first syllable of every
#: transmission is lost to the network round trip.
_START_BUFFER_BLOCKS = 25

#: Minimum jitter-buffer depth expressed in inbound packets. Three gives
#: room for one late packet plus one out-of-order arrival without starving
#: playback into a keyed transmitter.
_JITTER_PACKETS = 3


@dataclass(frozen=True)
class StreamMeta:
    """Metadata from an inbound Zello on_stream_start."""

    stream_id: int
    channel: str
    sender: str
    codec_header: CodecHeader


class ZelloTransport(Protocol):
    """The subset of the Zello client the controller depends on."""

    async def start_stream(self, codec_header: str, packet_duration_ms: int) -> int: ...
    async def stop_stream(self, stream_id: int) -> None: ...
    async def send_audio(self, stream_id: int, packet_id: int, payload: bytes) -> None: ...


class AudioSink(Protocol):
    """The playback side of the audio engine."""

    def play(self, pcm: np.ndarray) -> None: ...
    def drain_pending_s(self) -> float: ...
    def flush(self) -> None: ...


@dataclass
class BridgeStats:
    """Counters surfaced by the periodic metrics line and the soak test."""

    zello_to_rf_calls: int = 0
    rf_to_zello_calls: int = 0
    collisions: int = 0
    rejected_no_direction: int = 0
    rejected_guard: int = 0
    rejected_not_ready: int = 0
    ptt_timeouts: int = 0
    faults: int = 0
    concealed_frames: int = 0
    late_packets_dropped: int = 0
    capture_overflows: int = 0
    clipped_rx_samples: int = 0
    clipped_tx_samples: int = 0
    zello_to_rf_ms: float = 0.0
    rf_to_zello_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class BridgeController:
    def __init__(
        self,
        cfg: Any,
        *,
        zello: ZelloTransport,
        audio: AudioSink,
        ptt: Any,
        cos: Any,
        clock: Any = time.monotonic,
    ) -> None:
        self.cfg = cfg
        self.zello = zello
        self.audio = audio
        self.ptt = ptt
        self.cos = cos
        self._clock = clock

        self.log = bind("controller", instance=cfg.instance.name)
        self.stats = BridgeStats()

        self._state = State.IDLE
        self._lock = asyncio.Lock()
        self._guard_until = 0.0
        self._direction_started = 0.0

        # Zello -> RF
        self._inbound: StreamDecoder | None = None
        self._inbound_id: int | None = None
        self._inbound_resampler: Resampler | None = None
        self._tail_task: asyncio.Task[None] | None = None

        # RF -> Zello
        self._encoder: OpusEncoder | None = None
        self._outbound_id: int | None = None
        self._outbound_packet_id = 0
        self._outbound_resampler: Resampler | None = None
        # Resampled capture audio does not arrive in whole Opus frames (20 ms
        # at 11025 Hz is not even a whole sample), so it is accumulated here
        # and drained one exact frame at a time.
        self._encode_buf = np.zeros(0, dtype=np.int16)
        self._start_buffer: collections.deque[np.ndarray] = collections.deque(
            maxlen=_START_BUFFER_BLOCKS
        )

        self._closing = False

        #: Most recent capture-block measurement, for --showmonitor.
        #: Written on the capture path and only ever read for display.
        self.last_block_stats: Any = None

    # -- state ------------------------------------------------------------
    @property
    def state(self) -> State:
        return self._state

    @property
    def direction(self) -> Direction:
        return direction_of(self._state)

    def _set_state(self, new: State) -> None:
        if new is self._state:
            return
        if not is_legal(self._state, new):
            # A bug, not a runtime condition. Fail safe rather than continue
            # from a state we did not intend to be in.
            self.log.error("illegal transition %s -> %s", self._state.value, new.value)
            raise RuntimeError(f"illegal state transition {self._state.value} -> {new.value}")
        self.log.debug("state %s -> %s", self._state.value, new.value)
        self._state = new

    async def wait_for_idle(self, timeout: float = 5.0) -> bool:
        """Await completion of any in-flight tail/hang work.

        Used by shutdown and by tests: the Zello->RF tail runs as its own task
        so the event loop is not blocked while the transmitter drains.
        """
        task = self._tail_task
        if task is not None and not task.done():
            with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError, Exception):
                await asyncio.wait_for(asyncio.shield(task), timeout)
        return self._state is State.IDLE

    def _guard_active(self) -> bool:
        return self._clock() < self._guard_until

    def _arm_guard(self, ms: int) -> None:
        self._guard_until = self._clock() + ms / 1000.0

    # -- lifecycle --------------------------------------------------------
    async def start(self) -> None:
        self.ptt.open()
        self.cos.open()
        if self.cfg.bridge.rf_to_zello:
            self._encoder = OpusEncoder(
                self.cfg.opus.sample_rate,
                application=self.cfg.opus.application,
                bitrate=self.cfg.opus.bitrate,
                complexity=self.cfg.opus.complexity,
            )
            self._outbound_resampler = Resampler(
                self.cfg.sound.sample_rate, self.cfg.opus.sample_rate
            )
            if not self._outbound_resampler.passthrough:
                self.log.info("RF->ZELLO resampling %s", self._outbound_resampler)

        self.log.info("bridge started state=%s", self._state.value)

    async def stop(self) -> None:
        """Ordered shutdown. Never raises."""
        self._closing = True
        if self._tail_task is not None:
            self._tail_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._tail_task

        with contextlib.suppress(Exception):
            if self._outbound_id is not None:
                await self.zello.stop_stream(self._outbound_id)

        self.fail_safe("shutdown")

        with contextlib.suppress(Exception):
            self.cos.close()
        with contextlib.suppress(Exception):
            self.ptt.close()
        if self._encoder is not None:
            with contextlib.suppress(Exception):
                self._encoder.close()
        if self._inbound is not None:
            with contextlib.suppress(Exception):
                self._inbound.close()

    def fail_safe(self, reason: str) -> None:
        """Force the bridge to a safe state. Never raises."""
        self.stats.faults += 1
        self.log.error("FAILSAFE reason=%s state=%s", reason, self._state.value)

        with contextlib.suppress(Exception):
            self.ptt.fail_safe()
        with contextlib.suppress(Exception):
            self.audio.flush()
        with contextlib.suppress(Exception):
            self.cos.suppress(False)

        self._inbound_id = None
        self._outbound_id = None
        self._start_buffer.clear()
        self._state = State.FAILSAFE
        self._state = State.IDLE

    # -- Zello -> RF ------------------------------------------------------
    async def on_zello_stream_start(self, meta: StreamMeta) -> bool:
        """Decide whether to accept an inbound Zello stream and key the radio."""
        async with self._lock:
            if not self.cfg.bridge.zello_to_rf:
                self.stats.rejected_no_direction += 1
                return False

            if meta.channel != self.cfg.zello.channel:
                self.log.debug("ignoring stream for other channel=%s", meta.channel)
                return False

            if self._state is not State.IDLE:
                self.stats.collisions += 1
                if self.cfg.bridge.collision_log:
                    self.log.warning(
                        "collision incoming_zello_while_%s sender=%s stream=%d",
                        self.direction.value,
                        meta.sender,
                        meta.stream_id,
                    )
                return False

            if self._guard_active():
                self.stats.rejected_guard += 1
                self.log.info(
                    "rejected zello stream during guard sender=%s stream=%d",
                    meta.sender, meta.stream_id,
                )
                return False

            self._set_state(State.ZELLO_TO_RF_PREKEY)
            self._inbound_id = meta.stream_id
            # Parse the peer's header rather than assuming our own frame size.
            self._inbound = StreamDecoder(meta.codec_header)
            # The peer picks its own rate, so this is needed even when our own
            # device and Opus rates match. Without it a peer at 8 kHz would
            # play at the wrong pitch and speed into a keyed transmitter.
            self._inbound_resampler = Resampler(
                meta.codec_header.sample_rate, self.cfg.sound.sample_rate
            )

            # Size the jitter buffer in PACKETS, not milliseconds. The peer
            # chooses its own packet duration: a client sending 60 ms packets
            # into the 120 ms default has only two packets of slack, so a
            # single late packet starves playback. Observed live as three
            # underruns in one five-second transmission.
            packet_ms = meta.codec_header.packet_duration_ms
            setter = getattr(self.audio, "set_jitter_target_ms", None)
            if setter is not None and packet_ms > 0:
                setter(max(self.cfg.sound.jitter_ms, packet_ms * _JITTER_PACKETS))
            self.cos.suppress(True)
            self._direction_started = self._clock()

            try:
                await self.ptt.key()
            except Exception as e:
                self.log.error("PTT key failed: %s", e)
                self.fail_safe("ptt_key_failed")
                return False

        self.log.info(
            'ZELLO->RF start sender="%s" stream=%d rate=%d frame_ms=%d',
            meta.sender, meta.stream_id,
            meta.codec_header.sample_rate, meta.codec_header.frame_size_ms,
        )

        # Give the radio time to reach full transmit before audio is released.
        await asyncio.sleep(self.cfg.ptt.pre_key_ms / 1000.0)

        async with self._lock:
            if self._state is State.ZELLO_TO_RF_PREKEY:
                self._set_state(State.ZELLO_TO_RF_ACTIVE)
        return True

    async def on_zello_audio(self, stream_id: int, packet_id: int, payload: bytes) -> None:
        """Decode an inbound packet and queue it for playback into the radio."""
        if self._state is not State.ZELLO_TO_RF_ACTIVE:
            return
        if stream_id != self._inbound_id or self._inbound is None:
            return

        try:
            frames = self._inbound.push(packet_id, payload)
        except Exception:
            self.log.exception("decode failed stream=%d", stream_id)
            return

        for pcm in frames:
            # Resample to the device rate first, then apply gain, so
            # saturation stays the last step before the sound card.
            if self._inbound_resampler is not None:
                pcm = self._inbound_resampler.process(pcm)
                if pcm.size == 0:
                    continue
            out, clipped = apply_gain_db(pcm, self.cfg.sound.tx_gain_db)
            if clipped:
                self.stats.clipped_tx_samples += clipped
            self.audio.play(out)

    async def on_zello_stream_stop(self, stream_id: int) -> None:
        """Drain playback, honour the post-audio delay, then unkey."""
        async with self._lock:
            if self._state not in (State.ZELLO_TO_RF_PREKEY, State.ZELLO_TO_RF_ACTIVE):
                return
            if stream_id != self._inbound_id:
                return
            self._set_state(State.ZELLO_TO_RF_TAIL)

        self._tail_task = asyncio.create_task(self._finish_zello_to_rf(), name="zello-rf-tail")

    async def _finish_zello_to_rf(self) -> None:
        try:
            # Let queued audio actually reach the radio before unkeying,
            # bounded so a stuck device cannot hold the transmitter.
            pending = min(self.audio.drain_pending_s(), self.cfg.sound.jitter_max_ms / 1000.0)
            if pending > 0:
                await asyncio.sleep(pending)

            await asyncio.sleep(self.cfg.ptt.post_audio_ms / 1000.0)
            await self.ptt.unkey()

            elapsed_ms = (self._clock() - self._direction_started) * 1000.0
            self.stats.zello_to_rf_calls += 1
            self.stats.zello_to_rf_ms += elapsed_ms
            if self._inbound is not None:
                self.stats.concealed_frames += self._inbound.concealed_frames
                self.stats.late_packets_dropped += self._inbound.dropped_late
            self.log.info("ZELLO->RF stop duration_ms=%d", int(elapsed_ms))

        except asyncio.CancelledError:
            self.ptt.fail_safe()
            raise
        except Exception:
            self.log.exception("error finishing ZELLO->RF")
            self.fail_safe("zello_to_rf_tail")
            return
        finally:
            if self._inbound is not None:
                with contextlib.suppress(Exception):
                    self._inbound.close()
                self._inbound = None
            self._inbound_id = None
            self._inbound_resampler = None
            reset = getattr(self.audio, "reset_jitter_target", None)
            if reset is not None:
                with contextlib.suppress(Exception):
                    reset()

        async with self._lock:
            self._arm_guard(self.cfg.bridge.rx_guard_ms)
            if self._state is State.ZELLO_TO_RF_TAIL:
                self._set_state(State.IDLE)
            self.cos.suppress(False)

    # -- RF -> Zello ------------------------------------------------------
    async def on_capture_block(self, pcm: np.ndarray) -> None:
        """Handle one capture block: measure, detect COS, encode if streaming."""
        gained, clipped = apply_gain_db(pcm, self.cfg.sound.rx_gain_db)
        if clipped:
            self.stats.clipped_rx_samples += clipped

        try:
            stats, event = self.cos.feed(gained, clipped=clipped)
            self.last_block_stats = stats
        except Exception:
            self.log.exception("COS backend failed")
            self.fail_safe("cos_backend")
            return

        if event is CosEvent.ACTIVE:
            await self._open_rf_to_zello(stats)
        elif event is CosEvent.INACTIVE:
            await self._close_rf_to_zello()

        if self._state is State.RF_TO_ZELLO_START:
            # Hold audio captured while the server round trip completes.
            self._start_buffer.append(gained)
        elif self._state is State.RF_TO_ZELLO_ACTIVE:
            await self._send_block(gained)

    async def _open_rf_to_zello(self, stats: Any) -> None:
        async with self._lock:
            if not self.cfg.bridge.rf_to_zello:
                self.stats.rejected_no_direction += 1
                return
            if self._state is not State.IDLE:
                # Should not happen: COS is suppressed while Zello->RF runs.
                self.stats.collisions += 1
                if self.cfg.bridge.collision_log:
                    self.log.warning("collision cos_active_while_%s", self.direction.value)
                return
            if self._guard_active():
                self.stats.rejected_guard += 1
                return

            # Don't enter the transmit state for a channel that cannot accept
            # audio: the stream would be rejected and the RF audio discarded
            # anyway, and churning through START/HANG on every COS event
            # while disconnected just fills the log.
            if not getattr(self.zello, "channel_ready", True):
                self.stats.rejected_not_ready += 1
                if self.stats.rejected_not_ready == 1:
                    self.log.warning(
                        "COS active but the Zello channel is not ready; "
                        "RF audio is being discarded"
                    )
                return

            self._set_state(State.RF_TO_ZELLO_START)
            self._direction_started = self._clock()
            self._start_buffer.clear()
            self._outbound_packet_id = 0
            # Fresh filter state per transmission: blocks captured while idle
            # never reached the resampler, so its history is stale.
            if self._outbound_resampler is not None:
                self._outbound_resampler.reset()
            self._encode_buf = np.zeros(0, dtype=np.int16)

        header = pack_codec_header(
            self.cfg.opus.sample_rate,
            self.cfg.opus.frames_per_packet,
            self.cfg.opus.frame_ms,
        )
        duration = self.cfg.opus.frame_ms * self.cfg.opus.frames_per_packet

        try:
            stream_id = await self.zello.start_stream(header, duration)
        except Exception as e:
            self.log.error("start_stream failed: %s", e)
            async with self._lock:
                self._set_state(State.RF_TO_ZELLO_HANG)
                self._set_state(State.IDLE)
                self._start_buffer.clear()
            return

        async with self._lock:
            if self._state is not State.RF_TO_ZELLO_START:
                # COS closed while we waited; tear the stream back down.
                with contextlib.suppress(Exception):
                    await self.zello.stop_stream(stream_id)
                return
            self._outbound_id = stream_id
            self._set_state(State.RF_TO_ZELLO_ACTIVE)

        self.log.info(
            "RF->ZELLO start cos=%s level_dbfs=%.1f stream=%d",
            self.cos.name, stats.rms_dbfs, stream_id,
        )

        # Flush what was captured during the round trip so the first syllable
        # is not lost.
        buffered = list(self._start_buffer)
        self._start_buffer.clear()
        for block in buffered:
            await self._send_block(block)

    @property
    def _samples_per_packet(self) -> int:
        return self.cfg.opus.samples_per_frame * self.cfg.opus.frames_per_packet

    async def _send_block(self, pcm: np.ndarray) -> None:
        """Resample, accumulate, and emit whole Opus packets."""
        if self._encoder is None or self._outbound_id is None:
            return

        if self._outbound_resampler is not None:
            pcm = self._outbound_resampler.process(pcm)

        if pcm.size:
            self._encode_buf = (
                np.concatenate((self._encode_buf, pcm)) if self._encode_buf.size else pcm
            )

        await self._drain_encode_buffer()

    async def _drain_encode_buffer(self) -> None:
        """Encode and send every whole packet sitting in the buffer."""
        n = self._samples_per_packet
        while self._encode_buf.size >= n and self._outbound_id is not None:
            chunk = self._encode_buf[:n]
            self._encode_buf = self._encode_buf[n:]
            try:
                payload = self._encoder.encode(chunk)
                await self.zello.send_audio(
                    self._outbound_id, self._outbound_packet_id, payload
                )
                self._outbound_packet_id += 1
            except Exception as e:
                self.log.error("send failed on stream=%s: %s", self._outbound_id, e)
                await self._close_rf_to_zello()
                return

    async def _close_rf_to_zello(self) -> None:
        async with self._lock:
            if self._state not in (State.RF_TO_ZELLO_START, State.RF_TO_ZELLO_ACTIVE):
                return
            self._set_state(State.RF_TO_ZELLO_HANG)
            closing_id = self._outbound_id

        # Push the resampler tail through and send any whole packet it
        # completes, so the last few ms of a transmission are not truncated.
        if closing_id is not None and self._outbound_resampler is not None:
            tail = self._outbound_resampler.flush()
            if tail.size:
                self._encode_buf = np.concatenate((self._encode_buf, tail))
                await self._drain_encode_buffer()

        async with self._lock:
            stream_id = self._outbound_id
            self._outbound_id = None
            self._start_buffer.clear()
            # A sub-packet remainder is dropped: padding it to a full frame
            # would emit audible silence at the end of every transmission.
            self._encode_buf = np.zeros(0, dtype=np.int16)

        if stream_id is not None:
            try:
                await self.zello.stop_stream(stream_id)
            except Exception as e:
                self.log.error("stop_stream failed: %s", e)

            elapsed_ms = (self._clock() - self._direction_started) * 1000.0
            self.stats.rf_to_zello_calls += 1
            self.stats.rf_to_zello_ms += elapsed_ms
            self.log.info("RF->ZELLO stop duration_ms=%d", int(elapsed_ms))

        async with self._lock:
            self._arm_guard(self.cfg.bridge.tx_guard_ms)
            if self._state is State.RF_TO_ZELLO_HANG:
                self._set_state(State.IDLE)

    # -- faults -----------------------------------------------------------
    async def on_zello_disconnected(self) -> None:
        """Terminate outbound stream state, stop playback, unkey."""
        self.log.warning("Zello disconnected; forcing safe state")
        self._outbound_id = None
        self.fail_safe("zello_disconnected")

    def on_ptt_timeout(self, held_s: float) -> None:
        """Called by the PTT watchdog. Already unkeyed by the time we get here."""
        self.stats.ptt_timeouts += 1
        self.fail_safe(f"ptt_watchdog_{held_s:.1f}s")
