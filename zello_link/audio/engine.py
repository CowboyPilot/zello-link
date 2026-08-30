"""Audio capture and playback via PortAudio, with bounded queues.

The hard rule: PortAudio callbacks never block, never allocate unboundedly,
and never touch the network. They move fixed-size blocks across thread-safe
queues and return. Everything else happens on the asyncio side.

Queue policy is explicit rather than implicit:
  * Playback (Zello -> RF) is a jitter buffer. Playback does not begin until
    ``jitter_ms`` of audio has accumulated, so network jitter does not become
    dropouts on a keyed transmitter. Above ``jitter_max_ms`` the oldest audio
    is dropped -- the buffer is bounded, so it can never grow into latency.
  * Capture (RF -> Zello) drops per ``overflow_policy`` and counts every
    dropped block. Silent loss is not acceptable on a voice link.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import queue
import threading
from typing import Any, Callable

import numpy as np

from .devices import AudioDevice, check_device_settings, resolve_device

__all__ = ["AudioEngine", "AudioEngineError"]

log = logging.getLogger(__name__)


class AudioEngineError(Exception):
    """Audio device fault. Always treated as fail-safe by the controller."""


class AudioEngine:
    """Owns the input and output streams and the queues between them."""

    def __init__(
        self,
        cfg: Any,
        *,
        on_fault: Callable[[str], None] | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self.cfg = cfg
        self._on_fault = on_fault
        self._loop = loop

        self.block_samples = cfg.sound.samples_per_block
        self.sample_rate = cfg.sound.sample_rate
        self.channels = cfg.sound.channels

        blocks_per_s = self.sample_rate / self.block_samples
        capture_max = max(2, int(cfg.sound.capture_queue_ms * blocks_per_s / 1000))

        # Jitter thresholds are tracked in SAMPLES, not blocks. Once a
        # resampler is in the path, a queued chunk is no longer one block --
        # 20 ms at 11025 Hz does not even land on a whole sample. Counting
        # blocks would make the buffer depth depend on how the audio happened
        # to be chunked.
        self._jitter_target_samples = max(
            1, int(self.sample_rate * cfg.sound.jitter_ms / 1000)
        )
        self._jitter_max_samples = max(
            self.block_samples, int(self.sample_rate * cfg.sound.jitter_max_ms / 1000)
        )

        # Playback: a deque of chunks plus an exact sample count, read from
        # the PortAudio callback. The callback assembles exactly the frames
        # it was asked for, spanning chunk boundaries.
        self._play_buf: collections.deque[np.ndarray] = collections.deque()
        self._play_samples = 0
        self._play_lock = threading.Lock()
        self._priming = True

        # Capture: a bounded queue drained by the asyncio side.
        self._capture_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=capture_max)

        self._in_stream: Any = None
        self._out_stream: Any = None
        self._input_device: AudioDevice | None = None
        self._output_device: AudioDevice | None = None
        self._closing = False
        # Set when a PortAudio stream dies unexpectedly. capture_blocks
        # raises on it so the supervisor sees a device loss instead of
        # spinning forever on a queue that will never fill again.
        self._faulted: str | None = None

        # Counters surfaced through metrics.
        self.capture_overflows = 0
        self.playback_underruns = 0
        self.playback_drops = 0
        self.input_errors = 0
        self.output_errors = 0

    # -- lifecycle --------------------------------------------------------
    def open(self) -> None:
        import sounddevice as sd

        self._closing = False
        self._faulted = None

        self._loop = self._loop or asyncio.get_event_loop()

        if self.cfg.sound.input_device is not None:
            self._input_device = resolve_device(self.cfg.sound.input_device, "input")
            check_device_settings(
                self._input_device, "input",
                sample_rate=self.sample_rate, channels=self.channels,
            )
            self._in_stream = sd.InputStream(
                device=self._input_device.index,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.block_samples,
                dtype="int16",
                callback=self._capture_callback,
                finished_callback=self._input_finished,
            )
            self._in_stream.start()
            log.info("audio input opened device=%r", self._input_device.name)

        if self.cfg.sound.output_device is not None:
            self._output_device = resolve_device(self.cfg.sound.output_device, "output")
            check_device_settings(
                self._output_device, "output",
                sample_rate=self.sample_rate, channels=self.channels,
            )
            self._out_stream = sd.OutputStream(
                device=self._output_device.index,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.block_samples,
                dtype="int16",
                callback=self._playback_callback,
                finished_callback=self._output_finished,
            )
            self._out_stream.start()
            log.info("audio output opened device=%r", self._output_device.name)

    def reset_backend(self) -> None:
        """Re-initialise PortAudio so a replugged device is re-enumerated.

        PortAudio snapshots the device list when it initialises. After a USB
        unplug/replug the cached entry for the old device is stale, and
        opening it fails with paInternalError (-9986) *forever* -- even
        though the device is present again and the OS has re-enumerated it.
        Only Pa_Terminate followed by Pa_Initialize refreshes that list.

        Observed live: the AIOC dropped off the bus, came back as
        /dev/cu.usbmodemcab102015, and every reopen attempt still failed
        until PortAudio itself was restarted.
        """
        try:
            import sounddevice as sd

            sd._terminate()
            sd._initialize()
            log.info("PortAudio re-initialised; device list refreshed")
        except Exception:
            log.warning("could not re-initialise PortAudio", exc_info=True)

    def close(self) -> None:
        self._closing = True
        for stream, label in ((self._in_stream, "input"), (self._out_stream, "output")):
            if stream is None:
                continue
            try:
                stream.stop()
                stream.close()
            except Exception:
                log.error("error closing audio %s stream", label, exc_info=True)
        self._in_stream = None
        self._out_stream = None
        self.flush()

    # -- PortAudio callbacks (real-time thread; must not block) ----------
    def _capture_callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        if status:
            self.input_errors += 1
        try:
            block = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
            self._capture_q.put_nowait(block)
        except queue.Full:
            self.capture_overflows += 1
            if self.cfg.sound.overflow_policy == "drop_oldest":
                try:
                    self._capture_q.get_nowait()
                    self._capture_q.put_nowait(block)
                except (queue.Empty, queue.Full):
                    pass
            # drop_newest: the block is simply discarded.

    def _take_locked(self, frames: int) -> np.ndarray:
        """Pull exactly ``frames`` samples, spanning queued chunks.

        Caller holds ``_play_lock``. A partially consumed chunk is put back at
        the front, so a chunk boundary never becomes a gap in the output.
        """
        out = np.zeros(frames, dtype=np.int16)
        filled = 0

        while filled < frames and self._play_buf:
            chunk = self._play_buf.popleft()
            take = min(len(chunk), frames - filled)
            out[filled : filled + take] = chunk[:take]
            filled += take
            if take < len(chunk):
                self._play_buf.appendleft(chunk[take:])

        self._play_samples -= filled
        if filled < frames:
            # Ran dry mid-callback; the remainder stays silent.
            self.playback_underruns += 1
            self._priming = True
        return out

    def _playback_callback(self, outdata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        if status:
            self.output_errors += 1

        with self._play_lock:
            # Prime the jitter buffer before releasing the first sample, so
            # network jitter does not become dropouts on a keyed transmitter.
            if self._priming:
                if self._play_samples < self._jitter_target_samples:
                    outdata[:] = 0
                    return
                self._priming = False

            if self._play_samples == 0:
                self.playback_underruns += 1
                self._priming = True          # re-prime after a gap
                outdata[:] = 0
                return

            block = self._take_locked(frames)

        if outdata.ndim > 1:
            outdata[:, 0] = block
        else:
            outdata[:] = block

    def _input_finished(self) -> None:
        if not self._closing:
            self._fault("audio input stream finished unexpectedly")

    def _output_finished(self) -> None:
        if not self._closing:
            self._fault("audio output stream finished unexpectedly")

    def _fault(self, reason: str) -> None:
        log.error("audio fault: %s", reason)
        self._faulted = reason
        if self._on_fault is None or self._loop is None:
            return
        try:
            self._loop.call_soon_threadsafe(self._on_fault, reason)
        except RuntimeError:
            pass

    # -- playback side (AudioSink protocol) ------------------------------
    def play(self, pcm: np.ndarray) -> None:
        """Queue decoded audio for the radio. Never blocks, never grows."""
        if pcm.size == 0:
            return
        with self._play_lock:
            self._play_buf.append(pcm)
            self._play_samples += pcm.size

            # Bound in samples so the buffer can never grow into latency,
            # dropping the oldest audio rather than the newest.
            while self._play_samples > self._jitter_max_samples and self._play_buf:
                dropped = self._play_buf.popleft()
                self._play_samples -= dropped.size
                self.playback_drops += 1

    def set_jitter_target_ms(self, ms: float) -> None:
        """Raise the priming depth for the stream about to play.

        A remote client picks its own packet duration, and the buffer has to
        be measured in *packets*, not milliseconds: at the 120 ms default a
        peer sending 60 ms packets has only two packets of slack, so one late
        packet starves the output. Observed live as three underruns in a
        five-second transmission.

        Only ever raises within this stream; never above ``jitter_max_ms``.
        """
        target = max(1, int(self.sample_rate * ms / 1000))
        target = min(target, self._jitter_max_samples)
        if target != self._jitter_target_samples:
            log.info(
                "jitter target %d -> %d ms",
                int(self._jitter_target_samples * 1000 / self.sample_rate),
                int(target * 1000 / self.sample_rate),
            )
        with self._play_lock:
            self._jitter_target_samples = target

    def reset_jitter_target(self) -> None:
        """Return to the configured target after a stream ends."""
        with self._play_lock:
            self._jitter_target_samples = max(
                1, int(self.sample_rate * self.cfg.sound.jitter_ms / 1000)
            )

    def drain_pending_s(self) -> float:
        """Seconds of audio still queued for playback."""
        with self._play_lock:
            return self._play_samples / self.sample_rate

    def flush(self) -> None:
        """Discard queued playback audio and re-arm priming."""
        with self._play_lock:
            self._play_buf.clear()
            self._play_samples = 0
            self._priming = True

    # -- capture side -----------------------------------------------------
    async def capture_blocks(self):
        """Async iterator over captured blocks.

        The blocking queue read runs in a thread so the event loop stays free
        while waiting on the sound card.
        """
        loop = asyncio.get_running_loop()
        while not self._closing:
            if self._faulted:
                raise AudioEngineError(self._faulted)
            try:
                block = await loop.run_in_executor(None, self._capture_q.get, True, 0.5)
            except queue.Empty:
                continue
            except Exception:
                if self._closing:
                    return
                raise
            if block is None:
                return
            yield block

    def stats(self) -> dict[str, int]:
        return {
            "capture_overflows": self.capture_overflows,
            "playback_underruns": self.playback_underruns,
            "playback_drops": self.playback_drops,
            "input_errors": self.input_errors,
            "output_errors": self.output_errors,
        }
