"""Fake ZelloClient, AudioEngine, PTT, and COS backends.

Section 19 requires these so the complete controller can be integration
tested without RF hardware, a sound card, or a network.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import numpy as np

from zello_dmr_bridge.audio.levels import BlockStats, CosEvent
from zello_dmr_bridge.hardware.cos import CosBackend


class FakeClock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance_ms(self, ms: float) -> None:
        self.t += ms / 1000.0


@dataclass
class SentPacket:
    stream_id: int
    packet_id: int
    payload: bytes


class FakeZello:
    """Records control commands and audio; can be told to fail or stall."""

    def __init__(self, *, start_delay_s: float = 0.0) -> None:
        self.started: list[tuple[str, int]] = []
        self.stopped: list[int] = []
        self.sent: list[SentPacket] = []

        self.start_delay_s = start_delay_s
        self.fail_start = False
        self.fail_stop = False
        self.fail_send = False

        self._next_stream_id = 1000

    async def start_stream(self, codec_header: str, packet_duration_ms: int) -> int:
        if self.start_delay_s:
            await asyncio.sleep(self.start_delay_s)
        if self.fail_start:
            raise ConnectionError("simulated start_stream failure")
        self.started.append((codec_header, packet_duration_ms))
        self._next_stream_id += 1
        return self._next_stream_id

    async def stop_stream(self, stream_id: int) -> None:
        if self.fail_stop:
            raise ConnectionError("simulated stop_stream failure")
        self.stopped.append(stream_id)

    async def send_audio(self, stream_id: int, packet_id: int, payload: bytes) -> None:
        if self.fail_send:
            raise ConnectionError("simulated send failure")
        self.sent.append(SentPacket(stream_id, packet_id, payload))

    @property
    def open_streams(self) -> int:
        return len(self.started) - len(self.stopped)


class FakeAudioSink:
    """Collects playback blocks instead of writing to a device."""

    def __init__(self, *, pending_s: float = 0.0) -> None:
        self.played: list[np.ndarray] = []
        self.flushes = 0
        self._pending_s = pending_s

    def play(self, pcm: np.ndarray) -> None:
        self.played.append(pcm)

    def drain_pending_s(self) -> float:
        return self._pending_s

    def flush(self) -> None:
        self.flushes += 1
        self.played.clear()

    @property
    def total_samples(self) -> int:
        return sum(b.size for b in self.played)


class FakeCos(CosBackend):
    """COS backend driven explicitly by the test.

    Suppression is honoured faithfully -- that is the loopback-prevention
    behaviour under test, so faking it away would defeat the purpose.
    """

    name = "fake"

    def __init__(self) -> None:
        self._active = False
        self._suppressed = False
        self._pending: CosEvent | None = None
        self.opened = False
        self.closed = False
        self.fail_feed = False
        self.blocks_fed = 0

    def open(self) -> None:
        self.opened = True

    def close(self) -> None:
        self.closed = True

    def trigger_active(self) -> None:
        self._pending = CosEvent.ACTIVE

    def trigger_inactive(self) -> None:
        self._pending = CosEvent.INACTIVE

    def feed(self, pcm: np.ndarray, *, clipped: int = 0) -> tuple[BlockStats, CosEvent | None]:
        if self.fail_feed:
            raise RuntimeError("simulated COS backend failure")
        self.blocks_fed += 1
        stats = BlockStats(rms_dbfs=-20.0, peak_dbfs=-14.0, clipped_samples=clipped)

        event, self._pending = self._pending, None
        if self._suppressed:
            return stats, None
        if event is CosEvent.ACTIVE:
            self._active = True
        elif event is CosEvent.INACTIVE:
            self._active = False
        return stats, event

    def suppress(self, on: bool) -> CosEvent | None:
        if on == self._suppressed:
            return None
        self._suppressed = on
        if on and self._active:
            self._active = False
            return CosEvent.INACTIVE
        return None

    @property
    def suppressed(self) -> bool:
        return self._suppressed

    @property
    def active(self) -> bool:
        return self._active


class FakeEncoder:
    """Stand-in Opus encoder so controller tests do not require libopus."""

    def __init__(self) -> None:
        self.encoded = 0

    def encode(self, pcm: np.ndarray) -> bytes:
        self.encoded += 1
        return b"opus" + self.encoded.to_bytes(2, "big")

    def close(self) -> None:
        pass


def block(n: int = 320, value: int = 1000) -> np.ndarray:
    return np.full(n, value, dtype=np.int16)
