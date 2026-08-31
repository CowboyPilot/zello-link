"""Reorder buffer and transport together, over a real UDP socket.

The unit tests drive JitterBuffer directly. These check the wiring: that the
transport routes received audio through it, resets it per transmission, and
flushes it before closing the stream -- the last of which is the difference
between a clean over and one whose final words vanish.
"""

from __future__ import annotations

import asyncio

import pytest

from zello_link.usrp.protocol import VOICE_BYTES, pack_signal, pack_voice
from zello_link.usrp.transport import UsrpEvents, UsrpTransport

_next_port = 39000


def ports() -> tuple[int, int]:
    global _next_port
    _next_port += 2
    return _next_port, _next_port + 1


def frame(n: int) -> bytes:
    return bytes([n & 0xFF]) * VOICE_BYTES


class Recorder:
    def __init__(self):
        self.frames: list[bytes] = []
        self.keys = 0
        self.unkeys = 0

    def events(self):
        return UsrpEvents(
            on_key=lambda: setattr(self, "keys", self.keys + 1),
            on_audio=self.frames.append,
            on_unkey=lambda: setattr(self, "unkeys", self.unkeys + 1),
        )


@pytest.fixture
async def rig():
    a, b = ports()
    rec = Recorder()
    rx = UsrpTransport(
        bind_host="127.0.0.1", bind_port=b,
        asl_host="127.0.0.1", asl_port=a,
        events=rec.events(), jitter_buffer_ms=60,
    )
    tx = UsrpTransport(
        bind_host="127.0.0.1", bind_port=a,
        asl_host="127.0.0.1", asl_port=b,
    )
    await rx.start()
    await tx.start()
    yield tx, rx, rec
    await tx.stop()
    await rx.stop()


async def settle(t: float = 0.15) -> None:
    await asyncio.sleep(t)


class TestReorderingOverTheWire:
    async def test_out_of_order_arrivals_are_put_back_in_order(self, rig):
        tx, rx, rec = rig
        for seq in (1, 3, 2, 4):
            tx._sendto(pack_voice(seq, frame(seq)))
        await settle()
        tx._sendto(pack_signal(5, keyed=False))
        await settle()

        assert rec.frames == [frame(1), frame(2), frame(3), frame(4)]
        assert rx.stats.rx_concealed_frames == 0

    async def test_nothing_is_lost_at_end_of_transmission(self, rig):
        """Frames still held when the talker stops must be released.

        Without the flush the window's worth of audio at the end of every
        over -- the last 60 ms -- would be silently discarded.
        """
        tx, rx, rec = rig
        for seq in range(1, 4):
            tx._sendto(pack_voice(seq, frame(seq)))
        await settle()
        assert len(rec.frames) < 3, "window should still be holding some"

        tx._sendto(pack_signal(9, keyed=False))
        await settle()
        assert rec.frames == [frame(1), frame(2), frame(3)]
        assert rec.unkeys == 1

    async def test_audio_precedes_the_unkey_callback(self, rig):
        """The core closes its Zello stream on unkey; audio must land first."""
        tx, rx, rec = rig
        order: list[str] = []
        rec_events = rec.events()
        rx.events = UsrpEvents(
            on_key=rec_events.on_key,
            on_audio=lambda f: (order.append("audio"), rec.frames.append(f))[0],
            on_unkey=lambda: order.append("unkey"),
        )
        for seq in range(1, 4):
            tx._sendto(pack_voice(seq, frame(seq)))
        await settle()
        tx._sendto(pack_signal(9, keyed=False))
        await settle()

        assert "unkey" in order
        assert order.index("unkey") == len(order) - 1, "unkey must come last"


class TestPerTransmissionState:
    async def test_buffer_resets_between_overs(self, rig):
        """Sequence numbers from the previous over say nothing about this one."""
        tx, rx, rec = rig
        tx._sendto(pack_voice(1, frame(1)))
        await settle()
        tx._sendto(pack_signal(2, keyed=False))
        await settle()
        before = rx.stats.rx_concealed_frames

        # A new transmission starting far away must not look like a huge gap.
        tx._sendto(pack_signal(500, keyed=True))
        tx._sendto(pack_voice(501, frame(2)))
        await settle()
        tx._sendto(pack_signal(502, keyed=False))
        await settle()

        assert rx.stats.rx_concealed_frames == before
        assert rec.keys == 2
        assert rec.frames[-1] == frame(2)
