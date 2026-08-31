"""USRP UDP transport: real sockets on loopback, no ASL required."""

from __future__ import annotations

import asyncio

import pytest

from zello_link.usrp.protocol import VOICE_BYTES, pack_signal, pack_voice
from zello_link.usrp.transport import UsrpEvents, UsrpTransport

SILENCE = b"\x00" * VOICE_BYTES

_next_port = 35000


def ports() -> tuple[int, int]:
    global _next_port
    _next_port += 2
    return _next_port, _next_port + 1


class Recorder:
    def __init__(self):
        self.keys = 0
        self.unkeys = 0
        self.frames: list[bytes] = []

    def events(self) -> UsrpEvents:
        return UsrpEvents(
            on_key=lambda: setattr(self, "keys", self.keys + 1),
            on_audio=self.frames.append,
            on_unkey=lambda: setattr(self, "unkeys", self.unkeys + 1),
        )


async def pair(**kw):
    """A sender and a receiver pointed at each other on loopback."""
    a_port, b_port = ports()
    rec = Recorder()
    rx = UsrpTransport(
        bind_host="127.0.0.1", bind_port=b_port,
        asl_host="127.0.0.1", asl_port=a_port,
        events=rec.events(), **kw,
    )
    tx = UsrpTransport(
        bind_host="127.0.0.1", bind_port=a_port,
        asl_host="127.0.0.1", asl_port=b_port,
    )
    await rx.start()
    await tx.start()
    return tx, rx, rec


async def settle(n: float = 0.15) -> None:
    await asyncio.sleep(n)


class TestVoicePath:
    async def test_key_audio_unkey(self):
        tx, rx, rec = await pair()
        try:
            for _ in range(10):
                tx.send_voice(SILENCE)
            tx.send_unkey()
            await settle()

            assert rec.keys == 1
            assert len(rec.frames) == 10
            assert rec.unkeys == 1
            assert all(len(f) == VOICE_BYTES for f in rec.frames)
        finally:
            await tx.stop(); await rx.stop()

    async def test_payload_is_delivered_intact(self):
        tx, rx, rec = await pair()
        try:
            pcm = bytes(range(256)) + bytes(range(64))
            tx.send_voice(pcm)
            await settle()
            assert rec.frames == [pcm]
        finally:
            await tx.stop(); await rx.stop()

    async def test_counters(self):
        tx, rx, rec = await pair()
        try:
            for _ in range(5):
                tx.send_voice(SILENCE)
            tx.send_unkey()
            await settle()
            assert rx.stats.rx_voice == 5
            assert rx.stats.rx_signal == 1
            assert tx.stats.tx_packets == 6
        finally:
            await tx.stop(); await rx.stop()

    async def test_explicit_key_signal_opens_without_audio(self):
        tx, rx, rec = await pair()
        try:
            tx.send_key()
            await settle()
            assert rec.keys == 1 and not rec.frames
        finally:
            await tx.stop(); await rx.stop()


class TestKeyStateTracking:
    async def test_key_fires_once_per_transmission(self):
        tx, rx, rec = await pair()
        try:
            for _ in range(20):
                tx.send_voice(SILENCE)
            await settle()
            assert rec.keys == 1, "key must not re-fire on every frame"
        finally:
            await tx.stop(); await rx.stop()

    async def test_two_transmissions_give_two_key_events(self):
        tx, rx, rec = await pair()
        try:
            tx.send_voice(SILENCE); tx.send_unkey()
            await settle()
            tx.send_voice(SILENCE); tx.send_unkey()
            await settle()
            assert rec.keys == 2 and rec.unkeys == 2
        finally:
            await tx.stop(); await rx.stop()

    async def test_unkey_without_key_is_ignored(self):
        tx, rx, rec = await pair()
        try:
            tx.send_unkey()
            await settle()
            assert rec.unkeys == 0, "unkey while idle must not fire an event"
        finally:
            await tx.stop(); await rx.stop()


class TestWatchdog:
    """UDP can lose the explicit unkey; the stream must not stay open."""

    async def test_forced_unkey_when_audio_stops(self):
        tx, rx, rec = await pair(rx_unkey_timeout_ms=150)
        try:
            for _ in range(3):
                tx.send_voice(SILENCE)
            await settle(0.1)
            assert rec.unkeys == 0, "too early"

            await asyncio.sleep(0.4)          # let the watchdog fire
            assert rec.unkeys == 1
            assert rx.stats.forced_unkeys == 1
        finally:
            await tx.stop(); await rx.stop()

    async def test_no_forced_unkey_while_audio_flows(self):
        tx, rx, rec = await pair(rx_unkey_timeout_ms=200)
        try:
            for _ in range(12):
                tx.send_voice(SILENCE)
                await asyncio.sleep(0.02)
            assert rx.stats.forced_unkeys == 0
            assert rec.unkeys == 0
        finally:
            await tx.stop(); await rx.stop()


class TestSourceFiltering:
    async def test_foreign_source_is_dropped(self):
        """USRP has no authentication; strict_source is the only guard."""
        a_port, b_port = ports()
        rec = Recorder()
        rx = UsrpTransport(
            bind_host="127.0.0.1", bind_port=b_port,
            asl_host="10.255.255.1", asl_port=a_port,     # not us
            events=rec.events(), strict_source=True,
        )
        await rx.start()
        intruder = UsrpTransport(
            bind_host="127.0.0.1", bind_port=a_port,
            asl_host="127.0.0.1", asl_port=b_port,
        )
        await intruder.start()
        try:
            intruder.send_voice(SILENCE)
            await settle()
            assert not rec.frames, "accepted audio from an unexpected host"
            assert rx.stats.foreign_packets >= 1
        finally:
            await intruder.stop(); await rx.stop()

    async def test_any_source_accepts_it(self):
        a_port, b_port = ports()
        rec = Recorder()
        rx = UsrpTransport(
            bind_host="127.0.0.1", bind_port=b_port,
            asl_host="10.255.255.1", asl_port=a_port,
            events=rec.events(), strict_source=False,
        )
        await rx.start()
        intruder = UsrpTransport(
            bind_host="127.0.0.1", bind_port=a_port,
            asl_host="127.0.0.1", asl_port=b_port,
        )
        await intruder.start()
        try:
            intruder.send_voice(SILENCE)
            await settle()
            assert len(rec.frames) == 1
        finally:
            await intruder.stop(); await rx.stop()


class TestMalformedInput:
    async def _send_raw(self, rx, port, blob):
        sock = UsrpTransport(
            bind_host="127.0.0.1", bind_port=port,
            asl_host="127.0.0.1", asl_port=rx.bind_port,
        )
        await sock.start()
        sock._sendto(blob)
        await settle()
        await sock.stop()

    async def test_garbage_is_counted_not_crashed(self):
        a_port, b_port = ports()
        rec = Recorder()
        rx = UsrpTransport(
            bind_host="127.0.0.1", bind_port=b_port,
            asl_host="127.0.0.1", asl_port=a_port, events=rec.events(),
        )
        await rx.start()
        try:
            await self._send_raw(rx, a_port, b"not a usrp packet at all")
            assert rx.stats.malformed_packets >= 1
            assert not rec.frames
        finally:
            await rx.stop()

    async def test_bad_magic_rejected(self):
        a_port, b_port = ports()
        rec = Recorder()
        rx = UsrpTransport(
            bind_host="127.0.0.1", bind_port=b_port,
            asl_host="127.0.0.1", asl_port=a_port, events=rec.events(),
        )
        await rx.start()
        try:
            bad = b"XXXX" + pack_signal(0, keyed=True)[4:]
            await self._send_raw(rx, a_port, bad)
            assert rx.stats.malformed_packets >= 1
        finally:
            await rx.stop()

    async def test_odd_payload_length_rejected(self):
        a_port, b_port = ports()
        rec = Recorder()
        rx = UsrpTransport(
            bind_host="127.0.0.1", bind_port=b_port,
            asl_host="127.0.0.1", asl_port=a_port, events=rec.events(),
        )
        await rx.start()
        try:
            await self._send_raw(rx, a_port, pack_signal(0, keyed=True) + b"\x00" * 99)
            assert rx.stats.malformed_packets >= 1
            assert not rec.frames
        finally:
            await rx.stop()


class TestSequenceTracking:
    async def test_in_order_has_no_gaps(self):
        tx, rx, rec = await pair()
        try:
            for _ in range(20):
                tx.send_voice(SILENCE)
            await settle()
            assert rx.stats.sequence_gaps == 0
        finally:
            await tx.stop(); await rx.stop()

    async def test_gap_is_counted(self):
        a_port, b_port = ports()
        rec = Recorder()
        rx = UsrpTransport(
            bind_host="127.0.0.1", bind_port=b_port,
            asl_host="127.0.0.1", asl_port=a_port, events=rec.events(),
        )
        await rx.start()
        sender = UsrpTransport(
            bind_host="127.0.0.1", bind_port=a_port,
            asl_host="127.0.0.1", asl_port=b_port,
        )
        await sender.start()
        try:
            sender._sendto(pack_voice(1, SILENCE))
            await settle(0.05)
            sender._sendto(pack_voice(5, SILENCE))     # 2,3,4 lost
            await settle()
            assert rx.stats.sequence_gaps == 1
            assert len(rec.frames) == 2, "audio must still be delivered"
        finally:
            await sender.stop(); await rx.stop()

    async def test_duplicate_is_counted_not_delivered_as_a_gap(self):
        a_port, b_port = ports()
        rec = Recorder()
        rx = UsrpTransport(
            bind_host="127.0.0.1", bind_port=b_port,
            asl_host="127.0.0.1", asl_port=a_port, events=rec.events(),
        )
        await rx.start()
        sender = UsrpTransport(
            bind_host="127.0.0.1", bind_port=a_port,
            asl_host="127.0.0.1", asl_port=b_port,
        )
        await sender.start()
        try:
            sender._sendto(pack_voice(10, SILENCE))
            await settle(0.05)
            sender._sendto(pack_voice(10, SILENCE))
            await settle()
            assert rx.stats.duplicates == 1
            assert rx.stats.sequence_gaps == 0
        finally:
            await sender.stop(); await rx.stop()


class TestShutdown:
    async def test_stop_sends_an_unkey_if_we_were_keyed(self):
        """Never leave the far side latched when the process goes away."""
        tx, rx, rec = await pair()
        try:
            tx.send_voice(SILENCE)
            await settle()
            assert tx.tx_keyed
            await tx.stop()
            await settle()
            assert rec.unkeys == 1, "closing socket left ASL keyed"
        finally:
            await rx.stop()

    async def test_stop_is_quiet_when_idle(self):
        tx, rx, rec = await pair()
        try:
            await tx.stop()
            await settle()
            assert rec.unkeys == 0
        finally:
            await rx.stop()

    async def test_stop_twice_is_safe(self):
        tx, rx, _ = await pair()
        await tx.stop()
        await tx.stop()
        await rx.stop()
