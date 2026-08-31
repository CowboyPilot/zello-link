"""Asyncio UDP endpoint for chan_usrp.

Keeps the socket out of the bridge core: the backend above this speaks in
semantic events (rx_key / rx_audio / rx_unkey) and never sees a datagram.

Spec section 7 requirements implemented here: non-blocking receive, exact
bind, optional strict source filtering, independent TX/RX sequence tracking,
rate-limited diagnostics, and an explicit unkey on shutdown if we were keyed.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

from .jitter import JitterBuffer
from .protocol import (
    UsrpPacket,
    UsrpProtocolError,
    pack_signal,
    pack_voice,
    unpack,
)

__all__ = ["UsrpTransport", "UsrpStats", "UsrpEvents", "UsrpBindError"]


class UsrpBindError(OSError):
    """The local UDP socket could not be opened."""

log = logging.getLogger(__name__)

#: Seconds between repeats of the same complaint. USRP is a 50 packet/second
#: stream; a malformed sender would otherwise fill the disk.
_LOG_THROTTLE_S = 5.0

#: A sequence jump larger than this is treated as a restart rather than loss,
#: so a peer that restarts does not log a gap of four billion.
_SEQ_RESTART_THRESHOLD = 1000


@dataclass
class UsrpStats:
    rx_packets: int = 0
    tx_packets: int = 0
    rx_voice: int = 0
    rx_signal: int = 0
    malformed_packets: int = 0
    foreign_packets: int = 0
    sequence_gaps: int = 0
    duplicates: int = 0
    forced_unkeys: int = 0
    last_rx_monotonic: float = 0.0

    # Reorder buffer. Named apart from the controller's concealed_frames /
    # late_packets_dropped, which count the Zello->RF Opus path -- conflating
    # the two directions would make both useless for diagnosis.
    rx_concealed_frames: int = 0
    rx_late_dropped: int = 0
    rx_jitter_overflows: int = 0

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class UsrpEvents:
    """Semantic callbacks. The core bridge never sees a datagram."""

    on_key: Callable[[], None] | None = None
    on_audio: Callable[[bytes], None] | None = None
    on_unkey: Callable[[], None] | None = None


class _Protocol(asyncio.DatagramProtocol):
    def __init__(self, owner: "UsrpTransport") -> None:
        self._owner = owner

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self._owner._transport = transport

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        self._owner._on_datagram(data, addr)

    def error_received(self, exc: Exception) -> None:
        self._owner._on_socket_error(exc)

    def connection_lost(self, exc: Exception | None) -> None:
        if exc is not None:
            self._owner._on_socket_error(exc)


class UsrpTransport:
    """One UDP endpoint talking to one chan_usrp channel."""

    def __init__(
        self,
        *,
        bind_host: str,
        bind_port: int,
        asl_host: str,
        asl_port: int,
        strict_source: bool = True,
        rx_unkey_timeout_ms: int = 500,
        jitter_buffer_ms: int = 60,
        max_jitter_buffer_ms: int = 200,
        packet_loss_fill: str = "silence",
        events: UsrpEvents | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.bind_host = bind_host
        self.bind_port = bind_port
        self.asl_host = asl_host
        self.asl_port = asl_port
        self.strict_source = strict_source
        self.rx_unkey_timeout_s = rx_unkey_timeout_ms / 1000.0
        self.events = events or UsrpEvents()
        self._clock = clock

        self.stats = UsrpStats()
        self._jitter = JitterBuffer(
            target_ms=jitter_buffer_ms,
            max_ms=max_jitter_buffer_ms,
            fill=packet_loss_fill,
        )
        self._transport: Any = None
        self._tx_seq = 0
        self._rx_expected: int | None = None
        self._rx_keyed = False
        self._tx_keyed = False
        self._last_log: dict[str, float] = {}
        self._watchdog: asyncio.Task[None] | None = None
        self._resolved_peer: tuple[str, int] | None = None

    # -- lifecycle --------------------------------------------------------
    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        try:
            await loop.create_datagram_endpoint(
                lambda: _Protocol(self),
                local_addr=(self.bind_host, self.bind_port),
            )
        except OSError as e:
            # A stack trace here says nothing useful. The causes are few and
            # each has an obvious fix.
            raise UsrpBindError(
                f"cannot bind UDP {self.bind_host}:{self.bind_port}: {e}\n"
                "  - already in use? another bridge instance, or a leftover "
                "`python -m zello_link.usrp.probe`\n"
                "  - cannot assign requested address? usrp.bind_host must be an "
                "address on THIS host\n"
                "  - permission denied? ports below 1024 need privileges"
            ) from e
        self._watchdog = asyncio.create_task(self._watch_rx(), name="usrp-rx-watchdog")
        log.info(
            "usrp bound %s:%d -> ASL %s:%d",
            self.bind_host, self.bind_port, self.asl_host, self.asl_port,
        )

    async def stop(self) -> None:
        """Close the socket, telling ASL we have stopped talking first."""
        if self._watchdog is not None:
            self._watchdog.cancel()
            try:
                await self._watchdog
            except (asyncio.CancelledError, Exception):
                pass
            self._watchdog = None

        # Never leave the far side latched: if we were mid-transmission, send
        # an explicit unkey before the socket goes away.
        if self._tx_keyed:
            try:
                self.send_unkey()
            except Exception:
                log.error("could not send the closing unkey", exc_info=True)

        if self._transport is not None:
            self._transport.close()
            self._transport = None

    # -- transmit ---------------------------------------------------------
    def _next_seq(self) -> int:
        seq = self._tx_seq
        self._tx_seq = (self._tx_seq + 1) & 0xFFFFFFFF
        return seq

    def _sendto(self, payload: bytes) -> None:
        if self._transport is None:
            raise RuntimeError("USRP transport is not started")
        self._transport.sendto(payload, (self.asl_host, self.asl_port))
        self.stats.tx_packets += 1

    def send_voice(self, pcm: bytes) -> None:
        """Send one 20 ms frame. Marks us keyed on the first frame."""
        self._sendto(pack_voice(self._next_seq(), pcm, keyed=True))
        self._tx_keyed = True

    def send_key(self) -> None:
        self._sendto(pack_signal(self._next_seq(), keyed=True))
        self._tx_keyed = True

    def send_unkey(self) -> None:
        """Explicit end of transmission. Going silent alone is not enough."""
        self._sendto(pack_signal(self._next_seq(), keyed=False))
        self._tx_keyed = False

    @property
    def tx_keyed(self) -> bool:
        return self._tx_keyed

    @property
    def rx_keyed(self) -> bool:
        return self._rx_keyed

    # -- receive ----------------------------------------------------------
    def _throttled(self, key: str) -> bool:
        """True if this complaint was logged too recently."""
        now = self._clock()
        if now - self._last_log.get(key, 0.0) < _LOG_THROTTLE_S:
            return True
        self._last_log[key] = now
        return False

    def _on_datagram(self, data: bytes, addr: tuple[str, int]) -> None:
        if self.strict_source and not self._is_expected_source(addr):
            self.stats.foreign_packets += 1
            if not self._throttled("foreign"):
                log.warning(
                    "ignoring USRP datagram from %s:%d (expecting %s:%d); "
                    "set strict_source=false only on a trusted network",
                    addr[0], addr[1], self.asl_host, self.asl_port,
                )
            return

        try:
            packet = unpack(data)
        except UsrpProtocolError as e:
            self.stats.malformed_packets += 1
            if not self._throttled("malformed"):
                log.warning("malformed USRP datagram from %s:%d: %s", addr[0], addr[1], e)
            return

        self.stats.rx_packets += 1
        self.stats.last_rx_monotonic = self._clock()
        self._track_sequence(packet)

        if packet.is_voice:
            self.stats.rx_voice += 1
            self._deliver_voice(packet)
        else:
            self.stats.rx_signal += 1
            self._deliver_signal(packet)

    def _is_expected_source(self, addr: tuple[str, int]) -> bool:
        # ASL may send from an address that differs from the one we send to
        # (multi-homed hosts), so match the port and remember what we saw.
        if addr[0] == self.asl_host:
            return True
        if self._resolved_peer is not None and addr == self._resolved_peer:
            return True
        return False

    def _track_sequence(self, packet: UsrpPacket) -> None:
        if self._rx_expected is None:
            self._rx_expected = (packet.seq + 1) & 0xFFFFFFFF
            return

        if packet.seq == self._rx_expected:
            self._rx_expected = (packet.seq + 1) & 0xFFFFFFFF
            return

        delta = (packet.seq - self._rx_expected) & 0xFFFFFFFF
        if delta == 0xFFFFFFFF or delta > 0xFFFFFFFF - _SEQ_RESTART_THRESHOLD:
            self.stats.duplicates += 1
            if not self._throttled("duplicate"):
                log.debug("usrp rx duplicate/late seq=%d", packet.seq)
            return

        if delta < _SEQ_RESTART_THRESHOLD:
            self.stats.sequence_gaps += 1
            if not self._throttled("gap"):
                log.warning(
                    "usrp rx sequence gap expected=%d received=%d",
                    self._rx_expected, packet.seq,
                )
        # A huge jump is a restarted peer, not loss; resync quietly.
        self._rx_expected = (packet.seq + 1) & 0xFFFFFFFF

    def _start_rx(self) -> None:
        self._rx_keyed = True
        # Fresh ordering state per transmission: sequence numbers from the
        # previous over say nothing about this one.
        self._jitter.reset()
        log.info("usrp->zello key")
        if self.events.on_key:
            self.events.on_key()

    def _emit(self, frames: list[bytes]) -> None:
        self.stats.rx_concealed_frames = self._jitter.concealed
        self.stats.rx_late_dropped = self._jitter.late_dropped
        self.stats.rx_jitter_overflows = self._jitter.overflows
        if not self.events.on_audio:
            return
        for frame in frames:
            self.events.on_audio(frame)

    def _deliver_voice(self, packet: UsrpPacket) -> None:
        if packet.keyed and not self._rx_keyed:
            self._start_rx()

        # Through the reorder buffer rather than straight out: UDP can hand
        # us frames out of order, and encoding them that way garbles speech.
        self._emit(self._jitter.push(packet.seq, packet.payload))

        if not packet.keyed and self._rx_keyed:
            # A voice frame marked unkeyed ends the transmission.
            self._end_rx("unkey flag on a voice frame")

    def _deliver_signal(self, packet: UsrpPacket) -> None:
        if packet.keyed and not self._rx_keyed:
            self._start_rx()
        elif not packet.keyed and self._rx_keyed:
            self._end_rx("explicit unkey")

    def _end_rx(self, reason: str) -> None:
        # Release what the buffer still holds before closing the stream, or
        # the last few frames of every over would be dropped.
        self._emit(self._jitter.flush())
        self._rx_keyed = False
        log.info("usrp->zello unkey (%s)", reason)
        if self.events.on_unkey:
            self.events.on_unkey()

    async def _watch_rx(self) -> None:
        """Force an unkey if keyed audio stops without an unkey packet.

        UDP gives no delivery guarantee, so the explicit unkey can simply be
        lost. Without this the Zello stream would stay open indefinitely.
        """
        try:
            while True:
                await asyncio.sleep(self.rx_unkey_timeout_s / 2)
                if not self._rx_keyed:
                    continue
                idle = self._clock() - self.stats.last_rx_monotonic
                if idle >= self.rx_unkey_timeout_s:
                    self.stats.forced_unkeys += 1
                    log.warning(
                        "usrp forced unkey: %.0f ms without packet",
                        idle * 1000.0,
                    )
                    self._end_rx("watchdog")
        except asyncio.CancelledError:
            raise

    def _on_socket_error(self, exc: Exception) -> None:
        log.error("usrp socket error: %s", exc)
