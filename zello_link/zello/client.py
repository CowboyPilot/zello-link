"""Asyncio Zello Channel API client.

Connection health is enforced by WebSocket ping/pong alone. If the peer stops
answering pings within ``keepalive_timeout_s`` the library raises
ConnectionClosed, which covers the case that matters: a TCP connection dying
without ever delivering a FIN, leaving the bridge "connected" but deaf.

An earlier version *also* treated "no application frame within the keepalive
window" as half-open. That was wrong and is deliberately not here: a channel
with nobody talking sends no application frames at all, so it disconnected
healthy bridges every 45 s, and each reconnect made the server kick the
previous session (close code 3003).

Two further asymmetries this client exists to get right:

  * A successful logon does NOT mean the channel is usable. Readiness arrives
    separately via ``on_channel_status``, and sending early is rejected.
  * ``refresh_token`` substitutes for ``auth_token`` -- the *application*
    credential -- and not for username/password.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import random
from typing import Any, Awaitable, Callable

import websockets

from ..logging_setup import bind
from .auth import TokenStore, build_logon_credentials
from .protocol import (
    ProtocolError,
    build_logon,
    build_start_stream,
    build_stop_stream,
    pack_audio_packet,
    unpack_audio_packet,
    unpack_codec_header,
)

__all__ = ["ZelloClient", "ZelloError"]

log = logging.getLogger(__name__)

#: How long to wait for a command response before giving up.
_COMMAND_TIMEOUT_S = 10.0

#: WebSocket ping cadence. Kept under Zello's documented 30 s keepalive so the
#: connection stays alive through idle periods.
_PING_INTERVAL_S = 20.0


class ZelloError(Exception):
    """Protocol-level failure reported by the server or the transport."""


#: Zello's close code for "another session logged on with this account".
KICK_CLOSE_CODE = 3003


def _is_kick(exc: BaseException) -> bool:
    """True when the server closed us out for a duplicate logon."""
    code = getattr(exc, "code", None)
    if code == KICK_CLOSE_CODE:
        return True
    rcvd = getattr(exc, "rcvd", None)
    if getattr(rcvd, "code", None) == KICK_CLOSE_CODE:
        return True
    return "kicked" in str(exc).lower() and str(KICK_CLOSE_CODE) in str(exc)


class ZelloClient:
    """One WebSocket connection to one Zello channel."""

    def __init__(
        self,
        cfg: Any,
        *,
        on_stream_start: Callable[[Any], Awaitable[bool]] | None = None,
        on_audio: Callable[[int, int, bytes], Awaitable[None]] | None = None,
        on_stream_stop: Callable[[int], Awaitable[None]] | None = None,
        on_disconnected: Callable[[], Awaitable[None]] | None = None,
        version: str = "0.1.0",
    ) -> None:
        self.cfg = cfg
        self.log = bind("zello", instance=cfg.instance.name)
        self.version = version

        self._on_stream_start = on_stream_start
        self._on_audio = on_audio
        self._on_stream_stop = on_stream_stop
        self._on_disconnected = on_disconnected

        self.tokens = TokenStore(cfg.zello.refresh_token_file)
        self.tokens.load()

        self._ws: Any = None
        self._seq = 0
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._connected = asyncio.Event()
        self._stop = asyncio.Event()

        self._accepted_streams: set[int] = set()
        # Seqs of commands sent without blocking on the reply.
        self._unawaited: set[int] = set()

        # A successful logon does NOT mean the channel is usable. Zello
        # reports channel readiness separately, and asynchronously -- the
        # first on_channel_status after connect commonly says "offline" and
        # a later one flips to "online". Sending before then is rejected with
        # "channel is not ready".
        self._channel_ready = False

        self.connects = 0
        self.disconnects = 0
        self.half_open_detections = 0
        #: Times the server closed us out because another session logged on
        #: with this account. Repeats mean two bridges share credentials.
        self.kicks = 0

    def set_handlers(
        self,
        *,
        on_stream_start: Callable[[Any], Awaitable[bool]] | None = None,
        on_audio: Callable[[int, int, bytes], Awaitable[None]] | None = None,
        on_stream_stop: Callable[[int], Awaitable[None]] | None = None,
        on_disconnected: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        """Attach controller callbacks after construction.

        The client and the controller each need a reference to the other, so
        one of them has to be wired up second.
        """
        if on_stream_start is not None:
            self._on_stream_start = on_stream_start
        if on_audio is not None:
            self._on_audio = on_audio
        if on_stream_stop is not None:
            self._on_stream_stop = on_stream_stop
        if on_disconnected is not None:
            self._on_disconnected = on_disconnected

    @property
    def connected(self) -> bool:
        return self._ws is not None and self._connected.is_set()

    @property
    def channel_ready(self) -> bool:
        """True once the server reports the channel online.

        Authentication and channel readiness are separate: a bridge can be
        logged on and still unable to send.
        """
        return self._channel_ready

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    # -- connection lifecycle --------------------------------------------
    async def run(self) -> None:
        """Connect, serve, and reconnect until ``stop()`` is called."""
        backoff = self.cfg.zello.reconnect_initial_s
        use_refresh = self.tokens.token is not None

        while not self._stop.is_set():
            try:
                await self._connect_once(use_refresh=use_refresh)
                backoff = self.cfg.zello.reconnect_initial_s
                use_refresh = self.tokens.token is not None

            except asyncio.CancelledError:
                raise
            except Exception as e:
                if _is_kick(e):
                    self.kicks += 1
                    self._warn_about_kick()
                else:
                    self.log.warning("connection failed: %s", e)
                # A refresh token that did not work is worse than useless:
                # retrying with it just burns reconnect attempts.
                if use_refresh:
                    self.log.info("refresh-token logon failed; falling back to credentials")
                    use_refresh = False

            finally:
                await self._teardown()

            if self._stop.is_set():
                break

            # Jitter prevents a fleet of bridges reconnecting in lockstep
            # after a shared network event.
            delay = min(backoff, self.cfg.zello.reconnect_max_s)
            delay *= 0.75 + random.random() * 0.5
            self.log.info("reconnecting in %.1fs", delay)
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._stop.wait(), timeout=delay)
            backoff = min(backoff * 2, self.cfg.zello.reconnect_max_s)

    async def _connect_once(self, *, use_refresh: bool) -> None:
        self.log.info("connecting to %s", self.cfg.zello.server)

        # Liveness is the WebSocket layer's job. If the peer stops answering
        # pings the library raises ConnectionClosed, which is real half-open
        # detection -- it works on a silent channel, where no application
        # frame would ever arrive to time out against.
        async with websockets.connect(
            self.cfg.zello.server,
            ping_interval=_PING_INTERVAL_S,
            ping_timeout=self.cfg.zello.keepalive_timeout_s,
            close_timeout=5,
            max_size=2**20,
        ) as ws:
            self._ws = ws
            self.connects += 1

            # The receive loop MUST be running before logon is sent. _logon
            # awaits a future that only _handle_text resolves, and only the
            # receive loop calls it -- starting the loop after logon would
            # deadlock until the command timeout, with the server's reply
            # sitting unread in the socket.
            receiver = asyncio.create_task(self._receive_loop(), name="zello-recv")
            try:
                logon = asyncio.create_task(
                    self._logon(use_refresh=use_refresh), name="zello-logon"
                )
                done, _ = await asyncio.wait(
                    {receiver, logon}, return_when=asyncio.FIRST_COMPLETED
                )

                if receiver in done:
                    # The connection dropped while authenticating; surface
                    # that error rather than the logon timeout it would cause.
                    logon.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await logon
                    await receiver          # re-raises the real failure
                    return

                await logon                 # re-raises a rejected logon

                self._connected.set()
                self.log.info('Zello connected channel="%s"', self.cfg.zello.channel)
                await receiver
            finally:
                if not receiver.done():
                    receiver.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await receiver

    def _warn_about_kick(self) -> None:
        """Say what a 3003 close actually means.

        Zello permits ONE session per account. This close code means another
        session logged on and took it -- almost always a second bridge, or an
        app signed into the same account. Two bridges sharing credentials
        kick each other indefinitely: each reconnect steals the session back,
        and both spend most of their time in reconnect backoff with traffic
        failing in BOTH directions.

        The bare close-code warning was not enough. One deployment logged 483
        of them over two days while the cause went unrecognised.
        """
        if self.kicks == 1 or self.kicks % 10 == 0:
            self.log.error(
                "kicked by the server (close %d): another session logged on as "
                "%r. Zello allows ONE session per account -- this is the %d%s "
                "time. Check for a second bridge, or an app signed in with the "
                "same account; each will keep stealing the session from the "
                "other and both will be unusable. Give each bridge its own "
                "Zello account.",
                KICK_CLOSE_CODE, self.cfg.zello.username, self.kicks,
                {1: "st", 2: "nd", 3: "rd"}.get(self.kicks % 10, "th"),
            )
        else:
            self.log.warning(
                "kicked again (close %d, %d total): another session is using %r",
                KICK_CLOSE_CODE, self.kicks, self.cfg.zello.username,
            )

    async def _teardown(self) -> None:
        was_connected = self._connected.is_set()
        self._connected.clear()

        self._channel_ready = False

        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(ZelloError("connection closed"))
        self._pending.clear()
        self._unawaited.clear()
        self._accepted_streams.clear()

        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.close()
            self._ws = None

        if was_connected:
            self.disconnects += 1
            if self._on_disconnected is not None:
                with contextlib.suppress(Exception):
                    await self._on_disconnected()

    async def stop(self) -> None:
        self._stop.set()
        await self._teardown()

    # -- logon ------------------------------------------------------------
    async def _logon(self, *, use_refresh: bool) -> None:
        creds = build_logon_credentials(self.cfg, self.tokens, use_refresh=use_refresh)
        cmd = build_logon(
            seq=self._next_seq(),
            channels=[self.cfg.zello.channel],
            version=self.version,
            platform_name=self.cfg.zello.platform_name,
            **creds,
        )
        response = await self._send_command(cmd)

        # Log the shape of the logon result (never its token values) so a
        # partial success -- authenticated but not subscribed -- is visible.
        self.log.debug(
            "logon response: success=%s error=%s keys=%s refresh_used=%s",
            response.get("success"),
            response.get("error"),
            sorted(response.keys()),
            use_refresh,
        )

        if not response.get("success"):
            error = response.get("error", "unknown error")
            if use_refresh:
                self.tokens.clear()
            raise ZelloError(f"logon rejected: {error}")

        refresh = response.get("refresh_token")
        if refresh:
            self.tokens.save(refresh)

    # -- command/response --------------------------------------------------
    async def _send_command(self, cmd: dict[str, Any]) -> dict[str, Any]:
        if self._ws is None:
            raise ZelloError("not connected")

        seq = cmd["seq"]
        fut: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[seq] = fut

        try:
            await self._ws.send(json.dumps(cmd))
            return await asyncio.wait_for(fut, timeout=_COMMAND_TIMEOUT_S)
        except asyncio.TimeoutError as e:
            raise ZelloError(f"no response to {cmd['command']} (seq={seq})") from e
        finally:
            self._pending.pop(seq, None)

    # -- receive loop -----------------------------------------------------
    async def _receive_loop(self) -> None:
        # Deliberately NO application-level read timeout.
        #
        # An earlier version treated "no frame within keepalive_timeout_s" as
        # a half-open socket. That is wrong: a channel with nobody talking
        # sends no application frames at all, so the timeout fired on every
        # idle bridge, and each reconnect made the server kick the previous
        # session (close code 3003). Protocol-level ping/pong, configured on
        # the connection above, is what actually detects a dead peer.
        while not self._stop.is_set():
            message = await self._ws.recv()

            if isinstance(message, bytes):
                await self._handle_binary(message)
            else:
                await self._handle_text(message)

    async def _handle_binary(self, data: bytes) -> None:
        try:
            packet = unpack_audio_packet(data)
        except ProtocolError as e:
            self.log.debug("ignoring binary frame: %s", e)
            return

        if packet.stream_id not in self._accepted_streams:
            return
        if self._on_audio is not None:
            await self._on_audio(packet.stream_id, packet.packet_id, packet.payload)

    async def _handle_text(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            self.log.warning("ignoring malformed JSON frame")
            return

        # A response to one of our commands.
        seq = msg.get("seq")
        if seq is not None and seq in self._pending:
            fut = self._pending[seq]
            if not fut.done():
                fut.set_result(msg)
            return

        # A response to a command we deliberately did not block on. Not
        # awaiting it keeps the half-duplex turnaround fast, but the result
        # still matters: a failed stop_stream can leave a stream open on the
        # server, so it is checked here rather than ignored.
        if seq is not None and seq in self._unawaited:
            self._unawaited.discard(seq)
            if not msg.get("success", True):
                self.log.warning(
                    "command seq=%s failed: %s", seq, msg.get("error", "unknown")
                )
            return

        command = msg.get("command")
        if command == "on_stream_start":
            await self._handle_stream_start(msg)
        elif command == "on_stream_stop":
            await self._handle_stream_stop(msg)
        elif command == "on_channel_status":
            self._handle_channel_status(msg)
        elif command == "on_error":
            self.log.error("server error: %s", msg.get("error"))
        else:
            # Never drop a server message silently: an unhandled command is
            # exactly where a protocol change or an unexpected refusal hides.
            self.log.debug("unhandled server message: %s", json.dumps(msg)[:500])

    def _handle_channel_status(self, msg: dict[str, Any]) -> None:
        """Record channel readiness and surface why it is not ready.

        The error carried here is the only place the server explains a
        refusal. Logging just status/users hides it -- a channel can sit
        "offline" indefinitely while the server is plainly saying why.
        """
        ready = msg.get("status") == "online"
        error = msg.get("error")
        error_type = msg.get("error_type")
        changed = ready != self._channel_ready
        self._channel_ready = ready

        if ready:
            if changed:
                self.log.info(
                    "channel READY status=%s users=%s",
                    msg.get("status"), msg.get("users_online"),
                )
            return

        level = self.log.warning if error else self.log.info
        level(
            "channel NOT READY status=%s users=%s error=%r error_type=%r",
            msg.get("status"), msg.get("users_online"), error, error_type,
        )

        if changed or error:
            hint = self._channel_error_hint(error, error_type)
            if hint:
                self.log.error("channel cannot be joined: %s", hint)

    @staticmethod
    def _channel_error_hint(error: str | None, error_type: str | None) -> str | None:
        """Translate a channel refusal into something an operator can act on."""
        if not error:
            return None
        e = error.lower()

        if "password" in e:
            # The Channel API has no field for a channel password, so a
            # password-protected channel simply cannot be joined this way.
            return (
                "the channel appears to be password-protected. The Zello Channel "
                "API has no channel-password field, so a protected channel cannot "
                "be joined by this bridge. Remove the channel password, or use an "
                "unprotected channel. (This is NOT the account password: a wrong "
                "account password fails the logon itself.)"
            )
        if "not found" in e or "closed" in e:
            return f"the server does not have an open channel by that name ({error!r})"
        if error_type == "configuration":
            return (
                f"the channel's configuration rejects Channel API credentials "
                f"({error!r}). Check the channel's access settings."
            )
        return None

    async def _handle_stream_start(self, msg: dict[str, Any]) -> None:
        from ..controller import StreamMeta

        stream_id = msg.get("stream_id")
        if stream_id is None:
            return

        if msg.get("codec") != "opus" or msg.get("type") != "audio":
            self.log.info("ignoring non-opus stream=%s codec=%s", stream_id, msg.get("codec"))
            return

        try:
            header = unpack_codec_header(msg.get("codec_header", ""))
        except ProtocolError as e:
            self.log.warning("bad codec_header on stream=%s: %s", stream_id, e)
            return

        meta = StreamMeta(
            stream_id=stream_id,
            channel=msg.get("channel", ""),
            sender=msg.get("from", "unknown"),
            codec_header=header,
        )

        accepted = True
        if self._on_stream_start is not None:
            accepted = await self._on_stream_start(meta)
        if accepted:
            self._accepted_streams.add(stream_id)

    async def _handle_stream_stop(self, msg: dict[str, Any]) -> None:
        stream_id = msg.get("stream_id")
        if stream_id is None:
            return
        self._accepted_streams.discard(stream_id)
        if self._on_stream_stop is not None:
            await self._on_stream_stop(stream_id)

    # -- outbound stream (ZelloTransport protocol) -----------------------
    async def start_stream(self, codec_header: str, packet_duration_ms: int) -> int:
        # Fail fast rather than spending a round trip to be told
        # "channel is not ready". The controller uses this to keep RF audio
        # out of a stream that was never going to open.
        if not self._channel_ready:
            raise ZelloError("channel is not ready")

        response = await self._send_command(
            build_start_stream(
                seq=self._next_seq(),
                channel=self.cfg.zello.channel,
                codec_header=codec_header,
                packet_duration_ms=packet_duration_ms,
            )
        )
        if not response.get("success"):
            raise ZelloError(f"start_stream rejected: {response.get('error', 'unknown')}")

        stream_id = response.get("stream_id")
        if stream_id is None:
            raise ZelloError("start_stream response carried no stream_id")
        return int(stream_id)

    async def stop_stream(self, stream_id: int) -> None:
        """End an outbound stream.

        Sent without waiting for the reply: this runs on the half-duplex
        turnaround, and blocking here would hold the bridge out of IDLE for
        a network round trip. The reply is still checked, in _handle_text.
        """
        if self._ws is None:
            return
        seq = self._next_seq()
        self._unawaited.add(seq)
        await self._ws.send(
            json.dumps(
                build_stop_stream(
                    seq=seq,
                    stream_id=stream_id,
                    channel=self.cfg.zello.channel,
                )
            )
        )

    async def send_audio(self, stream_id: int, packet_id: int, payload: bytes) -> None:
        if self._ws is None:
            raise ZelloError("not connected")
        await self._ws.send(pack_audio_packet(stream_id, packet_id, payload))

    def stats(self) -> dict[str, int]:
        return {
            "kicks": self.kicks,
            "connects": self.connects,
            "disconnects": self.disconnects,
            "half_open_detections": self.half_open_detections,
        }
