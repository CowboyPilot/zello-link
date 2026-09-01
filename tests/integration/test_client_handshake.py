"""Logon handshake ordering.

Regression guard for a deadlock found only by connecting to the live server:
``_logon`` awaits a future that is resolved solely by ``_handle_text``, which
is called solely by ``_receive_loop``. Starting the receive loop *after* logon
meant the server's reply sat unread in the socket until the command timeout,
and the bridge never connected.

The fake transport below reproduces that faithfully: a response is delivered
only when something actually calls ``recv()``. A client that sends logon and
then waits without reading will hang here, exactly as it did in service.
"""

from __future__ import annotations

import asyncio
import json

import pytest
import yaml

from zello_link.config import load_config
from zello_link.zello.client import ZelloClient, ZelloError

CONFIG = {
    "config_version": 2,
    "instance": {"name": "handshake"},
    "zello": {
        "channel": "Test Channel",
        "username": "u",
        "auth_token": "tok-abcdef",
        "reconnect_initial_s": 0.05,
        "reconnect_max_s": 0.05,
    },
    "sound": {"input_device": "in", "output_device": "out"},
    "ptt": {"mode": "none"},
    "logging": {"console": False, "file": None},
}


@pytest.fixture
def cfg(tmp_path):
    p = tmp_path / "b.yaml"
    p.write_text(yaml.safe_dump(CONFIG))
    return load_config(p)


class FakeWebSocket:
    """Delivers queued frames only to a caller that reads.

    This is the property that matters: nothing is pushed. If the client does
    not run a receive loop, the logon response is never observed.
    """

    def __init__(self, responder=None):
        self.sent: list[str | bytes] = []
        self._inbox: asyncio.Queue[str | bytes] = asyncio.Queue()
        self._responder = responder
        self.closed = False
        self.recv_calls = 0

    async def send(self, message):
        self.sent.append(message)
        if self._responder is not None and isinstance(message, str):
            reply = self._responder(json.loads(message))
            if reply is not None:
                await self._inbox.put(json.dumps(reply))

    async def recv(self):
        self.recv_calls += 1
        return await self._inbox.get()

    async def close(self):
        self.closed = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.closed = True


def logon_ok(cmd):
    if cmd.get("command") == "logon":
        return {"seq": cmd["seq"], "success": True, "refresh_token": "rt-value-abc"}
    return None


def logon_rejected(cmd):
    if cmd.get("command") == "logon":
        return {"seq": cmd["seq"], "success": False, "error": "not authorized"}
    return None


class TestHandshakeOrdering:
    async def test_logon_completes_when_response_requires_reading(self, cfg):
        """The deadlock: this hangs if the receive loop starts after logon."""
        client = ZelloClient(cfg)
        ws = FakeWebSocket(responder=logon_ok)
        client._ws = ws

        receiver = asyncio.create_task(client._receive_loop())
        try:
            await asyncio.wait_for(client._logon(use_refresh=False), timeout=2.0)
        finally:
            receiver.cancel()
            try:
                await receiver
            except asyncio.CancelledError:
                pass

        assert ws.recv_calls > 0, "logon response was never read from the socket"

    async def test_logon_without_a_reader_times_out(self, cfg):
        """Proves the fake reproduces the bug rather than hiding it."""
        client = ZelloClient(cfg)
        client._ws = FakeWebSocket(responder=logon_ok)

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(client._logon(use_refresh=False), timeout=0.3)

    async def test_refresh_token_is_persisted_on_success(self, cfg):
        client = ZelloClient(cfg)
        client._ws = FakeWebSocket(responder=logon_ok)
        receiver = asyncio.create_task(client._receive_loop())
        try:
            await asyncio.wait_for(client._logon(use_refresh=False), timeout=2.0)
        finally:
            receiver.cancel()
            with pytest.raises(asyncio.CancelledError):
                await receiver

        assert client.tokens.token == "rt-value-abc"

    async def test_rejected_logon_raises(self, cfg):
        client = ZelloClient(cfg)
        client._ws = FakeWebSocket(responder=logon_rejected)
        receiver = asyncio.create_task(client._receive_loop())
        try:
            with pytest.raises(ZelloError, match="not authorized"):
                await asyncio.wait_for(client._logon(use_refresh=False), timeout=2.0)
        finally:
            receiver.cancel()
            with pytest.raises(asyncio.CancelledError):
                await receiver

    async def test_logon_command_carries_channel_and_token(self, cfg):
        client = ZelloClient(cfg)
        ws = FakeWebSocket(responder=logon_ok)
        client._ws = ws
        receiver = asyncio.create_task(client._receive_loop())
        try:
            await asyncio.wait_for(client._logon(use_refresh=False), timeout=2.0)
        finally:
            receiver.cancel()
            with pytest.raises(asyncio.CancelledError):
                await receiver

        cmd = json.loads(ws.sent[0])
        assert cmd["command"] == "logon"
        assert cmd["channels"] == ["Test Channel"]
        assert cmd["auth_token"] == "tok-abcdef"


class TestIdleConnection:
    """An idle channel must not be mistaken for a dead one."""

    async def test_silence_does_not_disconnect(self, cfg):
        """Regression: an idle-data timeout disconnected healthy bridges.

        Nobody talking means no application frames at all. Liveness is the
        WebSocket ping/pong layer's job, not this loop's.
        """
        client = ZelloClient(cfg)
        ws = FakeWebSocket(responder=logon_ok)
        client._ws = ws

        receiver = asyncio.create_task(client._receive_loop())
        await asyncio.wait_for(client._logon(use_refresh=False), timeout=2.0)

        # Stay completely silent for longer than any keepalive window.
        await asyncio.sleep(0.5)

        assert not receiver.done(), "receive loop gave up on an idle connection"
        assert client.half_open_detections == 0

        receiver.cancel()
        with pytest.raises(asyncio.CancelledError):
            await receiver

    async def test_keepalive_timeout_is_a_pong_deadline(self, cfg):
        """It configures ping_timeout; it is not an application-read timeout."""
        import inspect

        source = inspect.getsource(ZelloClient._connect_once)
        assert "ping_timeout=self.cfg.zello.keepalive_timeout_s" in source
        assert "wait_for" not in inspect.getsource(ZelloClient._receive_loop)


class TestCommandCorrelation:
    async def test_out_of_order_responses_match_by_seq(self, cfg):
        """A late response to an earlier command must not resolve a later one."""
        client = ZelloClient(cfg)
        ws = FakeWebSocket()
        client._ws = ws
        receiver = asyncio.create_task(client._receive_loop())
        try:
            first = asyncio.create_task(
                client._send_command({"command": "a", "seq": client._next_seq()})
            )
            second = asyncio.create_task(
                client._send_command({"command": "b", "seq": client._next_seq()})
            )
            await asyncio.sleep(0.05)

            # Answer the second command first.
            await ws._inbox.put(json.dumps({"seq": 2, "success": True, "which": "b"}))
            assert (await asyncio.wait_for(second, 1.0))["which"] == "b"
            assert not first.done()

            await ws._inbox.put(json.dumps({"seq": 1, "success": True, "which": "a"}))
            assert (await asyncio.wait_for(first, 1.0))["which"] == "a"
        finally:
            receiver.cancel()
            with pytest.raises(asyncio.CancelledError):
                await receiver

    async def test_unsolicited_event_does_not_resolve_a_command(self, cfg):
        client = ZelloClient(cfg)
        ws = FakeWebSocket()
        client._ws = ws
        receiver = asyncio.create_task(client._receive_loop())
        try:
            pending = asyncio.create_task(
                client._send_command({"command": "a", "seq": client._next_seq()})
            )
            await asyncio.sleep(0.05)
            await ws._inbox.put(
                json.dumps({"command": "on_channel_status", "status": "online"})
            )
            await asyncio.sleep(0.05)
            assert not pending.done()
            pending.cancel()
        finally:
            receiver.cancel()
            with pytest.raises(asyncio.CancelledError):
                await receiver


class TestChannelReadiness:
    """Logon success does not mean the channel can accept audio.

    Found live: the bridge connected, COS fired in the same second, and
    start_stream came back "channel is not ready". Zello reports readiness
    asynchronously -- the first on_channel_status after connect commonly says
    offline and a later one flips to online.
    """

    async def test_not_ready_until_status_online(self, cfg):
        client = ZelloClient(cfg)
        assert client.channel_ready is False

    async def test_start_stream_refuses_before_ready(self, cfg):
        client = ZelloClient(cfg)
        client._ws = FakeWebSocket()
        with pytest.raises(ZelloError, match="not ready"):
            await client.start_stream("hdr", 20)

    async def test_status_online_makes_it_ready(self, cfg):
        client = ZelloClient(cfg)
        await client._handle_text(
            json.dumps({"command": "on_channel_status", "status": "online",
                        "users_online": 1})
        )
        assert client.channel_ready is True

    async def test_status_offline_clears_readiness(self, cfg):
        client = ZelloClient(cfg)
        await client._handle_text(
            json.dumps({"command": "on_channel_status", "status": "online"})
        )
        await client._handle_text(
            json.dumps({"command": "on_channel_status", "status": "offline"})
        )
        assert client.channel_ready is False

    async def test_disconnect_clears_readiness(self, cfg):
        client = ZelloClient(cfg)
        await client._handle_text(
            json.dumps({"command": "on_channel_status", "status": "online"})
        )
        await client._teardown()
        assert client.channel_ready is False

    async def test_unhandled_commands_are_logged_not_dropped(self, cfg, caplog):
        """A silently dropped server message is where a protocol change hides."""
        import logging

        client = ZelloClient(cfg)
        with caplog.at_level(logging.DEBUG, logger="zello_link.zello"):
            await client._handle_text(json.dumps({"command": "on_some_new_thing"}))
        assert any("unhandled server message" in r.message for r in caplog.records)


class TestChannelErrorDiagnostics:
    """on_channel_status carries the only explanation of a refusal.

    Found live: the channel sat "offline" for the whole session while the
    server was saying error='invalid password', error_type='configuration'
    in every status message -- and the client logged only status and users.
    """

    async def test_channel_error_is_logged(self, cfg, caplog):
        import logging

        client = ZelloClient(cfg)
        with caplog.at_level(logging.WARNING, logger="zello_link.zello"):
            await client._handle_text(json.dumps({
                "command": "on_channel_status", "channel": "C",
                "status": "offline", "users_online": 0,
                "error": "invalid password", "error_type": "configuration",
            }))
        text = " ".join(r.message for r in caplog.records)
        assert "invalid password" in text
        assert "configuration" in text

    async def test_password_error_explains_the_api_limitation(self, cfg):
        hint = ZelloClient._channel_error_hint("invalid password", "configuration")
        assert hint is not None
        assert "password-protected" in hint
        assert "NOT the account password" in hint

    async def test_closed_channel_hint(self, cfg):
        hint = ZelloClient._channel_error_hint("channel_closed", "unknown")
        assert hint is not None and "open channel" in hint

    def test_generic_configuration_hint(self):
        hint = ZelloClient._channel_error_hint("something odd", "configuration")
        assert hint is not None and "access settings" in hint

    def test_no_hint_without_an_error(self):
        assert ZelloClient._channel_error_hint(None, None) is None

    async def test_online_status_still_marks_ready(self, cfg):
        client = ZelloClient(cfg)
        await client._handle_text(json.dumps({
            "command": "on_channel_status", "status": "online", "users_online": 2,
        }))
        assert client.channel_ready is True


class TestKickDiagnosis:
    """Close 3003 means another session took the account.

    Zello permits ONE session per account. Two bridges sharing credentials
    kick each other indefinitely: each reconnect steals the session back, and
    both spend most of their time in backoff with traffic failing in BOTH
    directions. That is what it looks like from the outside -- "it works, then
    stops, then works again" -- and nothing in the bare close-code warning
    pointed at the cause. One deployment logged 483 of them across two days
    before anyone recognised it.
    """

    def test_detects_the_close_code_on_the_exception(self):
        from zello_link.zello.client import _is_kick

        class Closed(Exception):
            code = 3003

        assert _is_kick(Closed()) is True

    def test_detects_it_on_the_received_frame(self):
        from zello_link.zello.client import _is_kick

        class Frame:
            code = 3003

        class Closed(Exception):
            rcvd = Frame()

        assert _is_kick(Closed()) is True

    def test_detects_the_library_message_form(self):
        from zello_link.zello.client import _is_kick

        exc = Exception("received 3003 (registered) kicked; then sent 3003 (registered) kicked")
        assert _is_kick(exc) is True

    def test_ordinary_disconnects_are_not_kicks(self):
        from zello_link.zello.client import _is_kick

        class Closed(Exception):
            code = 1006

        assert _is_kick(Closed()) is False
        assert _is_kick(OSError("connection reset")) is False

    async def test_first_kick_explains_the_cause(self, cfg, caplog):
        import logging

        client = ZelloClient(cfg)
        client.kicks = 1
        with caplog.at_level(logging.ERROR, logger="zello_link.zello"):
            client._warn_about_kick()
        msg = " ".join(r.getMessage() for r in caplog.records)
        assert "ONE session per account" in msg
        assert "second bridge" in msg
        assert cfg.zello.username in msg

    async def test_repeat_kicks_do_not_fill_the_disk(self, cfg, caplog):
        """A fight between two bridges produces one of these every minute."""
        import logging

        client = ZelloClient(cfg)
        with caplog.at_level(logging.WARNING, logger="zello_link.zello"):
            for n in range(1, 10):
                client.kicks = n
                client._warn_about_kick()
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert len(errors) == 1, "the full explanation should not repeat every time"

    async def test_kick_count_is_exposed_for_metrics(self, cfg):
        """So a fight is visible in the metrics line, not only in the log."""
        client = ZelloClient(cfg)
        client.kicks = 7
        assert client.stats()["kicks"] == 7
