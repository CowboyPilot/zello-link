"""U4: ASL -> Zello, the receive direction of the USRP backend.

Where a radio backend discovers a transmission by measuring audio levels,
chan_usrp is *told*: ASL signals key and unkey in the protocol header. These
tests drive the transport's callbacks directly, which is exactly what an
arriving datagram does, and assert the controller turns them into Zello
streams.

The loopback tests matter most. A USRP node repeats what we send it -- on a
live ASL3 node at duplex=3 a 201-packet transmission came back as 203 packets
-- so without suppression every Zello->RF over would bounce straight back
into the channel as an RF->Zello stream.
"""

from __future__ import annotations

import asyncio
import copy

import numpy as np
import pytest
import yaml

from tests.fakes import FakeEncoder, FakeZello
from zello_link.backends.base import BackendEvents
from zello_link.backends.usrp import UsrpBackend
from zello_link.config import load_config
from zello_link.controller import BridgeController, StreamMeta
from zello_link.state import State
from zello_link.usrp.protocol import VOICE_SAMPLES
from zello_link.zello.protocol import CodecHeader

BASE = {
    "config_version": 2,
    "instance": {"name": "u4", "log_level": "DEBUG"},
    "zello": {"channel": "C", "username": "u", "auth_token": "tok-abcdef"},
    "bridge": {
        "backend": "usrp",
        "rx_guard_ms": 0,
        "tx_guard_ms": 0,
        "min_stream_interval_ms": 0,
    },
    "logging": {"console": False, "file": None},
}

_next_port = 37000


def _ports() -> tuple[int, int]:
    global _next_port
    _next_port += 2
    return _next_port, _next_port + 1


def make_config(tmp_path, **overrides):
    data = copy.deepcopy(BASE)
    bind_port, asl_port = _ports()
    data["usrp"] = {
        "bind_host": "127.0.0.1",
        "bind_port": bind_port,
        "asl_host": "127.0.0.1",
        "asl_port": asl_port,
    }
    for section, values in overrides.items():
        data.setdefault(section, {}).update(values)
    p = tmp_path / "bridge.yaml"
    p.write_text(yaml.safe_dump(data))
    return load_config(p)


HEADER = CodecHeader(sample_rate=16000, frames_per_packet=1, frame_size_ms=20)


def meta(stream_id=4242, sender="Zello-User"):
    return StreamMeta(stream_id=stream_id, channel="C", sender=sender, codec_header=HEADER)


def voice(value: int = 1200) -> bytes:
    """One 20 ms chan_usrp voice frame."""
    return np.full(VOICE_SAMPLES, value, dtype="<i2").tobytes()


async def settle(backend) -> None:
    """Let every dispatched receive task run to completion."""
    for _ in range(10):
        if not backend._pending:
            await asyncio.sleep(0)
            if not backend._pending:
                return
        await asyncio.gather(*list(backend._pending), return_exceptions=True)


@pytest.fixture
async def rig(tmp_path):
    cfg = make_config(tmp_path)
    zello = FakeZello()
    backend = UsrpBackend(cfg)
    ctrl = BridgeController(cfg, zello=zello, backend=backend)
    await ctrl.start()
    ctrl._encoder = FakeEncoder()
    yield ctrl, zello, backend
    await ctrl.stop()


class TestReceiveDirection:
    async def test_rx_key_opens_a_zello_stream(self, rig):
        ctrl, zello, backend = rig
        backend._rx_key()
        await settle(backend)
        assert len(zello.started) == 1
        assert ctrl.state is State.RF_TO_ZELLO_ACTIVE

    async def test_audio_reaches_zello(self, rig):
        ctrl, zello, backend = rig
        backend._rx_key()
        await settle(backend)
        for _ in range(10):
            backend._rx_audio(voice())
        await settle(backend)
        assert zello.sent, "no audio was forwarded to Zello"

    async def test_unkey_closes_the_stream(self, rig):
        ctrl, zello, backend = rig
        backend._rx_key()
        await settle(backend)
        backend._rx_audio(voice())
        await settle(backend)
        backend._rx_unkey()
        await settle(backend)
        assert zello.open_streams == 0
        assert ctrl.state is State.IDLE
        assert ctrl.stats.rf_to_zello_calls == 1

    async def test_audio_before_key_is_ignored(self, rig):
        """A stray voice frame with no keyup must not open a stream."""
        ctrl, zello, backend = rig
        backend._rx_audio(voice())
        await settle(backend)
        assert zello.started == []
        assert ctrl.state is State.IDLE

    async def test_direction_can_be_disabled(self, tmp_path):
        cfg = make_config(tmp_path, bridge={"rf_to_zello": False})
        zello, backend = FakeZello(), UsrpBackend(cfg)
        ctrl = BridgeController(cfg, zello=zello, backend=backend)
        await ctrl.start()
        try:
            backend._rx_key()
            await settle(backend)
            assert zello.started == []
            assert ctrl.stats.rejected_no_direction == 1
        finally:
            await ctrl.stop()


class TestLoopbackSuppression:
    """The node repeats our own audio back at us; it must not reach Zello."""

    async def test_our_own_audio_does_not_open_a_stream(self, rig):
        ctrl, zello, backend = rig
        assert await ctrl.on_zello_stream_start(meta()) is True
        assert ctrl.state is State.ZELLO_TO_RF_ACTIVE

        # The node echoes what we just sent it.
        backend._rx_key()
        backend._rx_audio(voice())
        await settle(backend)

        assert zello.started == [], "our own audio was streamed back to Zello"
        assert ctrl.state is State.ZELLO_TO_RF_ACTIVE

    async def test_suppression_closes_an_rf_stream_already_open(self, rig):
        """Zello->RF winning arbitration must not strand an open RF stream.

        The unkey that would have closed it is about to be suppressed, so the
        backend has to synthesise one or the core sits in RF_TO_ZELLO forever.
        """
        ctrl, zello, backend = rig
        backend._rx_key()
        await settle(backend)
        assert ctrl.state is State.RF_TO_ZELLO_ACTIVE

        backend.suppress_rx(True)
        await settle(backend)

        assert zello.open_streams == 0
        assert ctrl.state is State.IDLE

    async def test_receive_resumes_after_suppression_lifts(self, rig):
        ctrl, zello, backend = rig
        backend.suppress_rx(True)
        backend._rx_key()
        await settle(backend)
        assert zello.started == []

        backend.suppress_rx(False)
        backend._rx_key()
        await settle(backend)
        assert len(zello.started) == 1

    async def test_suppression_is_idempotent(self, rig):
        _, zello, backend = rig
        backend.suppress_rx(True)
        backend.suppress_rx(True)
        backend.suppress_rx(False)
        backend.suppress_rx(False)
        await settle(backend)
        assert zello.stopped == []


class TestDispatch:
    async def test_pending_tasks_are_referenced(self, rig):
        """Without a strong reference the loop may collect a task mid-flight."""
        _, _, backend = rig
        backend._rx_key()
        assert backend._pending, "dispatched task was not retained"
        await settle(backend)
        assert not backend._pending, "task references leak after completion"

    async def test_dispatch_before_start_does_not_warn(self, tmp_path):
        cfg = make_config(tmp_path)
        backend = UsrpBackend(cfg)          # never started: no loop
        backend.set_events(BackendEvents(on_rx_key=lambda: asyncio.sleep(0)))
        backend._rx_key()                   # must close the coroutine, not warn
        assert not backend._pending
