"""USRP backend: the RadioBackend contract over a real loopback socket."""

from __future__ import annotations

import asyncio
import copy

import numpy as np
import pytest
import yaml

from zello_link.backends.base import BackendEvents, RadioBackend, create_backend
from zello_link.config import load_config
from zello_link.usrp.protocol import VOICE_SAMPLES
from zello_link.usrp.transport import UsrpEvents, UsrpTransport

BASE = {
    "config_version": 2,
    "instance": {"name": "usrp-test"},
    "zello": {"channel": "C", "username": "u", "auth_token": "tok-abcdef"},
    "sound": {"input_device": "in", "output_device": "out"},
    "ptt": {"mode": "none"},
    "bridge": {"backend": "usrp"},
    "logging": {"console": False, "file": None},
}

_next_port = 36000


def ports() -> tuple[int, int]:
    global _next_port
    _next_port += 2
    return _next_port, _next_port + 1


def make_config(tmp_path, **overrides):
    data = copy.deepcopy(BASE)
    for section, values in overrides.items():
        data.setdefault(section, {}).update(values)
    p = tmp_path / "bridge.yaml"
    p.write_text(yaml.safe_dump(data))
    return load_config(p)


class Peer:
    """Stands in for chan_usrp on the other end of the socket."""

    def __init__(self):
        self.keys = 0
        self.unkeys = 0
        self.frames: list[bytes] = []

    def events(self):
        return UsrpEvents(
            on_key=lambda: setattr(self, "keys", self.keys + 1),
            on_audio=self.frames.append,
            on_unkey=lambda: setattr(self, "unkeys", self.unkeys + 1),
        )


async def rig(tmp_path, **overrides):
    """Backend on one socket, a stand-in chan_usrp on the other.

    Both ends run with the reorder window off. These tests assert on frames
    the moment they are written, and a window legitimately holds frames back
    until flush -- which would make every assertion here about the buffer
    rather than the backend. JitterBuffer has its own tests.
    """
    ours, theirs = ports()
    usrp = {"bind_host": "127.0.0.1", "bind_port": ours,
            "asl_host": "127.0.0.1", "asl_port": theirs,
            "jitter_buffer_ms": 0}
    usrp.update(overrides.pop("usrp", {}))
    cfg = make_config(tmp_path, usrp=usrp, **overrides)
    peer = Peer()
    peer_tx = UsrpTransport(
        bind_host="127.0.0.1", bind_port=theirs,
        asl_host="127.0.0.1", asl_port=ours,
        events=peer.events(), jitter_buffer_ms=0,
    )
    await peer_tx.start()
    backend = create_backend(cfg)
    await backend.start()
    return backend, peer_tx, peer


class TestContract:
    async def test_create_backend_picks_usrp(self, tmp_path):
        b, peer_tx, _ = await rig(tmp_path)
        try:
            assert isinstance(b, RadioBackend)
            assert b.name == "usrp"
            assert b.sample_rate == 8000
        finally:
            await b.stop(); await peer_tx.stop()

    async def test_create_backend_defaults_to_aioc(self, tmp_path):
        cfg = make_config(tmp_path, bridge={"backend": "aioc"})
        assert cfg.bridge.backend == "aioc"

    async def test_starts_idle(self, tmp_path):
        b, peer_tx, peer = await rig(tmp_path)
        try:
            assert not b.keyed
            await asyncio.sleep(0.1)
            assert peer.keys == 0
        finally:
            await b.stop(); await peer_tx.stop()


class TestTransmitPath:
    async def test_key_audio_unkey(self, tmp_path):
        b, peer_tx, peer = await rig(tmp_path)
        try:
            await b.key()
            for _ in range(5):
                await b.write_audio(np.zeros(VOICE_SAMPLES, dtype=np.int16))
            await b.unkey()
            await asyncio.sleep(0.2)

            assert peer.keys == 1
            assert len(peer.frames) == 5
            assert peer.unkeys == 1
            assert not b.keyed
        finally:
            await b.stop(); await peer_tx.stop()

    async def test_repacketises_to_160_samples(self, tmp_path):
        """The core's block size must not leak onto the wire."""
        b, peer_tx, peer = await rig(tmp_path)
        try:
            await b.key()
            # 500 samples is not a multiple of 160.
            await b.write_audio(np.zeros(500, dtype=np.int16))
            await asyncio.sleep(0.2)
            assert len(peer.frames) == 3, "expected floor(500/160) whole frames"
            assert all(len(f) == VOICE_SAMPLES * 2 for f in peer.frames)
        finally:
            await b.stop(); await peer_tx.stop()

    async def test_remainder_carries_into_the_next_write(self, tmp_path):
        b, peer_tx, peer = await rig(tmp_path)
        try:
            await b.key()
            await b.write_audio(np.zeros(100, dtype=np.int16))
            assert len(peer.frames) == 0
            await b.write_audio(np.zeros(60, dtype=np.int16))   # now 160
            await asyncio.sleep(0.2)
            assert len(peer.frames) == 1
        finally:
            await b.stop(); await peer_tx.stop()

    async def test_payload_is_little_endian_int16(self, tmp_path):
        b, peer_tx, peer = await rig(tmp_path)
        try:
            await b.key()
            pcm = np.arange(VOICE_SAMPLES, dtype=np.int16)
            await b.write_audio(pcm)
            await asyncio.sleep(0.2)
            assert np.array_equal(
                np.frombuffer(peer.frames[0], dtype="<i2"), pcm
            )
        finally:
            await b.stop(); await peer_tx.stop()

    async def test_partial_tail_is_dropped_not_padded(self, tmp_path):
        """Padding would append audible silence to every transmission."""
        b, peer_tx, peer = await rig(tmp_path)
        try:
            await b.key()
            await b.write_audio(np.ones(80, dtype=np.int16))    # half a frame
            await b.unkey()
            await asyncio.sleep(0.2)
            assert len(peer.frames) == 0
            assert peer.unkeys == 1
        finally:
            await b.stop(); await peer_tx.stop()


class TestReceivePath:
    async def _wire(self, backend):
        got = {"key": 0, "unkey": 0, "frames": []}

        async def on_key():
            got["key"] += 1

        async def on_audio(pcm):
            got["frames"].append(pcm)

        async def on_unkey():
            got["unkey"] += 1

        backend.set_events(
            BackendEvents(on_rx_key=on_key, on_rx_audio=on_audio, on_rx_unkey=on_unkey)
        )
        return got

    async def test_inbound_becomes_semantic_events(self, tmp_path):
        b, peer_tx, _ = await rig(tmp_path)
        got = await self._wire(b)
        try:
            for _ in range(4):
                peer_tx.send_voice(b"\x00" * (VOICE_SAMPLES * 2))
            peer_tx.send_unkey()
            await asyncio.sleep(0.3)

            assert got["key"] == 1
            assert len(got["frames"]) == 4
            assert got["unkey"] == 1
        finally:
            await b.stop(); await peer_tx.stop()

    async def test_audio_arrives_as_int16_array(self, tmp_path):
        b, peer_tx, _ = await rig(tmp_path)
        got = await self._wire(b)
        try:
            pcm = np.arange(VOICE_SAMPLES, dtype=np.int16)
            peer_tx.send_voice(pcm.tobytes())
            await asyncio.sleep(0.3)
            assert got["frames"], "no audio delivered"
            assert got["frames"][0].dtype == np.int16
            assert np.array_equal(got["frames"][0], pcm)
        finally:
            await b.stop(); await peer_tx.stop()

    async def test_no_events_without_handlers(self, tmp_path):
        """A backend with no core attached must not explode."""
        b, peer_tx, _ = await rig(tmp_path)
        try:
            peer_tx.send_voice(b"\x00" * (VOICE_SAMPLES * 2))
            await asyncio.sleep(0.2)
        finally:
            await b.stop(); await peer_tx.stop()


class TestSafety:
    async def test_fail_safe_unkeys(self, tmp_path):
        b, peer_tx, peer = await rig(tmp_path)
        try:
            await b.key()
            await asyncio.sleep(0.1)
            b.fail_safe()
            await asyncio.sleep(0.2)
            assert peer.unkeys == 1
            assert not b.keyed
        finally:
            await b.stop(); await peer_tx.stop()

    async def test_fail_safe_never_raises(self, tmp_path):
        b, peer_tx, _ = await rig(tmp_path)
        await b.stop()                      # socket gone
        b.fail_safe()                       # must not raise
        await peer_tx.stop()

    async def test_fail_safe_is_idempotent(self, tmp_path):
        b, peer_tx, _ = await rig(tmp_path)
        try:
            await b.key()
            b.fail_safe()
            b.fail_safe()
            assert not b.keyed
        finally:
            await b.stop(); await peer_tx.stop()

    async def test_stop_unkeys_if_transmitting(self, tmp_path):
        b, peer_tx, peer = await rig(tmp_path)
        try:
            await b.key()
            await b.write_audio(np.zeros(VOICE_SAMPLES, dtype=np.int16))
            await asyncio.sleep(0.1)
            await b.stop()
            await asyncio.sleep(0.2)
            assert peer.unkeys == 1, "stop left ASL keyed"
        finally:
            await peer_tx.stop()


class TestConfigGating:
    """A USRP config must not be forced to describe hardware it has none of."""

    async def test_usrp_needs_no_sound_devices(self, tmp_path):
        data = copy.deepcopy(BASE)
        data["bridge"] = {"backend": "usrp"}
        data.pop("sound")
        data.pop("ptt")
        p = tmp_path / "b.yaml"
        p.write_text(yaml.safe_dump(data))
        cfg = load_config(p)
        assert cfg.bridge.backend == "usrp"

    async def test_aioc_still_requires_them(self, tmp_path):
        from zello_link.config import ConfigError

        data = copy.deepcopy(BASE)
        data["bridge"] = {"backend": "aioc"}
        data.pop("sound")
        p = tmp_path / "b.yaml"
        p.write_text(yaml.safe_dump(data))
        with pytest.raises(ConfigError):
            load_config(p)

    async def test_non_loopback_bind_is_refused_by_default(self, tmp_path):
        from zello_link.config import ConfigError

        with pytest.raises(ConfigError, match="unauthenticated"):
            make_config(tmp_path, usrp={"bind_host": "0.0.0.0"})

    async def test_non_loopback_bind_allowed_when_acknowledged(self, tmp_path):
        cfg = make_config(
            tmp_path, usrp={"bind_host": "0.0.0.0", "allow_remote_host": True}
        )
        assert cfg.usrp.bind_host == "0.0.0.0"

    async def test_same_host_and_port_refused(self, tmp_path):
        from zello_link.config import ConfigError

        with pytest.raises(ConfigError, match="must differ"):
            make_config(
                tmp_path,
                usrp={"bind_host": "127.0.0.1", "bind_port": 34001,
                      "asl_host": "127.0.0.1", "asl_port": 34001},
            )

    async def test_rates_are_fixed_by_chan_usrp(self, tmp_path):
        from zello_link.config import ConfigError

        with pytest.raises(ConfigError):
            make_config(tmp_path, usrp={"sample_rate": 16000})
        with pytest.raises(ConfigError):
            make_config(tmp_path, usrp={"frame_ms": 40})
