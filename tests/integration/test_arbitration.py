"""Half-duplex arbitration through the real controller (AT-03, AT-04, AT-06).

Uses the fake Zello/audio/PTT/COS backends, so the full controller path runs
with no radio, no sound card, and no network.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
import yaml

from tests.fakes import FakeAudioSink, FakeCos, FakeEncoder, FakeZello, block
from zello_link.config import load_config
from zello_link.controller import BridgeController, StreamMeta
from zello_link.hardware.ptt import NullPtt, SafePtt
from zello_link.state import State
from zello_link.zello.protocol import CodecHeader

CONFIG = {
    "config_version": 2,
    "instance": {"name": "test-bridge", "log_level": "DEBUG"},
    "zello": {"channel": "Event Security", "username": "u", "auth_token": "tok-abcdef"},
    "sound": {"input_device": "fake-in", "output_device": "fake-out"},
    "ptt": {"mode": "none", "pre_key_ms": 10, "post_audio_ms": 10, "max_tx_s": 5.0},
    "cos": {"mode": "internal_audio"},
    "bridge": {"rx_guard_ms": 0, "tx_guard_ms": 0, "min_stream_interval_ms": 0},
}


def make_config(tmp_path, **overrides):
    import copy

    data = copy.deepcopy(CONFIG)
    for section, values in overrides.items():
        data.setdefault(section, {}).update(values)
    p = tmp_path / "bridge.yaml"
    p.write_text(yaml.safe_dump(data))
    return load_config(p)


HEADER = CodecHeader(sample_rate=16000, frames_per_packet=1, frame_size_ms=20)


def meta(stream_id=22695, sender="Gate-2", channel="Event Security"):
    return StreamMeta(stream_id=stream_id, channel=channel, sender=sender, codec_header=HEADER)


@pytest.fixture
async def rig(tmp_path):
    """Controller wired to fakes, with the real SafePtt in the loop."""
    cfg = make_config(tmp_path)
    zello, audio, cos = FakeZello(), FakeAudioSink(), FakeCos()
    ptt = SafePtt(NullPtt(), max_tx_s=cfg.ptt.max_tx_s)

    ctrl = BridgeController(cfg, zello=zello, audio=audio, ptt=ptt, cos=cos)
    await ctrl.start()
    ctrl._encoder = FakeEncoder()      # avoid a libopus dependency here

    yield ctrl, zello, audio, cos, ptt
    await ctrl.stop()


class TestZelloToRf:
    """AT-03: a Zello call keys the radio, delivers audio, then unkeys."""

    async def test_accepts_and_keys(self, rig):
        ctrl, _, _, _, ptt = rig
        assert await ctrl.on_zello_stream_start(meta()) is True
        assert ptt.is_keyed()
        assert ctrl.state is State.ZELLO_TO_RF_ACTIVE

    async def test_keys_before_audio_is_released(self, rig):
        """Pre-key must complete before playback starts, or the radio clips it."""
        ctrl, _, audio, _, ptt = rig
        await ctrl.on_zello_stream_start(meta())
        assert ptt.is_keyed()
        assert audio.played == [], "no audio should have been queued during pre-key"

    async def test_audio_reaches_the_sink(self, rig):
        ctrl, _, audio, _, _ = rig
        await ctrl.on_zello_stream_start(meta())
        ctrl._inbound = _StubDecoder()
        await ctrl.on_zello_audio(22695, 0, b"payload")
        assert audio.total_samples == 320

    async def test_stop_unkeys_after_tail(self, rig):
        ctrl, _, _, _, ptt = rig
        await ctrl.on_zello_stream_start(meta())
        await ctrl.on_zello_stream_stop(22695)
        await ctrl.wait_for_idle()
        assert not ptt.is_keyed()
        assert ctrl.state is State.IDLE

    async def test_records_call_stats(self, rig):
        ctrl, _, _, _, _ = rig
        await ctrl.on_zello_stream_start(meta())
        await ctrl.on_zello_stream_stop(22695)
        await ctrl.wait_for_idle()
        assert ctrl.stats.zello_to_rf_calls == 1

    async def test_ignores_other_channel(self, rig):
        ctrl, _, _, _, ptt = rig
        assert await ctrl.on_zello_stream_start(meta(channel="Other Channel")) is False
        assert not ptt.is_keyed()

    async def test_ignores_audio_for_unknown_stream(self, rig):
        ctrl, _, audio, _, _ = rig
        await ctrl.on_zello_stream_start(meta(stream_id=1))
        ctrl._inbound = _StubDecoder()
        await ctrl.on_zello_audio(999, 0, b"payload")
        assert audio.played == []

    async def test_stop_for_wrong_stream_is_ignored(self, rig):
        ctrl, _, _, _, ptt = rig
        await ctrl.on_zello_stream_start(meta(stream_id=1))
        await ctrl.on_zello_stream_stop(999)
        assert ptt.is_keyed(), "an unrelated stop must not unkey the radio"

    async def test_direction_disabled(self, tmp_path):
        cfg = make_config(tmp_path, bridge={"zello_to_rf": False})
        ptt = SafePtt(NullPtt(), max_tx_s=5.0)
        ctrl = BridgeController(
            cfg, zello=FakeZello(), audio=FakeAudioSink(), ptt=ptt, cos=FakeCos()
        )
        await ctrl.start()
        try:
            assert await ctrl.on_zello_stream_start(meta()) is False
            assert not ptt.is_keyed()
        finally:
            await ctrl.stop()


class TestRfToZello:
    """AT-04: COS opens exactly one stream and closes it on hang."""

    async def test_cos_active_opens_a_stream(self, rig):
        ctrl, zello, _, cos, _ = rig
        cos.trigger_active()
        await ctrl.on_capture_block(block())
        assert len(zello.started) == 1
        assert ctrl.state is State.RF_TO_ZELLO_ACTIVE

    async def test_audio_is_sent_while_active(self, rig):
        ctrl, zello, _, cos, _ = rig
        cos.trigger_active()
        await ctrl.on_capture_block(block())
        for _ in range(5):
            await ctrl.on_capture_block(block())
        assert len(zello.sent) >= 5

    async def test_packet_ids_increment(self, rig):
        ctrl, zello, _, cos, _ = rig
        cos.trigger_active()
        await ctrl.on_capture_block(block())
        for _ in range(4):
            await ctrl.on_capture_block(block())
        ids = [p.packet_id for p in zello.sent]
        assert ids == sorted(ids) and len(set(ids)) == len(ids)

    async def test_cos_inactive_closes_the_stream(self, rig):
        ctrl, zello, _, cos, _ = rig
        cos.trigger_active()
        await ctrl.on_capture_block(block())
        cos.trigger_inactive()
        await ctrl.on_capture_block(block())
        assert len(zello.stopped) == 1
        assert ctrl.state is State.IDLE

    async def test_one_stream_per_transmission(self, rig):
        """Brief pauses must not split a call into several streams."""
        ctrl, zello, _, cos, _ = rig
        cos.trigger_active()
        await ctrl.on_capture_block(block())
        for _ in range(20):
            await ctrl.on_capture_block(block())
        assert len(zello.started) == 1

    async def test_start_buffer_preserves_leading_audio(self, tmp_path):
        """Audio captured during the server round trip must not be lost."""
        cfg = make_config(tmp_path)
        zello = FakeZello(start_delay_s=0.05)
        cos = FakeCos()
        ctrl = BridgeController(
            cfg, zello=zello, audio=FakeAudioSink(),
            ptt=SafePtt(NullPtt(), max_tx_s=5.0), cos=cos,
        )
        await ctrl.start()
        ctrl._encoder = FakeEncoder()
        try:
            cos.trigger_active()
            opening = asyncio.create_task(ctrl.on_capture_block(block()))
            await asyncio.sleep(0.01)
            for _ in range(3):
                await ctrl.on_capture_block(block())   # captured while waiting
            await opening
            assert len(zello.sent) >= 4, "leading blocks were dropped"
        finally:
            await ctrl.stop()

    async def test_start_stream_failure_returns_to_idle(self, rig):
        ctrl, zello, _, cos, _ = rig
        zello.fail_start = True
        cos.trigger_active()
        await ctrl.on_capture_block(block())
        assert ctrl.state is State.IDLE
        assert not zello.started

    async def test_direction_disabled(self, tmp_path):
        cfg = make_config(tmp_path, bridge={"rf_to_zello": False})
        zello, cos = FakeZello(), FakeCos()
        ctrl = BridgeController(
            cfg, zello=zello, audio=FakeAudioSink(),
            ptt=SafePtt(NullPtt(), max_tx_s=5.0), cos=cos,
        )
        await ctrl.start()
        try:
            cos.trigger_active()
            await ctrl.on_capture_block(block())
            assert not zello.started
        finally:
            await ctrl.stop()


class TestCollisions:
    """AT-06: neither direction may interrupt the other."""

    async def test_incoming_zello_while_rf_active_does_not_key(self, rig):
        ctrl, zello, _, cos, ptt = rig
        cos.trigger_active()
        await ctrl.on_capture_block(block())
        assert ctrl.state is State.RF_TO_ZELLO_ACTIVE

        assert await ctrl.on_zello_stream_start(meta()) is False
        assert not ptt.is_keyed(), "radio must not key during RF->Zello"
        assert ctrl.stats.collisions == 1

    async def test_cos_suppressed_while_zello_to_rf(self, rig):
        """The loopback guard: gateway TX audio must not open a Zello stream."""
        ctrl, zello, _, cos, _ = rig
        await ctrl.on_zello_stream_start(meta())
        assert cos.suppressed

        cos.trigger_active()
        for _ in range(10):
            await ctrl.on_capture_block(block())
        assert not zello.started, "gateway self-audio opened a Zello stream"

    async def test_suppression_released_after_tail(self, rig):
        ctrl, _, _, cos, _ = rig
        await ctrl.on_zello_stream_start(meta())
        await ctrl.on_zello_stream_stop(22695)
        await ctrl.wait_for_idle()
        assert not cos.suppressed

    async def test_competing_stream_is_discarded_not_queued(self, rig):
        """v0.1 never replays a competing call late."""
        ctrl, _, audio, cos, _ = rig
        cos.trigger_active()
        await ctrl.on_capture_block(block())

        await ctrl.on_zello_stream_start(meta())
        await ctrl.on_zello_audio(22695, 0, b"payload")
        assert audio.played == [], "competing Zello audio must be discarded"

    async def test_second_zello_stream_during_first_is_rejected(self, rig):
        ctrl, _, _, _, _ = rig
        assert await ctrl.on_zello_stream_start(meta(stream_id=1)) is True
        assert await ctrl.on_zello_stream_start(meta(stream_id=2, sender="Command")) is False
        assert ctrl.stats.collisions == 1

    async def test_collision_counter_tracks_every_rejection(self, rig):
        ctrl, _, _, _, _ = rig
        await ctrl.on_zello_stream_start(meta(stream_id=1))
        for i in range(3):
            await ctrl.on_zello_stream_start(meta(stream_id=10 + i))
        assert ctrl.stats.collisions == 3


class TestGuardIntervals:
    async def test_guard_blocks_opposite_direction(self, tmp_path):
        cfg = make_config(tmp_path, bridge={"rx_guard_ms": 5000, "tx_guard_ms": 5000})
        zello, cos = FakeZello(), FakeCos()
        ptt = SafePtt(NullPtt(), max_tx_s=5.0)
        ctrl = BridgeController(cfg, zello=zello, audio=FakeAudioSink(), ptt=ptt, cos=cos)
        await ctrl.start()
        ctrl._encoder = FakeEncoder()
        try:
            await ctrl.on_zello_stream_start(meta())
            await ctrl.on_zello_stream_stop(22695)
            await ctrl.wait_for_idle()
            assert ctrl.state is State.IDLE

            cos.trigger_active()
            await ctrl.on_capture_block(block())
            assert not zello.started, "guard interval was not honoured"
            assert ctrl.stats.rejected_guard == 1
        finally:
            await ctrl.stop()

    async def test_zero_guard_allows_immediate_turnaround(self, rig):
        ctrl, zello, _, cos, _ = rig
        await ctrl.on_zello_stream_start(meta())
        await ctrl.on_zello_stream_stop(22695)
        await ctrl.wait_for_idle()

        cos.trigger_active()
        await ctrl.on_capture_block(block())
        assert len(zello.started) == 1


class TestFaultHandling:
    async def test_cos_backend_failure_triggers_failsafe(self, rig):
        ctrl, _, _, cos, ptt = rig
        cos.fail_feed = True
        await ctrl.on_capture_block(block())
        assert ctrl.state is State.IDLE
        assert not ptt.is_keyed()
        assert ctrl.stats.faults >= 1

    async def test_disconnect_unkeys(self, rig):
        ctrl, _, _, _, ptt = rig
        await ctrl.on_zello_stream_start(meta())
        assert ptt.is_keyed()
        await ctrl.on_zello_disconnected()
        assert not ptt.is_keyed()
        assert ctrl.state is State.IDLE

    async def test_send_failure_closes_the_stream(self, rig):
        ctrl, zello, _, cos, _ = rig
        cos.trigger_active()
        await ctrl.on_capture_block(block())
        zello.fail_send = True
        await ctrl.on_capture_block(block())
        assert ctrl.state is State.IDLE

    async def test_stop_is_safe_when_idle(self, rig):
        ctrl, _, _, _, ptt = rig
        await ctrl.stop()
        assert not ptt.is_keyed()

    async def test_ptt_timeout_forces_failsafe(self, rig):
        ctrl, _, _, _, ptt = rig
        await ctrl.on_zello_stream_start(meta())
        ctrl.on_ptt_timeout(5.0)
        assert ctrl.state is State.IDLE
        assert ctrl.stats.ptt_timeouts == 1


class _StubDecoder:
    """Minimal inbound decoder for playback-path tests."""

    concealed_frames = 0
    dropped_late = 0

    def push(self, packet_id, payload):
        return [np.zeros(320, dtype=np.int16)]

    def close(self):
        pass


class TestOutboundPacing:
    """Zello refuses a client that opens streams too fast.

    Observed live: "start_stream rejected: woodpecker prohibited" after COS
    chattered. Without pacing the bridge retries on every COS event and keeps
    hammering a server that is already refusing it.
    """

    async def _rig(self, tmp_path, **bridge):
        cfg = make_config(tmp_path, bridge=bridge)
        zello, audio, cos = FakeZello(), FakeAudioSink(), FakeCos()
        ptt = SafePtt(NullPtt(), max_tx_s=5.0)
        ctrl = BridgeController(cfg, zello=zello, audio=audio, ptt=ptt, cos=cos)
        await ctrl.start()
        ctrl._encoder = FakeEncoder()
        return ctrl, zello, cos

    async def _one_stream(self, ctrl, cos):
        cos.trigger_active()
        await ctrl.on_capture_block(block())
        cos.trigger_inactive()
        await ctrl.on_capture_block(block())

    async def test_rapid_streams_are_suppressed(self, tmp_path):
        ctrl, zello, cos = await self._rig(tmp_path, min_stream_interval_ms=1000)
        try:
            for _ in range(5):
                await self._one_stream(ctrl, cos)
            assert len(zello.started) == 1, "pacing did not suppress the repeats"
            assert ctrl.stats.rejected_too_soon >= 1
        finally:
            await ctrl.stop()

    async def test_zero_interval_disables_pacing(self, tmp_path):
        ctrl, zello, cos = await self._rig(tmp_path, min_stream_interval_ms=0)
        try:
            for _ in range(4):
                await self._one_stream(ctrl, cos)
            assert len(zello.started) == 4
        finally:
            await ctrl.stop()

    async def test_first_stream_is_never_delayed(self, tmp_path):
        """Pacing must not add latency to the first transmission."""
        ctrl, zello, cos = await self._rig(tmp_path, min_stream_interval_ms=5000)
        try:
            await self._one_stream(ctrl, cos)
            assert len(zello.started) == 1
        finally:
            await ctrl.stop()


class TestRateLimitBackoff:
    async def _rig(self, tmp_path, **bridge):
        bridge.setdefault("min_stream_interval_ms", 0)
        cfg = make_config(tmp_path, bridge=bridge)
        zello, audio, cos = FakeZello(), FakeAudioSink(), FakeCos()
        ptt = SafePtt(NullPtt(), max_tx_s=5.0)
        ctrl = BridgeController(cfg, zello=zello, audio=audio, ptt=ptt, cos=cos)
        await ctrl.start()
        ctrl._encoder = FakeEncoder()
        return ctrl, zello, cos

    async def _attempt(self, ctrl, cos):
        cos.trigger_active()
        await ctrl.on_capture_block(block())
        cos.trigger_inactive()
        await ctrl.on_capture_block(block())

    async def test_woodpecker_refusal_triggers_a_cooldown(self, tmp_path):
        ctrl, zello, cos = await self._rig(tmp_path, rate_limit_cooldown_s=30.0)
        try:
            zello.fail_start = True
            zello.start_error = "start_stream rejected: woodpecker prohibited"
            await self._attempt(ctrl, cos)
            assert ctrl.stats.rate_limit_refusals == 1
            assert ctrl._cooldown_until > ctrl._clock()
        finally:
            await ctrl.stop()

    async def test_cooldown_suppresses_further_attempts(self, tmp_path):
        ctrl, zello, cos = await self._rig(tmp_path, rate_limit_cooldown_s=30.0)
        try:
            zello.fail_start = True
            zello.start_error = "woodpecker prohibited"
            await self._attempt(ctrl, cos)

            zello.fail_start = False          # server would accept now
            for _ in range(3):
                await self._attempt(ctrl, cos)
            assert not zello.started, "kept hammering while rate-limited"
            assert ctrl.stats.rejected_cooldown >= 1
        finally:
            await ctrl.stop()

    async def test_cooldown_doubles_on_repeat_refusals(self, tmp_path):
        ctrl, zello, cos = await self._rig(
            tmp_path, rate_limit_cooldown_s=10.0, rate_limit_cooldown_max_s=100.0
        )
        try:
            zello.fail_start = True
            zello.start_error = "woodpecker prohibited"
            await self._attempt(ctrl, cos)
            first = ctrl._cooldown_s
            ctrl._cooldown_until = 0.0        # let another attempt through
            await self._attempt(ctrl, cos)
            assert ctrl._cooldown_s == first * 2
        finally:
            await ctrl.stop()

    async def test_cooldown_is_capped(self, tmp_path):
        ctrl, zello, cos = await self._rig(
            tmp_path, rate_limit_cooldown_s=10.0, rate_limit_cooldown_max_s=25.0
        )
        try:
            zello.fail_start = True
            zello.start_error = "woodpecker prohibited"
            for _ in range(6):
                ctrl._cooldown_until = 0.0
                await self._attempt(ctrl, cos)
            assert ctrl._cooldown_s <= 25.0
        finally:
            await ctrl.stop()

    async def test_a_successful_stream_clears_the_penalty(self, tmp_path):
        ctrl, zello, cos = await self._rig(tmp_path, rate_limit_cooldown_s=10.0)
        try:
            zello.fail_start = True
            zello.start_error = "woodpecker prohibited"
            await self._attempt(ctrl, cos)
            assert ctrl._cooldown_s > 0

            zello.fail_start = False
            ctrl._cooldown_until = 0.0
            await self._attempt(ctrl, cos)
            assert ctrl._cooldown_s == 0.0
        finally:
            await ctrl.stop()

    async def test_ordinary_errors_do_not_trigger_a_cooldown(self, tmp_path):
        """A transient network failure should retry, not back off for minutes."""
        ctrl, zello, cos = await self._rig(tmp_path)
        try:
            zello.fail_start = True
            zello.start_error = "connection reset"
            await self._attempt(ctrl, cos)
            assert ctrl.stats.rate_limit_refusals == 0
            assert ctrl._cooldown_until == 0.0
        finally:
            await ctrl.stop()
