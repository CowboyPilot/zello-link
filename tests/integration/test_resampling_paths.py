"""Rate conversion through the real controller, both directions.

The inbound case is the one that matters most: a Zello peer picks its own
sample rate, so a bridge with a perfectly matched device and Opus rate can
still receive 8 kHz or 48 kHz audio. Playing that unconverted puts wrong-pitch
audio into a keyed transmitter.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import yaml

from tests.fakes import FakeAudioSink, FakeCos, FakeEncoder, FakeZello, block
from zello_link.config import load_config
from zello_link.controller import BridgeController, StreamMeta
from zello_link.hardware.ptt import NullPtt, SafePtt
from zello_link.zello.protocol import CodecHeader

BASE = {
    "config_version": 1,
    "instance": {"name": "rate-test"},
    "zello": {"channel": "C", "username": "u", "auth_token": "tok-abcdef"},
    "sound": {"input_device": "in", "output_device": "out"},
    "ptt": {"mode": "none", "pre_key_ms": 0, "post_audio_ms": 0, "max_tx_s": 5.0},
    "cos": {"mode": "internal_audio"},
    "bridge": {"rx_guard_ms": 0, "tx_guard_ms": 0},
    "logging": {"console": False, "file": None},
}


def make_config(tmp_path, **overrides):
    data = copy.deepcopy(BASE)
    for section, values in overrides.items():
        data.setdefault(section, {}).update(values)
    p = tmp_path / "bridge.yaml"
    p.write_text(yaml.safe_dump(data))
    return load_config(p)


def build(cfg):
    zello, audio, cos = FakeZello(), FakeAudioSink(), FakeCos()
    ptt = SafePtt(NullPtt(), max_tx_s=cfg.ptt.max_tx_s)
    ctrl = BridgeController(cfg, zello=zello, audio=audio, ptt=ptt, cos=cos)
    return ctrl, zello, audio, cos, ptt


def meta(rate: int, frame_ms: int = 20, stream_id: int = 1):
    return StreamMeta(
        stream_id=stream_id,
        channel="C",
        sender="Peer",
        codec_header=CodecHeader(rate, 1, frame_ms),
    )


class _RateDecoder:
    """Stands in for the Opus decoder, emitting frames at a given rate."""

    concealed_frames = 0
    dropped_late = 0

    def __init__(self, rate: int, frame_ms: int = 20):
        self.frame_samples = int(rate * frame_ms / 1000)

    def push(self, packet_id, payload):
        return [np.full(self.frame_samples, 1000, dtype=np.int16)]

    def close(self):
        pass


class TestInboundRateConversion:
    """Peer rate -> device rate."""

    @pytest.mark.parametrize("peer_rate", [8000, 12000, 16000, 24000, 48000])
    async def test_playback_lands_at_device_rate(self, tmp_path, peer_rate):
        """20 ms from any peer must become 20 ms at the device rate."""
        cfg = make_config(tmp_path, sound={"sample_rate": 16000})
        ctrl, _, audio, _, _ = build(cfg)
        await ctrl.start()
        try:
            await ctrl.on_zello_stream_start(meta(peer_rate))
            ctrl._inbound = _RateDecoder(peer_rate)

            for i in range(50):                 # 1 second of peer audio
                await ctrl.on_zello_audio(1, i, b"payload")

            expected = 16000                    # 1 s at the device rate
            assert audio.total_samples == pytest.approx(expected, abs=64)
        finally:
            await ctrl.stop()

    async def test_8k_peer_is_not_played_at_half_speed(self, tmp_path):
        """The specific bug: unconverted 8 kHz on a 16 kHz device."""
        cfg = make_config(tmp_path, sound={"sample_rate": 16000})
        ctrl, _, audio, _, _ = build(cfg)
        await ctrl.start()
        try:
            await ctrl.on_zello_stream_start(meta(8000))
            ctrl._inbound = _RateDecoder(8000)
            for i in range(10):
                await ctrl.on_zello_audio(1, i, b"payload")

            # 10 frames x 160 samples = 1600 at 8 kHz; must become ~3200.
            assert audio.total_samples > 3000, "8 kHz audio was not upsampled"
        finally:
            await ctrl.stop()

    async def test_matching_rate_is_passthrough(self, tmp_path):
        cfg = make_config(tmp_path, sound={"sample_rate": 16000})
        ctrl, _, audio, _, _ = build(cfg)
        await ctrl.start()
        try:
            await ctrl.on_zello_stream_start(meta(16000))
            assert ctrl._inbound_resampler.passthrough
            ctrl._inbound = _RateDecoder(16000)
            await ctrl.on_zello_audio(1, 0, b"payload")
            assert audio.total_samples == 320
        finally:
            await ctrl.stop()

    async def test_resampler_is_rebuilt_per_stream(self, tmp_path):
        """Consecutive callers at different rates must each be converted."""
        cfg = make_config(tmp_path, sound={"sample_rate": 16000})
        ctrl, _, _, _, _ = build(cfg)
        await ctrl.start()
        try:
            await ctrl.on_zello_stream_start(meta(8000, stream_id=1))
            assert ctrl._inbound_resampler.in_rate == 8000
            await ctrl.on_zello_stream_stop(1)
            await ctrl.wait_for_idle()

            await ctrl.on_zello_stream_start(meta(48000, stream_id=2))
            assert ctrl._inbound_resampler.in_rate == 48000
        finally:
            await ctrl.stop()

    async def test_device_at_48k_with_16k_peer(self, tmp_path):
        cfg = make_config(
            tmp_path, sound={"sample_rate": 48000}, opus={"sample_rate": 48000}
        )
        ctrl, _, audio, _, _ = build(cfg)
        await ctrl.start()
        try:
            await ctrl.on_zello_stream_start(meta(16000))
            ctrl._inbound = _RateDecoder(16000)
            for i in range(50):
                await ctrl.on_zello_audio(1, i, b"payload")
            assert audio.total_samples == pytest.approx(48000, abs=200)
        finally:
            await ctrl.stop()


class TestOutboundRateConversion:
    """Device rate -> Opus rate, with whole-packet framing."""

    async def test_48k_capture_encodes_16k_packets(self, tmp_path):
        cfg = make_config(
            tmp_path,
            sound={"sample_rate": 48000, "block_ms": 20},
            opus={"sample_rate": 16000, "frame_ms": 20},
        )
        ctrl, zello, _, cos, _ = build(cfg)
        await ctrl.start()
        ctrl._encoder = FakeEncoder()
        try:
            cos.trigger_active()
            await ctrl.on_capture_block(block(960))     # 20 ms at 48 kHz
            for _ in range(49):
                await ctrl.on_capture_block(block(960))

            # 1 s of audio at one 20 ms packet each = ~50 packets.
            assert len(zello.sent) == pytest.approx(50, abs=2)
        finally:
            await ctrl.stop()

    async def test_every_packet_is_a_whole_frame(self, tmp_path):
        """Ragged resampler output must not produce short Opus frames."""
        cfg = make_config(
            tmp_path,
            sound={"sample_rate": 44100, "block_ms": 20},
            opus={"sample_rate": 16000, "frame_ms": 20},
        )
        ctrl, zello, _, cos, _ = build(cfg)
        await ctrl.start()

        sizes: list[int] = []

        class SizeRecordingEncoder(FakeEncoder):
            def encode(self, pcm):
                sizes.append(len(pcm))
                return super().encode(pcm)

        ctrl._encoder = SizeRecordingEncoder()
        try:
            cos.trigger_active()
            await ctrl.on_capture_block(block(882))     # 20 ms at 44.1 kHz
            for _ in range(30):
                await ctrl.on_capture_block(block(882))
            assert sizes, "nothing was encoded"
            assert set(sizes) == {320}, f"non-frame-sized encodes: {sorted(set(sizes))}"
        finally:
            await ctrl.stop()

    async def test_matching_rates_are_passthrough(self, tmp_path):
        cfg = make_config(tmp_path, sound={"sample_rate": 16000})
        ctrl, zello, _, cos, _ = build(cfg)
        await ctrl.start()
        ctrl._encoder = FakeEncoder()
        try:
            assert ctrl._outbound_resampler.passthrough
            cos.trigger_active()
            await ctrl.on_capture_block(block(320))
            for _ in range(4):
                await ctrl.on_capture_block(block(320))
            assert len(zello.sent) == 5, "one packet per block when rates match"
        finally:
            await ctrl.stop()

    async def test_resampler_reset_between_transmissions(self, tmp_path):
        """Idle blocks never reach the resampler, so its state must be fresh."""
        cfg = make_config(tmp_path, sound={"sample_rate": 48000, "block_ms": 20})
        ctrl, zello, _, cos, _ = build(cfg)
        await ctrl.start()
        ctrl._encoder = FakeEncoder()
        try:
            cos.trigger_active()
            await ctrl.on_capture_block(block(960))
            for _ in range(10):
                await ctrl.on_capture_block(block(960))
            first = len(zello.sent)

            cos.trigger_inactive()
            await ctrl.on_capture_block(block(960))

            cos.trigger_active()
            await ctrl.on_capture_block(block(960))
            for _ in range(10):
                await ctrl.on_capture_block(block(960))
            second = len(zello.sent) - first

            assert abs(first - second) <= 2, "second transmission encoded differently"
        finally:
            await ctrl.stop()

    async def test_no_unbounded_buffer_growth(self, tmp_path):
        """AT-11: the encode accumulator must not grow over a long stream."""
        cfg = make_config(tmp_path, sound={"sample_rate": 44100, "block_ms": 20})
        ctrl, _, _, cos, _ = build(cfg)
        await ctrl.start()
        ctrl._encoder = FakeEncoder()
        try:
            cos.trigger_active()
            await ctrl.on_capture_block(block(882))
            for _ in range(500):
                await ctrl.on_capture_block(block(882))
            assert ctrl._encode_buf.size < ctrl._samples_per_packet
        finally:
            await ctrl.stop()

    async def test_buffer_cleared_between_streams(self, tmp_path):
        cfg = make_config(tmp_path, sound={"sample_rate": 44100, "block_ms": 20})
        ctrl, _, _, cos, _ = build(cfg)
        await ctrl.start()
        ctrl._encoder = FakeEncoder()
        try:
            cos.trigger_active()
            await ctrl.on_capture_block(block(882))
            await ctrl.on_capture_block(block(882))
            cos.trigger_inactive()
            await ctrl.on_capture_block(block(882))
            assert ctrl._encode_buf.size == 0
        finally:
            await ctrl.stop()


class TestAdaptiveJitterBuffer:
    """Buffer depth must follow the PEER's packet duration, not just config.

    Found live: a peer negotiating 60 ms packets against the 120 ms default
    had only two packets of slack and starved playback three times in one
    five-second transmission.
    """

    class RecordingSink(FakeAudioSink):
        def __init__(self):
            super().__init__()
            self.jitter_targets: list[float] = []
            self.resets = 0

        def set_jitter_target_ms(self, ms):
            self.jitter_targets.append(ms)

        def reset_jitter_target(self):
            self.resets += 1

    async def _run(self, tmp_path, frame_ms, frames_per_packet=1, jitter_ms=120):
        cfg = make_config(tmp_path, sound={"jitter_ms": jitter_ms, "jitter_max_ms": 800})
        sink = self.RecordingSink()
        ctrl = BridgeController(
            cfg, zello=FakeZello(), audio=sink,
            ptt=SafePtt(NullPtt(), max_tx_s=5.0), cos=FakeCos(),
        )
        await ctrl.start()
        try:
            await ctrl.on_zello_stream_start(
                StreamMeta(1, "C", "Peer",
                           CodecHeader(16000, frames_per_packet, frame_ms))
            )
            return ctrl, sink
        finally:
            pass

    async def test_60ms_peer_raises_the_target(self, tmp_path):
        ctrl, sink = await self._run(tmp_path, frame_ms=60)
        try:
            assert sink.jitter_targets == [180], "60 ms packets need 3 packets of depth"
        finally:
            await ctrl.stop()

    async def test_20ms_peer_keeps_the_configured_target(self, tmp_path):
        """3 x 20 ms is below the 120 ms default, so config wins."""
        ctrl, sink = await self._run(tmp_path, frame_ms=20)
        try:
            assert sink.jitter_targets == [120]
        finally:
            await ctrl.stop()

    async def test_multi_frame_packets_counted_whole(self, tmp_path):
        ctrl, sink = await self._run(tmp_path, frame_ms=60, frames_per_packet=2)
        try:
            assert sink.jitter_targets == [360], "120 ms packets need 360 ms"
        finally:
            await ctrl.stop()

    async def test_target_is_restored_after_the_stream(self, tmp_path):
        ctrl, sink = await self._run(tmp_path, frame_ms=60)
        try:
            await ctrl.on_zello_stream_stop(1)
            await ctrl.wait_for_idle()
            assert sink.resets == 1
        finally:
            await ctrl.stop()
