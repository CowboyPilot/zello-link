"""Opus codec and inbound stream sequencing.

Codec tests are skipped when libopus is absent so CI stays green on a bare
host; the StreamDecoder sequencing tests use a fake decoder and always run,
because packet-loss handling is pure logic.
"""

from __future__ import annotations

import numpy as np
import pytest

from zello_dmr_bridge.zello import opus as op
from zello_dmr_bridge.zello.protocol import CodecHeader

requires_opus = pytest.mark.skipif(not op.is_available(), reason="libopus not installed")


def speech_like(n: int, freq: float = 300.0, rate: int = 16000, amp: float = 8000.0):
    t = np.arange(n) / rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.int16)


@requires_opus
class TestCodecRoundTrip:
    def test_encode_produces_a_packet(self):
        with op.OpusEncoder(16000) as enc:
            packet = enc.encode(speech_like(320))
        assert isinstance(packet, bytes)
        assert 0 < len(packet) < 4000

    def test_roundtrip_preserves_frame_length(self):
        with op.OpusEncoder(16000) as enc, op.OpusDecoder(16000) as dec:
            pcm = dec.decode(enc.encode(speech_like(320)))
        assert pcm.shape == (320,)
        assert pcm.dtype == np.int16

    def test_roundtrip_preserves_signal_energy(self):
        """Opus is lossy, but a steady tone must survive at a similar level."""
        src = speech_like(320)
        with op.OpusEncoder(16000) as enc, op.OpusDecoder(16000) as dec:
            # Prime the codec: the first frame or two carry startup transients.
            for _ in range(5):
                out = dec.decode(enc.encode(src))
        src_rms = np.sqrt(np.mean(src.astype(float) ** 2))
        out_rms = np.sqrt(np.mean(out.astype(float) ** 2))
        assert out_rms == pytest.approx(src_rms, rel=0.5)

    def test_silence_encodes_small(self):
        with op.OpusEncoder(16000) as enc:
            packets = [enc.encode(np.zeros(320, dtype=np.int16)) for _ in range(10)]
        assert len(packets[-1]) < 20, "silence should compress to almost nothing"

    @pytest.mark.parametrize("frame_ms", [10, 20, 40, 60])
    def test_supported_frame_sizes(self, frame_ms):
        n = int(16000 * frame_ms / 1000)
        with op.OpusEncoder(16000) as enc, op.OpusDecoder(16000) as dec:
            assert dec.decode(enc.encode(speech_like(n))).shape == (n,)

    def test_accepts_non_int16_input(self):
        with op.OpusEncoder(16000) as enc:
            assert enc.encode(speech_like(320).astype(np.float64)) is not None

    def test_unknown_application_rejected(self):
        with pytest.raises(op.OpusError, match="application"):
            op.OpusEncoder(16000, application="nonsense")

    def test_use_after_close_raises(self):
        enc = op.OpusEncoder(16000)
        enc.close()
        with pytest.raises(op.OpusError, match="closed"):
            enc.encode(speech_like(320))

    def test_double_close_is_safe(self):
        enc = op.OpusEncoder(16000)
        enc.close()
        enc.close()


@requires_opus
class TestEncoderSettingsActuallyApply:
    """Regression guard for the variadic-ctl ABI.

    opus_encoder_ctl is variadic, and on arm64 a wrong ctypes signature makes
    every ctl fail. Constructing the encoder would still succeed, so without
    reading the values back the bridge would silently run at libopus defaults
    instead of the configured bitrate and complexity.
    """

    OPUS_GET_BITRATE = 4003
    OPUS_GET_COMPLEXITY = 4011

    def test_bitrate_is_applied(self):
        with op.OpusEncoder(16000, bitrate=24000) as enc:
            assert enc.get_ctl(self.OPUS_GET_BITRATE) == 24000

    def test_complexity_is_applied(self):
        with op.OpusEncoder(16000, complexity=8) as enc:
            assert enc.get_ctl(self.OPUS_GET_COMPLEXITY) == 8

    def test_distinct_bitrates_are_distinct(self):
        with op.OpusEncoder(16000, bitrate=12000) as low:
            low_v = low.get_ctl(self.OPUS_GET_BITRATE)
        with op.OpusEncoder(16000, bitrate=32000) as high:
            high_v = high.get_ctl(self.OPUS_GET_BITRATE)
        assert low_v == 12000 and high_v == 32000

    def test_lower_bitrate_produces_smaller_packets(self):
        """End-to-end proof the setting reaches the codec, not just the ctl."""
        def total(bitrate):
            with op.OpusEncoder(16000, bitrate=bitrate) as enc:
                return sum(len(enc.encode(speech_like(320, freq=440 + i))) for i in range(25))

        assert total(6000) < total(64000)


@requires_opus
class TestConcealment:
    def test_conceal_returns_a_full_frame(self):
        """The NULL-payload decode path -- the reason for the ctypes binding."""
        with op.OpusEncoder(16000) as enc, op.OpusDecoder(16000) as dec:
            for _ in range(3):
                dec.decode(enc.encode(speech_like(320)))
            concealed = dec.conceal(320)
        assert concealed.shape == (320,)
        assert concealed.dtype == np.int16

    def test_conceal_extrapolates_rather_than_emitting_silence(self):
        """A gap must not become a click or a hole in a keyed transmission."""
        with op.OpusEncoder(16000) as enc, op.OpusDecoder(16000) as dec:
            for _ in range(5):
                dec.decode(enc.encode(speech_like(320)))
            concealed = dec.conceal(320)
        assert np.abs(concealed).max() > 0, "concealment produced pure silence"

    def test_conceal_before_any_audio_is_safe(self):
        with op.OpusDecoder(16000) as dec:
            assert dec.conceal(320).shape == (320,)


class FakeDecoder:
    """Records the call sequence so sequencing can be tested without libopus."""

    def __init__(self):
        self.calls: list[str] = []

    def decode(self, payload: bytes) -> np.ndarray:
        self.calls.append(f"decode:{payload.decode()}")
        return np.zeros(320, dtype=np.int16)

    def conceal(self, frame_size: int) -> np.ndarray:
        self.calls.append("conceal")
        return np.zeros(frame_size, dtype=np.int16)

    def close(self) -> None:
        self.calls.append("close")


@pytest.fixture
def header():
    return CodecHeader(sample_rate=16000, frames_per_packet=1, frame_size_ms=20)


class TestStreamSequencing:
    def make(self, header):
        fake = FakeDecoder()
        return op.StreamDecoder(header, decoder=fake), fake

    def test_in_order_packets_just_decode(self, header):
        sd, fake = self.make(header)
        for i in range(1, 5):
            sd.push(i, f"p{i}".encode())
        assert fake.calls == ["decode:p1", "decode:p2", "decode:p3", "decode:p4"]
        assert sd.concealed_frames == 0
        assert sd.decoded_packets == 4

    def test_single_gap_conceals_one_frame(self, header):
        sd, fake = self.make(header)
        sd.push(1, b"a")
        sd.push(3, b"c")            # packet 2 lost
        assert fake.calls == ["decode:a", "conceal", "decode:c"]
        assert sd.concealed_frames == 1

    def test_multi_packet_gap(self, header):
        sd, fake = self.make(header)
        sd.push(1, b"a")
        sd.push(4, b"d")            # 2 and 3 lost
        assert fake.calls.count("conceal") == 2
        assert sd.concealed_frames == 2

    def test_concealment_is_capped(self, header):
        """A huge gap must not synthesize unbounded audio."""
        sd, fake = self.make(header)
        sd.push(1, b"a")
        sd.push(10_000, b"z")
        assert fake.calls.count("conceal") == op.MAX_CONSECUTIVE_PLC
        assert sd.concealed_frames == op.MAX_CONSECUTIVE_PLC

    def test_late_packet_is_dropped(self, header):
        sd, fake = self.make(header)
        sd.push(5, b"e")
        sd.push(3, b"c")            # arrives late
        assert fake.calls == ["decode:e"]
        assert sd.dropped_late == 1

    def test_duplicate_packet_is_dropped(self, header):
        sd, fake = self.make(header)
        sd.push(1, b"a")
        sd.push(2, b"b")
        sd.push(2, b"b")
        assert fake.calls == ["decode:a", "decode:b"]
        assert sd.dropped_late == 1

    def test_all_zero_packet_ids_disable_sequencing(self, header):
        """Zello documents that a client may send packet_id 0 throughout."""
        sd, fake = self.make(header)
        for i in range(5):
            sd.push(0, f"p{i}".encode())
        assert fake.calls == [f"decode:p{i}" for i in range(5)]
        assert sd.concealed_frames == 0
        assert sd.dropped_late == 0

    def test_returns_frames_in_playback_order(self, header):
        sd, _ = self.make(header)
        sd.push(1, b"a")
        frames = sd.push(3, b"c")
        assert len(frames) == 2, "one concealed frame then the real one"

    def test_first_packet_establishes_the_sequence(self, header):
        sd, fake = self.make(header)
        sd.push(100, b"a")
        sd.push(101, b"b")
        assert fake.calls == ["decode:a", "decode:b"]
        assert sd.concealed_frames == 0

    def test_frame_size_from_header(self, header):
        sd, _ = self.make(header)
        assert sd.frame_size == 320
