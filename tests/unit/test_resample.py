"""Rational resampling: rate accuracy, fidelity, and streaming continuity.

The continuity tests are the important ones. A resampler that treats each
block independently passes every single-block test and still ticks audibly
fifty times a second in service.
"""

from __future__ import annotations

import numpy as np
import pytest

from zello_dmr_bridge.audio.resample import (
    Resampler,
    ResamplerError,
    resample_block,
)

# Rate pairs the bridge can actually encounter: AIOC device rates crossed with
# the Opus rates a Zello peer may declare.
RATE_PAIRS = [
    (8000, 16000), (16000, 8000),
    (48000, 16000), (16000, 48000),
    (24000, 16000), (16000, 24000),
    (12000, 16000), (16000, 12000),
    (44100, 16000), (16000, 44100),
    (32000, 16000),
]


def tone(freq: float, n: int, rate: int, amp: float = 8000.0, phase: float = 0.0) -> np.ndarray:
    t = np.arange(n) / rate
    return (amp * np.sin(2 * np.pi * freq * t + phase)).astype(np.int16)


def dominant_freq(pcm: np.ndarray, rate: int) -> float:
    """Peak of the magnitude spectrum, ignoring DC."""
    windowed = pcm.astype(np.float64) * np.hanning(len(pcm))
    spectrum = np.abs(np.fft.rfft(windowed))
    spectrum[0] = 0.0
    return float(np.argmax(spectrum)) * rate / len(pcm)


def rms(pcm: np.ndarray) -> float:
    return float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2)))


class TestConstruction:
    def test_identity_is_passthrough(self):
        r = Resampler(16000, 16000)
        assert r.passthrough
        assert r.delay_ms == 0.0

    def test_passthrough_returns_input_untouched(self):
        r = Resampler(16000, 16000)
        block = tone(440, 320, 16000)
        assert np.array_equal(r.process(block), block)

    @pytest.mark.parametrize("in_rate,out_rate", RATE_PAIRS)
    def test_factors_are_reduced(self, in_rate, out_rate):
        r = Resampler(in_rate, out_rate)
        assert r.L * in_rate == r.M * out_rate

    def test_common_ratios_are_small(self):
        assert (Resampler(8000, 16000).L, Resampler(8000, 16000).M) == (2, 1)
        assert (Resampler(48000, 16000).L, Resampler(48000, 16000).M) == (1, 3)
        assert (Resampler(12000, 16000).L, Resampler(12000, 16000).M) == (4, 3)

    def test_rejects_invalid_rates(self):
        with pytest.raises(ResamplerError):
            Resampler(0, 16000)
        with pytest.raises(ResamplerError):
            Resampler(16000, -1)

    def test_rejects_too_few_taps(self):
        with pytest.raises(ResamplerError, match="taps_per_phase"):
            Resampler(8000, 16000, taps_per_phase=1)

    def test_rejects_pathological_ratio(self):
        """A coprime pair would need one branch per output sample."""
        with pytest.raises(ResamplerError, match="polyphase branches"):
            Resampler(48000, 44101)

    def test_delay_is_reported(self):
        assert Resampler(48000, 16000).delay_ms > 0

    def test_repr_is_informative(self):
        assert "48000" in repr(Resampler(48000, 16000))
        assert "passthrough" in repr(Resampler(16000, 16000))


class TestOutputLength:
    @pytest.mark.parametrize("in_rate,out_rate", RATE_PAIRS)
    def test_length_tracks_the_ratio(self, in_rate, out_rate):
        r = Resampler(in_rate, out_rate)
        n_in = in_rate  # one second
        out = r.process(tone(440, n_in, in_rate))
        expected = n_in * out_rate / in_rate
        assert abs(len(out) - expected) < 2 * r.taps_per_phase

    def test_streaming_length_accumulates_correctly(self):
        """3:4 does not divide evenly per block; totals must still track."""
        r = Resampler(12000, 16000)
        total = sum(len(r.process(tone(440, 240, 12000))) for _ in range(50))
        expected = 50 * 240 * 16000 / 12000
        assert abs(total - expected) < 2 * r.taps_per_phase

    def test_block_lengths_vary_but_average_out(self):
        """A block size that is not a multiple of M yields a ragged length.

        (441 samples at 44100->16000 would be exactly M, giving exactly L
        outputs every time -- which is correct, but tests nothing here.)
        """
        r = Resampler(44100, 16000)
        lengths = {len(r.process(tone(440, 320, 44100))) for _ in range(20)}
        assert len(lengths) > 1, "expected varying per-block output lengths"
        assert max(lengths) - min(lengths) == 1, "lengths should differ by at most one"

    def test_empty_input(self):
        assert len(Resampler(8000, 16000).process(np.zeros(0, dtype=np.int16))) == 0


class TestFidelity:
    @pytest.mark.parametrize("in_rate,out_rate", RATE_PAIRS)
    def test_tone_frequency_is_preserved(self, in_rate, out_rate):
        """The core property: a 440 Hz tone stays 440 Hz."""
        out = resample_block(tone(440, in_rate, in_rate), in_rate, out_rate)
        assert dominant_freq(out, out_rate) == pytest.approx(440, abs=15)

    @pytest.mark.parametrize("freq", [300, 440, 1000, 2000, 3000])
    def test_voice_band_frequencies(self, freq):
        out = resample_block(tone(freq, 16000, 16000), 16000, 8000)
        assert dominant_freq(out, 8000) == pytest.approx(freq, abs=20)

    @pytest.mark.parametrize("in_rate,out_rate", RATE_PAIRS)
    def test_amplitude_is_preserved(self, in_rate, out_rate):
        src = tone(1000, in_rate, in_rate, amp=8000.0)
        out = resample_block(src, in_rate, out_rate)
        # Ignore filter ramp-up at the edges.
        assert rms(out[100:-100]) == pytest.approx(rms(src), rel=0.10)

    def test_dc_is_preserved(self):
        """Unity DC gain: a constant must not be scaled or offset."""
        out = resample_block(np.full(4000, 1000, dtype=np.int16), 8000, 16000)
        assert np.mean(out[200:-200]) == pytest.approx(1000, abs=10)

    def test_silence_stays_silent(self):
        out = resample_block(np.zeros(4000, dtype=np.int16), 48000, 16000)
        assert np.abs(out).max() == 0

    def test_downsampling_attenuates_above_nyquist(self):
        """Anti-aliasing: 6 kHz into an 8 kHz output must not fold back."""
        out = resample_block(tone(6000, 16000, 16000), 16000, 8000)
        assert rms(out[100:-100]) < 0.15 * 8000.0

    def test_no_aliasing_artifact_frequency(self):
        """6 kHz at 16k downsampled to 8k would alias to 2 kHz if unfiltered."""
        out = resample_block(tone(6000, 16000, 16000), 16000, 8000)
        segment = out[200:-200]
        if rms(segment) > 100:
            assert not (1800 < dominant_freq(segment, 8000) < 2200)

    def test_output_is_int16(self):
        assert resample_block(tone(440, 1000, 8000), 8000, 16000).dtype == np.int16

    def test_loud_input_saturates_without_wrapping(self):
        loud = np.full(2000, 32000, dtype=np.int16)
        out = resample_block(loud, 8000, 16000)
        assert out.max() <= 32767
        assert out.min() >= -32768
        assert np.mean(out[200:-200]) > 30000, "should stay near full scale, not wrap"


class TestStreamingContinuity:
    """A block boundary must not be observable in the output."""

    @pytest.mark.parametrize("in_rate,out_rate", RATE_PAIRS)
    def test_blocked_matches_whole(self, in_rate, out_rate):
        """Feeding a signal in blocks must equal feeding it in one go."""
        n = in_rate // 2
        src = tone(500, n, in_rate)

        whole = Resampler(in_rate, out_rate).process(src)

        streamed_r = Resampler(in_rate, out_rate)
        block = max(1, in_rate // 50)          # 20 ms blocks
        streamed = np.concatenate(
            [streamed_r.process(src[i : i + block]) for i in range(0, n, block)]
        )

        assert len(streamed) == len(whole)
        assert np.array_equal(streamed, whole), "block boundaries changed the output"

    def test_no_discontinuity_at_block_joins(self):
        """The failure mode a naive resampler has: a tick every block.

        A continuous sine resampled in blocks must stay smooth. Any joint
        artefact shows up as a sample-to-sample jump far larger than the
        signal's own maximum slew rate.
        """
        in_rate, out_rate, freq = 48000, 16000, 400.0
        r = Resampler(in_rate, out_rate)
        block = in_rate // 50

        chunks = []
        for b in range(40):
            t0 = b * block
            t = (np.arange(block) + t0) / in_rate
            chunks.append((8000 * np.sin(2 * np.pi * freq * t)).astype(np.int16))

        out = np.concatenate([r.process(c) for c in chunks])
        settled = out[200:-10].astype(np.float64)

        max_jump = np.max(np.abs(np.diff(settled)))
        # Expected slew of the tone itself, with generous headroom.
        expected = 8000 * 2 * np.pi * freq / out_rate
        assert max_jump < expected * 2.0, (
            f"discontinuity at a block join: max jump {max_jump:.0f} "
            f"vs expected slew {expected:.0f}"
        )

    def test_uneven_block_sizes(self):
        """Real capture can hand over short or ragged blocks."""
        r = Resampler(48000, 16000)
        src = tone(440, 48000, 48000)
        out, i = [], 0
        for size in [960, 1, 4800, 17, 960, 2400] * 6:
            if i >= len(src):
                break
            out.append(r.process(src[i : i + size]))
            i += size
        joined = np.concatenate(out)
        assert dominant_freq(joined[200:], 16000) == pytest.approx(440, abs=20)

    def test_reset_clears_state(self):
        r = Resampler(8000, 16000)
        first = r.process(tone(440, 800, 8000))
        r.reset()
        second = r.process(tone(440, 800, 8000))
        assert np.array_equal(first, second)

    def test_state_persists_without_reset(self):
        """Second block must differ from the first: history is carried."""
        r = Resampler(8000, 16000)
        src = tone(440, 800, 8000)
        first = r.process(src)
        second = r.process(src)
        assert not np.array_equal(first, second)

    def test_flush_drains_the_tail(self):
        r = Resampler(8000, 16000)
        r.process(tone(440, 800, 8000))
        assert len(r.flush()) > 0

    def test_flush_on_passthrough_is_empty(self):
        assert len(Resampler(16000, 16000).flush()) == 0


class TestRealWorldScenarios:
    def test_peer_at_8k_into_16k_device(self):
        """A Zello client choosing 8 kHz, played on a 16 kHz AIOC."""
        r = Resampler(8000, 16000)
        out = np.concatenate([r.process(tone(440, 160, 8000, phase=2 * np.pi * 440 * (i * 160) / 8000))
                              for i in range(50)])
        assert dominant_freq(out[200:], 16000) == pytest.approx(440, abs=15)

    def test_peer_at_48k_into_16k_device(self):
        r = Resampler(48000, 16000)
        out = np.concatenate([r.process(tone(440, 960, 48000, phase=2 * np.pi * 440 * (i * 960) / 48000))
                              for i in range(50)])
        assert dominant_freq(out[200:], 16000) == pytest.approx(440, abs=15)

    def test_device_at_48k_capture_to_16k_opus(self):
        """AIOC running at its preferred 48 kHz, encoding Opus at 16 kHz."""
        r = Resampler(48000, 16000)
        out = r.process(tone(1000, 48000, 48000))
        assert dominant_freq(out, 16000) == pytest.approx(1000, abs=15)
        assert len(out) == pytest.approx(16000, abs=32)

    def test_round_trip_preserves_a_tone(self):
        up = resample_block(tone(800, 16000, 16000), 16000, 48000)
        down = resample_block(up, 48000, 16000)
        assert dominant_freq(down[200:-200], 16000) == pytest.approx(800, abs=15)

    def test_sustained_stream_does_not_drift(self):
        """Ten seconds of blocks must not accumulate a length error."""
        r = Resampler(44100, 16000)
        total = sum(len(r.process(np.zeros(441, dtype=np.int16))) for _ in range(1000))
        expected = 1000 * 441 * 16000 / 44100
        assert abs(total - expected) < 2 * r.taps_per_phase

    def test_history_buffer_does_not_grow(self):
        """AT-11: no unbounded memory growth over a long run."""
        r = Resampler(48000, 16000)
        for _ in range(500):
            r.process(np.zeros(960, dtype=np.int16))
        assert r._hist.size <= r.taps_per_phase + 960
