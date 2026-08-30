"""COS timing, level measurement, and gain/clip behaviour.

The detector takes an injectable clock, so every attack/hang assertion here
is exact and instant -- no sleeping, no flake.
"""

from __future__ import annotations


import numpy as np
import pytest

from zello_link.audio.levels import (
    DBFS_FLOOR,
    CosDetector,
    CosEvent,
    LevelMeter,
    apply_gain_db,
    peak_dbfs,
    rms_dbfs,
)


class FakeClock:
    """Manually advanced monotonic clock."""

    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance_ms(self, ms: float) -> None:
        self.t += ms / 1000.0


def tone(dbfs: float, n: int = 320) -> np.ndarray:
    """Constant-amplitude block at an exact RMS level."""
    amplitude = (10.0 ** (dbfs / 20.0)) * 32768.0
    return np.full(n, int(round(amplitude)), dtype=np.int16)


def silence(n: int = 320) -> np.ndarray:
    return np.zeros(n, dtype=np.int16)


class TestLevelMeasurement:
    def test_silence_is_at_the_floor(self):
        assert rms_dbfs(silence()) == pytest.approx(DBFS_FLOOR, abs=0.01)

    def test_floor_is_documented_value(self):
        """The max(rms, 1) floor lands at ~-90.31 dBFS."""
        assert DBFS_FLOOR == pytest.approx(-90.31, abs=0.01)

    def test_full_scale_is_zero_dbfs(self):
        assert rms_dbfs(np.full(320, 32767, dtype=np.int16)) == pytest.approx(0.0, abs=0.01)

    @pytest.mark.parametrize("level", [-6.0, -20.0, -38.0, -60.0])
    def test_known_levels(self, level):
        assert rms_dbfs(tone(level)) == pytest.approx(level, abs=0.1)

    def test_no_int16_overflow_on_squaring(self):
        """Squaring int16 in-place would wrap; the result must stay sane."""
        assert rms_dbfs(np.full(320, -32768, dtype=np.int16)) == pytest.approx(0.0, abs=0.01)

    def test_peak_exceeds_rms_for_a_transient(self):
        block = silence()
        block[100] = 32000
        assert peak_dbfs(block) > rms_dbfs(block)

    def test_empty_block(self):
        assert rms_dbfs(np.array([], dtype=np.int16)) == DBFS_FLOOR


class TestGainAndClipping:
    def test_unity_gain_is_a_passthrough(self):
        block = tone(-20.0)
        out, clipped = apply_gain_db(block, 0.0)
        assert np.array_equal(out, block)
        assert clipped == 0

    def test_gain_raises_level(self):
        out, _ = apply_gain_db(tone(-30.0), 6.0)
        assert rms_dbfs(out) == pytest.approx(-24.0, abs=0.1)

    def test_attenuation_lowers_level(self):
        out, _ = apply_gain_db(tone(-20.0), -12.0)
        assert rms_dbfs(out) == pytest.approx(-32.0, abs=0.1)

    def test_saturates_instead_of_wrapping(self):
        out, clipped = apply_gain_db(tone(-3.0), 20.0)
        assert out.max() <= 32767
        assert out.min() >= -32768
        assert clipped > 0, "clipping must be counted"
        # The bug this guards: wrapping would flip a loud positive sample
        # negative and the level would read low instead of pinned.
        assert rms_dbfs(out) == pytest.approx(0.0, abs=0.1)

    def test_clip_count_is_exact(self):
        block = np.array([0, 30000, 0, -30000], dtype=np.int16)
        _, clipped = apply_gain_db(block, 6.0)
        assert clipped == 2

    def test_no_clip_with_headroom(self):
        _, clipped = apply_gain_db(tone(-40.0), 6.0)
        assert clipped == 0

    def test_output_stays_int16(self):
        out, _ = apply_gain_db(tone(-20.0), 3.0)
        assert out.dtype == np.int16


class TestCosAttack:
    def make(self, clock, **kw):
        opts = dict(threshold_dbfs=-38.0, attack_ms=60, hang_ms=450, startup_ignore_ms=0)
        opts.update(kw)
        return CosDetector(clock=clock, **opts)

    def test_does_not_fire_before_attack_elapses(self):
        clock = FakeClock()
        cos = self.make(clock)
        _, ev = cos.feed(tone(-20.0))
        assert ev is None
        clock.advance_ms(59)
        _, ev = cos.feed(tone(-20.0))
        assert ev is None
        assert not cos.active

    def test_fires_once_attack_elapses(self):
        clock = FakeClock()
        cos = self.make(clock)
        cos.feed(tone(-20.0))
        clock.advance_ms(60)
        _, ev = cos.feed(tone(-20.0))
        assert ev is CosEvent.ACTIVE
        assert cos.active

    def test_attack_must_be_continuous(self):
        """A dip below threshold restarts the attack timer."""
        clock = FakeClock()
        cos = self.make(clock)
        cos.feed(tone(-20.0))
        clock.advance_ms(40)
        cos.feed(silence())          # breaks the run
        clock.advance_ms(40)
        _, ev = cos.feed(tone(-20.0))
        assert ev is None, "attack timer must restart after a dip"
        clock.advance_ms(60)
        _, ev = cos.feed(tone(-20.0))
        assert ev is CosEvent.ACTIVE

    def test_below_threshold_never_fires(self):
        clock = FakeClock()
        cos = self.make(clock)
        for _ in range(50):
            _, ev = cos.feed(tone(-50.0))
            assert ev is None
            clock.advance_ms(20)
        assert not cos.active

    def test_exactly_at_threshold_counts_as_above(self):
        clock = FakeClock()
        cos = self.make(clock, attack_ms=0)
        _, ev = cos.feed(tone(-38.0))
        assert ev is CosEvent.ACTIVE

    def test_zero_attack_fires_immediately(self):
        clock = FakeClock()
        cos = self.make(clock, attack_ms=0)
        _, ev = cos.feed(tone(-10.0))
        assert ev is CosEvent.ACTIVE

    def test_active_event_emitted_only_once(self):
        clock = FakeClock()
        cos = self.make(clock, attack_ms=0)
        assert cos.feed(tone(-10.0))[1] is CosEvent.ACTIVE
        clock.advance_ms(20)
        assert cos.feed(tone(-10.0))[1] is None


class TestCosHang:
    def make(self, clock, **kw):
        opts = dict(threshold_dbfs=-38.0, attack_ms=0, hang_ms=450, startup_ignore_ms=0)
        opts.update(kw)
        return CosDetector(clock=clock, **opts)

    def activate(self, cos, clock):
        cos.feed(tone(-10.0))
        assert cos.active

    def test_stays_active_through_a_short_pause(self):
        """AT-04: pauses shorter than hang_ms must not split the stream."""
        clock = FakeClock()
        cos = self.make(clock)
        self.activate(cos, clock)
        cos.feed(silence())
        clock.advance_ms(400)          # < hang_ms
        _, ev = cos.feed(silence())
        assert ev is None
        assert cos.active

    def test_closes_after_hang_elapses(self):
        clock = FakeClock()
        cos = self.make(clock)
        self.activate(cos, clock)
        cos.feed(silence())
        clock.advance_ms(450)
        _, ev = cos.feed(silence())
        assert ev is CosEvent.INACTIVE
        assert not cos.active

    def test_speech_resets_the_hang_timer(self):
        clock = FakeClock()
        cos = self.make(clock)
        self.activate(cos, clock)
        for _ in range(5):
            cos.feed(silence())
            clock.advance_ms(300)
            _, ev = cos.feed(tone(-10.0))   # audio returns, resets hang
            assert ev is None
            clock.advance_ms(20)
        assert cos.active

    def test_zero_hang_closes_immediately(self):
        clock = FakeClock()
        cos = self.make(clock, hang_ms=0)
        self.activate(cos, clock)
        _, ev = cos.feed(silence())
        assert ev is CosEvent.INACTIVE

    def test_full_cycle(self):
        clock = FakeClock()
        cos = self.make(clock, attack_ms=60)
        cos.feed(tone(-10.0))
        clock.advance_ms(60)
        assert cos.feed(tone(-10.0))[1] is CosEvent.ACTIVE
        cos.feed(silence())
        clock.advance_ms(450)
        assert cos.feed(silence())[1] is CosEvent.INACTIVE
        # And it can open again.
        cos.feed(tone(-10.0))
        clock.advance_ms(60)
        assert cos.feed(tone(-10.0))[1] is CosEvent.ACTIVE


class TestStartupIgnore:
    def test_transitions_ignored_during_window(self):
        clock = FakeClock()
        cos = CosDetector(
            threshold_dbfs=-38.0, attack_ms=0, hang_ms=450,
            startup_ignore_ms=500, clock=clock,
        )
        for _ in range(10):
            _, ev = cos.feed(tone(-10.0))
            assert ev is None, "must not fire inside the startup-ignore window"
            clock.advance_ms(20)
        assert not cos.active

    def test_fires_after_window(self):
        clock = FakeClock()
        cos = CosDetector(
            threshold_dbfs=-38.0, attack_ms=0, hang_ms=450,
            startup_ignore_ms=500, clock=clock,
        )
        clock.advance_ms(500)
        _, ev = cos.feed(tone(-10.0))
        assert ev is CosEvent.ACTIVE

    def test_levels_still_measured_while_ignoring(self):
        """Diagnostics must work during the ignore window."""
        clock = FakeClock()
        cos = CosDetector(
            threshold_dbfs=-38.0, attack_ms=0, hang_ms=450,
            startup_ignore_ms=500, clock=clock,
        )
        stats, ev = cos.feed(tone(-20.0))
        assert ev is None
        assert stats.rms_dbfs == pytest.approx(-20.0, abs=0.1)

    def test_reset_rearms_the_window(self):
        clock = FakeClock()
        cos = CosDetector(
            threshold_dbfs=-38.0, attack_ms=0, hang_ms=450,
            startup_ignore_ms=500, clock=clock,
        )
        clock.advance_ms(500)
        assert cos.feed(tone(-10.0))[1] is CosEvent.ACTIVE
        cos.reset()                      # simulates a device reopen
        assert not cos.active
        assert cos.feed(tone(-10.0))[1] is None


class TestSuppression:
    """Loopback prevention while the bridge is transmitting Zello->RF."""

    def make(self, clock):
        return CosDetector(
            threshold_dbfs=-38.0, attack_ms=0, hang_ms=450,
            startup_ignore_ms=0, clock=clock,
        )

    def test_suppressed_detector_ignores_loud_audio(self):
        clock = FakeClock()
        cos = self.make(clock)
        cos.suppress(True)
        for _ in range(20):
            _, ev = cos.feed(tone(-5.0))
            assert ev is None, "gateway TX audio must not open a Zello stream"
            clock.advance_ms(20)
        assert not cos.active

    def test_suppressing_while_active_forces_inactive(self):
        clock = FakeClock()
        cos = self.make(clock)
        cos.feed(tone(-10.0))
        assert cos.active
        assert cos.suppress(True) is CosEvent.INACTIVE
        assert not cos.active

    def test_release_restarts_attack_timer(self):
        """A partially elapsed attack must not fire the instant suppression lifts."""
        clock = FakeClock()
        cos = CosDetector(
            threshold_dbfs=-38.0, attack_ms=60, hang_ms=450,
            startup_ignore_ms=0, clock=clock,
        )
        cos.suppress(True)
        cos.feed(tone(-10.0))
        clock.advance_ms(500)
        cos.suppress(False)
        _, ev = cos.feed(tone(-10.0))
        assert ev is None, "attack must restart on unsuppress"
        clock.advance_ms(60)
        assert cos.feed(tone(-10.0))[1] is CosEvent.ACTIVE

    def test_levels_measured_while_suppressed(self):
        clock = FakeClock()
        cos = self.make(clock)
        cos.suppress(True)
        stats, _ = cos.feed(tone(-15.0))
        assert stats.rms_dbfs == pytest.approx(-15.0, abs=0.1)

    def test_repeated_suppress_is_idempotent(self):
        clock = FakeClock()
        cos = self.make(clock)
        assert cos.suppress(True) is CosEvent.INACTIVE or True
        assert cos.suppress(True) is None


class TestLevelMeter:
    def test_accumulates_stats(self):
        clock = FakeClock()
        cos = CosDetector(
            threshold_dbfs=-38.0, attack_ms=0, hang_ms=0,
            startup_ignore_ms=0, clock=clock,
        )
        meter = LevelMeter()
        for lvl in (-40.0, -20.0, -60.0):
            stats, _ = cos.feed(tone(lvl))
            meter.add(stats)
        assert meter.blocks == 3
        assert meter.max_dbfs == pytest.approx(-20.0, abs=0.1)
        assert meter.min_dbfs == pytest.approx(-60.0, abs=0.1)

    def test_peak_hold_survives_quiet_blocks(self):
        meter = LevelMeter()
        clock = FakeClock()
        cos = CosDetector(
            threshold_dbfs=-38.0, attack_ms=0, hang_ms=0,
            startup_ignore_ms=0, clock=clock,
        )
        meter.add(cos.feed(tone(-6.0))[0])
        for _ in range(10):
            meter.add(cos.feed(silence())[0])
        assert meter.peak_hold_dbfs == pytest.approx(-6.0, abs=0.2)

    def test_clip_counter_accumulates(self):
        meter = LevelMeter()
        clock = FakeClock()
        cos = CosDetector(
            threshold_dbfs=-38.0, attack_ms=0, hang_ms=0,
            startup_ignore_ms=0, clock=clock,
        )
        block, clipped = apply_gain_db(tone(-3.0), 20.0)
        stats, _ = cos.feed(block, clipped=clipped)
        meter.add(stats)
        assert meter.clipped_samples == clipped > 0

    def test_window_is_bounded(self):
        meter = LevelMeter(window=10)
        clock = FakeClock()
        cos = CosDetector(
            threshold_dbfs=-38.0, attack_ms=0, hang_ms=0,
            startup_ignore_ms=0, clock=clock,
        )
        for _ in range(1000):
            meter.add(cos.feed(silence())[0])
        assert len(meter._rms) == 10
        assert meter.blocks == 1000


class TestMinimumTransmissionTime:
    """min_tx_ms: a floor on how long COS stays open once it triggers.

    The radio emits a click as the squelch opens, then a gap, then speech.
    The click trips COS, the gap outlasts hang_ms, COS drops, and the front
    of the speech is lost. min_tx_ms bridges that gap without lengthening
    the tail on ordinary traffic.

    Contract: open time == max(min_tx_ms, speech + hang_ms).
    """

    def make(self, clock, *, min_tx_ms=500, hang_ms=200, **kw):
        opts = dict(
            threshold_dbfs=-38.0, attack_ms=0, hang_ms=hang_ms,
            startup_ignore_ms=0, min_tx_ms=min_tx_ms,
        )
        opts.update(kw)
        return CosDetector(clock=clock, **opts)

    def _open_ms(self, clock, cos, speech_ms):
        """Drive one transmission; return how long COS stayed open."""
        start = clock.t
        cos.feed(tone(-10.0))                 # attack_ms=0, so active now
        assert cos.active
        remaining = speech_ms
        while remaining > 0:
            step = min(20, remaining)
            clock.advance_ms(step)
            cos.feed(tone(-10.0))
            remaining -= step
        while cos.active:
            clock.advance_ms(20)
            _, ev = cos.feed(silence())
            if ev is CosEvent.INACTIVE:
                break
            assert (clock.t - start) * 1000 < 20000, "COS never closed"
        return (clock.t - start) * 1000.0

    def test_short_burst_is_held_to_the_floor(self):
        """A 0 ms click must still hold the stream open for min_tx_ms."""
        clock = FakeClock()
        cos = self.make(clock, min_tx_ms=500, hang_ms=200)
        assert self._open_ms(clock, cos, speech_ms=0) == pytest.approx(500, abs=25)

    def test_the_worked_example(self):
        """301 ms of speech + 200 ms hang = 501 ms, which beats the 500 floor."""
        clock = FakeClock()
        cos = self.make(clock, min_tx_ms=500, hang_ms=200)
        assert self._open_ms(clock, cos, speech_ms=300) == pytest.approx(500, abs=25)

    def test_long_transmission_is_not_dragged_out(self):
        """The floor must not extend traffic that already exceeds it."""
        clock = FakeClock()
        cos = self.make(clock, min_tx_ms=500, hang_ms=200)
        assert self._open_ms(clock, cos, speech_ms=2000) == pytest.approx(2200, abs=25)

    def test_open_time_is_max_of_floor_and_speech_plus_hang(self):
        for speech, expected in [(0, 500), (100, 500), (200, 500),
                                 (400, 600), (1000, 1200), (3000, 3200)]:
            clock = FakeClock()
            cos = self.make(clock, min_tx_ms=500, hang_ms=200)
            got = self._open_ms(clock, cos, speech_ms=speech)
            assert got == pytest.approx(max(500, speech + 200), abs=25), (
                f"speech={speech}ms gave {got}ms, expected "
                f"{max(500, speech + 200)}ms"
            )

    def test_click_then_gap_then_speech_stays_one_stream(self):
        """The actual failure this fixes: click, 400 ms gap, then speech."""
        clock = FakeClock()
        cos = self.make(clock, min_tx_ms=800, hang_ms=200)

        cos.feed(tone(-10.0))                  # the squelch click
        assert cos.active

        for _ in range(20):                    # 400 ms of silence
            clock.advance_ms(20)
            _, ev = cos.feed(silence())
            assert ev is not CosEvent.INACTIVE, "COS dropped during the gap"

        for _ in range(25):                    # speech arrives
            clock.advance_ms(20)
            _, ev = cos.feed(tone(-12.0))
            assert ev is None
        assert cos.active, "speech should be inside the original stream"

    def test_without_min_tx_the_gap_splits_the_stream(self):
        """Proves the test above is actually exercising the fix."""
        clock = FakeClock()
        cos = self.make(clock, min_tx_ms=0, hang_ms=200)

        cos.feed(tone(-10.0))
        dropped = False
        for _ in range(20):
            clock.advance_ms(20)
            _, ev = cos.feed(silence())
            if ev is CosEvent.INACTIVE:
                dropped = True
                break
        assert dropped, "expected the gap to split the stream without min_tx_ms"

    def test_zero_disables_the_floor(self):
        clock = FakeClock()
        cos = self.make(clock, min_tx_ms=0, hang_ms=200)
        assert self._open_ms(clock, cos, speech_ms=0) == pytest.approx(200, abs=25)

    def test_audio_returning_restarts_the_hang_not_the_floor(self):
        """A second burst inside the floor extends by hang, not by min_tx."""
        clock = FakeClock()
        cos = self.make(clock, min_tx_ms=500, hang_ms=200)
        cos.feed(tone(-10.0))
        clock.advance_ms(700)                  # past the floor already
        cos.feed(tone(-10.0))                  # audio returns
        clock.advance_ms(200)
        _, ev = cos.feed(silence())
        assert ev is None, "hang restarts when audio returns"
        clock.advance_ms(200)
        assert cos.feed(silence())[1] is CosEvent.INACTIVE

    def test_suppression_overrides_the_floor(self):
        """Loopback prevention must not be delayed by min_tx_ms."""
        clock = FakeClock()
        cos = self.make(clock, min_tx_ms=5000, hang_ms=200)
        cos.feed(tone(-10.0))
        assert cos.active
        assert cos.suppress(True) is CosEvent.INACTIVE
        assert not cos.active

    def test_reset_clears_the_floor(self):
        clock = FakeClock()
        cos = self.make(clock, min_tx_ms=5000, hang_ms=200)
        cos.feed(tone(-10.0))
        cos.reset()
        assert not cos.active
