"""Level measurement, gain staging, and the internal audio COS detector.

Everything here is driven by an injectable clock and operates on plain int16
arrays, so the whole COS state machine is unit-testable with no sound card,
no radio, and no wall-clock sleeping.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import numpy as np

__all__ = [
    "FULL_SCALE",
    "DBFS_FLOOR",
    "BlockStats",
    "CosEvent",
    "CosDetector",
    "LevelMeter",
    "apply_gain_db",
    "rms_dbfs",
    "peak_dbfs",
]

#: int16 full scale used as the 0 dBFS reference.
FULL_SCALE = 32768.0

#: The RMS is floored at 1 LSB before the logarithm, so this is the lowest
#: value the detector can ever report. A threshold below it never triggers.
DBFS_FLOOR = 20.0 * math.log10(1.0 / FULL_SCALE)  # ~= -90.31 dBFS

_INT16_MIN = -32768
_INT16_MAX = 32767

#: Tolerance for elapsed-time comparisons, in milliseconds.
#:
#: Monotonic timestamps are large floats, so a difference that is exactly
#: 60 ms in decimal can come back as 59.99999999995 ms in binary floating
#: point. Without a tolerance an attack/hang boundary would be met or missed
#: depending on the clock's absolute value, delaying the decision by a whole
#: block at random. Sub-microsecond precision is meaningless here anyway --
#: COS decisions are only ever evaluated on block boundaries.
_TIME_EPS_MS = 1e-6


def rms_dbfs(pcm: np.ndarray) -> float:
    """RMS level of an int16 block in dBFS.

    Computed in float64 to avoid the int16 overflow that squaring would
    otherwise cause.
    """
    if pcm.size == 0:
        return DBFS_FLOOR
    x = pcm.astype(np.float64)
    rms = math.sqrt(float(np.mean(x * x)))
    return 20.0 * math.log10(max(rms, 1.0) / FULL_SCALE)


def peak_dbfs(pcm: np.ndarray) -> float:
    """Absolute peak of an int16 block in dBFS."""
    if pcm.size == 0:
        return DBFS_FLOOR
    peak = float(np.max(np.abs(pcm.astype(np.int32))))
    return 20.0 * math.log10(max(peak, 1.0) / FULL_SCALE)


def apply_gain_db(pcm: np.ndarray, gain_db: float) -> tuple[np.ndarray, int]:
    """Apply gain to an int16 block, saturating rather than wrapping.

    Saturation is applied *after* the gain, per the audio pipeline rules.
    Returns the processed block and the number of samples that hit the rail,
    which the caller surfaces as a clip counter -- without it, setting
    rx_gain_db/tx_gain_db is guesswork.
    """
    if gain_db == 0.0:
        clipped = int(np.count_nonzero((pcm <= _INT16_MIN) | (pcm >= _INT16_MAX)))
        return pcm, clipped

    scaled = pcm.astype(np.float64) * (10.0 ** (gain_db / 20.0))
    clipped = int(np.count_nonzero((scaled <= _INT16_MIN) | (scaled >= _INT16_MAX)))
    return np.clip(scaled, _INT16_MIN, _INT16_MAX).astype(np.int16), clipped


@dataclass(frozen=True, slots=True)
class BlockStats:
    """Per-block measurements, surfaced to diagnostics and metrics."""

    rms_dbfs: float
    peak_dbfs: float
    clipped_samples: int


class CosEvent(str, Enum):
    ACTIVE = "cos_active"
    INACTIVE = "cos_inactive"


@dataclass
class CosDetector:
    """Threshold/attack/hang carrier-operated-squelch detector.

    Transition rules:
      * Inactive -> Active only after the level has been at or above the
        threshold *continuously* for ``attack_ms``.
      * Active stays Active while any block is at or above the threshold.
      * Active -> Inactive only once BOTH are true: the level has stayed
        below the threshold for ``hang_ms``, AND ``min_tx_ms`` has elapsed
        since the detector went active. So a transmission is held open for
        ``max(min_tx_ms, speech + hang_ms)``.
      * All transitions are ignored for ``startup_ignore_ms`` after
        ``reset()``, which the audio engine calls on every device open.

    ``min_tx_ms`` exists for a specific radio behaviour: many radios emit a
    loud click as the squelch opens, then a gap of silence, then speech. The
    click trips COS, the gap outlasts ``hang_ms``, COS drops, and the front
    of the speech is lost re-triggering. Holding open for a floor duration
    bridges that gap without lengthening the tail on normal traffic --
    unlike raising ``hang_ms``, which delays the turnaround on every single
    transmission.

    ``suppress()`` is asserted by the controller for the whole Zello->RF
    direction. It is what stops the gateway's own transmit audio, bleeding
    back through the radio's receive path, from opening a Zello stream.
    """

    threshold_dbfs: float
    attack_ms: int
    hang_ms: int
    startup_ignore_ms: int = 0
    min_tx_ms: int = 0
    clock: Callable[[], float] = time.monotonic

    _active: bool = field(default=False, init=False)
    _suppressed: bool = field(default=False, init=False)
    _above_since: float | None = field(default=None, init=False)
    _below_since: float | None = field(default=None, init=False)
    _active_since: float | None = field(default=None, init=False)
    _ignore_until: float = field(default=0.0, init=False)
    _last: BlockStats | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.reset()

    # -- state -----------------------------------------------------------
    @property
    def active(self) -> bool:
        return self._active

    @property
    def suppressed(self) -> bool:
        return self._suppressed

    @property
    def last_stats(self) -> BlockStats | None:
        return self._last

    def reset(self) -> None:
        """Re-arm after a capture device open/reopen."""
        now = self.clock()
        self._active = False
        self._above_since = None
        self._below_since = None
        self._active_since = None
        self._ignore_until = now + self.startup_ignore_ms / 1000.0

    def suppress(self, on: bool) -> CosEvent | None:
        """Assert or release loopback suppression.

        Asserting forces the detector inactive immediately. Releasing
        restarts the attack timer so a partially-elapsed attack from before
        the transmission cannot fire the instant suppression lifts.
        """
        if on == self._suppressed:
            return None
        self._suppressed = on
        if on:
            # Loopback prevention outranks min_tx_ms: the gateway's own
            # transmit audio must never hold a Zello stream open.
            was_active = self._active
            self._active = False
            self._above_since = None
            self._below_since = None
            self._active_since = None
            return CosEvent.INACTIVE if was_active else None
        self._above_since = None
        self._below_since = None
        return None

    # -- detection -------------------------------------------------------
    def feed(self, pcm: np.ndarray, *, clipped: int = 0) -> tuple[BlockStats, CosEvent | None]:
        """Measure one capture block and advance the state machine.

        Levels are always measured -- diagnostics and the level monitor rely
        on them even while suppressed or inside the startup-ignore window.
        """
        stats = BlockStats(
            rms_dbfs=rms_dbfs(pcm),
            peak_dbfs=peak_dbfs(pcm),
            clipped_samples=clipped,
        )
        self._last = stats

        now = self.clock()
        if self._suppressed or now < self._ignore_until - _TIME_EPS_MS / 1000.0:
            return stats, None

        above = stats.rms_dbfs >= self.threshold_dbfs

        if above:
            self._below_since = None
            if self._active:
                return stats, None
            if self._above_since is None:
                self._above_since = now
            if (now - self._above_since) * 1000.0 >= self.attack_ms - _TIME_EPS_MS:
                self._active = True
                self._above_since = None
                self._active_since = now
                return stats, CosEvent.ACTIVE
            return stats, None

        # Below threshold.
        self._above_since = None
        if not self._active:
            return stats, None
        if self._below_since is None:
            self._below_since = now

        hang_elapsed = (now - self._below_since) * 1000.0 >= self.hang_ms - _TIME_EPS_MS
        min_tx_elapsed = self._active_since is None or (
            (now - self._active_since) * 1000.0 >= self.min_tx_ms - _TIME_EPS_MS
        )

        # Both gates, so the open time is max(min_tx_ms, speech + hang_ms):
        # short bursts are stretched to the floor, long transmissions are not
        # extended past their own tail.
        if hang_elapsed and min_tx_elapsed:
            self._active = False
            self._below_since = None
            self._active_since = None
            return stats, CosEvent.INACTIVE
        return stats, None


@dataclass
class LevelMeter:
    """Rolling level statistics for the --cos-monitor diagnostic and metrics."""

    window: int = 100

    _rms: list[float] = field(default_factory=list, init=False)
    _peak_hold: float = field(default=DBFS_FLOOR, init=False)
    _clips: int = field(default=0, init=False)
    _blocks: int = field(default=0, init=False)

    def add(self, stats: BlockStats) -> None:
        self._blocks += 1
        self._clips += stats.clipped_samples
        self._peak_hold = max(self._peak_hold, stats.peak_dbfs)
        self._rms.append(stats.rms_dbfs)
        if len(self._rms) > self.window:
            del self._rms[: len(self._rms) - self.window]

    @property
    def blocks(self) -> int:
        return self._blocks

    @property
    def clipped_samples(self) -> int:
        return self._clips

    @property
    def peak_hold_dbfs(self) -> float:
        return self._peak_hold

    @property
    def mean_dbfs(self) -> float:
        return sum(self._rms) / len(self._rms) if self._rms else DBFS_FLOOR

    @property
    def max_dbfs(self) -> float:
        return max(self._rms) if self._rms else DBFS_FLOOR

    @property
    def min_dbfs(self) -> float:
        return min(self._rms) if self._rms else DBFS_FLOOR

    def reset_peak(self) -> None:
        self._peak_hold = DBFS_FLOOR
