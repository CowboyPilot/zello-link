"""Streaming rational sample-rate conversion.

Two places need this, and the second is not optional:

  1. ``sound.sample_rate`` differing from ``opus.sample_rate``. Avoidable by
     running the AIOC at 16 kHz, which is why the config warns about it.
  2. An inbound Zello stream whose ``codec_header`` declares a rate other than
     the playback device's. Another Zello client chooses its own rate, so this
     happens with a perfectly-configured bridge. Without conversion the audio
     plays at the wrong pitch and speed into a keyed transmitter.

Design notes
------------
**Polyphase, not zero-stuffing.** The textbook "upsample by L, filter,
decimate by M" is easy to write but materialises L times more samples than it
keeps. The polyphase decomposition computes only the output samples actually
wanted, which matters on a Pi doing this every 20 ms.

**Stateful, not per-block.** A resampler that treats each block independently
produces a discontinuity -- an audible tick -- at every block boundary, fifty
times a second. This one carries both the filter delay line and a phase
accumulator across calls, so a block boundary is not observable in the output.
That also means output length varies block to block (e.g. 3 kHz-worth of
input does not divide evenly), which is correct and expected.

**No scipy.** A windowed-sinc FIR in numpy keeps the dependency footprint
suitable for an appliance image.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np

__all__ = ["Resampler", "ResamplerError", "resample_block", "SUPPORTED_DEVICE_RATES"]

#: Rates the AIOC's USB sound card advertises, per its README.
SUPPORTED_DEVICE_RATES: Final[tuple[int, ...]] = (
    8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000,
)

_INT16_MIN: Final = -32768
_INT16_MAX: Final = 32767

#: Taps per polyphase branch. 16 gives roughly 60 dB of stopband rejection
#: with a Kaiser window, which is well beyond what a 16 kHz voice link over
#: DMR can resolve, at a cost of 16 multiply-accumulates per output sample.
_DEFAULT_TAPS_PER_PHASE: Final = 16

#: Kaiser beta. 8.6 is the classic choice for ~-65 dB sidelobes.
_KAISER_BETA: Final = 8.6

#: Guard against a pathological ratio (e.g. a peer at some exotic rate)
#: allocating an enormous filter. 44100<->48000 needs L=160, so this is roomy.
_MAX_PHASES: Final = 1000


class ResamplerError(Exception):
    """Unsupported or unreasonable rate conversion."""


def _design_lowpass(num_taps: int, cutoff: float, *, beta: float = _KAISER_BETA) -> np.ndarray:
    """Windowed-sinc lowpass.

    ``cutoff`` is normalised to the *upsampled* rate (cycles per sample), so
    the anti-imaging and anti-aliasing requirements collapse into one filter.
    """
    n = np.arange(num_taps, dtype=np.float64)
    centre = (num_taps - 1) / 2.0
    # np.sinc(x) is sin(pi x)/(pi x), so the 2*fc factors give a cutoff at fc.
    h = 2.0 * cutoff * np.sinc(2.0 * cutoff * (n - centre))
    h *= np.kaiser(num_taps, beta)

    total = h.sum()
    if total != 0.0:
        h /= total          # unity DC gain
    return h


class Resampler:
    """Stateful rational resampler for mono int16 audio.

    Call :meth:`process` with successive blocks of one continuous stream, and
    :meth:`reset` between streams. Output length varies per call.
    """

    def __init__(
        self,
        in_rate: int,
        out_rate: int,
        *,
        taps_per_phase: int = _DEFAULT_TAPS_PER_PHASE,
    ) -> None:
        if in_rate <= 0 or out_rate <= 0:
            raise ResamplerError(f"invalid rates {in_rate} -> {out_rate}")
        if taps_per_phase < 2:
            raise ResamplerError("taps_per_phase must be at least 2")

        self.in_rate = in_rate
        self.out_rate = out_rate
        self.taps_per_phase = taps_per_phase

        # Identity conversion: skip all the machinery.
        self.passthrough = in_rate == out_rate
        if self.passthrough:
            self.L = self.M = 1
            self._phases = np.ones((1, 1))
            self.reset()
            return

        g = math.gcd(in_rate, out_rate)
        self.L = out_rate // g          # interpolation factor
        self.M = in_rate // g           # decimation factor

        if self.L > _MAX_PHASES:
            raise ResamplerError(
                f"{in_rate} Hz -> {out_rate} Hz needs {self.L} polyphase branches "
                f"(limit {_MAX_PHASES}); choose rates with a common factor"
            )

        # The filter must suppress both the images from interpolation and the
        # aliases from decimation, so take the tighter of the two cutoffs.
        cutoff = 0.5 / max(self.L, self.M)
        num_taps = self.taps_per_phase * self.L
        h = _design_lowpass(num_taps, cutoff)

        # Polyphase decomposition: phase p holds h[k*L + p] for k in 0..Lp-1.
        # Scaling by L restores the energy that zero-stuffing would have lost,
        # so each branch has unity DC gain.
        self._phases = (h.reshape(self.taps_per_phase, self.L).T * self.L).copy()

        self.reset()

    # -- properties -------------------------------------------------------
    @property
    def ratio(self) -> float:
        return self.out_rate / self.in_rate

    @property
    def delay_ms(self) -> float:
        """Group delay added to the audio path, in milliseconds.

        Counts toward ``bridge.latency_budget_ms``.
        """
        if self.passthrough:
            return 0.0
        num_taps = self.taps_per_phase * self.L
        delay_input_samples = (num_taps - 1) / (2.0 * self.L)
        return 1000.0 * delay_input_samples / self.in_rate

    # -- state ------------------------------------------------------------
    def reset(self) -> None:
        """Clear filter state. Call between distinct streams."""
        history = self.taps_per_phase - 1
        self._hist = np.zeros(history, dtype=np.float64)
        # Position of the next output sample, in 1/L input-sample units,
        # relative to the start of the working buffer. Starts past the
        # zero-padded lead-in so the first real sample is centred correctly.
        self._pos = history * self.L

    # -- conversion -------------------------------------------------------
    def process(self, pcm: np.ndarray) -> np.ndarray:
        """Convert one block, carrying filter state across calls."""
        if self.passthrough:
            return pcm

        if pcm.dtype != np.float64:
            x = pcm.astype(np.float64)
        else:
            x = pcm

        buf = np.concatenate((self._hist, x)) if self._hist.size else x
        L, M, Lp = self.L, self.M, self.taps_per_phase

        # Highest 1/L-unit position whose input index is inside buf.
        max_pos = (len(buf) - 1) * L + (L - 1)
        if self._pos > max_pos:
            self._hist = buf
            return np.zeros(0, dtype=np.int16)

        n_out = (max_pos - self._pos) // M + 1
        positions = self._pos + np.arange(n_out, dtype=np.int64) * M
        idx = positions // L
        phase = positions % L

        # Only outputs with a full history window are computable. The lead-in
        # padding makes this true from the first call onward.
        valid = idx >= (Lp - 1)
        if not valid.all():
            idx, phase = idx[valid], phase[valid]
            n_out = int(valid.sum())

        if n_out == 0:
            self._hist = buf
            return np.zeros(0, dtype=np.int16)

        # Gather x[i], x[i-1], ... x[i-Lp+1] for each output, then apply that
        # output's polyphase branch.
        windows = buf[idx[:, None] - np.arange(Lp)[None, :]]
        out = np.einsum("ij,ij->i", windows, self._phases[phase])

        # Advance state: keep only what the next output still needs.
        consumed = self._pos + (len(positions)) * M
        next_idx = consumed // L
        drop = max(0, int(next_idx) - (Lp - 1))
        drop = min(drop, len(buf))
        self._hist = buf[drop:]
        self._pos = int(consumed) - drop * L

        return np.clip(np.rint(out), _INT16_MIN, _INT16_MAX).astype(np.int16)

    def flush(self) -> np.ndarray:
        """Drain the filter tail at end of stream.

        Feeds in enough zeros to push the remaining history through, so the
        last few milliseconds of a transmission are not truncated.
        """
        if self.passthrough:
            return np.zeros(0, dtype=np.int16)
        tail = np.zeros(self.taps_per_phase, dtype=np.int16)
        return self.process(tail)

    def __repr__(self) -> str:
        if self.passthrough:
            return f"Resampler({self.in_rate} Hz, passthrough)"
        return (
            f"Resampler({self.in_rate} -> {self.out_rate} Hz, "
            f"L={self.L} M={self.M}, {self.delay_ms:.1f} ms)"
        )


def resample_block(pcm: np.ndarray, in_rate: int, out_rate: int) -> np.ndarray:
    """One-shot conversion for tests and diagnostics.

    Not for streaming: it discards filter state, so consecutive calls would
    click at the joins. Use a :class:`Resampler` instance for a live path.
    """
    r = Resampler(in_rate, out_rate)
    return np.concatenate((r.process(pcm), r.flush()))
