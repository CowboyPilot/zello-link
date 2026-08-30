"""Operator-facing diagnostics: config validation and the level monitor."""

from __future__ import annotations

import shutil
import sys
import time
from typing import Any

from ..audio.levels import DBFS_FLOOR

__all__ = [
    "render_validation", "cos_monitor", "render_bar", "render_meter",
    "RollingPeak", "TARGET_PEAK_LOW", "TARGET_PEAK_HIGH", "PEAK_HOLD_S",
]


def render_validation(cfg: Any, *, device_report: list[str] | None = None) -> str:
    """Human-readable --validate output. Never prints a secret."""
    lines: list[str] = []
    add = lines.append

    add(f"instance          {cfg.instance.name}")
    add(f"log level         {cfg.instance.log_level}")
    add("")
    add(f"zello server      {cfg.zello.server}")
    add(f"zello channel     {cfg.zello.channel}")
    add(f"zello username    {cfg.zello.username}")
    add(f"credentials       {_describe_credentials(cfg)}")
    add(f"refresh token     {cfg.zello.refresh_token_file or '(not persisted)'}")
    add("")
    add(f"opus              {cfg.opus.sample_rate} Hz, {cfg.opus.frame_ms} ms/frame, "
        f"{cfg.opus.bitrate} bps, complexity {cfg.opus.complexity}")
    add(f"audio in          {cfg.sound.input_device!r}")
    add(f"audio out         {cfg.sound.output_device!r}")
    add(f"audio format      {cfg.sound.sample_rate} Hz, {cfg.sound.channels} ch, "
        f"{cfg.sound.block_ms} ms blocks")
    add(f"rx gain           {cfg.sound.rx_gain_db:+.1f} dB   (radio receive -> Zello)")
    add(f"tx gain           {cfg.sound.tx_gain_db:+.1f} dB   (Zello -> radio transmit)")
    add(f"jitter buffer     {cfg.sound.jitter_ms} ms target, {cfg.sound.jitter_max_ms} ms max")
    add("")
    add(f"ptt mode          {cfg.ptt.mode}")
    if cfg.ptt.tty_device:
        add(f"ptt tty           {cfg.ptt.tty_device}")
    if cfg.ptt.hid_device:
        add(f"ptt hid           {cfg.ptt.hid_device}")
    add(f"ptt timing        pre-key {cfg.ptt.pre_key_ms} ms, "
        f"post-audio {cfg.ptt.post_audio_ms} ms, max TX {cfg.ptt.max_tx_s:.0f} s")
    add("")
    add(f"cos mode          {cfg.cos.mode}")
    if cfg.cos.mode == "internal_audio":
        add(f"cos threshold     {cfg.cos.threshold_dbfs:.1f} dBFS "
            f"(detector floor {DBFS_FLOOR:.1f})")
        add(f"cos timing        attack {cfg.cos.attack_ms} ms, hang {cfg.cos.hang_ms} ms, "
            f"startup ignore {cfg.cos.startup_ignore_ms} ms")
        if cfg.cos.min_tx_ms:
            add(f"cos min TX        {cfg.cos.min_tx_ms} ms floor "
                f"(held for max({cfg.cos.min_tx_ms}, speech + {cfg.cos.hang_ms}))")
    if cfg.cos.mode in ("aioc_virtual", "aioc_hardware"):
        add(f"cos hid           {cfg.cos.hid_device}")
    add("")
    add(f"directions        rf->zello {_yn(cfg.bridge.rf_to_zello)}, "
        f"zello->rf {_yn(cfg.bridge.zello_to_rf)}")
    add(f"arbitration       {cfg.bridge.arbitration}")
    add(f"guards            rx {cfg.bridge.rx_guard_ms} ms, tx {cfg.bridge.tx_guard_ms} ms")

    resample_ms = cfg.resampler_delay_ms()
    if cfg.sound.sample_rate != cfg.opus.sample_rate:
        add(f"resampler         {cfg.sound.sample_rate} -> {cfg.opus.sample_rate} Hz "
            f"on RF->Zello (+{resample_ms:.1f} ms)")
    else:
        add("resampler         none on RF->Zello (device and Opus rates match)")
    add("                  inbound Zello streams are converted per stream, "
        "using the peer's declared rate")

    budget = cfg.ptt.pre_key_ms + cfg.sound.jitter_ms + cfg.sound.block_ms + resample_ms
    add(f"latency estimate  {budget:.0f} ms of {cfg.bridge.latency_budget_ms} ms budget")

    if device_report:
        add("")
        add("devices")
        lines.extend(f"  {line}" for line in device_report)

    warnings = cfg.warnings()
    if warnings:
        add("")
        add(f"{len(warnings)} warning(s):")
        for w in warnings:
            add(f"  ! {w}")
    else:
        add("")
        add("no warnings")

    return "\n".join(lines)


def _describe_credentials(cfg: Any) -> str:
    parts = []
    if cfg.zello.auth_token is not None:
        parts.append("auth_token set")
    if cfg.zello.password is not None:
        parts.append("password set")
    return ", ".join(parts) if parts else "none"


def _yn(v: bool) -> str:
    return "yes" if v else "no"


#: Meter scale. -60 dBFS is below any usable interface noise floor and 0 is
#: full scale, so the whole useful adjustment range is on screen.
METER_LO = -60.0
METER_HI = 0.0
METER_WIDTH = 56

#: Target window for the PEAK level when setting a radio's volume knob.
#: Below this and COS margin suffers; above it there is no headroom and a
#: strong signal clips.
TARGET_PEAK_LOW = -12.0
TARGET_PEAK_HIGH = -6.0

#: How long the peak marker holds, in seconds.
PEAK_HOLD_S = 5.0

_RED = "\033[31m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_DIM = "\033[2m"
_RESET = "\033[0m"


def _pos(dbfs: float, width: int = METER_WIDTH) -> int:
    """Map a dBFS value to a column, clamped to the meter."""
    frac = (dbfs - METER_LO) / (METER_HI - METER_LO)
    return max(0, min(width - 1, int(frac * (width - 1))))


def render_bar(dbfs: float, *, threshold: float | None = None, width: int = 40) -> str:
    """Single-line meter. Retained for simple/non-interactive output."""
    frac = max(0.0, min(1.0, (dbfs - METER_LO) / (METER_HI - METER_LO)))
    filled = int(frac * width)
    bar = ["=" if i < filled else "-" for i in range(width)]

    if threshold is not None:
        t = _pos(threshold, width)
        if 0 <= t < width:
            bar[t] = "|"

    return "".join(bar)


def render_meter(
    level_dbfs: float,
    peak_dbfs: float,
    *,
    threshold: float | None = None,
    width: int = METER_WIDTH,
    color: bool = True,
) -> tuple[str, str]:
    """Render the level meter and its bounding-caret line.

    Returns ``(meter, carets)``. Both strings are ``width`` visible columns
    wide, so the caller can print them one above the other and have the
    markers line up.

    The meter reads left (quiet) to right (full scale):
      ``=``  filled to the current level
      ``-``  unfilled
      ``|``  peak hold (see PEAK_HOLD_S)
      red ``|``  the COS threshold

    The caret line marks the target window for the peak, so the operator
    turns the knob until the peak marker sits between the carets.
    """
    cells = ["=" if i <= _pos(level_dbfs, width) else "-" for i in range(width)]

    # Threshold first, so a coincident peak marker wins the cell -- the peak
    # is what the operator is actively adjusting.
    thresh_pos = _pos(threshold, width) if threshold is not None else None
    if thresh_pos is not None:
        cells[thresh_pos] = "|"

    peak_pos = _pos(peak_dbfs, width)
    cells[peak_pos] = "|"

    if color:
        if thresh_pos is not None and thresh_pos != peak_pos:
            cells[thresh_pos] = f"{_RED}|{_RESET}"
        in_window = TARGET_PEAK_LOW <= peak_dbfs <= TARGET_PEAK_HIGH
        too_hot = peak_dbfs > TARGET_PEAK_HIGH
        peak_colour = _GREEN if in_window else (_RED if too_hot else _YELLOW)
        cells[peak_pos] = f"{peak_colour}|{_RESET}"

    carets = [" "] * width
    carets[_pos(TARGET_PEAK_LOW, width)] = "^"
    carets[_pos(TARGET_PEAK_HIGH, width)] = "^"

    return "".join(cells), "".join(carets)


#: Narrowest usable bar. Below this the meter conveys nothing.
_MIN_BAR = 12
#: Widest bar, regardless of how roomy the terminal is.
_MAX_BAR = 44


def render_status_line(
    rms_dbfs: float,
    peak_dbfs: float,
    *,
    threshold: float | None = None,
    clipped: int = 0,
    cos_active: bool = False,
    keyed: bool = False,
    total_width: int | None = None,
    color: bool = True,
) -> str:
    """One-line live level readout, for --showmonitor.

    Sized to fit the terminal. This is not cosmetic: the line is redrawn in
    place with a bare carriage return, and CR only returns to the start of
    the last *physical* line. If the text is wider than the terminal it wraps,
    every redraw leaves the wrapped remnant behind, and the meter appears to
    scroll instead of holding still.

      ``=``      filled to the current RMS level
      ``-``      unfilled
      ``[`` ``]``  bracket the target window for the peak
      ``|``      5-second peak hold (green inside the window)
      red ``|``  the COS threshold

    Turn the radio's volume until the peak marker sits inside the brackets.
    """
    if total_width is None:
        total_width = shutil.get_terminal_size((100, 24)).columns

    state_txt = _state_text(cos_active, keyed, clipped)
    verdict_txt = _verdict_text(peak_dbfs, clipped)

    # Leave one spare column: writing into the last cell makes some terminals
    # wrap immediately, which is the very thing this sizing prevents.
    budget = max(40, total_width - 1)
    fixed = len(state_txt) + len(" rms ") + 6 + len("  pk ") + 6 + 2 + 2

    width = budget - fixed - len(verdict_txt)
    if width < _MIN_BAR:
        verdict_txt = _verdict_text(peak_dbfs, clipped, short=True)
        width = budget - fixed - len(verdict_txt)
    if width < _MIN_BAR:
        verdict_txt = ""
        width = budget - fixed
    width = max(_MIN_BAR, min(_MAX_BAR, width))

    cells = ["=" if i <= _pos(rms_dbfs, width) else "-" for i in range(width)]

    if threshold is not None:
        t = _pos(threshold, width)
        cells[t] = f"{_RED}|{_RESET}" if color else "|"

    lo, hi = _pos(TARGET_PEAK_LOW, width), _pos(TARGET_PEAK_HIGH, width)
    cells[lo], cells[hi] = "[", "]"

    # Peak last: it is the marker the operator is actively steering.
    peak_pos = _pos(peak_dbfs, width)
    if color:
        in_window = TARGET_PEAK_LOW <= peak_dbfs <= TARGET_PEAK_HIGH
        hot = peak_dbfs > TARGET_PEAK_HIGH or clipped
        cells[peak_pos] = f"{_GREEN if in_window else (_RED if hot else _YELLOW)}|{_RESET}"
    else:
        cells[peak_pos] = "|"

    state = _colour_state(state_txt, cos_active, keyed, clipped) if color else state_txt
    verdict = _colour_verdict(verdict_txt, peak_dbfs, clipped) if color else verdict_txt

    tail = f"  {verdict}" if verdict else ""
    return (
        f"{state} rms {rms_dbfs:6.1f}  pk {peak_dbfs:6.1f}  {''.join(cells)}{tail}"
    )


def _state_text(cos_active: bool, keyed: bool, clipped: int) -> str:
    if clipped:
        return "CLIP"
    if keyed:
        return "TX  "
    if cos_active:
        return "RX  "
    return "    "


def _colour_state(text: str, cos_active: bool, keyed: bool, clipped: int) -> str:
    if clipped or keyed:
        return f"{_RED}{text}{_RESET}"
    if cos_active:
        return f"{_GREEN}{text}{_RESET}"
    return text


def _verdict_text(peak_dbfs: float, clipped: int, *, short: bool = False) -> str:
    """Plain-text guidance for which way to turn the knob."""
    if clipped:
        return "CLIPPING" if short else f"CLIPPING ({clipped}) - turn DOWN"
    if peak_dbfs > TARGET_PEAK_HIGH:
        return "DOWN" if short else "too hot - turn DOWN"
    if peak_dbfs < TARGET_PEAK_LOW:
        return "UP" if short else "too quiet - turn UP"
    return "OK" if short else "level OK"


def _colour_verdict(text: str, peak_dbfs: float, clipped: int) -> str:
    if clipped or peak_dbfs > TARGET_PEAK_HIGH:
        return f"{_RED}{text}{_RESET}"
    if peak_dbfs < TARGET_PEAK_LOW:
        return f"{_YELLOW}{text}{_RESET}"
    return f"{_GREEN}{text}{_RESET}"


def _verdict(peak_dbfs: float, clipped: int, color: bool) -> str:
    """One-word guidance for which way to turn the knob."""
    text = _verdict_text(peak_dbfs, clipped)
    return _colour_verdict(text, peak_dbfs, clipped) if color else text


class RollingPeak:
    """Peak-hold over a sliding time window."""

    def __init__(self, window_s: float = PEAK_HOLD_S) -> None:
        self.window_s = window_s
        self._samples: list[tuple[float, float]] = []

    def add(self, dbfs: float, now: float) -> float:
        self._samples.append((now, dbfs))
        cutoff = now - self.window_s
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.pop(0)
        return self.value

    @property
    def value(self) -> float:
        return max((v for _, v in self._samples), default=METER_LO)


async def cos_monitor(cfg: Any, *, engine: Any, cos: Any, duration_s: float | None = None) -> int:
    """Print rolling levels without transmitting anything.

    This is how an operator calibrates ``cos.threshold_dbfs`` and the gain
    settings against their actual radio. It never keys PTT and never opens a
    Zello stream -- section 5 requires the diagnostic paths to be inert.
    """
    threshold = cfg.cos.threshold_dbfs if cfg.cos.mode == "internal_audio" else None
    live = sys.stderr.isatty()

    print(f"COS monitor: {cfg.sound.input_device!r} @ {cfg.sound.sample_rate} Hz",
          file=sys.stderr)
    print(f"rx_gain_db={cfg.sound.rx_gain_db:+.1f}  threshold={threshold} dBFS  "
          f"(PTT is NOT armed)", file=sys.stderr)
    print(f"Turn the radio's volume until the peak marker sits between the carets "
          f"({TARGET_PEAK_LOW:.0f} to {TARGET_PEAK_HIGH:.0f} dBFS).", file=sys.stderr)
    if threshold is not None:
        red = f"{_RED}|{_RESET}" if live else "|"
        print(f"  {red} = COS threshold      | = {PEAK_HOLD_S:.0f}s peak      "
              f"^ = target window", file=sys.stderr)
    print(f"  {METER_LO:.0f}{' ' * (METER_WIDTH - 8)}{METER_HI:.0f} dBFS",
          file=sys.stderr)

    from ..audio.levels import apply_gain_db

    started = time.monotonic()
    total_clipped = 0
    blocks = 0
    peak_hold = RollingPeak(PEAK_HOLD_S)
    drawn = False

    async for pcm in engine.capture_blocks():
        gained, clipped = apply_gain_db(pcm, cfg.sound.rx_gain_db)
        total_clipped += clipped
        stats, _ = cos.feed(gained, clipped=clipped)
        blocks += 1

        now = time.monotonic()
        peak = peak_hold.add(stats.peak_dbfs, now)

        # ~10 redraws a second is readable; every block is not.
        if blocks % max(1, int(100 / cfg.sound.block_ms)) == 0:
            meter, carets = render_meter(
                stats.rms_dbfs, peak, threshold=threshold, color=live
            )
            verdict = _verdict(peak, total_clipped, live)
            flag = "RX" if cos.active else "  "

            line1 = f" [{meter}] {flag}"
            line2 = f"  {carets}  rms {stats.rms_dbfs:6.1f}  pk {peak:6.1f}  {verdict}"

            if live:
                if drawn:
                    sys.stderr.write("\033[2A")     # redraw in place
                sys.stderr.write(f"\033[2K{line1}\n\033[2K{line2}\n")
                sys.stderr.flush()
                drawn = True
            else:
                print(f"{line1}{line2}", file=sys.stderr)

        if duration_s is not None and now - started >= duration_s:
            break

    print(f"\n{blocks} blocks, {total_clipped} clipped samples", file=sys.stderr)
    if total_clipped:
        print("clipping detected: reduce rx_gain_db or the radio's output level", file=sys.stderr)
    return 0
