"""COS backend abstraction.

Every backend presents the same interface to ``BridgeController``: it is fed
each capture block and returns a transition event or nothing. The controller
never learns which mode is in use.

Level statistics are computed in *all* modes, including the AIOC ones. The
levels do not drive the decision there, but the operator still needs them to
set ``rx_gain_db`` and to see clipping, and ``--cos-monitor`` is the only
practical way to calibrate an interface.
"""

from __future__ import annotations

import abc
import logging
from typing import Any

import numpy as np

from ..audio.levels import BlockStats, CosDetector, CosEvent, LevelMeter, rms_dbfs, peak_dbfs

__all__ = ["CosBackend", "InternalAudioCos", "AiocCos", "DisabledCos", "create_cos_backend"]

log = logging.getLogger(__name__)


class CosBackend(abc.ABC):
    """Detects whether the radio is currently receiving."""

    name = "abstract"

    def open(self) -> None:
        """Acquire resources and arm the detector."""

    def close(self) -> None:
        """Release resources."""

    @abc.abstractmethod
    def feed(self, pcm: np.ndarray, *, clipped: int = 0) -> tuple[BlockStats, CosEvent | None]:
        """Process one capture block; return stats and any transition."""

    @abc.abstractmethod
    def suppress(self, on: bool) -> CosEvent | None:
        """Assert/release loopback suppression while transmitting to RF."""

    @property
    @abc.abstractmethod
    def active(self) -> bool: ...

    @property
    def meter(self) -> LevelMeter | None:
        return None


class DisabledCos(CosBackend):
    """Never reports receive activity. Only legal when rf_to_zello is off."""

    name = "disabled"

    def feed(self, pcm: np.ndarray, *, clipped: int = 0) -> tuple[BlockStats, CosEvent | None]:
        return BlockStats(rms_dbfs(pcm), peak_dbfs(pcm), clipped), None

    def suppress(self, on: bool) -> CosEvent | None:
        return None

    @property
    def active(self) -> bool:
        return False


class InternalAudioCos(CosBackend):
    """Software COS: threshold, attack, and hang on the captured audio level."""

    name = "internal_audio"

    def __init__(self, cfg: Any) -> None:
        self._detector = CosDetector(
            threshold_dbfs=cfg.cos.threshold_dbfs,
            attack_ms=cfg.cos.attack_ms,
            hang_ms=cfg.cos.hang_ms,
            startup_ignore_ms=cfg.cos.startup_ignore_ms,
            min_tx_ms=cfg.cos.min_tx_ms,
        )
        self._meter = LevelMeter()

    def open(self) -> None:
        self._detector.reset()

    def feed(self, pcm: np.ndarray, *, clipped: int = 0) -> tuple[BlockStats, CosEvent | None]:
        stats, event = self._detector.feed(pcm, clipped=clipped)
        self._meter.add(stats)
        return stats, event

    def suppress(self, on: bool) -> CosEvent | None:
        return self._detector.suppress(on)

    @property
    def active(self) -> bool:
        return self._detector.active

    @property
    def meter(self) -> LevelMeter:
        return self._meter

    @property
    def detector(self) -> CosDetector:
        return self._detector


class AiocCos(CosBackend):
    """COS from the AIOC's own indication, read over HID.

    The AIOC's state is authoritative: no software hang is layered on top.
    ``cos.aioc_hang_ms`` programs the device, it does not post-process the
    result. Stacking a second tail here would silently double the hold time
    an operator thinks they configured.
    """

    name = "aioc"

    def __init__(self, cfg: Any, *, hid: Any = None) -> None:
        self._cfg = cfg
        self._hid = hid
        self._meter = LevelMeter()
        self._active = False
        self._suppressed = False
        self._faults = 0

    def open(self) -> None:
        if self._hid is None:
            from .aioc_hid import HidCos

            self._hid = HidCos(
                self._cfg.cos.hid_device, button=self._cfg.cos.hid_button
            )
        self._hid.open()

        if self._cfg.cos.configure_aioc_on_start:
            self._configure()

        self._active = False

    def _configure(self) -> None:
        """Program threshold/tail into the AIOC.

        Firmware 1.3.0 added an HID configuration interface, but the register
        map is not published in the README. Rather than write speculative
        registers at a radio interface, this fails clearly -- section 10.2
        requires failing startup over silently falling back.
        """
        raise NotImplementedError(
            "cos.configure_aioc_on_start is not implemented: the AIOC HID "
            "configuration register map is not published, and writing "
            "speculative registers to a radio interface is unsafe. Program "
            "threshold and tail with the vendor tool, then set "
            "configure_aioc_on_start: false."
        )

    def feed(self, pcm: np.ndarray, *, clipped: int = 0) -> tuple[BlockStats, CosEvent | None]:
        stats = BlockStats(rms_dbfs(pcm), peak_dbfs(pcm), clipped)
        self._meter.add(stats)

        if self._suppressed or self._hid is None:
            return stats, None

        state = self._hid.poll()

        if state and not self._active:
            self._active = True
            return stats, CosEvent.ACTIVE
        if not state and self._active:
            self._active = False
            return stats, CosEvent.INACTIVE
        return stats, None

    def suppress(self, on: bool) -> CosEvent | None:
        if on == self._suppressed:
            return None
        self._suppressed = on
        if on and self._active:
            self._active = False
            return CosEvent.INACTIVE
        return None

    @property
    def active(self) -> bool:
        return self._active

    @property
    def meter(self) -> LevelMeter:
        return self._meter

    def close(self) -> None:
        if self._hid is not None:
            self._hid.close()
        self._active = False


def create_cos_backend(cfg: Any) -> CosBackend:
    mode = cfg.cos.mode
    if mode == "internal_audio":
        return InternalAudioCos(cfg)
    if mode == "aioc_hardware":
        return AiocCos(cfg)
    if mode == "disabled":
        return DisabledCos()
    raise ValueError(f"unknown cos.mode {mode!r}")
