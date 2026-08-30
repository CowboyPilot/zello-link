"""Sound device enumeration and selection.

Selection rules, from the spec's multi-instance requirements:
  * A device may be given as a numeric index or as a name substring.
  * An ambiguous partial name match is an error, never a guess. Two bridge
    processes must not silently land on the same card.
  * There is no default. If a device is not configured for a direction the
    bridge does not open one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

__all__ = ["AudioDevice", "DeviceError", "list_devices", "resolve_device", "format_device_table"]

log = logging.getLogger(__name__)


class DeviceError(Exception):
    """No such device, an ambiguous match, or an unusable configuration."""


@dataclass(frozen=True)
class AudioDevice:
    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float
    hostapi_name: str = ""

    def supports(self, kind: Literal["input", "output"]) -> bool:
        return (
            self.max_input_channels > 0 if kind == "input" else self.max_output_channels > 0
        )

    def __str__(self) -> str:
        io = f"in:{self.max_input_channels} out:{self.max_output_channels}"
        api = f" [{self.hostapi_name}]" if self.hostapi_name else ""
        return f"{self.index:>3}  {self.name}{api}  ({io}, {self.default_samplerate:.0f} Hz)"


def _import_sounddevice() -> Any:
    try:
        import sounddevice
    except (ImportError, OSError) as e:
        raise DeviceError(
            "sounddevice/PortAudio is required for audio I/O. Install with:\n"
            "  pip install 'zello-dmr-bridge[audio]'\n"
            "On Debian/Raspberry Pi OS also: sudo apt install libportaudio2"
        ) from e
    return sounddevice


def list_devices() -> list[AudioDevice]:
    sd = _import_sounddevice()
    try:
        raw = sd.query_devices()
        hostapis = sd.query_hostapis()
    except Exception as e:
        raise DeviceError(f"cannot enumerate audio devices: {e}") from e

    out: list[AudioDevice] = []
    for i, d in enumerate(raw):
        api = ""
        try:
            api = hostapis[d["hostapi"]]["name"]
        except (IndexError, KeyError, TypeError):
            pass
        out.append(
            AudioDevice(
                index=i,
                name=d["name"],
                max_input_channels=d["max_input_channels"],
                max_output_channels=d["max_output_channels"],
                default_samplerate=d["default_samplerate"],
                hostapi_name=api,
            )
        )
    return out


def resolve_device(
    selector: str | int | None,
    kind: Literal["input", "output"],
    *,
    devices: list[AudioDevice] | None = None,
) -> AudioDevice:
    """Resolve a config selector to exactly one device.

    Raises rather than guessing when the selector is ambiguous -- on a host
    with several AIOCs, guessing would silently cross-wire two bridges.
    """
    if selector is None:
        raise DeviceError(f"no {kind} device configured")

    if devices is None:
        devices = list_devices()

    candidates = [d for d in devices if d.supports(kind)]
    if not candidates:
        raise DeviceError(f"no audio devices with {kind} channels are present")

    # Numeric index: exact, but still checked for direction support.
    if isinstance(selector, int) or (isinstance(selector, str) and selector.strip().isdigit()):
        idx = int(selector)
        for d in devices:
            if d.index == idx:
                if not d.supports(kind):
                    raise DeviceError(
                        f"device {idx} ({d.name!r}) has no {kind} channels"
                    )
                return d
        raise DeviceError(f"no audio device with index {idx}")

    needle = str(selector).strip()
    if not needle:
        raise DeviceError(f"empty {kind} device selector")

    exact = [d for d in candidates if d.name == needle]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise DeviceError(
            f"{len(exact)} {kind} devices are named {needle!r}; select by index instead:\n"
            + "\n".join(f"  {d}" for d in exact)
        )

    partial = [d for d in candidates if needle.lower() in d.name.lower()]
    if len(partial) == 1:
        return partial[0]
    if len(partial) > 1:
        raise DeviceError(
            f"{kind} device selector {needle!r} is ambiguous ({len(partial)} matches); "
            "use a longer name or a numeric index:\n"
            + "\n".join(f"  {d}" for d in partial)
        )

    raise DeviceError(
        f"no {kind} device matching {needle!r}. Available:\n"
        + "\n".join(f"  {d}" for d in candidates)
    )


def check_device_settings(
    device: AudioDevice,
    kind: Literal["input", "output"],
    *,
    sample_rate: int,
    channels: int,
) -> None:
    """Verify the device will actually accept the configured format."""
    sd = _import_sounddevice()
    try:
        sd.check_input_settings(
            device=device.index, channels=channels, samplerate=sample_rate, dtype="int16"
        ) if kind == "input" else sd.check_output_settings(
            device=device.index, channels=channels, samplerate=sample_rate, dtype="int16"
        )
    except Exception as e:
        raise DeviceError(
            f"{kind} device {device.name!r} rejects {sample_rate} Hz / {channels} ch int16: {e}"
        ) from e


def format_device_table(devices: list[AudioDevice] | None = None) -> str:
    """Render --list-audio-devices output."""
    if devices is None:
        devices = list_devices()
    if not devices:
        return "no audio devices found"
    lines = ["idx  name  (channels, default rate)", "-" * 60]
    lines.extend(str(d) for d in devices)
    return "\n".join(lines)
