"""Typed configuration schema, environment expansion, and validation.

Every operational variable lives here. Nothing outside this module may
hard-code a device path, threshold, timing constant, channel name, or
credential -- defaults belong in the models below and nowhere else.

Secrets are held in ``pydantic.SecretStr`` so that a stray ``repr()`` in a log
line or an uncaught traceback cannot leak them. See ``logging_setup.py`` for
the complementary log-record scrubber.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)

__all__ = [
    "BridgeConfig",
    "ConfigError",
    "CONFIG_VERSION",
    "DBFS_FLOOR",
    "load_config",
    "expand_env",
]

# The schema version this build understands. Bumped on any breaking config
# change so an old file fails loudly instead of being silently mis-parsed.
CONFIG_VERSION = 1

# The internal COS detector floors RMS at 1 LSB before the log, so the lowest
# level it can ever report is 20*log10(1/32768). A threshold below this can
# never be crossed -- validation warns rather than accepting a dead config.
DBFS_FLOOR = -90.3

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class ConfigError(Exception):
    """Raised for unreadable, unparseable, or semantically invalid config."""


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys.

    PyYAML's default is last-one-wins, silently. In a hand-edited config a
    second ``sound:`` block would discard every setting from the first one
    with no diagnostic -- including a device selection or a gain. Refusing to
    load is far better than starting with settings the operator believes are
    applied but are not.
    """


def _no_duplicate_keys(loader: yaml.Loader, node: yaml.MappingNode, deep: bool = False) -> dict:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ConfigError(
                f"duplicate key {key!r} at line {key_node.start_mark.line + 1}; "
                "a repeated section silently overrides the earlier one"
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicate_keys
)


def expand_env(value: Any, *, path: str = "") -> Any:
    """Recursively expand ``${VAR}`` references against the environment.

    A reference to an unset variable is an error rather than an empty string:
    silently authenticating with an empty password is worse than refusing to
    start.
    """
    if isinstance(value, str):

        def _sub(match: re.Match[str]) -> str:
            name = match.group(1)
            try:
                return os.environ[name]
            except KeyError:
                raise ConfigError(
                    f"{path or 'config'}: environment variable ${{{name}}} is not set"
                ) from None

        return _ENV_PATTERN.sub(_sub, value)
    if isinstance(value, dict):
        return {k: expand_env(v, path=f"{path}.{k}" if path else k) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v, path=f"{path}[{i}]") for i, v in enumerate(value)]
    return value


class _Model(BaseModel):
    """Base: reject unknown keys so typos fail at startup, not at 3am."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class InstanceConfig(_Model):
    name: str = Field(min_length=1, description="Identifier tagged onto every log record.")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"


class ZelloConfig(_Model):
    server: str = "wss://zello.io/ws"
    channel: str = Field(min_length=1)
    username: str = Field(min_length=1)
    password: SecretStr | None = None
    auth_token: SecretStr | None = None
    refresh_token_file: Path | None = None
    platform_name: str = "zello-dmr-bridge"

    reconnect_initial_s: float = Field(default=1.0, gt=0, le=60)
    reconnect_max_s: float = Field(default=60.0, gt=0, le=3600)

    # Deadline for a WebSocket pong before the peer is declared dead. This is
    # the half-open detection: a TCP connection can die without ever
    # delivering a FIN, and ping/pong is what notices.
    #
    # It is deliberately a *pong* deadline, not an idle-data timeout. An idle
    # channel sends no application frames at all, so timing out on those
    # disconnects a perfectly healthy bridge every time nobody is talking.
    keepalive_timeout_s: float = Field(default=45.0, ge=10, le=300)

    @model_validator(mode="after")
    def _need_some_credential(self) -> "ZelloConfig":
        if self.auth_token is None and self.password is None:
            raise ValueError("zello: at least one of auth_token or password must be set")

        # Verified against the live server 2026-08-30: on the consumer network
        # a logon carrying only username+password is rejected with
        # "not enough params". The auth_token authenticates the *application*
        # and is separate from the user credential, so catch it here rather
        # than letting the bridge fail on its first connect.
        # Zello Work (wss://<network>.zellowork.com/ws) authenticates
        # differently, so this applies only to the consumer host.
        if self.auth_token is None and "zello.io" in self.server:
            raise ValueError(
                "zello.auth_token is required on the consumer Zello network "
                "(zello.io): username+password alone is rejected with 'not enough "
                "params'. Obtain a development token from https://developers.zello.com/ "
                "(Keys -> Add Key). Development tokens expire after 30 days."
            )

        if self.reconnect_max_s < self.reconnect_initial_s:
            raise ValueError("zello: reconnect_max_s must be >= reconnect_initial_s")
        return self

    @field_validator("server")
    @classmethod
    def _must_be_tls(cls, v: str) -> str:
        if not v.startswith("wss://"):
            raise ValueError("zello.server must use wss:// (TLS); plaintext ws:// is not permitted")
        return v


class OpusConfig(_Model):
    sample_rate: int = 16000
    frame_ms: Literal[5, 10, 20, 40, 60] = 20
    # API.md documents frames_per_packet as 1 or 2.
    frames_per_packet: Literal[1, 2] = 1
    bitrate: int = Field(default=16000, ge=6000, le=510000)
    complexity: int = Field(default=5, ge=0, le=10)
    application: Literal["voip", "audio", "lowdelay"] = "voip"

    @field_validator("sample_rate")
    @classmethod
    def _opus_rate(cls, v: int) -> int:
        if v not in (8000, 12000, 16000, 24000, 48000):
            raise ValueError(f"opus.sample_rate must be one of 8000/12000/16000/24000/48000, got {v}")
        return v

    @property
    def samples_per_frame(self) -> int:
        return int(self.sample_rate * self.frame_ms / 1000)


class SoundConfig(_Model):
    """Audio device selection and levels.

    Gain names are radio-centric, not sound-card-centric. ``rx_gain_db``
    scales audio arriving FROM the radio's receive path (radio speaker ->
    bridge -> Zello). ``tx_gain_db`` scales audio the bridge sends TO the
    radio's microphone input, i.e. what gets transmitted over RF. An operator
    adjusting "TX level" wants ``tx_gain_db``.
    """

    backend: Literal["sounddevice"] = "sounddevice"

    # Device selectors: numeric index or a substring of the device name.
    # Ambiguous partial name matches are a startup error, never a guess.
    input_device: str | int | None = None
    output_device: str | int | None = None

    sample_rate: int = 16000
    channels: Literal[1] = 1
    block_ms: Literal[5, 10, 20, 40, 60] = 20

    @field_validator("sample_rate")
    @classmethod
    def _device_rate(cls, v: int) -> int:
        # Catches typos like 1600 or 480000, which would otherwise open a
        # stream the card silently reinterprets.
        from .audio.resample import SUPPORTED_DEVICE_RATES

        if v not in SUPPORTED_DEVICE_RATES:
            allowed = "/".join(str(r) for r in SUPPORTED_DEVICE_RATES)
            raise ValueError(f"sound.sample_rate must be one of {allowed}, got {v}")
        return v

    rx_gain_db: float = Field(default=0.0, ge=-40.0, le=40.0)
    tx_gain_db: float = Field(default=0.0, ge=-40.0, le=40.0)

    # Playback jitter buffer for the Zello -> RF path. Target depth is the
    # steady-state latency cost; max is the hard bound above which we drop.
    jitter_ms: int = Field(default=120, ge=0, le=2000)
    jitter_max_ms: int = Field(default=400, ge=20, le=4000)

    # Capture-side bound. Overflow increments a counter and drops per policy.
    capture_queue_ms: int = Field(default=400, ge=40, le=4000)
    overflow_policy: Literal["drop_oldest", "drop_newest"] = "drop_oldest"

    @model_validator(mode="after")
    def _check(self) -> "SoundConfig":
        if self.jitter_max_ms < self.jitter_ms:
            raise ValueError("sound.jitter_max_ms must be >= sound.jitter_ms")
        if self.input_device is None and self.output_device is None:
            raise ValueError(
                "sound: at least one of input_device/output_device must be set; "
                "the bridge never guesses a default sound card"
            )
        return self

    @property
    def samples_per_block(self) -> int:
        return int(self.sample_rate * self.block_ms / 1000)


class PttConfig(_Model):
    mode: Literal["serial", "cm108_hid", "none"] = "serial"
    tty_device: str | None = None
    hid_device: str | None = None

    pre_key_ms: int = Field(default=150, ge=0, le=2000)
    post_audio_ms: int = Field(default=120, ge=0, le=2000)

    # Hard RF transmit ceiling, enforced by a watchdog independent of any
    # application state or Zello control message.
    max_tx_s: float = Field(default=180.0, gt=0, le=3600)

    @model_validator(mode="after")
    def _device_present(self) -> "PttConfig":
        if self.mode == "serial" and not self.tty_device:
            raise ValueError("ptt.tty_device is required when ptt.mode='serial'")
        if self.mode == "cm108_hid" and not self.hid_device:
            raise ValueError("ptt.hid_device is required when ptt.mode='cm108_hid'")
        return self


class CosConfig(_Model):
    mode: Literal["internal_audio", "aioc_virtual", "aioc_hardware", "disabled"] = "internal_audio"
    hid_device: str | None = None

    # internal_audio only.
    threshold_dbfs: float = Field(default=-38.0, ge=-96.0, le=0.0)
    attack_ms: int = Field(default=60, ge=0, le=2000)
    hang_ms: int = Field(default=450, ge=0, le=5000)
    startup_ignore_ms: int = Field(default=500, ge=0, le=10000)

    # Floor on how long COS stays open once it triggers. A transmission is
    # held for max(min_tx_ms, speech + hang_ms).
    #
    # This is for radios that emit a click as the squelch opens, then a gap,
    # then speech: the click trips COS, the gap outlasts hang_ms, COS drops,
    # and the front of the speech is lost. Set min_tx_ms longer than that
    # click-to-speech gap. Unlike raising hang_ms it costs nothing on normal
    # traffic, because a transmission longer than the floor is unaffected.
    min_tx_ms: int = Field(default=0, ge=0, le=10000)

    # AIOC-programmed values. Used only when configuring the device; the
    # AIOC's own COS indication stays authoritative at runtime, so no second
    # software hang is applied on top of these.
    configure_aioc_on_start: bool = False
    aioc_threshold: int | None = None
    aioc_hang_ms: int = Field(default=450, ge=0, le=5000)

    @model_validator(mode="after")
    def _check(self) -> "CosConfig":
        if self.mode in ("aioc_virtual", "aioc_hardware") and not self.hid_device:
            raise ValueError(f"cos.hid_device is required when cos.mode='{self.mode}'")
        if self.configure_aioc_on_start and self.mode not in ("aioc_virtual", "aioc_hardware"):
            raise ValueError("cos.configure_aioc_on_start requires an AIOC COS mode")
        return self


class BridgeConfigSection(_Model):
    rf_to_zello: bool = True
    zello_to_rf: bool = True

    # Only one arbitration policy exists in v0.1. The field is an enum from
    # the start so adding e.g. "zello_priority" later is not a breaking change.
    arbitration: Literal["first_talker_wins"] = "first_talker_wins"
    collision_log: bool = True

    rx_guard_ms: int = Field(default=150, ge=0, le=5000)
    tx_guard_ms: int = Field(default=150, ge=0, le=5000)

    # v0.1 discards a competing Zello stream rather than replaying it late.
    # Stale traffic on a security net is worse than a logged collision.
    queue_incoming_zello: bool = False

    # Advisory one-way mouth-to-ear target. Validation warns when the
    # configured pre-key + jitter + block budget exceeds it.
    latency_budget_ms: int = Field(default=300, ge=50, le=5000)

    @model_validator(mode="after")
    def _at_least_one_direction(self) -> "BridgeConfigSection":
        if not self.rf_to_zello and not self.zello_to_rf:
            raise ValueError("bridge: at least one of rf_to_zello/zello_to_rf must be true")
        return self


class HardwareConfig(_Model):
    """Recovery policy for losing the USB interface mid-run.

    A USB device can vanish outright -- observed live as
    ``OSError: [Errno 6] Device not configured`` when the AIOC dropped off the
    bus. The bridge must fail safe *and* come back when the device returns,
    rather than exiting and leaning on a service supervisor to restart it.
    """

    retry_initial_s: float = Field(default=1.0, gt=0, le=60)
    retry_max_s: float = Field(default=30.0, gt=0, le=600)

    #: 0 means retry forever. A bridge appliance should keep trying; a bench
    #: run may prefer to give up and surface the fault.
    max_attempts: int = Field(default=0, ge=0, le=1000)

    @model_validator(mode="after")
    def _check(self) -> "HardwareConfig":
        if self.retry_max_s < self.retry_initial_s:
            raise ValueError("hardware.retry_max_s must be >= hardware.retry_initial_s")
        return self


class LoggingConfig(_Model):
    console: bool = True
    file: Path | None = None
    max_bytes: int = Field(default=10 * 1024 * 1024, ge=0)
    backup_count: int = Field(default=5, ge=0, le=100)
    include_audio_levels: bool = False


class MetricsConfig(_Model):
    """v0.1 metrics surface: a periodic structured log line.

    Deliberately not an HTTP endpoint -- it keeps the bridge free of a web
    server, and gives the 24-hour soak test something concrete to assert on.
    """

    enabled: bool = True
    interval_s: float = Field(default=60.0, ge=5, le=3600)


class BridgeConfig(_Model):
    config_version: int = CONFIG_VERSION
    instance: InstanceConfig
    zello: ZelloConfig
    opus: OpusConfig = Field(default_factory=OpusConfig)
    sound: SoundConfig
    ptt: PttConfig = Field(default_factory=PttConfig)
    cos: CosConfig = Field(default_factory=CosConfig)
    bridge: BridgeConfigSection = Field(default_factory=BridgeConfigSection)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)

    @field_validator("config_version")
    @classmethod
    def _version(cls, v: int) -> int:
        if v != CONFIG_VERSION:
            raise ValueError(
                f"config_version {v} is not supported by this build "
                f"(expected {CONFIG_VERSION}); the schema changed incompatibly"
            )
        return v

    @model_validator(mode="after")
    def _cross_section(self) -> "BridgeConfig":
        if self.sound.sample_rate != self.opus.sample_rate:
            # Legal, but it forces a resampler into the hot path. Surfaced as
            # a warning by warnings() rather than a hard failure.
            pass
        if self.sound.block_ms != self.opus.frame_ms:
            raise ValueError(
                f"sound.block_ms ({self.sound.block_ms}) must equal opus.frame_ms "
                f"({self.opus.frame_ms}); a capture block maps 1:1 to an Opus frame in v0.1"
            )
        if self.bridge.rf_to_zello and self.cos.mode == "disabled":
            raise ValueError(
                "bridge.rf_to_zello is true but cos.mode is 'disabled'; "
                "RF-to-Zello has no way to detect receive activity"
            )
        if self.cos.min_tx_ms and self.cos.min_tx_ms <= self.cos.hang_ms:
            raise ValueError(
                f"cos.min_tx_ms ({self.cos.min_tx_ms}) must exceed cos.hang_ms "
                f"({self.cos.hang_ms}) to have any effect: a transmission is held "
                "for max(min_tx_ms, speech + hang_ms), so a floor at or below the "
                "hang is never the binding constraint"
            )
        if self.cos.mode == "internal_audio" and self.sound.input_device is None:
            raise ValueError("cos.mode='internal_audio' requires sound.input_device")
        if self.bridge.zello_to_rf and self.sound.output_device is None:
            raise ValueError("bridge.zello_to_rf requires sound.output_device")
        return self

    def resampler_delay_ms(self) -> float:
        """Group delay the RF->Zello resampler adds, or 0.0 when unneeded.

        The inbound (Zello->RF) resampler's delay depends on the peer's rate
        and so cannot be known until a stream starts.
        """
        if self.sound.sample_rate == self.opus.sample_rate:
            return 0.0
        from .audio.resample import Resampler, ResamplerError

        try:
            return Resampler(self.sound.sample_rate, self.opus.sample_rate).delay_ms
        except ResamplerError:
            return 0.0

    def warnings(self) -> list[str]:
        """Non-fatal advisories, surfaced by --validate and at startup."""
        out: list[str] = []

        if self.cos.mode == "internal_audio" and self.cos.threshold_dbfs < DBFS_FLOOR:
            out.append(
                f"cos.threshold_dbfs ({self.cos.threshold_dbfs}) is below the detector's "
                f"floor of {DBFS_FLOOR} dBFS and can never be crossed; COS will never open"
            )

        resample_ms = self.resampler_delay_ms()
        if self.sound.sample_rate != self.opus.sample_rate:
            out.append(
                f"sound.sample_rate ({self.sound.sample_rate}) differs from opus.sample_rate "
                f"({self.opus.sample_rate}); a resampler adds {resample_ms:.1f} ms to the "
                "RF->Zello path. Running the AIOC at 16 kHz avoids this."
            )

        budget = (
            self.ptt.pre_key_ms + self.sound.jitter_ms + self.sound.block_ms + resample_ms
        )
        if budget > self.bridge.latency_budget_ms:
            out.append(
                f"estimated one-way latency {budget:.0f} ms (pre_key {self.ptt.pre_key_ms} + "
                f"jitter {self.sound.jitter_ms} + block {self.sound.block_ms} + "
                f"resample {resample_ms:.1f}) exceeds "
                f"bridge.latency_budget_ms ({self.bridge.latency_budget_ms})"
            )

        exact_block = self.sound.sample_rate * self.sound.block_ms / 1000.0
        if exact_block != int(exact_block):
            out.append(
                f"{self.sound.block_ms} ms at {self.sound.sample_rate} Hz is "
                f"{exact_block} samples, not a whole number; blocks are truncated to "
                f"{int(exact_block)}. Nothing breaks, but each block is slightly short "
                "of its nominal duration and end-of-transmission audio is ragged. "
                "Prefer a rate where the block divides evenly (16000 Hz is exact)."
            )

        for field, selector in (
            ("input_device", self.sound.input_device),
            ("output_device", self.sound.output_device),
        ):
            if isinstance(selector, int) or (
                isinstance(selector, str) and selector.strip().isdigit()
            ):
                out.append(
                    f"sound.{field} is the numeric index {selector}. Indices are NOT "
                    "stable: if the USB interface is unplugged, that index can point "
                    "at a different card, and after automatic recovery the bridge "
                    "could open the host's built-in microphone or speakers instead of "
                    "the radio. Select by device name."
                )

        if self.cos.mode in ("aioc_virtual", "aioc_hardware"):
            out.append(
                f"cos.mode='{self.cos.mode}' uses the AIOC's CM108-style HID report, "
                "whose byte layout is not published by the vendor and is NOT yet "
                "bench-verified in this project. If COS never asserts (or never "
                "releases), check Cm108Report before suspecting the radio. "
                "cos.mode='internal_audio' is the proven path."
            )

        if self.cos.mode == "aioc_hardware":
            out.append(
                "cos.mode='aioc_hardware' needs a COS wire from the radio into the "
                "AIOC's external input. A 2-pin (Kenwood K1) connector has no such "
                "signal, so on an HT this will never assert."
            )

        if self.ptt.mode == "none":
            out.append("ptt.mode='none': the radio will never be keyed. Bench/test use only.")

        if self.zello.password is not None and self.zello.refresh_token_file is None:
            out.append(
                "zello.refresh_token_file is unset; every reconnect will re-authenticate "
                "with full credentials instead of a refresh token"
            )

        if abs(self.sound.rx_gain_db) > 20 or abs(self.sound.tx_gain_db) > 20:
            out.append(
                "a gain beyond +/-20 dB usually means the radio's own level pots are "
                "mis-set; prefer fixing levels at the interface"
            )

        return out


def load_config(path: str | os.PathLike[str]) -> BridgeConfig:
    """Read, env-expand, and validate a YAML config file."""
    p = Path(path)
    try:
        raw_text = p.read_text(encoding="utf-8")
    except OSError as e:
        raise ConfigError(f"cannot read config {p}: {e}") from e

    try:
        raw = yaml.load(raw_text, Loader=_StrictLoader)
    except ConfigError as e:
        raise ConfigError(f"cannot parse config {p}: {e}") from e
    except yaml.YAMLError as e:
        raise ConfigError(f"cannot parse config {p}: {e}") from e

    if not isinstance(raw, dict):
        raise ConfigError(f"config {p} must be a YAML mapping at the top level")

    expanded = expand_env(raw)

    try:
        return BridgeConfig.model_validate(expanded)
    except Exception as e:  # pydantic.ValidationError, but keep the surface narrow
        raise ConfigError(f"invalid config {p}:\n{e}") from e
