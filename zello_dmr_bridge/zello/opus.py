"""libopus encoder/decoder via ctypes, plus packet-loss concealment.

A direct ctypes binding is used instead of a wrapper package for one specific
reason: concealment requires calling ``opus_decode`` with a NULL data pointer
and length 0, which the common Python wrappers do not expose. Without it, a
dropped packet becomes an audible discontinuity fed straight into a keyed
transmitter.

The library is loaded lazily so that config validation, unit tests, and CI
run on hosts with no libopus installed.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import platform
from typing import Any, Final

import numpy as np

__all__ = [
    "OpusError",
    "OpusEncoder",
    "OpusDecoder",
    "StreamDecoder",
    "APPLICATIONS",
    "load_libopus",
    "is_available",
]

log = logging.getLogger(__name__)

# opus_defines.h
OPUS_APPLICATION_VOIP: Final = 2048
OPUS_APPLICATION_AUDIO: Final = 2049
OPUS_APPLICATION_RESTRICTED_LOWDELAY: Final = 2051

APPLICATIONS: Final[dict[str, int]] = {
    "voip": OPUS_APPLICATION_VOIP,
    "audio": OPUS_APPLICATION_AUDIO,
    "lowdelay": OPUS_APPLICATION_RESTRICTED_LOWDELAY,
}

_OPUS_SET_BITRATE: Final = 4002
_OPUS_SET_COMPLEXITY: Final = 4010
_OPUS_SET_INBAND_FEC: Final = 4012
_OPUS_SET_PACKET_LOSS_PERC: Final = 4014
_OPUS_SET_SIGNAL: Final = 4024
_OPUS_SIGNAL_VOICE: Final = 3001

_OPUS_OK: Final = 0

#: Opus tolerates at most 120 ms of audio per call; 4000 bytes is the
#: conventional safe ceiling for a single encoded packet.
_MAX_PACKET_BYTES: Final = 4000

#: Cap on consecutive concealment frames. Beyond roughly this much loss the
#: stream is gone and inventing more audio just prolongs noise on the air.
MAX_CONSECUTIVE_PLC: Final = 5

_lib: Any = None


class OpusError(Exception):
    """libopus is missing, or an opus_* call returned an error code."""


def _candidate_names() -> list[str]:
    system = platform.system()
    if system == "Darwin":
        return ["libopus.0.dylib", "libopus.dylib", "/opt/homebrew/lib/libopus.dylib",
                "/usr/local/lib/libopus.dylib"]
    if system == "Windows":
        return ["opus.dll", "libopus-0.dll", "libopus.dll"]
    return ["libopus.so.0", "libopus.so"]


def load_libopus() -> Any:
    """Load and prototype libopus once. Raises OpusError if unavailable."""
    global _lib
    if _lib is not None:
        return _lib

    found = ctypes.util.find_library("opus")
    names = ([found] if found else []) + _candidate_names()

    lib = None
    errors: list[str] = []
    for name in names:
        try:
            lib = ctypes.CDLL(name)
            break
        except OSError as e:
            errors.append(f"{name}: {e}")

    if lib is None:
        raise OpusError(
            "libopus not found. Install it:\n"
            "  Debian / Raspberry Pi OS:  sudo apt install libopus0\n"
            "  macOS:                     brew install opus\n"
            "Tried: " + "; ".join(errors)
        )

    # -- prototypes ------------------------------------------------------
    lib.opus_strerror.argtypes = [ctypes.c_int]
    lib.opus_strerror.restype = ctypes.c_char_p

    lib.opus_encoder_get_size.argtypes = [ctypes.c_int]
    lib.opus_encoder_get_size.restype = ctypes.c_int

    lib.opus_encoder_create.argtypes = [
        ctypes.c_int32, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)
    ]
    lib.opus_encoder_create.restype = ctypes.c_void_p

    lib.opus_encoder_destroy.argtypes = [ctypes.c_void_p]
    lib.opus_encoder_destroy.restype = None

    lib.opus_encode.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_int16), ctypes.c_int,
        ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int32,
    ]
    lib.opus_encode.restype = ctypes.c_int32

    # opus_encoder_ctl is variadic: int opus_encoder_ctl(OpusEncoder*, int, ...)
    #
    # argtypes must list ONLY the fixed parameters. ctypes treats every
    # argument beyond len(argtypes) as variadic, and on arm64 (Apple Silicon,
    # aarch64 Linux) variadic arguments use a different calling convention
    # than fixed ones. Listing all three here makes ctypes pass the value in a
    # register instead of on the stack, and every ctl returns OPUS_BAD_ARG.
    lib.opus_encoder_ctl.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.opus_encoder_ctl.restype = ctypes.c_int

    lib.opus_decoder_ctl.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.opus_decoder_ctl.restype = ctypes.c_int

    lib.opus_decoder_create.argtypes = [
        ctypes.c_int32, ctypes.c_int, ctypes.POINTER(ctypes.c_int)
    ]
    lib.opus_decoder_create.restype = ctypes.c_void_p

    lib.opus_decoder_destroy.argtypes = [ctypes.c_void_p]
    lib.opus_decoder_destroy.restype = None

    lib.opus_decode.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int16), ctypes.c_int, ctypes.c_int,
    ]
    lib.opus_decode.restype = ctypes.c_int

    _lib = lib
    return _lib


def is_available() -> bool:
    try:
        load_libopus()
        return True
    except OpusError:
        return False


def _strerror(code: int) -> str:
    try:
        return load_libopus().opus_strerror(code).decode("utf-8", "replace")
    except Exception:
        return f"opus error {code}"


def _check(code: int, what: str) -> int:
    if code < _OPUS_OK:
        raise OpusError(f"{what}: {_strerror(code)}")
    return code


class OpusEncoder:
    """Stateful mono Opus encoder."""

    def __init__(
        self,
        sample_rate: int,
        *,
        application: str = "voip",
        bitrate: int = 16000,
        complexity: int = 5,
        channels: int = 1,
    ) -> None:
        lib = load_libopus()
        self._lib = lib
        self.sample_rate = sample_rate
        self.channels = channels

        err = ctypes.c_int(0)
        try:
            app = APPLICATIONS[application]
        except KeyError:
            raise OpusError(f"unknown opus application {application!r}") from None

        st = lib.opus_encoder_create(sample_rate, channels, app, ctypes.byref(err))
        if not st or err.value != _OPUS_OK:
            raise OpusError(f"opus_encoder_create failed: {_strerror(err.value)}")
        self._st = ctypes.c_void_p(st)

        self._ctl(_OPUS_SET_BITRATE, bitrate)
        self._ctl(_OPUS_SET_COMPLEXITY, complexity)
        self._ctl(_OPUS_SET_SIGNAL, _OPUS_SIGNAL_VOICE)
        # Inband FEC costs a little bitrate and lets the far end recover a
        # lost frame; worthwhile on a voice link that may cross poor networks.
        self._ctl(_OPUS_SET_INBAND_FEC, 1)
        self._ctl(_OPUS_SET_PACKET_LOSS_PERC, 10)

    def _ctl(self, request: int, value: int) -> None:
        _check(
            self._lib.opus_encoder_ctl(self._st, request, ctypes.c_int32(value)),
            f"opus_encoder_ctl({request})",
        )

    def get_ctl(self, request: int) -> int:
        """Read back a ctl value. Used by tests and --diagnose output."""
        out = ctypes.c_int32(0)
        _check(
            self._lib.opus_encoder_ctl(self._st, request, ctypes.byref(out)),
            f"opus_encoder_ctl({request})",
        )
        return out.value

    def encode(self, pcm: np.ndarray) -> bytes:
        """Encode one frame of mono int16 PCM."""
        if self._st is None:
            raise OpusError("encoder is closed")
        if pcm.dtype != np.int16:
            pcm = pcm.astype(np.int16)
        pcm = np.ascontiguousarray(pcm)

        frame_size = pcm.size // self.channels
        out = (ctypes.c_ubyte * _MAX_PACKET_BYTES)()
        n = self._lib.opus_encode(
            self._st,
            pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            frame_size,
            out,
            _MAX_PACKET_BYTES,
        )
        _check(n, "opus_encode")
        return bytes(out[:n])

    def close(self) -> None:
        if getattr(self, "_st", None) is not None:
            self._lib.opus_encoder_destroy(self._st)
            self._st = None

    def __enter__(self) -> "OpusEncoder":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class OpusDecoder:
    """Stateful mono Opus decoder with packet-loss concealment."""

    def __init__(self, sample_rate: int, *, channels: int = 1, max_frame_ms: int = 120) -> None:
        lib = load_libopus()
        self._lib = lib
        self.sample_rate = sample_rate
        self.channels = channels
        self._max_samples = int(sample_rate * max_frame_ms / 1000)

        err = ctypes.c_int(0)
        st = lib.opus_decoder_create(sample_rate, channels, ctypes.byref(err))
        if not st or err.value != _OPUS_OK:
            raise OpusError(f"opus_decoder_create failed: {_strerror(err.value)}")
        self._st = ctypes.c_void_p(st)

    def decode(self, payload: bytes) -> np.ndarray:
        """Decode one Opus packet to mono int16 PCM."""
        return self._decode(payload)

    def conceal(self, frame_size: int) -> np.ndarray:
        """Synthesize one lost frame.

        This is the NULL-pointer call that the wrapper packages do not
        expose: libopus extrapolates from decoder state rather than emitting
        a click or a gap.
        """
        return self._decode(None, frame_size=frame_size)

    def _decode(self, payload: bytes | None, frame_size: int | None = None) -> np.ndarray:
        if self._st is None:
            raise OpusError("decoder is closed")

        n_samples = frame_size if frame_size is not None else self._max_samples
        buf = np.empty(n_samples * self.channels, dtype=np.int16)

        if payload is None:
            data_ptr = ctypes.cast(None, ctypes.POINTER(ctypes.c_ubyte))
            data_len = 0
        else:
            data_ptr = (ctypes.c_ubyte * len(payload)).from_buffer_copy(payload)
            data_ptr = ctypes.cast(data_ptr, ctypes.POINTER(ctypes.c_ubyte))
            data_len = len(payload)

        n = self._lib.opus_decode(
            self._st,
            data_ptr,
            data_len,
            buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            n_samples,
            0,
        )
        _check(n, "opus_decode")
        return buf[: n * self.channels]

    def close(self) -> None:
        if getattr(self, "_st", None) is not None:
            self._lib.opus_decoder_destroy(self._st)
            self._st = None

    def __enter__(self) -> "OpusDecoder":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class StreamDecoder:
    """One inbound Zello stream: decoder state plus packet sequencing.

    Uses the 32-bit ``packet_id`` that the wire format carries but that
    nothing otherwise consumes: a gap drives concealment, and a late or
    duplicate packet is discarded rather than played out of order.

    Zello documents that a client may send packet_id 0. If the peer's ids
    are not actually monotonic, sequencing disables itself for that stream
    and every packet is simply decoded in arrival order.
    """

    def __init__(self, header: Any, *, decoder: OpusDecoder | None = None) -> None:
        self.header = header
        self.frame_size = header.samples_per_frame
        self.decoder = decoder or OpusDecoder(header.sample_rate)

        self._expected: int | None = None
        self._sequencing = True
        self._seen_nonzero = False

        self.decoded_packets = 0
        self.concealed_frames = 0
        self.dropped_late = 0

    def push(self, packet_id: int, payload: bytes) -> list[np.ndarray]:
        """Feed one received packet; return PCM frames ready for playback.

        A gap yields concealment frames ahead of the real frame.
        """
        frames: list[np.ndarray] = []

        if packet_id != 0:
            self._seen_nonzero = True

        if not self._sequencing or self._expected is None:
            # First packet of the stream, or sequencing already disabled.
            if self._expected is None and self._seen_nonzero:
                self._expected = packet_id + 1
            frames.append(self.decoder.decode(payload))
            self.decoded_packets += 1
            return frames

        if packet_id < self._expected:
            # Late or duplicate. If the peer is simply sending 0 for every
            # packet, stop sequencing rather than discarding the whole call.
            if packet_id == 0 and not self._seen_nonzero:
                self._sequencing = False
                frames.append(self.decoder.decode(payload))
                self.decoded_packets += 1
                return frames
            self.dropped_late += 1
            return frames

        missing = packet_id - self._expected
        if missing:
            for _ in range(min(missing, MAX_CONSECUTIVE_PLC)):
                frames.append(self.decoder.conceal(self.frame_size))
                self.concealed_frames += 1

        frames.append(self.decoder.decode(payload))
        self.decoded_packets += 1
        self._expected = packet_id + 1
        return frames

    def close(self) -> None:
        self.decoder.close()
