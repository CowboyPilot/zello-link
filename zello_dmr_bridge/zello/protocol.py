"""Zello Channel API wire format: serialization, parsing, framing.

All protocol-shaped knowledge lives here so that an upstream change to a beta
API can be absorbed in one file. Nothing in this module performs I/O.

Two byte orders are in play and they are NOT the same -- this is the single
easiest thing to get wrong in the whole protocol:

  * The binary audio packet header uses NETWORK byte order (big-endian):
        {type(8) = 0x01, stream_id(32), packet_id(32), data[]}
  * The base64 codec_header uses LITTLE-endian for the sample rate:
        {sample_rate_hz(16LE), frames_per_packet(8), frame_size_ms(8)}

Reference: https://github.com/zelloptt/zello-channel-api/blob/master/API.md
"""

from __future__ import annotations

import base64
import struct
from dataclasses import dataclass
from typing import Any, Final

__all__ = [
    "PACKET_TYPE_AUDIO",
    "CodecHeader",
    "AudioPacket",
    "ProtocolError",
    "pack_codec_header",
    "unpack_codec_header",
    "pack_audio_packet",
    "unpack_audio_packet",
    "build_logon",
    "build_start_stream",
    "build_stop_stream",
]

PACKET_TYPE_AUDIO: Final[int] = 0x01

# type(8) + stream_id(32) + packet_id(32), big-endian.
_AUDIO_HEADER = struct.Struct("!BII")
_AUDIO_HEADER_LEN: Final[int] = _AUDIO_HEADER.size  # 9

# sample_rate(16LE) + frames_per_packet(8) + frame_size_ms(8).
_CODEC_HEADER = struct.Struct("<HBB")

_UINT32_MAX: Final[int] = 0xFFFFFFFF


class ProtocolError(Exception):
    """Malformed or unsupported protocol data received from the peer."""


@dataclass(frozen=True, slots=True)
class CodecHeader:
    sample_rate: int
    frames_per_packet: int
    frame_size_ms: int

    @property
    def samples_per_frame(self) -> int:
        return int(self.sample_rate * self.frame_size_ms / 1000)

    @property
    def packet_duration_ms(self) -> int:
        return self.frame_size_ms * self.frames_per_packet


@dataclass(frozen=True, slots=True)
class AudioPacket:
    stream_id: int
    packet_id: int
    payload: bytes


def pack_codec_header(sample_rate: int, frames_per_packet: int, frame_size_ms: int) -> str:
    """Build the base64 ``codec_header`` sent with ``start_stream``."""
    if not 0 <= sample_rate <= 0xFFFF:
        raise ValueError(f"sample_rate {sample_rate} does not fit in uint16")
    if not 1 <= frames_per_packet <= 0xFF:
        raise ValueError(f"frames_per_packet {frames_per_packet} does not fit in uint8")
    if not 1 <= frame_size_ms <= 0xFF:
        raise ValueError(f"frame_size_ms {frame_size_ms} does not fit in uint8")
    raw = _CODEC_HEADER.pack(sample_rate, frames_per_packet, frame_size_ms)
    return base64.b64encode(raw).decode("ascii")


def unpack_codec_header(value: str | bytes) -> CodecHeader:
    """Parse a peer's ``codec_header``.

    Always parse this rather than assuming the incoming stream matches our
    own configured frame size -- other Zello clients negotiate their own.
    """
    if isinstance(value, str):
        try:
            raw = base64.b64decode(value, validate=True)
        except Exception as e:
            raise ProtocolError(f"codec_header is not valid base64: {value!r}") from e
    else:
        raw = bytes(value)

    if len(raw) != _CODEC_HEADER.size:
        raise ProtocolError(
            f"codec_header must decode to {_CODEC_HEADER.size} bytes, got {len(raw)}"
        )

    sample_rate, frames_per_packet, frame_size_ms = _CODEC_HEADER.unpack(raw)

    if sample_rate <= 0:
        raise ProtocolError(f"codec_header sample_rate is {sample_rate}")
    if frames_per_packet < 1:
        raise ProtocolError(f"codec_header frames_per_packet is {frames_per_packet}")
    if frame_size_ms < 1:
        raise ProtocolError(f"codec_header frame_size_ms is {frame_size_ms}")

    return CodecHeader(sample_rate, frames_per_packet, frame_size_ms)


def pack_audio_packet(stream_id: int, packet_id: int, payload: bytes) -> bytes:
    """Frame one Opus payload as a Zello binary audio packet."""
    if not 0 <= stream_id <= _UINT32_MAX:
        raise ValueError(f"stream_id {stream_id} out of uint32 range")
    if not 0 <= packet_id <= _UINT32_MAX:
        raise ValueError(f"packet_id {packet_id} out of uint32 range")
    return _AUDIO_HEADER.pack(PACKET_TYPE_AUDIO, stream_id, packet_id) + payload


def unpack_audio_packet(data: bytes) -> AudioPacket:
    """Parse an inbound binary frame. Raises for non-audio or truncated data."""
    if len(data) < _AUDIO_HEADER_LEN:
        raise ProtocolError(
            f"binary frame too short: {len(data)} bytes, need at least {_AUDIO_HEADER_LEN}"
        )
    ptype, stream_id, packet_id = _AUDIO_HEADER.unpack_from(data, 0)
    if ptype != PACKET_TYPE_AUDIO:
        raise ProtocolError(f"unsupported binary packet type 0x{ptype:02x}")
    return AudioPacket(stream_id, packet_id, data[_AUDIO_HEADER_LEN:])


def build_logon(
    *,
    seq: int,
    channels: list[str],
    username: str | None = None,
    password: str | None = None,
    auth_token: str | None = None,
    refresh_token: str | None = None,
    version: str,
    platform_name: str,
    platform_type: str = "linux",
) -> dict[str, Any]:
    """Build the ``logon`` command.

    ``channels`` is an array of channel names -- confirmed against API.md.
    Fields that are None are omitted entirely rather than sent as null.
    """
    cmd: dict[str, Any] = {
        "command": "logon",
        "seq": seq,
        "channels": list(channels),
        "version": version,
        "platform_type": platform_type,
        "platform_name": platform_name,
    }
    if refresh_token is not None:
        cmd["refresh_token"] = refresh_token
    if auth_token is not None:
        cmd["auth_token"] = auth_token
    if username is not None:
        cmd["username"] = username
    if password is not None:
        cmd["password"] = password
    return cmd


def build_start_stream(
    *,
    seq: int,
    channel: str,
    codec_header: str,
    packet_duration_ms: int,
    codec: str = "opus",
    stream_type: str = "audio",
) -> dict[str, Any]:
    return {
        "command": "start_stream",
        "seq": seq,
        "channel": channel,
        "type": stream_type,
        "codec": codec,
        "codec_header": codec_header,
        "packet_duration": packet_duration_ms,
    }


def build_stop_stream(*, seq: int, stream_id: int, channel: str) -> dict[str, Any]:
    return {
        "command": "stop_stream",
        "seq": seq,
        "stream_id": stream_id,
        "channel": channel,
    }
