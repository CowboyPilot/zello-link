"""chan_usrp wire format: header packing, parsing, validation.

All protocol-shaped knowledge for the AllStarLink side lives here, mirroring
how ``zello/protocol.py`` isolates the Zello wire format. Nothing here does
I/O.

Verified against AllStarLink/ASL-Asterisk ``chan_usrp.h`` and ``chan_usrp.c``
(develop branch) rather than taken on trust:

    struct _chan_usrp_bufhdr {
        char     eye[4];      // verification string, "USRP"
        uint32_t seq;         // sequence counter
        uint32_t memory;      // memory ID or zero (default)
        uint32_t keyup;       // tracks PTT state
        uint32_t talkgroup;   // trunk TG id
        uint32_t type;        // USRP_TYPE_VOICE=0, DTMF, TEXT
        uint32_t mpxid;       // for future use
        uint32_t reserved;    // for future use
    };
    #define USRP_VOICE_FRAME_SIZE (160*sizeof(short))   // 0.02 * 8k

BYTE ORDER -- and chan_usrp is not self-consistent about it:

  * ``seq``   is written ``htonl(p->send_seqno++)`` and read ``ntohl(...)``.
  * ``keyup`` is written ``htonl(1)``.
  * ``type``  is compared RAW: ``if (bufhdrp->type == USRP_TYPE_TEXT)`` with
    no ``ntohl``.

So the header is network byte order except for ``type``, which chan_usrp
effectively reads in host order. This is harmless for us only because
``USRP_TYPE_VOICE == 0`` and zero is identical in both orders. Anything that
later sends DTMF or TEXT must send ``type`` in HOST order to be recognised --
see ``pack_host_order_type``.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Final

__all__ = [
    "USRP_MAGIC",
    "HEADER_BYTES",
    "VOICE_SAMPLES",
    "VOICE_BYTES",
    "SAMPLE_RATE",
    "FRAME_MS",
    "USRP_TYPE_VOICE",
    "USRP_TYPE_DTMF",
    "USRP_TYPE_TEXT",
    "UsrpPacket",
    "UsrpProtocolError",
    "pack_voice",
    "pack_signal",
    "unpack",
    "pack_host_order_type",
]

USRP_MAGIC: Final[bytes] = b"USRP"

#: char[4] + 7 x uint32, network byte order. Explicit format rather than a
#: native struct: C alignment/padding must not leak into the wire layout.
_HEADER: Final[struct.Struct] = struct.Struct("!4s7I")
HEADER_BYTES: Final[int] = _HEADER.size          # 32

#: chan_usrp is signed-linear only, at a fixed rate and frame size.
SAMPLE_RATE: Final[int] = 8000
FRAME_MS: Final[int] = 20
VOICE_SAMPLES: Final[int] = 160                  # 20 ms at 8 kHz
VOICE_BYTES: Final[int] = VOICE_SAMPLES * 2      # USRP_VOICE_FRAME_SIZE

USRP_TYPE_VOICE: Final[int] = 0
USRP_TYPE_DTMF: Final[int] = 1
USRP_TYPE_TEXT: Final[int] = 2

_UINT32_MAX: Final[int] = 0xFFFFFFFF


class UsrpProtocolError(Exception):
    """Malformed or unsupported USRP datagram."""


@dataclass(frozen=True, slots=True)
class UsrpPacket:
    seq: int
    keyed: bool
    ptype: int
    talkgroup: int
    memory: int
    payload: bytes

    @property
    def is_voice(self) -> bool:
        """True for a datagram carrying a full audio frame.

        chan_usrp decides this by length -- ``datalen == USRP_VOICE_FRAME_SIZE``
        -- not by the type field, so this mirrors that.
        """
        return len(self.payload) == VOICE_BYTES

    @property
    def is_signal(self) -> bool:
        """True for a header-only datagram: key/unkey signalling."""
        return len(self.payload) == 0


def _check_seq(seq: int) -> int:
    if not 0 <= seq <= _UINT32_MAX:
        raise ValueError(f"seq {seq} out of uint32 range")
    return seq


def pack_voice(
    seq: int,
    pcm: bytes,
    *,
    keyed: bool = True,
    talkgroup: int = 0,
    memory: int = 0,
) -> bytes:
    """Frame one 20 ms audio block as a USRP voice datagram.

    ``pcm`` must be exactly 320 bytes of 8 kHz mono signed 16-bit LE audio.
    Anything else is a bug upstream in the packetizer, so it raises rather
    than padding -- a short frame sent to ASL would be heard as a click.
    """
    _check_seq(seq)
    if len(pcm) != VOICE_BYTES:
        raise ValueError(
            f"voice payload must be exactly {VOICE_BYTES} bytes "
            f"({VOICE_SAMPLES} samples at {SAMPLE_RATE} Hz), got {len(pcm)}"
        )
    header = _HEADER.pack(
        USRP_MAGIC, seq, memory, 1 if keyed else 0, talkgroup, USRP_TYPE_VOICE, 0, 0
    )
    return header + pcm


def pack_signal(
    seq: int,
    *,
    keyed: bool,
    talkgroup: int = 0,
    memory: int = 0,
) -> bytes:
    """Build a header-only datagram: the explicit key/unkey signal.

    The spec's Zello->ASL teardown is an explicit ``keyup=0`` packet after the
    last voice frame; simply going silent is not enough, because ASL would sit
    waiting on its own receive timeout.
    """
    _check_seq(seq)
    return _HEADER.pack(
        USRP_MAGIC, seq, memory, 1 if keyed else 0, talkgroup, USRP_TYPE_VOICE, 0, 0
    )


def unpack(data: bytes) -> UsrpPacket:
    """Parse an inbound datagram. Raises for anything not a USRP packet."""
    if len(data) < HEADER_BYTES:
        raise UsrpProtocolError(
            f"datagram too short: {len(data)} bytes, need at least {HEADER_BYTES}"
        )

    eye, seq, memory, keyup, talkgroup, ptype, _mpxid, _reserved = _HEADER.unpack_from(
        data, 0
    )
    if eye != USRP_MAGIC:
        raise UsrpProtocolError(f"bad magic {eye!r}, expected {USRP_MAGIC!r}")

    payload = data[HEADER_BYTES:]
    if payload and len(payload) != VOICE_BYTES:
        raise UsrpProtocolError(
            f"payload is {len(payload)} bytes; expected 0 (signalling) "
            f"or {VOICE_BYTES} (voice)"
        )

    return UsrpPacket(
        seq=seq,
        keyed=bool(keyup),
        ptype=ptype,
        talkgroup=talkgroup,
        memory=memory,
        payload=payload,
    )


def pack_host_order_type(ptype: int) -> bytes:
    """Encode ``type`` the way chan_usrp actually compares it.

    chan_usrp reads ``type`` without ``ntohl`` while reading ``seq`` with it,
    so a non-zero type must go on the wire in HOST order to be recognised.
    Unused today -- the bridge only sends voice, and ``USRP_TYPE_VOICE == 0``
    is identical either way -- but here so that adding DTMF or TEXT later does
    not silently send a type ASL will never match.
    """
    return struct.pack("=I", ptype)
