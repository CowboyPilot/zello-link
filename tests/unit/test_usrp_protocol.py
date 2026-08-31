"""chan_usrp wire format.

Byte-level assertions are taken from the AllStarLink source rather than from
our own round trips: a test that only round-trips our packer would pass even
if the whole layout were wrong.

    struct _chan_usrp_bufhdr {
        char eye[4]; uint32_t seq, memory, keyup, talkgroup, type, mpxid, reserved;
    };
    #define USRP_VOICE_FRAME_SIZE (160*sizeof(short))
"""

from __future__ import annotations

import struct

import pytest

from zello_link.usrp import protocol as u


class TestConstants:
    def test_header_is_32_bytes(self):
        """4 + 7*4. Matches sizeof(struct _chan_usrp_bufhdr)."""
        assert u.HEADER_BYTES == 32

    def test_voice_frame_matches_the_c_define(self):
        """USRP_VOICE_FRAME_SIZE == 160 * sizeof(short) == 320."""
        assert u.VOICE_SAMPLES == 160
        assert u.VOICE_BYTES == 320

    def test_frame_is_20ms_at_8k(self):
        assert u.VOICE_SAMPLES == u.SAMPLE_RATE * u.FRAME_MS // 1000

    def test_type_enum_values(self):
        """enum { USRP_TYPE_VOICE=0, USRP_TYPE_DTMF, USRP_TYPE_TEXT }"""
        assert (u.USRP_TYPE_VOICE, u.USRP_TYPE_DTMF, u.USRP_TYPE_TEXT) == (0, 1, 2)

    def test_magic(self):
        assert u.USRP_MAGIC == b"USRP"


class TestHeaderLayout:
    """Field order and endianness, asserted against raw bytes."""

    def _hdr(self, packet):
        return packet[: u.HEADER_BYTES]

    def test_eye_is_first_four_raw_bytes(self):
        """chan_usrp does memcmp(bufhdrp->eye, "USRP", 4) -- raw, not encoded."""
        assert self._hdr(u.pack_signal(0, keyed=True))[:4] == b"USRP"

    def test_field_order(self):
        packet = u.pack_signal(0x11223344, keyed=True, talkgroup=0x55667788,
                               memory=0x99AABBCC)
        eye, seq, memory, keyup, tg, ptype, mpxid, reserved = struct.unpack(
            "!4s7I", self._hdr(packet)
        )
        assert eye == b"USRP"
        assert seq == 0x11223344
        assert memory == 0x99AABBCC
        assert keyup == 1
        assert tg == 0x55667788
        assert ptype == u.USRP_TYPE_VOICE
        assert (mpxid, reserved) == (0, 0)

    def test_seq_is_network_byte_order(self):
        """chan_usrp: bufhdrp->seq = htonl(...) / seq = ntohl(...)."""
        packet = u.pack_signal(1, keyed=True)
        assert packet[4:8] == struct.pack("!I", 1)
        assert packet[4:8] != struct.pack("<I", 1), "seq must NOT be little-endian"

    def test_keyup_is_network_byte_order(self):
        """chan_usrp sends bufhdrp->keyup = htonl(1)."""
        packet = u.pack_signal(0, keyed=True)
        assert packet[12:16] == struct.pack("!I", 1)

    def test_unkey_writes_zero(self):
        assert u.pack_signal(0, keyed=False)[12:16] == struct.pack("!I", 0)

    def test_no_padding_between_fields(self):
        """Explicit '!' format: C struct alignment must not leak into the wire."""
        assert struct.calcsize("!4s7I") == 32


class TestVoicePackets:
    def test_total_length(self):
        assert len(u.pack_voice(0, b"\x00" * 320)) == 352

    def test_payload_is_appended_verbatim(self):
        pcm = bytes(range(256)) + bytes(range(64))
        assert u.pack_voice(0, pcm)[u.HEADER_BYTES:] == pcm

    def test_type_is_voice(self):
        packet = u.pack_voice(0, b"\x00" * 320)
        assert struct.unpack_from("!I", packet, 20)[0] == u.USRP_TYPE_VOICE

    def test_keyed_by_default(self):
        assert u.unpack(u.pack_voice(0, b"\x00" * 320)).keyed is True

    def test_can_be_marked_unkeyed(self):
        packet = u.pack_voice(0, b"\x00" * 320, keyed=False)
        assert u.unpack(packet).keyed is False

    @pytest.mark.parametrize("size", [0, 1, 159, 319, 321, 640])
    def test_wrong_payload_size_rejected(self, size):
        """A short frame would be heard as a click; pad upstream, not here."""
        with pytest.raises(ValueError, match="exactly 320 bytes"):
            u.pack_voice(0, b"\x00" * size)

    def test_seq_range_enforced(self):
        with pytest.raises(ValueError, match="uint32"):
            u.pack_voice(2**32, b"\x00" * 320)

    def test_max_seq_is_allowed(self):
        assert u.unpack(u.pack_voice(0xFFFFFFFF, b"\x00" * 320)).seq == 0xFFFFFFFF


class TestSignalPackets:
    def test_is_header_only(self):
        assert len(u.pack_signal(0, keyed=False)) == u.HEADER_BYTES

    def test_unkey_roundtrip(self):
        pkt = u.unpack(u.pack_signal(7, keyed=False))
        assert pkt.is_signal and not pkt.is_voice and pkt.keyed is False

    def test_key_roundtrip(self):
        assert u.unpack(u.pack_signal(7, keyed=True)).keyed is True


class TestUnpack:
    def test_roundtrip_voice(self):
        pcm = bytes(range(256)) + bytes(range(64))
        pkt = u.unpack(u.pack_voice(42, pcm, talkgroup=9, memory=3))
        assert pkt.seq == 42
        assert pkt.payload == pcm
        assert pkt.talkgroup == 9
        assert pkt.memory == 3
        assert pkt.is_voice

    def test_rejects_short_datagram(self):
        with pytest.raises(u.UsrpProtocolError, match="too short"):
            u.unpack(b"USRP" + b"\x00" * 10)

    def test_rejects_bad_magic(self):
        bad = b"XXXX" + b"\x00" * 28
        with pytest.raises(u.UsrpProtocolError, match="bad magic"):
            u.unpack(bad)

    def test_rejects_empty(self):
        with pytest.raises(u.UsrpProtocolError, match="too short"):
            u.unpack(b"")

    @pytest.mark.parametrize("extra", [1, 100, 319, 321, 700])
    def test_rejects_odd_payload_length(self, extra):
        """chan_usrp treats only 320 as voice; anything else is malformed."""
        with pytest.raises(u.UsrpProtocolError, match="payload is"):
            u.unpack(u.pack_signal(0, keyed=True) + b"\x00" * extra)

    def test_exactly_header_is_signalling(self):
        assert u.unpack(u.pack_signal(0, keyed=True)).is_signal

    def test_keyup_any_nonzero_counts_as_keyed(self):
        """chan_usrp tracks PTT by non-zero, not by == 1."""
        raw = struct.pack("!4s7I", b"USRP", 0, 0, 0x0BADF00D, 0, 0, 0, 0)
        assert u.unpack(raw).keyed is True


class TestTypeFieldByteOrderQuirk:
    """chan_usrp reads `type` without ntohl while reading `seq` with it.

    Harmless for us because USRP_TYPE_VOICE == 0 is identical in both orders,
    but a non-zero type must go out in HOST order to be recognised. Pinned so
    that adding DTMF/TEXT later does not silently send an unmatched type.
    """

    def test_voice_type_is_order_agnostic(self):
        assert struct.pack("!I", 0) == struct.pack("=I", 0)

    def test_nonzero_type_differs_between_orders(self):
        assert struct.pack("!I", u.USRP_TYPE_TEXT) != struct.pack(
            "=I", u.USRP_TYPE_TEXT
        )

    def test_helper_emits_host_order(self):
        assert u.pack_host_order_type(u.USRP_TYPE_TEXT) == struct.pack("=I", 2)

    def test_helper_documents_the_hazard(self):
        assert "host" in (u.pack_host_order_type.__doc__ or "").lower()
