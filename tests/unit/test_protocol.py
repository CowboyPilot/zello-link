"""Wire-format tests.

The endianness split is the whole point of this file: the codec header is
little-endian, the binary packet header is network (big) byte order. A test
that only round-trips our own code would pass even if both were wrong, so the
byte-level assertions use the literal example from the published API doc.
"""

from __future__ import annotations

import base64
import struct

import pytest

from zello_dmr_bridge.zello import protocol as p


class TestCodecHeader:
    def test_matches_published_example(self):
        """API.md: "gD4BPA==" == {0x80,0x3e,0x01,0x3c} == 16000 Hz, 1 frame, 60 ms."""
        header = p.unpack_codec_header("gD4BPA==")
        assert header.sample_rate == 16000
        assert header.frames_per_packet == 1
        assert header.frame_size_ms == 60

    def test_packs_to_published_example(self):
        assert p.pack_codec_header(16000, 1, 60) == "gD4BPA=="

    def test_sample_rate_is_little_endian(self):
        raw = base64.b64decode(p.pack_codec_header(16000, 1, 20))
        assert raw[0:2] == struct.pack("<H", 16000)
        assert raw[0:2] != struct.pack(">H", 16000), "sample rate must NOT be big-endian"

    def test_field_order(self):
        raw = base64.b64decode(p.pack_codec_header(48000, 2, 40))
        assert struct.unpack("<H", raw[0:2])[0] == 48000
        assert raw[2] == 2
        assert raw[3] == 40

    @pytest.mark.parametrize("rate,fpp,ms", [(8000, 1, 10), (16000, 1, 20), (48000, 2, 60)])
    def test_roundtrip(self, rate, fpp, ms):
        h = p.unpack_codec_header(p.pack_codec_header(rate, fpp, ms))
        assert (h.sample_rate, h.frames_per_packet, h.frame_size_ms) == (rate, fpp, ms)

    def test_derived_properties(self):
        h = p.unpack_codec_header(p.pack_codec_header(16000, 2, 20))
        assert h.samples_per_frame == 320
        assert h.packet_duration_ms == 40

    def test_rejects_bad_base64(self):
        with pytest.raises(p.ProtocolError, match="base64"):
            p.unpack_codec_header("not valid base64!!")

    def test_rejects_wrong_length(self):
        with pytest.raises(p.ProtocolError, match="4 bytes"):
            p.unpack_codec_header(base64.b64encode(b"\x01\x02").decode())

    def test_rejects_zero_fields(self):
        with pytest.raises(p.ProtocolError, match="sample_rate"):
            p.unpack_codec_header(base64.b64encode(struct.pack("<HBB", 0, 1, 20)).decode())
        with pytest.raises(p.ProtocolError, match="frame_size_ms"):
            p.unpack_codec_header(base64.b64encode(struct.pack("<HBB", 16000, 1, 0)).decode())

    def test_rejects_out_of_range_pack(self):
        with pytest.raises(ValueError, match="uint16"):
            p.pack_codec_header(70000, 1, 20)
        with pytest.raises(ValueError, match="uint8"):
            p.pack_codec_header(16000, 1, 300)


class TestAudioPacket:
    def test_header_is_network_byte_order(self):
        data = p.pack_audio_packet(0x11223344, 0x55667788, b"")
        assert data == bytes([0x01, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88])

    def test_layout_type_stream_packet_payload(self):
        data = p.pack_audio_packet(22695, 7, b"\xde\xad\xbe\xef")
        assert data[0] == p.PACKET_TYPE_AUDIO
        assert struct.unpack("!I", data[1:5])[0] == 22695
        assert struct.unpack("!I", data[5:9])[0] == 7
        assert data[9:] == b"\xde\xad\xbe\xef"

    def test_roundtrip(self):
        pkt = p.unpack_audio_packet(p.pack_audio_packet(22695, 42, b"opus-payload"))
        assert pkt.stream_id == 22695
        assert pkt.packet_id == 42
        assert pkt.payload == b"opus-payload"

    def test_packet_id_zero_is_legal(self):
        """Zello documents that packet_id sent to the server may be zero."""
        pkt = p.unpack_audio_packet(p.pack_audio_packet(1, 0, b"x"))
        assert pkt.packet_id == 0

    def test_empty_payload_roundtrips(self):
        assert p.unpack_audio_packet(p.pack_audio_packet(1, 1, b"")).payload == b""

    def test_rejects_truncated_frame(self):
        with pytest.raises(p.ProtocolError, match="too short"):
            p.unpack_audio_packet(b"\x01\x00\x00")

    def test_rejects_unknown_type(self):
        bad = bytes([0x02]) + b"\x00" * 8
        with pytest.raises(p.ProtocolError, match="0x02"):
            p.unpack_audio_packet(bad)

    def test_rejects_out_of_range_ids(self):
        with pytest.raises(ValueError, match="stream_id"):
            p.pack_audio_packet(2**32, 0, b"")
        with pytest.raises(ValueError, match="packet_id"):
            p.pack_audio_packet(0, 2**32, b"")

    def test_max_uint32_roundtrips(self):
        pkt = p.unpack_audio_packet(p.pack_audio_packet(0xFFFFFFFF, 0xFFFFFFFF, b"z"))
        assert pkt.stream_id == 0xFFFFFFFF
        assert pkt.packet_id == 0xFFFFFFFF


class TestCommands:
    def test_logon_uses_channels_array(self):
        """Confirmed against API.md: the field is `channels`, an array."""
        cmd = p.build_logon(
            seq=1, channels=["Event Security"], auth_token="tok",
            version="0.1.0", platform_name="DMR Gateway",
        )
        assert cmd["command"] == "logon"
        assert cmd["channels"] == ["Event Security"]
        assert "channel" not in cmd

    def test_logon_omits_absent_credentials(self):
        cmd = p.build_logon(
            seq=1, channels=["c"], auth_token="tok",
            version="0.1.0", platform_name="n",
        )
        assert "password" not in cmd
        assert "username" not in cmd
        assert "refresh_token" not in cmd

    def test_logon_with_refresh_token(self):
        cmd = p.build_logon(
            seq=3, channels=["c"], refresh_token="rt",
            version="0.1.0", platform_name="n",
        )
        assert cmd["refresh_token"] == "rt"

    def test_start_stream_fields(self):
        cmd = p.build_start_stream(
            seq=2, channel="Event Security",
            codec_header=p.pack_codec_header(16000, 1, 20),
            packet_duration_ms=20,
        )
        assert cmd == {
            "command": "start_stream",
            "seq": 2,
            "channel": "Event Security",
            "type": "audio",
            "codec": "opus",
            "codec_header": "gD4BFA==",
            "packet_duration": 20,
        }

    def test_stop_stream_fields(self):
        cmd = p.build_stop_stream(seq=9, stream_id=22695, channel="Event Security")
        assert cmd == {
            "command": "stop_stream",
            "seq": 9,
            "stream_id": 22695,
            "channel": "Event Security",
        }
