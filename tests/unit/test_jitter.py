"""The chan_usrp reorder buffer.

A reorder buffer, not a playout buffer: frames leave as soon as ordering
allows, so in-order arrival costs no latency at all. What it must get right
is disorder, loss, duplication, and a peer that restarts its counter.
"""

from __future__ import annotations

import pytest

from zello_link.usrp.jitter import JitterBuffer
from zello_link.usrp.protocol import VOICE_BYTES

SILENCE = bytes(VOICE_BYTES)


def frame(n: int) -> bytes:
    """A frame whose contents identify its sequence number."""
    return bytes([n & 0xFF]) * VOICE_BYTES


class TestInOrder:
    def test_no_window_releases_immediately(self):
        jb = JitterBuffer(target_ms=0)
        assert jb.push(1, frame(1)) == [frame(1)]
        assert jb.push(2, frame(2)) == [frame(2)]

    def test_a_window_costs_nothing_when_ordered(self):
        """The window is a tolerance for disorder, not a delay."""
        jb = JitterBuffer(target_ms=60)          # depth 3
        out = []
        for n in range(1, 11):
            out.extend(jb.push(n, frame(n)))
        # Three frames are still held as reorder slack; the rest came through.
        assert out == [frame(n) for n in range(1, 8)]
        assert jb.held == 3

    def test_first_sequence_number_sets_the_origin(self):
        jb = JitterBuffer(target_ms=0)
        assert jb.push(5000, frame(1)) == [frame(1)]


class TestReordering:
    def test_swapped_pair_is_put_back_in_order(self):
        jb = JitterBuffer(target_ms=60)
        assert jb.push(1, frame(1)) == []
        assert jb.push(3, frame(3)) == []
        assert jb.push(2, frame(2)) == []
        assert jb.push(4, frame(4)) == [frame(1)]
        assert jb.flush() == [frame(2), frame(3), frame(4)]

    def test_disorder_beyond_the_window_is_not_recovered(self):
        """Honest limit: a frame later than the window is already conceded."""
        jb = JitterBuffer(target_ms=20)          # depth 1
        jb.push(1, frame(1))
        jb.push(3, frame(3))
        out = jb.push(4, frame(4))               # forces 2 to be declared lost
        assert SILENCE in out
        assert jb.push(2, frame(2)) == []        # too late now
        assert jb.late_dropped == 1


class TestLoss:
    def test_missing_frame_becomes_silence(self):
        jb = JitterBuffer(target_ms=0)
        jb.push(1, frame(1))
        out = jb.push(3, frame(3))
        assert out == [SILENCE, frame(3)]
        assert jb.concealed == 1

    def test_drop_shortens_instead(self):
        jb = JitterBuffer(target_ms=0, fill="drop")
        jb.push(1, frame(1))
        assert jb.push(3, frame(3)) == [frame(3)]
        assert jb.concealed == 1

    def test_unknown_fill_is_refused(self):
        with pytest.raises(ValueError, match="packet_loss_fill"):
            JitterBuffer(fill="interpolate")


class TestDuplicatesAndLateArrivals:
    def test_duplicate_is_not_re_emitted(self):
        jb = JitterBuffer(target_ms=0)
        assert jb.push(1, frame(1)) == [frame(1)]
        assert jb.push(1, frame(1)) == []
        assert jb.late_dropped == 1

    def test_already_released_sequence_is_dropped(self):
        jb = JitterBuffer(target_ms=0)
        jb.push(10, frame(10))
        jb.push(11, frame(11))
        assert jb.push(10, frame(10)) == []
        assert jb.late_dropped == 1


class TestPeerRestart:
    def test_counter_reset_resyncs_without_concealing_billions(self):
        """chan_usrp restarts its counter at zero on a channel rebuild."""
        jb = JitterBuffer(target_ms=0)
        jb.push(900_000, frame(1))
        out = jb.push(0, frame(2))
        assert out == [frame(2)]
        assert jb.concealed == 0, "a restart is not four billion lost frames"

    def test_forward_jump_resyncs(self):
        jb = JitterBuffer(target_ms=0)
        jb.push(1, frame(1))
        out = jb.push(500_000, frame(2))
        assert out == [frame(2)]
        assert jb.concealed == 0


class TestWraparound:
    def test_sequence_wrap_is_not_seen_as_loss(self):
        jb = JitterBuffer(target_ms=0)
        assert jb.push(0xFFFFFFFE, frame(1)) == [frame(1)]
        assert jb.push(0xFFFFFFFF, frame(2)) == [frame(2)]
        assert jb.push(0, frame(3)) == [frame(3)]
        assert jb.concealed == 0
        assert jb.late_dropped == 0


class TestLifecycle:
    def test_flush_drains_without_concealing(self):
        """The talker stopped; inventing silence past the end is wrong."""
        jb = JitterBuffer(target_ms=100)
        jb.push(1, frame(1))
        jb.push(3, frame(3))
        assert jb.flush() == [frame(1), frame(3)]
        assert jb.concealed == 0

    def test_reset_forgets_the_previous_transmission(self):
        jb = JitterBuffer(target_ms=60)
        jb.push(1, frame(1))
        jb.reset()
        assert jb.held == 0
        assert jb.push(9000, frame(2)) == []      # a fresh origin, not a gap
        assert jb.concealed == 0

    def test_capacity_is_never_below_the_window(self):
        """Otherwise the overflow guard would evict frames still being waited on."""
        jb = JitterBuffer(target_ms=200, max_ms=20)
        assert jb.capacity > jb.depth
