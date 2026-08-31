"""Reorder and conceal for the chan_usrp receive path.

UDP may deliver datagrams out of order, twice, or not at all. Encoding them
in arrival order would garble speech, and simply skipping a lost frame
shortens the audio -- audible as a click and a clipped syllable.

This is a REORDER buffer, not a paced playout buffer, and the distinction is
deliberate. Nothing downstream needs a real-time clock: received audio is
resampled, Opus-encoded and handed to Zello, whose own receiver buffers
before playing. A paced buffer would add its full depth to end-to-end latency
and buy nothing, so frames are released as soon as ordering permits.

The cost of the buffer is therefore bounded by disorder, not by depth: with
in-order arrival -- which is what loopback gives -- every frame is released
immediately and the buffer adds no latency at all.
"""

from __future__ import annotations

from typing import Final

from .protocol import VOICE_BYTES

__all__ = ["JitterBuffer"]

_MASK: Final[int] = 0xFFFFFFFF

#: Beyond this, a sequence difference is a restarted peer rather than
#: reordering. chan_usrp restarts its counter at zero on every channel
#: rebuild, which would otherwise look like three billion lost frames.
_RESTART_THRESHOLD: Final[int] = 1000


def _sdiff(a: int, b: int) -> int:
    """``a - b`` as a signed difference across the uint32 wrap."""
    d = (a - b) & _MASK
    return d - 0x100000000 if d >= 0x80000000 else d


class JitterBuffer:
    """Order incoming voice frames by sequence, concealing what never comes.

    ``target_ms`` is how much disorder to tolerate before declaring a frame
    lost: the buffer waits until it holds that many frames' worth before
    giving up on a missing sequence number. ``0`` disables waiting, which is
    the right setting on loopback where reordering cannot occur.
    """

    def __init__(
        self,
        *,
        target_ms: int = 60,
        max_ms: int = 200,
        frame_ms: int = 20,
        fill: str = "silence",
        frame_bytes: int = VOICE_BYTES,
    ) -> None:
        if frame_ms <= 0:
            raise ValueError("frame_ms must be positive")
        if fill not in ("silence", "drop"):
            raise ValueError(f"unknown packet_loss_fill {fill!r}")

        self.frame_ms = frame_ms
        self.fill = fill
        self.depth = max(0, target_ms // frame_ms)
        # Never below the target, or the overflow guard would fight the
        # reorder window and drop frames the buffer is still waiting on.
        self.capacity = max(self.depth + 1, max_ms // frame_ms)

        self._silence = bytes(frame_bytes)
        self._frames: dict[int, bytes] = {}
        self._next: int | None = None

        self.concealed = 0
        self.late_dropped = 0
        self.overflows = 0

    def __repr__(self) -> str:
        return (
            f"JitterBuffer(depth={self.depth} frames, "
            f"capacity={self.capacity}, fill={self.fill!r})"
        )

    @property
    def held(self) -> int:
        return len(self._frames)

    def reset(self) -> None:
        """Forget all state. Called at the start of each transmission."""
        self._frames.clear()
        self._next = None

    def push(self, seq: int, payload: bytes) -> list[bytes]:
        """Accept one frame; return whatever is now releasable, in order."""
        if self._next is None:
            self._next = seq

        d = _sdiff(seq, self._next)

        if d < 0:
            if d < -_RESTART_THRESHOLD:
                # Not late by any sane margin: the peer restarted its counter.
                out = self._drain()
                self._next = seq
                out.extend(self._push_ordered(seq, payload))
                return out
            # Already released; re-emitting it would stutter the audio.
            self.late_dropped += 1
            return []

        if d > _RESTART_THRESHOLD:
            out = self._drain()
            self._next = seq
            out.extend(self._push_ordered(seq, payload))
            return out

        return self._push_ordered(seq, payload)

    def _push_ordered(self, seq: int, payload: bytes) -> list[bytes]:
        self._frames[seq] = payload

        out: list[bytes] = []
        while len(self._frames) > self.depth:
            out.extend(self._pop_next())

        # Defensive: reordering alone cannot exceed the window above, so this
        # only trips on a peer emitting wildly scattered sequence numbers.
        while len(self._frames) > self.capacity:
            self.overflows += 1
            out.extend(self._pop_next())
        return out

    def _pop_next(self) -> list[bytes]:
        assert self._next is not None
        seq = self._next
        self._next = (seq + 1) & _MASK

        frame = self._frames.pop(seq, None)
        if frame is not None:
            return [frame]

        # Nothing arrived for this slot. Emitting silence keeps the timeline
        # intact so the words either side stay where they belong.
        self.concealed += 1
        return [self._silence] if self.fill == "silence" else []

    def _drain(self) -> list[bytes]:
        """Release everything held, in sequence order, without concealing."""
        if not self._frames:
            return []
        anchor = self._next if self._next is not None else min(self._frames)
        ordered = sorted(self._frames, key=lambda s: _sdiff(s, anchor))
        out = [self._frames[s] for s in ordered]
        self._frames.clear()
        return out

    def flush(self) -> list[bytes]:
        """Drain at end of transmission. No concealment: the talker stopped."""
        out = self._drain()
        self._next = None
        return out
