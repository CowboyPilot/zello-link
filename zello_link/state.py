"""Bridge state machine definitions.

``BridgeController`` is the sole authority for transmission direction. This
module holds only the vocabulary -- states, the legal transition graph, and
the direction each state belongs to -- so the transition rules can be unit
tested without any hardware, network, or asyncio machinery.
"""

from __future__ import annotations

from enum import Enum

__all__ = ["State", "Direction", "LEGAL_TRANSITIONS", "is_legal", "direction_of"]


class Direction(str, Enum):
    """Which way audio is flowing, for logging and arbitration."""

    NONE = "none"
    ZELLO_TO_RF = "zello->rf"
    RF_TO_ZELLO = "rf->zello"


class State(str, Enum):
    IDLE = "IDLE"

    # Zello -> RF: key the radio, wait for it to come up, play, tail, unkey.
    ZELLO_TO_RF_PREKEY = "ZELLO_TO_RF_PREKEY"
    ZELLO_TO_RF_ACTIVE = "ZELLO_TO_RF_ACTIVE"
    ZELLO_TO_RF_TAIL = "ZELLO_TO_RF_TAIL"

    # RF -> Zello: open a stream, encode, hang, close.
    RF_TO_ZELLO_START = "RF_TO_ZELLO_START"
    RF_TO_ZELLO_ACTIVE = "RF_TO_ZELLO_ACTIVE"
    RF_TO_ZELLO_HANG = "RF_TO_ZELLO_HANG"

    # Terminal-ish: PTT forced off, hardware being torn down or retried.
    FAILSAFE = "FAILSAFE"


ZELLO_TO_RF_STATES = frozenset(
    {State.ZELLO_TO_RF_PREKEY, State.ZELLO_TO_RF_ACTIVE, State.ZELLO_TO_RF_TAIL}
)
RF_TO_ZELLO_STATES = frozenset(
    {State.RF_TO_ZELLO_START, State.RF_TO_ZELLO_ACTIVE, State.RF_TO_ZELLO_HANG}
)

#: Legal successor states. FAILSAFE is reachable from anywhere and is applied
#: outside this table, because a fail-safe must never be blocked by a
#: transition check.
LEGAL_TRANSITIONS: dict[State, frozenset[State]] = {
    State.IDLE: frozenset({State.ZELLO_TO_RF_PREKEY, State.RF_TO_ZELLO_START}),
    State.ZELLO_TO_RF_PREKEY: frozenset({State.ZELLO_TO_RF_ACTIVE, State.ZELLO_TO_RF_TAIL}),
    State.ZELLO_TO_RF_ACTIVE: frozenset({State.ZELLO_TO_RF_TAIL}),
    State.ZELLO_TO_RF_TAIL: frozenset({State.IDLE}),
    State.RF_TO_ZELLO_START: frozenset({State.RF_TO_ZELLO_ACTIVE, State.RF_TO_ZELLO_HANG}),
    State.RF_TO_ZELLO_ACTIVE: frozenset({State.RF_TO_ZELLO_HANG}),
    State.RF_TO_ZELLO_HANG: frozenset({State.IDLE}),
    State.FAILSAFE: frozenset({State.IDLE}),
}


def is_legal(src: State, dst: State) -> bool:
    """True if ``src -> dst`` is an allowed transition.

    Any state may enter FAILSAFE, and a state may always re-enter itself
    (idempotent re-assertion is not an error).
    """
    if dst is State.FAILSAFE or src is dst:
        return True
    return dst in LEGAL_TRANSITIONS.get(src, frozenset())


def direction_of(state: State) -> Direction:
    if state in ZELLO_TO_RF_STATES:
        return Direction.ZELLO_TO_RF
    if state in RF_TO_ZELLO_STATES:
        return Direction.RF_TO_ZELLO
    return Direction.NONE
