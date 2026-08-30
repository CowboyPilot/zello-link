"""Transition-graph invariants for the bridge state machine."""

from __future__ import annotations

import pytest

from zello_dmr_bridge.state import (
    LEGAL_TRANSITIONS,
    RF_TO_ZELLO_STATES,
    ZELLO_TO_RF_STATES,
    Direction,
    State,
    direction_of,
    is_legal,
)


class TestDirections:
    def test_idle_has_no_direction(self):
        assert direction_of(State.IDLE) is Direction.NONE

    @pytest.mark.parametrize("s", sorted(ZELLO_TO_RF_STATES, key=lambda x: x.value))
    def test_zello_to_rf_states(self, s):
        assert direction_of(s) is Direction.ZELLO_TO_RF

    @pytest.mark.parametrize("s", sorted(RF_TO_ZELLO_STATES, key=lambda x: x.value))
    def test_rf_to_zello_states(self, s):
        assert direction_of(s) is Direction.RF_TO_ZELLO

    def test_direction_sets_are_disjoint(self):
        assert not (ZELLO_TO_RF_STATES & RF_TO_ZELLO_STATES)

    def test_failsafe_has_no_direction(self):
        assert direction_of(State.FAILSAFE) is Direction.NONE


class TestTransitions:
    def test_every_state_has_an_entry(self):
        for s in State:
            assert s in LEGAL_TRANSITIONS, f"{s} missing from the transition table"

    def test_idle_can_start_either_direction(self):
        assert is_legal(State.IDLE, State.ZELLO_TO_RF_PREKEY)
        assert is_legal(State.IDLE, State.RF_TO_ZELLO_START)

    def test_cannot_jump_between_directions(self):
        """The core half-duplex invariant."""
        for a in ZELLO_TO_RF_STATES:
            for b in RF_TO_ZELLO_STATES:
                assert not is_legal(a, b), f"{a} -> {b} would cross directions"
                assert not is_legal(b, a), f"{b} -> {a} would cross directions"

    def test_cannot_skip_prekey(self):
        """PTT must be asserted and settle before audio is released."""
        assert not is_legal(State.IDLE, State.ZELLO_TO_RF_ACTIVE)

    def test_cannot_skip_the_tail(self):
        assert not is_legal(State.ZELLO_TO_RF_ACTIVE, State.IDLE)

    def test_cannot_skip_the_hang(self):
        assert not is_legal(State.RF_TO_ZELLO_ACTIVE, State.IDLE)

    def test_prekey_may_abort_straight_to_tail(self):
        """A stream that stops during pre-key still has to unkey."""
        assert is_legal(State.ZELLO_TO_RF_PREKEY, State.ZELLO_TO_RF_TAIL)

    def test_start_may_abort_straight_to_hang(self):
        assert is_legal(State.RF_TO_ZELLO_START, State.RF_TO_ZELLO_HANG)

    def test_failsafe_reachable_from_everywhere(self):
        """A fail-safe must never be blocked by a transition check."""
        for s in State:
            assert is_legal(s, State.FAILSAFE), f"cannot fail-safe from {s}"

    def test_self_transition_is_idempotent(self):
        for s in State:
            assert is_legal(s, s)

    def test_failsafe_returns_only_to_idle(self):
        assert is_legal(State.FAILSAFE, State.IDLE)
        assert not is_legal(State.FAILSAFE, State.ZELLO_TO_RF_ACTIVE)
        assert not is_legal(State.FAILSAFE, State.RF_TO_ZELLO_ACTIVE)

    def test_tail_and_hang_return_to_idle(self):
        assert is_legal(State.ZELLO_TO_RF_TAIL, State.IDLE)
        assert is_legal(State.RF_TO_ZELLO_HANG, State.IDLE)

    def test_no_state_transitions_directly_to_a_start_state_except_idle(self):
        for src in State:
            if src in (State.IDLE, State.ZELLO_TO_RF_PREKEY, State.RF_TO_ZELLO_START):
                continue
            assert not is_legal(src, State.ZELLO_TO_RF_PREKEY)
            assert not is_legal(src, State.RF_TO_ZELLO_START)

    def test_every_path_reaches_idle(self):
        """No state may be a dead end -- a stuck state means a stuck PTT."""
        for start in State:
            seen, frontier = {start}, [start]
            while frontier:
                cur = frontier.pop()
                for nxt in LEGAL_TRANSITIONS[cur]:
                    if nxt not in seen:
                        seen.add(nxt)
                        frontier.append(nxt)
            assert State.IDLE in seen, f"{start} cannot reach IDLE"
