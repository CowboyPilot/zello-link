"""PTT safety: startup state, fail-safe, and the transmit watchdog (AT-02, AT-07)."""

from __future__ import annotations

import asyncio

import pytest

from zello_link.hardware.aioc_hid import Cm108Report
from zello_link.hardware.ptt import NullPtt, PttBackend, PttError, SafePtt


class FlakyPtt(NullPtt):
    """Backend that can be told to fail, to exercise the fail-safe paths."""

    def __init__(self, *, fail_unkey: bool = False, fail_key: bool = False) -> None:
        super().__init__()
        self.fail_unkey = fail_unkey
        self.fail_key = fail_key
        self.unkey_attempts = 0

    def key(self) -> None:
        if self.fail_key:
            raise PttError("simulated key failure")
        super().key()

    def unkey(self) -> None:
        self.unkey_attempts += 1
        if self.fail_unkey:
            raise PttError("simulated unkey failure")
        super().unkey()


@pytest.fixture
def ptt():
    backend = NullPtt()
    safe = SafePtt(backend, max_tx_s=180.0)
    safe.open()
    yield safe
    safe.close()


class TestStartupSafety:
    """AT-02: the radio must never key during service start."""

    def test_not_keyed_after_open(self, ptt):
        assert not ptt.is_keyed()

    def test_open_forces_unkey_even_if_backend_starts_keyed(self):
        backend = NullPtt()
        backend.open()
        backend.key()               # simulates a driver asserting the line
        safe = SafePtt(backend, max_tx_s=180.0)
        safe.open()
        assert not safe.is_keyed()

    def test_backend_rejects_key_before_open(self):
        with pytest.raises(PttError, match="before open"):
            NullPtt().key()

    def test_open_aborts_if_safe_state_cannot_be_established(self):
        """If PTT state is unknown at startup, refuse to run at all.

        Continuing would mean playing Zello audio into a radio whose transmit
        state we cannot control -- worse than not starting.
        """
        safe = SafePtt(FlakyPtt(fail_unkey=True), max_tx_s=180.0)
        with pytest.raises(PttError, match="simulated unkey failure"):
            safe.open()


class TestKeyUnkey:
    async def test_key_then_unkey(self, ptt):
        await ptt.key()
        assert ptt.is_keyed()
        await ptt.unkey()
        assert not ptt.is_keyed()

    async def test_double_key_is_idempotent(self, ptt):
        await ptt.key()
        await ptt.key()
        assert ptt._backend.key_count == 1

    async def test_unkey_when_idle_is_safe(self, ptt):
        await ptt.unkey()
        assert not ptt.is_keyed()

    async def test_cycle_accounting(self, ptt):
        for _ in range(3):
            await ptt.key()
            await ptt.unkey()
        assert ptt.key_cycles == 3
        assert ptt.total_keyed_s >= 0.0


class TestFailSafe:
    def test_fail_safe_unkeys(self):
        backend = NullPtt()
        backend.open()
        backend.key()
        safe = SafePtt(backend, max_tx_s=180.0)
        safe.fail_safe()
        assert not backend.is_keyed()

    def test_fail_safe_never_raises(self):
        """It is called from exception handlers and finally blocks."""
        backend = FlakyPtt(fail_unkey=True)
        backend.open()
        safe = SafePtt(backend, max_tx_s=180.0)
        safe.fail_safe()            # must not propagate
        assert backend.unkey_attempts == 1

    def test_close_never_raises(self):
        backend = FlakyPtt(fail_unkey=True)
        backend.open()
        SafePtt(backend, max_tx_s=180.0).close()

    def test_fail_safe_is_idempotent(self, ptt):
        ptt.fail_safe()
        ptt.fail_safe()
        assert not ptt.is_keyed()

    async def test_fail_safe_disarms_the_watchdog(self):
        backend = NullPtt()
        safe = SafePtt(backend, max_tx_s=0.05)
        safe.open()
        await safe.key()
        safe.fail_safe()
        await asyncio.sleep(0.12)
        assert safe.timeouts == 0, "a disarmed watchdog must not fire"


class TestTransmitWatchdog:
    """AT-07: a missing stop event cannot hold PTT past max_tx_s."""

    async def test_watchdog_unkeys_after_timeout(self):
        backend = NullPtt()
        safe = SafePtt(backend, max_tx_s=0.05)
        safe.open()
        await safe.key()
        assert safe.is_keyed()

        await asyncio.sleep(0.15)
        assert not backend.is_keyed(), "watchdog did not force PTT off"
        assert safe.timeouts == 1

    async def test_watchdog_does_not_fire_early(self):
        backend = NullPtt()
        safe = SafePtt(backend, max_tx_s=1.0)
        safe.open()
        await safe.key()
        await asyncio.sleep(0.05)
        assert backend.is_keyed()
        assert safe.timeouts == 0
        await safe.unkey()

    async def test_unkey_disarms_the_watchdog(self):
        backend = NullPtt()
        safe = SafePtt(backend, max_tx_s=0.05)
        safe.open()
        await safe.key()
        await safe.unkey()
        await asyncio.sleep(0.12)
        assert safe.timeouts == 0

    async def test_watchdog_rearms_on_each_key(self):
        backend = NullPtt()
        safe = SafePtt(backend, max_tx_s=0.08)
        safe.open()
        for _ in range(3):
            await safe.key()
            await asyncio.sleep(0.02)
            await safe.unkey()
        assert safe.timeouts == 0

        await safe.key()
        await asyncio.sleep(0.16)
        assert safe.timeouts == 1

    async def test_timeout_callback_fires(self):
        fired: list[float] = []
        safe = SafePtt(NullPtt(), max_tx_s=0.05, on_timeout=fired.append)
        safe.open()
        await safe.key()
        await asyncio.sleep(0.15)
        assert len(fired) == 1
        assert fired[0] >= 0.05

    async def test_timeout_callback_exception_does_not_break_unkey(self):
        def boom(_):
            raise RuntimeError("callback exploded")

        backend = NullPtt()
        safe = SafePtt(backend, max_tx_s=0.05, on_timeout=boom)
        safe.open()
        await safe.key()
        await asyncio.sleep(0.15)
        assert not backend.is_keyed(), "unkey must happen before the callback"

    async def test_watchdog_survives_a_failing_backend(self):
        backend = FlakyPtt()
        safe = SafePtt(backend, max_tx_s=0.05)
        safe.open()
        backend.fail_unkey = True       # hardware goes bad mid-transmission
        await safe.key()
        await asyncio.sleep(0.15)
        assert safe.timeouts == 1, "watchdog must count the timeout even if unkey fails"


class TestCm108Report:
    """The HID layout is bench-unverified, so its logic is pinned by tests."""

    def test_key_sets_the_gpio_bit(self):
        r = Cm108Report(gpio_pin=3)
        assert r.build(True) == bytes([0x00, 0x00, 0b100, 0b100])

    def test_unkey_clears_data_but_keeps_mask(self):
        """The line is driven low, not floated."""
        r = Cm108Report(gpio_pin=3)
        assert r.build(False) == bytes([0x00, 0x00, 0x00, 0b100])

    @pytest.mark.parametrize("pin,bit", [(1, 0b1), (2, 0b10), (3, 0b100), (4, 0b1000)])
    def test_pin_to_bit_mapping_is_one_based(self, pin, bit):
        assert Cm108Report(gpio_pin=pin).bit == bit

    def test_report_length(self):
        assert len(Cm108Report().build(True)) == 4

    def test_reads_cos_bit(self):
        r = Cm108Report()
        assert r.read_button(bytes([0b1000, 0, 0, 0]), button=4) is True
        assert r.read_button(bytes([0b0000, 0, 0, 0]), button=4) is False

    def test_cos_ignores_other_pins(self):
        r = Cm108Report()
        assert r.read_button(bytes([0b0111, 0, 0, 0]), button=4) is False

    def test_rejects_bad_pin(self):
        with pytest.raises(ValueError, match="gpio_pin"):
            Cm108Report(gpio_pin=0)
        with pytest.raises(ValueError, match="gpio_pin"):
            Cm108Report(gpio_pin=9)

    def test_rejects_overlapping_indices(self):
        with pytest.raises(ValueError, match="share a byte"):
            Cm108Report(data_index=2, mask_index=2)

    def test_rejects_index_outside_report(self):
        with pytest.raises(ValueError, match="outside"):
            Cm108Report(data_index=9)

    def test_rejects_short_report(self):
        with pytest.raises(ValueError, match="at least 4"):
            Cm108Report(length=2)

    def test_layout_is_swappable(self):
        """A bench finding must be a one-class change, not a code hunt."""
        r = Cm108Report(gpio_pin=1, data_index=3, mask_index=2)
        assert r.build(True) == bytes([0x00, 0x00, 0b1, 0b1])

    def test_truncated_input_report_raises(self):
        with pytest.raises(PttError, match="input report"):
            Cm108Report().read_button(b"", button=2)


class TestBackendContract:
    """Every backend must satisfy the same safety contract."""

    def test_null_backend_implements_the_abc(self):
        assert isinstance(NullPtt(), PttBackend)

    def test_close_is_idempotent(self):
        b = NullPtt()
        b.open()
        b.close()
        b.close()
        assert not b.is_keyed()

    def test_context_manager_closes(self):
        with NullPtt() as b:
            b.key()
            assert b.is_keyed()
        assert not b.is_keyed()
