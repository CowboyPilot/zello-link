"""AT-09: lose the USB interface mid-run, fail safe, and come back.

Modelled on a real failure: the AIOC dropped off the USB bus, every PTT call
raised OSError [Errno 6] Device not configured, and the bridge exited instead
of recovering.
"""

from __future__ import annotations

import asyncio
import copy

import numpy as np
import yaml

from tests.fakes import FakeAudioSink, FakeCos, FakeEncoder, FakeZello
from zello_link.config import load_config
from zello_link.controller import BridgeController
from zello_link.hardware.ptt import NullPtt, PttError, SafePtt
from zello_link.hardware.supervisor import HardwareSupervisor

BASE = {
    "config_version": 1,
    "instance": {"name": "recovery"},
    "zello": {"channel": "C", "username": "u", "auth_token": "tok-abcdef"},
    "sound": {"input_device": "in", "output_device": "out"},
    "ptt": {"mode": "none", "max_tx_s": 5.0},
    "hardware": {"retry_initial_s": 0.01, "retry_max_s": 0.05},
    "logging": {"console": False, "file": None},
}


def make_config(tmp_path, **overrides):
    data = copy.deepcopy(BASE)
    for section, values in overrides.items():
        data.setdefault(section, {}).update(values)
    p = tmp_path / "bridge.yaml"
    p.write_text(yaml.safe_dump(data))
    return load_config(p)


class VanishingEngine(FakeAudioSink):
    """Audio engine whose device disappears after N blocks.

    Reproduces the real fault: the generator raises, and every subsequent
    open() fails until the device is declared back.
    """

    def __init__(self, *, fail_after=3, blocks_before_end=1000):
        super().__init__()
        self.fail_after = fail_after
        self.blocks_before_end = blocks_before_end
        self.present = True
        self.opens = 0
        self.closes = 0
        self.open_failures = 0
        self._served = 0

    def open(self):
        if not self.present:
            self.open_failures += 1
            raise OSError(6, "Device not configured")
        self.opens += 1

    def close(self):
        self.closes += 1

    async def capture_blocks(self):
        served_this_session = 0
        while served_this_session < self.blocks_before_end:
            if self.fail_after is not None and self._served >= self.fail_after:
                self.fail_after = None
                raise OSError(6, "Device not configured")
            self._served += 1
            served_this_session += 1
            await asyncio.sleep(0)
            yield np.zeros(320, dtype=np.int16)

    def stats(self):
        return {}


class VanishingPtt(NullPtt):
    """PTT backend that fails like a serial port whose device is gone."""

    def __init__(self):
        super().__init__()
        self.present = True
        self.opens = 0
        self.unkey_failures = 0

    def open(self):
        if not self.present:
            raise PttError("cannot open: [Errno 6] Device not configured")
        self.opens += 1
        super().open()

    def unkey(self):
        if not self.present:
            self.unkey_failures += 1
            raise PttError("cannot release PTT: [Errno 6] Device not configured")
        super().unkey()

    def close(self):
        if not self.present:
            return
        super().close()


def build(cfg, engine, ptt_backend):
    ptt = SafePtt(ptt_backend, max_tx_s=cfg.ptt.max_tx_s)
    ctrl = BridgeController(
        cfg, zello=FakeZello(), audio=engine, ptt=ptt, cos=FakeCos()
    )
    ctrl._encoder = FakeEncoder()
    sup = HardwareSupervisor(cfg, engine=engine, ptt=ptt, controller=ctrl)
    return ctrl, sup, ptt


class TestDeviceLossAndRecovery:
    async def test_recovers_when_the_device_returns(self, tmp_path):
        cfg = make_config(tmp_path)
        engine, backend = VanishingEngine(fail_after=3), VanishingPtt()
        ctrl, sup, ptt = build(cfg, engine, backend)
        ptt.open()

        shutdown = asyncio.Event()
        task = asyncio.create_task(sup.run(shutdown))
        await asyncio.sleep(0.15)

        assert sup.device_losses == 1
        assert sup.recoveries >= 1, "supervisor never reopened the device"
        assert sup.healthy

        shutdown.set()
        await asyncio.wait_for(task, timeout=2)

    async def test_keeps_retrying_while_the_device_is_absent(self, tmp_path):
        cfg = make_config(tmp_path)
        engine, backend = VanishingEngine(fail_after=2), VanishingPtt()
        ctrl, sup, ptt = build(cfg, engine, backend)
        ptt.open()

        engine.present = False          # device stays gone
        backend.present = False

        shutdown = asyncio.Event()
        task = asyncio.create_task(sup.run(shutdown))
        await asyncio.sleep(0.2)

        assert sup.failed_attempts >= 2, "supervisor gave up too early"
        assert not sup.healthy

        # Device comes back.
        engine.present = True
        backend.present = True
        await asyncio.sleep(0.2)
        assert sup.recoveries >= 1

        shutdown.set()
        await asyncio.wait_for(task, timeout=2)

    async def test_ptt_is_released_before_teardown(self, tmp_path):
        """The transmitter must not stay keyed across a device fault."""
        cfg = make_config(tmp_path)
        engine, backend = VanishingEngine(fail_after=2), VanishingPtt()
        ctrl, sup, ptt = build(cfg, engine, backend)
        ptt.open()
        await ptt.key()
        assert ptt.is_keyed()

        shutdown = asyncio.Event()
        task = asyncio.create_task(sup.run(shutdown))
        await asyncio.sleep(0.15)

        assert not ptt.is_keyed(), "PTT still keyed after a device loss"

        shutdown.set()
        await asyncio.wait_for(task, timeout=2)

    async def test_failing_unkey_does_not_stop_recovery(self, tmp_path):
        """The exact live failure: unkey raises because the device is gone."""
        cfg = make_config(tmp_path)
        engine, backend = VanishingEngine(fail_after=2), VanishingPtt()
        ctrl, sup, ptt = build(cfg, engine, backend)
        ptt.open()
        await ptt.key()

        backend.present = False         # unkey will now raise
        engine.present = False

        shutdown = asyncio.Event()
        task = asyncio.create_task(sup.run(shutdown))
        await asyncio.sleep(0.15)

        assert backend.unkey_failures >= 1, "test did not exercise the failing unkey"
        assert sup.device_losses >= 1

        engine.present = backend.present = True
        await asyncio.sleep(0.2)
        assert sup.recoveries >= 1, "a failing unkey blocked recovery"

        shutdown.set()
        await asyncio.wait_for(task, timeout=2)

    async def test_never_leaves_a_half_open_interface(self, tmp_path):
        """Audio opening but PTT failing must not leave audio running."""
        cfg = make_config(tmp_path)
        engine, backend = VanishingEngine(fail_after=2), VanishingPtt()
        ctrl, sup, ptt = build(cfg, engine, backend)
        ptt.open()

        backend.present = False         # audio recovers, PTT does not
        shutdown = asyncio.Event()
        task = asyncio.create_task(sup.run(shutdown))
        await asyncio.sleep(0.15)

        assert engine.closes >= engine.opens, (
            "audio device left open while PTT is unavailable"
        )

        shutdown.set()
        await asyncio.wait_for(task, timeout=2)

    async def test_gives_up_after_max_attempts(self, tmp_path):
        cfg = make_config(tmp_path, hardware={"max_attempts": 3})
        engine, backend = VanishingEngine(fail_after=1), VanishingPtt()
        ctrl, sup, ptt = build(cfg, engine, backend)
        ptt.open()
        engine.present = backend.present = False

        shutdown = asyncio.Event()
        task = asyncio.create_task(sup.run(shutdown))
        await asyncio.wait_for(task, timeout=3)

        assert shutdown.is_set(), "supervisor should signal shutdown after giving up"
        assert sup.failed_attempts == 3

    async def test_unlimited_retries_by_default(self, tmp_path):
        assert make_config(tmp_path).hardware.max_attempts == 0

    async def test_shutdown_stops_retrying_promptly(self, tmp_path):
        cfg = make_config(tmp_path, hardware={"retry_initial_s": 5.0, "retry_max_s": 5.0})
        engine, backend = VanishingEngine(fail_after=1), VanishingPtt()
        ctrl, sup, ptt = build(cfg, engine, backend)
        ptt.open()
        engine.present = backend.present = False

        shutdown = asyncio.Event()
        task = asyncio.create_task(sup.run(shutdown))
        await asyncio.sleep(0.05)
        shutdown.set()
        # Must not wait out the full 5 s backoff.
        await asyncio.wait_for(task, timeout=1.0)


class TestControllerErrorsAreNotDeviceLosses:
    async def test_handler_error_does_not_tear_down_hardware(self, tmp_path):
        """A controller bug must fail safe, not reopen working hardware."""
        cfg = make_config(tmp_path)
        engine, backend = VanishingEngine(fail_after=None), VanishingPtt()
        ctrl, sup, ptt = build(cfg, engine, backend)
        ptt.open()

        calls = {"n": 0}

        async def boom(_block):
            calls["n"] += 1
            raise RuntimeError("controller bug")

        sup._on_block = boom
        shutdown = asyncio.Event()
        task = asyncio.create_task(sup.run(shutdown))
        await asyncio.sleep(0.1)

        assert calls["n"] > 1, "supervisor stopped serving after one handler error"
        assert sup.device_losses == 0, "a handler error was misread as device loss"
        assert ctrl.stats.faults >= 1

        shutdown.set()
        await asyncio.wait_for(task, timeout=2)


class TestStats:
    async def test_stats_surface_recovery_counters(self, tmp_path):
        cfg = make_config(tmp_path)
        engine, backend = VanishingEngine(fail_after=2), VanishingPtt()
        _, sup, ptt = build(cfg, engine, backend)
        ptt.open()

        shutdown = asyncio.Event()
        task = asyncio.create_task(sup.run(shutdown))
        await asyncio.sleep(0.15)
        s = sup.stats()
        assert s["hw_device_losses"] == 1
        assert s["hw_recoveries"] >= 1
        assert s["hw_healthy"] is True

        shutdown.set()
        await asyncio.wait_for(task, timeout=2)


class TestPortAudioReinit:
    """A replugged USB device needs PortAudio itself restarted.

    Observed live: the AIOC returned as /dev/cu.usbmodemcab102015, but every
    reopen still failed with paInternalError (-9986) because PortAudio had
    cached the device list from before the unplug.
    """

    class ReinitEngine(VanishingEngine):
        """Only opens successfully after the backend has been reset."""

        def __init__(self, **kw):
            super().__init__(**kw)
            self.backend_resets = 0
            self.reset_since_close = False

        def reset_backend(self):
            self.backend_resets += 1
            self.reset_since_close = True

        def open(self):
            if not self.reset_since_close:
                raise OSError("Internal PortAudio error [PaErrorCode -9986]")
            super().open()

        def close(self):
            super().close()
            self.reset_since_close = False

    async def test_backend_is_reset_before_reopen(self, tmp_path):
        cfg = make_config(tmp_path)
        engine, backend = self.ReinitEngine(fail_after=2), VanishingPtt()
        _, sup, ptt = build(cfg, engine, backend)
        ptt.open()

        shutdown = asyncio.Event()
        task = asyncio.create_task(sup.run(shutdown))
        await asyncio.sleep(0.2)

        assert engine.backend_resets >= 1, "PortAudio was never re-initialised"
        assert sup.recoveries >= 1, "recovery failed without a backend reset"

        shutdown.set()
        await asyncio.wait_for(task, timeout=2)

    async def test_reset_happens_on_every_attempt(self, tmp_path):
        """A device that returns late must still get a fresh device list."""
        cfg = make_config(tmp_path)
        engine, backend = self.ReinitEngine(fail_after=1), VanishingPtt()
        _, sup, ptt = build(cfg, engine, backend)
        ptt.open()
        engine.present = False

        shutdown = asyncio.Event()
        task = asyncio.create_task(sup.run(shutdown))
        await asyncio.sleep(0.2)
        assert engine.backend_resets >= 2, "reset must precede each retry"

        engine.present = True
        await asyncio.sleep(0.2)
        assert sup.recoveries >= 1

        shutdown.set()
        await asyncio.wait_for(task, timeout=2)

    async def test_engine_without_reset_backend_still_recovers(self, tmp_path):
        """The hook is optional; a fake engine lacking it must not break."""
        cfg = make_config(tmp_path)
        engine, backend = VanishingEngine(fail_after=2), VanishingPtt()
        assert not hasattr(engine, "reset_backend")
        _, sup, ptt = build(cfg, engine, backend)
        ptt.open()

        shutdown = asyncio.Event()
        task = asyncio.create_task(sup.run(shutdown))
        await asyncio.sleep(0.2)
        assert sup.recoveries >= 1

        shutdown.set()
        await asyncio.wait_for(task, timeout=2)
