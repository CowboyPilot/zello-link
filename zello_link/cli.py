"""Command-line entry point.

Safety rules for the diagnostic modes (spec section 5):
  * ``--validate`` parses, checks, and reports. It never connects and never
    keys PTT.
  * ``--diagnose-aioc`` forces PTT OFF first and never keys the radio unless
    ``--ptt-test`` is passed explicitly.
  * ``--cos-monitor`` reads audio levels only. PTT is never armed.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import signal
import sys
import time
from typing import Any

from . import __version__
from .config import ConfigError, load_config
from .logging_setup import StatusLine, setup_logging

__all__ = ["main", "build_parser"]

log = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_CONFIG = 2
EXIT_DEVICE = 3
EXIT_RUNTIME = 4


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="zello-link",
        description="Bridge a Zello channel to a radio through a CM108-class USB interface.",
    )
    p.add_argument("--config", required=True, metavar="PATH", help="YAML configuration file")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    modes = p.add_argument_group("diagnostic modes (mutually exclusive)")
    g = modes.add_mutually_exclusive_group()
    g.add_argument(
        "--validate", action="store_true",
        help="parse and check the configuration, then exit without connecting or keying PTT",
    )
    g.add_argument(
        "--list-audio-devices", action="store_true",
        help="print available audio devices and exit",
    )
    g.add_argument(
        "--diagnose-aioc", action="store_true",
        help="report AIOC serial/HID state; forces PTT OFF and does not key the radio",
    )
    g.add_argument(
        "--cos-monitor", action="store_true",
        help="print rolling RX levels for threshold and gain calibration; never transmits",
    )

    p.add_argument(
        "--ptt-test", action="store_true",
        help="with --diagnose-aioc only: briefly key the transmitter. THE RADIO WILL TRANSMIT.",
    )
    p.add_argument(
        "--showmonitor", action="store_true",
        help="pin a live audio level meter to the bottom of this terminal while "
             "the bridge runs, for setting the radio's volume against real traffic",
    )
    p.add_argument(
        "--duration", type=float, metavar="SECONDS",
        help="stop --cos-monitor after this many seconds",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return EXIT_CONFIG

    # Diagnostic modes never touch the production log file: they are routinely
    # run by an operator who is not the service user.
    diagnostic = bool(
        args.validate or args.list_audio_devices or args.diagnose_aioc or args.cos_monitor
    )

    # The meter shares stderr with the log, so the console handler has to know
    # about it or log records would smear across the pinned line.
    status = StatusLine() if (args.showmonitor and not diagnostic) else None
    setup_logging(cfg, console_only=diagnostic, status=status)

    if args.ptt_test and not args.diagnose_aioc:
        print("--ptt-test is only valid with --diagnose-aioc", file=sys.stderr)
        return EXIT_CONFIG

    try:
        if args.list_audio_devices:
            return _list_audio_devices(cfg, args.config)
        if args.validate:
            return _validate(cfg)
        if args.diagnose_aioc:
            return asyncio.run(_diagnose_aioc(cfg, ptt_test=args.ptt_test))
        if args.cos_monitor:
            return asyncio.run(_cos_monitor(cfg, duration_s=args.duration))
        return asyncio.run(_run_bridge(cfg, status=status))
    except KeyboardInterrupt:
        return EXIT_OK


# -- diagnostic modes -----------------------------------------------------
def _list_audio_devices(cfg: Any, config_path: str) -> int:
    """List devices, and on a terminal offer to write the selection back.

    Non-interactive (piped, redirected, CI) prints the table and stops --
    blocking on input would hang a script that only wanted the listing.
    """
    from .diagnostics.setup_wizard import run_device_setup

    try:
        return run_device_setup(cfg, config_path)
    except Exception as e:
        print(f"device error: {e}", file=sys.stderr)
        return EXIT_DEVICE


def _validate(cfg: Any) -> int:
    """Check everything checkable without connecting or keying."""
    from .audio.devices import DeviceError, resolve_device
    from .diagnostics.status import render_validation

    report: list[str] = []
    problems = 0

    for selector, kind in (
        (cfg.sound.input_device, "input"),
        (cfg.sound.output_device, "output"),
    ):
        if selector is None:
            continue
        try:
            device = resolve_device(selector, kind)
            report.append(f"{kind:<7} OK   [{device.index}] {device.name}")
        except DeviceError as e:
            problems += 1
            report.append(f"{kind:<7} FAIL {e}")

    if cfg.ptt.mode == "serial" and cfg.ptt.tty_device:
        from pathlib import Path

        exists = Path(cfg.ptt.tty_device).exists()
        report.append(f"ptt tty {'OK  ' if exists else 'FAIL'} {cfg.ptt.tty_device}")
        problems += 0 if exists else 1

    print(render_validation(cfg, device_report=report))

    if problems:
        print(f"\n{problems} device problem(s) found", file=sys.stderr)
        return EXIT_DEVICE
    return EXIT_OK


async def _probe_hid_input(cfg: Any, *, seconds: float = 3.0) -> list[str]:
    """Check that CM108 input reports actually reach us.

    Reading them is not universally permitted. On macOS the AIOC enumerates
    as a Consumer Control device (usage page 0x0c), which the system gates
    behind Privacy & Security > Input Monitoring; writing output reports for
    PTT is unrestricted, so PTT works while COS stays silent forever.
    """
    import platform

    out: list[str] = []
    try:
        from .hardware.aioc_hid import HidCos
    except Exception as e:                                  # pragma: no cover
        return [f"HID COS unavailable: {e}"]

    cos = HidCos(cfg.cos.hid_device, button=cfg.cos.hid_button)
    try:
        cos.open()
    except Exception as e:
        return [f"cannot open HID COS device: {e}"]

    out.append(f"COS over HID: watching button {cfg.cos.hid_button} "
               f"for {seconds:.0f}s -- open the radio's squelch now")
    seen_any = False
    states: set[bool] = set()
    loop = asyncio.get_running_loop()
    end = loop.time() + seconds
    while loop.time() < end:
        try:
            state = cos.poll()
        except Exception as e:
            out.append(f"  HID read failed: {e}")
            break
        if cos.reads:
            seen_any = True
            states.add(state)
        await asyncio.sleep(0.02)
    cos.close()

    if not seen_any:
        out.append("  NO input reports received.")
        if platform.system() == "Darwin":
            out.append("  On macOS the AIOC is a Consumer Control HID device, which")
            out.append("  needs Privacy & Security > Input Monitoring for your")
            out.append("  terminal. PTT still works without it; only COS is blocked.")
            out.append("  If that does not help, use cos.mode='internal_audio' here")
            out.append("  and verify HID COS on a Linux host.")
        else:
            out.append("  Check that VCOS is enabled and mapped to a CM108 button in")
            out.append("  the AIOC's configuration, and that the button number matches")
            out.append(f"  cos.hid_button ({cfg.cos.hid_button}).")
    elif len(states) < 2:
        out.append(f"  input reports arriving, but COS never changed "
                   f"(stuck {'closed' if False in states else 'open'}).")
        out.append(f"  Check VCOS is mapped to button {cfg.cos.hid_button} "
                   f"and its level threshold is low enough to trip.")
    else:
        out.append("  OK: COS both asserted and released.")
    return out


async def _diagnose_aioc(cfg: Any, *, ptt_test: bool) -> int:
    """Report AIOC state. PTT is forced OFF before anything else happens."""
    from .hardware.ptt import PttError, SafePtt, create_ptt_backend

    print(f"ptt mode: {cfg.ptt.mode}")

    try:
        backend = create_ptt_backend(cfg)
    except PttError as e:
        print(f"cannot create PTT backend: {e}", file=sys.stderr)
        return EXIT_DEVICE

    ptt = SafePtt(backend, max_tx_s=cfg.ptt.max_tx_s)

    try:
        ptt.open()          # establishes PTT OFF before anything else
        print("PTT forced OFF and device opened successfully")
    except PttError as e:
        print(f"cannot open PTT device: {e}", file=sys.stderr)
        return EXIT_DEVICE

    try:
        from .hardware.aioc_hid import find_cm108_hid_devices

        devices = find_cm108_hid_devices()
        if devices:
            print("CM108-class HID interfaces:")
            for d in devices:
                print(f"  {d['path']}  {d['vid']:04x}:{d['pid']:04x}  {d['name']}")
        else:
            print("CM108-class HID interfaces: none found")
    except Exception as e:
        print(f"HID enumeration unavailable: {e}")

    # COS over HID either works or fails silently: poll() keeps returning the
    # last known state, so a bridge that can never hear the radio looks
    # perfectly healthy. Prove input reports actually arrive.
    if cfg.cos.mode in ("aioc_virtual", "aioc_hardware"):
        print()
        for line in await _probe_hid_input(cfg):
            print(line)

    try:
        if ptt_test:
            print("\n*** --ptt-test: THE RADIO WILL TRANSMIT FOR 1 SECOND ***")
            print("Ctrl-C within 3 seconds to abort.")
            await asyncio.sleep(3.0)
            print("keying...")
            await ptt.key()
            await asyncio.sleep(1.0)
            await ptt.unkey()
            print("unkeyed")
        else:
            print("\nPTT was not keyed. Pass --ptt-test to transmit.")
    except KeyboardInterrupt:
        print("\naborted")
    finally:
        ptt.close()

    return EXIT_OK


async def _cos_monitor(cfg: Any, *, duration_s: float | None) -> int:
    """Level monitor for threshold/gain calibration. Never keys, never sends."""
    from .audio.engine import AudioEngine
    from .diagnostics.status import cos_monitor
    from .hardware.cos import create_cos_backend

    engine = AudioEngine(cfg)
    cos = create_cos_backend(cfg)

    try:
        engine.open()
        cos.open()
    except Exception as e:
        print(f"cannot open audio: {e}", file=sys.stderr)
        return EXIT_DEVICE

    try:
        return await cos_monitor(cfg, engine=engine, cos=cos, duration_s=duration_s)
    except KeyboardInterrupt:
        return EXIT_OK
    finally:
        cos.close()
        engine.close()


async def _status_display(
    cfg: Any, *, controller: Any, ptt: Any, status: Any, shutdown: Any
) -> None:
    """Repaint the pinned level meter ~10x a second.

    Deliberately a poller rather than a hook on the capture path: the audio
    path must not take on terminal I/O, and a dropped frame of the display
    costs nothing.
    """
    from .diagnostics.status import PEAK_HOLD_S, RollingPeak, render_status_line

    peak_hold = RollingPeak(PEAK_HOLD_S)
    threshold = cfg.cos.threshold_dbfs if cfg.cos.mode == "internal_audio" else None
    last_clipped = 0

    try:
        while not shutdown.is_set():
            await asyncio.sleep(0.1)
            stats = getattr(controller, "last_block_stats", None)
            if stats is None:
                continue

            peak = peak_hold.add(stats.peak_dbfs, time.monotonic())
            total_clipped = controller.stats.clipped_rx_samples
            recent_clip = total_clipped - last_clipped
            last_clipped = total_clipped

            status.set(
                render_status_line(
                    stats.rms_dbfs,
                    peak,
                    threshold=threshold,
                    clipped=recent_clip,
                    cos_active=controller.cos.active,
                    keyed=ptt.is_keyed(),
                    color=status.enabled,
                )
            )
    except asyncio.CancelledError:
        raise
    finally:
        status.finish()


# -- normal operation -----------------------------------------------------
async def _run_bridge(cfg: Any, *, status: StatusLine | None = None) -> int:
    from .audio.engine import AudioEngine
    from .controller import BridgeController
    from .diagnostics.metrics import MetricsReporter
    from .hardware.cos import create_cos_backend
    from .hardware.ptt import SafePtt, create_ptt_backend
    from .hardware.supervisor import HardwareSupervisor
    from .zello.client import ZelloClient

    for warning in cfg.warnings():
        log.warning("config: %s", warning)

    loop = asyncio.get_running_loop()
    shutdown = asyncio.Event()

    # The USRP backend has no sound card, no PTT line and no COS, so none of
    # that hardware is constructed for it -- a USRP host need not even have
    # PortAudio installed.
    usrp_mode = cfg.bridge.backend == "usrp"

    engine = cos = ptt = None
    if not usrp_mode:
        # A device fault is handed to the supervisor, which fails safe and
        # retries. Setting shutdown here would exit on a recoverable USB glitch.
        engine = AudioEngine(cfg)
        cos = create_cos_backend(cfg)
        ptt = SafePtt(create_ptt_backend(cfg), max_tx_s=cfg.ptt.max_tx_s)

    controller: BridgeController | None = None
    client: ZelloClient | None = None

    def _signal_handler(signame: str) -> None:
        log.info("received %s; shutting down", signame)
        shutdown.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, _signal_handler, sig.name)

    try:
        client = ZelloClient(cfg)

        if usrp_mode:
            from .backends.usrp import UsrpBackend

            radio = UsrpBackend(cfg)
        else:
            from .backends.aioc import AiocBackend

            radio = AiocBackend(cfg, engine=engine, ptt=ptt, cos=cos)

        controller = BridgeController(
            cfg, zello=client, audio=engine, ptt=ptt, cos=cos, backend=radio
        )

        # Wire the client's callbacks to the controller now that both exist.
        client.set_handlers(
            on_stream_start=controller.on_zello_stream_start,
            on_audio=controller.on_zello_audio,
            on_stream_stop=controller.on_zello_stream_stop,
            on_disconnected=controller.on_zello_disconnected,
        )
        if ptt is not None:
            ptt.set_timeout_callback(controller.on_ptt_timeout)
        if engine is not None:
            engine.open()
        await controller.start()

        metrics = MetricsReporter(
            cfg, controller=controller, audio=engine, zello=client, ptt=ptt
        )
        metrics.start()

        # The hardware supervisor watches a USB device; USRP has none, so
        # its failure mode is the socket, handled by the transport itself.
        supervisor = None
        if not usrp_mode:
            supervisor = HardwareSupervisor(
                cfg, engine=engine, ptt=ptt, controller=controller
            )
            metrics.supervisor = supervisor

        tasks = [
            asyncio.create_task(client.run(), name="zello"),
            *([asyncio.create_task(supervisor.run(shutdown), name="hardware")]
              if supervisor is not None else []),
            asyncio.create_task(shutdown.wait(), name="shutdown"),
        ]
        if status is not None:
            log.info("live level meter enabled (--showmonitor)")
            tasks.append(
                asyncio.create_task(
                    _status_display(
                        cfg, controller=controller, ptt=ptt,
                        status=status, shutdown=shutdown,
                    ),
                    name="status",
                )
            )

        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

        for task in done:
            exc = task.exception() if not task.cancelled() else None
            if exc is not None:
                log.error("task %s failed: %s", task.get_name(), exc, exc_info=exc)

        await metrics.stop()

    except Exception as e:
        from .usrp.transport import UsrpBindError

        if isinstance(e, UsrpBindError):
            log.critical("%s", e)
            return EXIT_DEVICE
        log.critical("fatal error", exc_info=True)
        return EXIT_RUNTIME

    finally:
        # Shutdown order: stop streams, stop audio, unkey PTT, close hardware.
        if controller is not None:
            with contextlib.suppress(Exception):
                await controller.stop()
        if client is not None:
            with contextlib.suppress(Exception):
                await client.stop()
        if engine is not None:
            with contextlib.suppress(Exception):
                engine.close()
        if ptt is not None:
            ptt.close()
        if status is not None:
            status.finish()
        log.info("shutdown complete")

    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
