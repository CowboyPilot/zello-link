"""Standalone chan_usrp loopback/probe utility (spec milestone U1).

Exercises the USRP transport against a real AllStarLink node before any of it
is wired into the bridge, so a failure here is unambiguously the UDP path.

  python -m zello_link.usrp.probe --asl-host 10.0.0.5

Listening is the default and is completely inert. Transmitting is behind an
explicit flag because audio sent to a node's USRP rxchannel arrives as
RECEIVED audio: with duplex>=2 it is repeated locally, and if the node is
linked, it reaches every connected node and their RF.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import struct
import sys
import time

from .protocol import FRAME_MS, SAMPLE_RATE, VOICE_SAMPLES
from .transport import UsrpEvents, UsrpTransport


def tone_frame(freq_hz: float, phase: float, amplitude: float = 0.25) -> tuple[bytes, float]:
    """One 20 ms frame of a sine, carrying phase across frames.

    Phase continuity matters: restarting the sine each frame produces a click
    every 20 ms, which sounds like a fault rather than a test tone.
    """
    step = 2.0 * math.pi * freq_hz / SAMPLE_RATE
    samples = []
    for _ in range(VOICE_SAMPLES):
        samples.append(int(amplitude * 32767 * math.sin(phase)))
        phase = (phase + step) % (2.0 * math.pi)
    return struct.pack(f"<{VOICE_SAMPLES}h", *samples), phase


def _rms_dbfs(pcm: bytes) -> float:
    if not pcm:
        return -99.0
    vals = struct.unpack(f"<{len(pcm) // 2}h", pcm)
    mean_sq = sum(v * v for v in vals) / len(vals)
    return 20.0 * math.log10(max(math.sqrt(mean_sq), 1.0) / 32768.0)


async def run(args: argparse.Namespace) -> int:
    rx_frames = 0
    rx_bytes = 0
    key_events = 0
    levels: list[float] = []
    started_rx = 0.0

    def on_key() -> None:
        nonlocal key_events, started_rx, rx_frames
        key_events += 1
        rx_frames = 0
        started_rx = time.monotonic()
        print("  << KEY   from ASL", file=sys.stderr)

    def on_audio(pcm: bytes) -> None:
        nonlocal rx_frames, rx_bytes
        rx_frames += 1
        rx_bytes += len(pcm)
        level = _rms_dbfs(pcm)
        levels.append(level)
        if rx_frames % 25 == 0:          # ~every 500 ms
            print(f"     .. {rx_frames:4d} frames  {level:6.1f} dBFS", file=sys.stderr)

    def on_unkey() -> None:
        dur = time.monotonic() - started_rx if started_rx else 0.0
        peak = max(levels[-rx_frames:], default=-99.0) if rx_frames else -99.0
        print(
            f"  << UNKEY  {rx_frames} frames, {dur:.2f}s, peak {peak:.1f} dBFS",
            file=sys.stderr,
        )

    transport = UsrpTransport(
        bind_host=args.bind_host,
        bind_port=args.bind_port,
        asl_host=args.asl_host,
        asl_port=args.asl_port,
        strict_source=not args.any_source,
        rx_unkey_timeout_ms=args.rx_unkey_timeout_ms,
        events=UsrpEvents(on_key=on_key, on_audio=on_audio, on_unkey=on_unkey),
    )

    try:
        await transport.start()
    except OSError as e:
        print(f"cannot bind {args.bind_host}:{args.bind_port}: {e}", file=sys.stderr)
        return 3

    print(
        f"listening on {args.bind_host}:{args.bind_port}, "
        f"peer {args.asl_host}:{args.asl_port}"
        + ("" if args.transmit else "   (listen-only)"),
        file=sys.stderr,
    )
    print("Key up on the AllStar node to see receive traffic. Ctrl-C to stop.\n",
          file=sys.stderr)

    try:
        if args.transmit:
            await _transmit(transport, args)
        if args.duration:
            await asyncio.sleep(args.duration)
        else:
            await asyncio.Event().wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await transport.stop()

    s = transport.stats
    print("\n--- summary ---", file=sys.stderr)
    print(f"  tx packets        {s.tx_packets}", file=sys.stderr)
    print(f"  rx packets        {s.rx_packets}  ({s.rx_voice} voice, {s.rx_signal} signal)",
          file=sys.stderr)
    print(f"  key events        {key_events}", file=sys.stderr)
    print(f"  sequence gaps     {s.sequence_gaps}", file=sys.stderr)
    print(f"  duplicates        {s.duplicates}", file=sys.stderr)
    print(f"  malformed         {s.malformed_packets}", file=sys.stderr)
    print(f"  from other hosts  {s.foreign_packets}", file=sys.stderr)
    print(f"  forced unkeys     {s.forced_unkeys}", file=sys.stderr)
    if levels:
        print(f"  rx level          mean {sum(levels)/len(levels):.1f} dBFS, "
              f"peak {max(levels):.1f} dBFS", file=sys.stderr)
    if s.rx_packets == 0:
        print("\n  No packets received. Check that the node's rxchannel is the USRP\n"
              "  one (a second rxchannel line will shadow it), that MYPORT matches\n"
              "  --bind-port, and that nothing is filtering UDP between the hosts.",
              file=sys.stderr)
    return 0


async def _transmit(transport: UsrpTransport, args: argparse.Namespace) -> None:
    """Send a phase-continuous test tone, then an explicit unkey."""
    print(f"*** TRANSMITTING {args.tone}Hz for {args.transmit}s to the node ***",
          file=sys.stderr)
    print("*** This arrives as RECEIVED audio and reaches any linked nodes. ***",
          file=sys.stderr)
    print("Ctrl-C within 3 seconds to abort.", file=sys.stderr)
    await asyncio.sleep(3.0)

    frames = int(args.transmit * 1000 / FRAME_MS)
    phase = 0.0
    tick = time.monotonic()
    for i in range(frames):
        pcm, phase = tone_frame(args.tone, phase, args.amplitude)
        transport.send_voice(pcm)
        if (i + 1) % 25 == 0:
            print(f"     >> {i + 1:4d}/{frames} frames sent", file=sys.stderr)
        tick += FRAME_MS / 1000.0
        await asyncio.sleep(max(0.0, tick - time.monotonic()))

    transport.send_unkey()
    print("  >> UNKEY sent", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m zello_link.usrp.probe",
        description="Probe an AllStarLink chan_usrp channel over UDP.",
    )
    p.add_argument("--bind-host", default="0.0.0.0",
                   help="local address to bind (default: all interfaces)")
    p.add_argument("--bind-port", type=int, default=34001,
                   help="local UDP port; the HISPORT in the node's rxchannel")
    p.add_argument("--asl-host", required=True, help="the AllStarLink host")
    p.add_argument("--asl-port", type=int, default=32001,
                   help="the node's MYPORT, i.e. the port Asterisk listens on")
    p.add_argument("--any-source", action="store_true",
                   help="accept datagrams from any host (disables source checking)")
    p.add_argument("--rx-unkey-timeout-ms", type=int, default=500)
    p.add_argument("--duration", type=float,
                   help="exit after this many seconds")
    p.add_argument("--transmit", type=float, metavar="SECONDS",
                   help="SEND a test tone for this long. The node treats it as "
                        "received audio and passes it to any linked nodes.")
    p.add_argument("--tone", type=float, default=1000.0, help="test tone Hz")
    p.add_argument("--amplitude", type=float, default=0.25,
                   help="test tone amplitude, 0..1 (default 0.25)")
    return p


def main(argv: list[str] | None = None) -> int:
    import logging

    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
