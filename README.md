# zello-link

Headless bridge between a Zello channel and a radio, over a CM108-class USB
interface such as the AIOC.

The radio side stays entirely the radio's business — RF, modulation, vocoding
(DMR, analogue FM, whatever it runs), talkgroups, and encryption. This service
transports audio and PTT/COS only. It never contains, requests, or handles an
encryption key.

Originally built against an encrypted DMR handheld, but nothing in it is
DMR-specific: any radio reachable through a CM108-style interface works. There
is also an AllStarLink backend over `chan_usrp`, which needs no radio attached
at all — it bridges a Zello channel straight onto an ASL node.

## What it does

```
Zello channel  <--TLS WebSocket-->  zello-link  <--USB audio + PTT/COS-->  radio
```

Second backend, replacing the radio side entirely:

```
Zello channel  <--TLS WebSocket-->  zello-link  <--UDP/chan_usrp-->  AllStarLink
```

- Connects directly to the Zello Channel API. No Android app, no Waydroid, no
  GUI automation, no VOX, no MMDVM modem.
- Backend-agnostic core: arbitration, Opus, and the state machine do not know
  what is on the radio side.
- Half-duplex, first-talker-wins arbitration. The bridge never keys RF and
  transmits to Zello at the same time.
- Fails safe: PTT is driven OFF on startup, shutdown, exception, disconnect,
  device failure, SIGTERM, and watchdog timeout.

## Faster Funnier

**One shot install — copy/paste**

**Linux / Raspberry Pi OS** (no `git` needed):

```bash
wget -qO- https://github.com/CowboyPilot/zello-link/archive/refs/heads/main.tar.gz | tar xz && cd zello-link-main && ./quickstart.sh
```

**macOS** (ships `curl`, not `wget`):

```bash
curl -fsSL https://github.com/CowboyPilot/zello-link/archive/refs/heads/main.tar.gz | tar xz && cd zello-link-main && ./quickstart.sh
```

**With `git`, if you have it** — better if you intend to pull updates:

```bash
git clone https://github.com/CowboyPilot/zello-link.git && cd zello-link && ./quickstart.sh
```

That fetches the code, builds a virtualenv, installs the package with its
hardware extras, and then offers to run the setup wizard — which asks for your
Zello channel and account, which interface the radio is on, and which
directions to carry, and writes:

```
<name>.yaml     the config
<name>.env      your credentials, mode 600
run-<name>.sh   a launcher
```

Nothing is written until the last question is answered, and the wizard
validates its own output before telling you it worked. Then start the bridge:

```bash
./run-<name>.sh
```

The archive is **extracted before anything runs** — nothing is piped into a
shell. `quickstart.sh` lands on disk where you can read it, and if you would
rather look first, stop after `tar xz` and run it yourself. Every line of it
is also visible in this repo.

It **never sudo's**. `libopus`, PortAudio and hidapi are C libraries that need
root, so the installer detects what is missing and prints the exact command
for your platform, leaving that decision to you.

### Why a virtualenv

Not a development convenience — it is required on the platforms this targets,
and you never have to activate it.

Debian 12 and Raspberry Pi OS ship `/usr/lib/python3.11/EXTERNALLY-MANAGED`
(PEP 668), so `pip install` into the system Python is refused outright. Their
packaged dependencies are also too old: `python3-pydantic` is 1.10 where this
needs 2.6+, a different major API, and `python3-websockets` is 10.4 against a
12.0 floor.

`--break-system-packages` would install pydantic 2.x over the apt-managed
1.10 and break anything else on the host that depends on it — a poor trade on
an AllStarLink node. `pipx` works, but creates a virtualenv itself.

So the installer makes one, and everything points at it by absolute path: the
generated `run-<name>.sh`, and the systemd unit. There is nothing to activate,
and no shell profile to edit. It only shows up if you invoke the CLI by hand,
as `.venv/bin/zello-link` rather than `zello-link`.

Skip the wizard with `./quickstart.sh --no-wizard`, or add the test tooling
with `./quickstart.sh --dev`.

## Install

If you would rather do it in steps, or already have a checkout:

**Linux / macOS:**

```bash
./install.sh
```

**Windows:**

```
install.bat
```

The script checks for Python 3.11+, creates a virtual environment, installs
the package with its hardware extras, and verifies each component. Add `--dev`
for the test tooling.

It does **not** install system libraries silently — that needs root, and a
script that sudo's behind your back is not one you should run. `libopus`,
PortAudio and hidapi are C libraries, not Python wheels; the installer detects
which are missing and prints the exact command for your platform:

```bash
sudo apt install libopus0 libportaudio2 libhidapi-hidraw0   # Debian / RPi OS
brew install opus portaudio hidapi                          # macOS
```

On Windows, PortAudio ships inside the `sounddevice` wheel, but `opus.dll` has
to be obtained separately — the installer says so if it is missing.

The bridge imports all of these lazily, so `--validate` and the test suite run
without them.

### Choosing devices

With a config file in hand, this lists the audio devices and offers to write
your selection into it:

```bash
.venv/bin/zello-link --config my-bridge.yaml --list-audio-devices
```

It prompts for the input device, the output device, and the COS source —
offering the AIOC's HID-based options only when an AIOC is actually detected.
Edits are surgical: exactly the lines that change are rewritten, so the
comments in your config survive. The previous version is kept as `.bak`, and
the result is validated before anything is written.

Piped or redirected, it just prints the table and exits, so it stays usable
from scripts.

## Run

```bash
zello-link --config /etc/zello-link/west.yaml
```

Start from [`examples/bridge.yaml`](examples/bridge.yaml) — every option is
documented inline. The config path is mandatory; the bridge never assumes
`/dev/ttyACM0` or a default sound card.

### AllStarLink (USRP)

Set `bridge.backend: usrp` and the radio side becomes an ASL node over
`chan_usrp`. No sound card, no PTT line, no COS tuning: key state is carried
in the protocol. `zello-link --setup` will ask for the node number and ports
and then print the AllStarLink side for you.

Three things decide whether this works, and all three fail silently:

- **Run it on the AllStarLink host, over loopback.** `chan_usrp` has no
  authentication or encryption whatsoever, and loopback also sidesteps the
  firewall. If you must go over a network, `usrp.strict_source` and
  `usrp.allow_remote_host` exist, and the path should be a VPN.
- **The node needs `duplex = 3`.** At `duplex` 0 or 1 an idle node never
  writes to the channel, and `chan_usrp` only delivers received audio during a
  write — so audio reaches ASL and the node never keys. Nothing logs a
  problem.
- **Open the port if the bridge is remote.** ASL3's `firewalld` zone permits
  `iax2` and `echolink` but not the USRP port, and rejected datagrams still
  show up in `tcpdump`, which makes this look like a `chan_usrp` bug.

`asterisk -rx "usrp show"` is the fastest diagnosis: `Read` stuck at 0 means
the packets never reached the socket (firewall), while `Read` climbing with
`Write` at 0 is the `duplex` problem.

A node has exactly **one** `rxchannel`. Adding a second line does not give it
two inputs — one silently wins — so the bridge wants its own node, linked to
your RF node.

### Diagnostics

All of these are inert with respect to the transmitter except `--ptt-test`.

| Command | Purpose |
|---|---|
| `--validate` | Parse, check ranges, resolve devices. Never connects, never keys. |
| `--list-audio-devices` | Exact device names and indices for the config. |
| `--cos-monitor` | Rolling RX level, peak, and clip count. Never transmits. |
| `--diagnose-aioc` | Forces PTT OFF, reports serial/HID state. |
| `--diagnose-aioc --ptt-test` | **Transmits for 1 second.** 3-second abort window. |

## Calibrating levels

Gain names are **radio-centric**, not sound-card-centric:

| Setting | Path |
|---|---|
| `sound.rx_gain_db` | Radio receive audio *into* the bridge → Zello |
| `sound.tx_gain_db` | Bridge audio *out to* the radio's mic → transmitted over RF |

Run `--cos-monitor` with the radio attached and squelched, note the idle floor,
then key a test call and note the peak. Set `cos.threshold_dbfs` between them.
The monitor reports a running clip count; any clipping at all means
`rx_gain_db` (or the radio's own output level) is too high.

The detector floors at **−90.3 dBFS**, so a threshold below that can never
trigger. `--validate` warns about this.

## Sample rates and resampling

Keeping `sound.sample_rate` equal to `opus.sample_rate` (both 16 kHz by
default) keeps a resampler out of the RF→Zello path. Other rates work — the
AIOC advertises 8 k through 48 k — and `--validate` reports the exact added
latency.

Inbound Zello audio is a different matter: **the far end chooses its own sample
rate**, declared in its `codec_header`. A peer transmitting at 8 kHz or 48 kHz
is resampled to the device rate per stream, automatically, even on a bridge
whose own rates match. Without that, remote audio would play at the wrong pitch
and speed into a keyed transmitter.

The converter is a streaming polyphase rational resampler (`audio/resample.py`)
carrying filter state across blocks, so a block boundary is never audible. It
adds group delay, which counts toward `bridge.latency_budget_ms`.

## Multiple bridges on one host

One process per channel/radio pair. Give each instance its own config with a
distinct `instance.name`, audio devices, TTY/HID path, log file, and
`refresh_token_file`.

```bash
sudo cp systemd/zello-link@.service /etc/systemd/system/
sudo systemctl enable --now zello-link@west
sudo systemctl enable --now zello-link@east
```

Prefer stable identifiers (`/dev/serial/by-id/...`) over `/dev/ttyACM0`, which
renumbers. An ambiguous partial audio-device name match is a startup error, not
a guess — two instances must never silently land on the same card.

## Security

- Secrets support `${ENV_VAR}` indirection. An unset variable is a startup
  error, never an empty credential.
- Passwords and tokens are held in `SecretStr` and scrubbed from all log
  output, including tracebacks, by the log formatter.
- The refresh token is written atomically with `0600` permissions.
- TLS verification is always on; there is no insecure bypass option.
- Run as an unprivileged user in only the `audio`, `dialout`, and `plugdev`
  groups. Prefer udev rules over root.
- Production Friends & Family auth should follow Zello's guidance: private
  signing keys stay on a token service, not in this application. Development
  tokens expire after 30 days.

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest
```

The suite runs with no radio, no sound card, and no network. Codec tests skip
themselves when libopus is absent. `tests/fakes.py` provides fake Zello, audio,
PTT, and COS backends so the full controller is integration-tested without RF
hardware.

## Known issues

### RF ingress on the USB cable (hardware)

**Observed twice on the bench, both times while transmitting.** The AIOC
disappeared from the USB bus mid-run — `OSError: [Errno 6] Device not
configured`, with the device gone from `/dev/` entirely — then re-enumerated
normally after a replug. This is the classic symptom of RF energy from the
transmitter getting into the USB cable and crashing the device, and it happened
even into a dummy load.

Worth knowing: **the AIOC must have USB power to hold the radio keyed**, so a
dropout releases PTT in hardware. The worst case of losing the interface
mid-transmission is a truncated over, not a latched transmitter — verified on
the bench, including while RF-locked. `ptt.max_tx_s` is a second line of
defence rather than the only one.

The bridge survives it: the hardware supervisor fails safe (PTT off first),
then retries audio and PTT together with exponential backoff until the device
returns. `hw_recoveries` in the metrics line counts how often that has happened,
and a non-zero value in normal operation is a signal to go fix the cabling.

**But recovery is mitigation, not a fix.** A bridge that loses its interface
every few transmissions is not deployable. Remedies, in order of effort:

- Ferrite chokes on the USB cable, close to the AIOC end
- Keep the USB cable away from the antenna, feedline, and dummy load
- Shorter and/or better-shielded USB cable; avoid unshielded hubs
- Improve station grounding
- Reduce transmit power if the deployment permits

Worth resolving before an event: the failure is silent from the operator's
point of view apart from a log line, and while the bridge recovers, traffic
during the outage is lost.

### COS on a 2-pin (K1) connector

A Kenwood K1 connector carries speaker audio and mic/PTT only — there is no
squelch or COS line on it, so no interface can provide hardware COS from that
radio. Audio-level COS (`cos.mode: internal_audio`) is the only option, and it
works well once calibrated (see the notes in `examples/bridge.yaml`, and
`cos.min_tx_ms` for radios that click before speaking). Real carrier detect
needs a radio that exposes squelch on a multi-pin accessory connector.

## Status and known unknowns

v0.1. The Zello Channel API is labelled beta by Zello, so all protocol
serialization is isolated in `zello/protocol.py` and covered by tests against
the published byte layouts.

**Verified against real hardware:**

| path | notes |
|---|---|
| Serial PTT — DTR | AIOC, on macOS and Linux |
| Serial PTT — RTS | Digirig Mobile |
| CM108 HID PTT — GPIO 3 | AIOC and Digirig Lite; two vendors |
| Audio-level COS | the proven detection path on every interface |
| AllStarLink over `chan_usrp` | bidirectional, against a live ASL3 node |

Interfaces are auto-detected by USB id (AIOC, C-Media CM108/CM108AH/CM119),
so `ptt.hid_device` should normally be left unset — a hardcoded path is
specific to the hidapi build that produced it.

**Not verified:**

- **`cos.mode: aioc_hardware`** needs a real COS wire brought into the
  interface — a RIM-style adapter, or an AIOC with the COS pad soldered — and
  no device here has produced a HID input report. `--validate` warns.
- **`cos.configure_aioc_on_start`** raises `NotImplementedError` rather than
  writing speculative registers to a radio interface. Program the AIOC's
  threshold and tail with the vendor tool.

The AIOC's firmware VCOS (`cos.mode: aioc_virtual`) was **removed**: it does
not work on the hardware, confirmed on two units and independently at the
device's own registers. Use `internal_audio`.

[`docs/hardware-notes.md`](docs/hardware-notes.md) has the byte layouts, the
failure modes, and the traps that cost the most time.
