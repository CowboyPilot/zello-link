# zello-dmr-bridge

Headless bridge between a Zello Friends & Family channel and a dedicated
encrypted DMR radio, over an AIOC USB radio interface.

The radio remains responsible for RF, DMR vocoding, talkgroup operation, and
encryption. This service transports audio and PTT/COS only. It never contains,
requests, or handles a DMR encryption key.

## What it does

```
Zello channel  <--TLS WebSocket-->  bridge  <--USB audio + PTT/COS-->  DMR radio
```

- Connects directly to the Zello Channel API. No Android app, no Waydroid, no
  GUI automation, no VOX, no MMDVM modem.
- Half-duplex, first-talker-wins arbitration. The bridge never keys RF and
  transmits to Zello at the same time.
- Fails safe: PTT is driven OFF on startup, shutdown, exception, disconnect,
  device failure, SIGTERM, and watchdog timeout.

## Install

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
.venv/bin/zello-dmr-bridge --config my-bridge.yaml --list-audio-devices
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
zello-dmr-bridge --config /etc/zello-dmr/west.yaml
```

Start from [`examples/bridge.yaml`](examples/bridge.yaml) — every option is
documented inline. The config path is mandatory; the bridge never assumes
`/dev/ttyACM0` or a default sound card.

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
sudo cp systemd/zello-dmr-bridge@.service /etc/systemd/system/
sudo systemctl enable --now zello-dmr-bridge@west
sudo systemctl enable --now zello-dmr-bridge@east
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

Two things still need bench verification against real hardware:

- **CM108/HID report layout.** The AIOC README documents that firmware ≥ 1.2.0
  exposes a CM108-compatible HID endpoint, but not its byte layout. What is
  implemented is the conventional CM108 GPIO report, isolated in
  `Cm108Report` and unit tested, so a bench finding is a one-class change.
  Serial PTT (`DTR=1, RTS=0`) is documented and is the safer default.
- **`cos.configure_aioc_on_start`** raises `NotImplementedError` rather than
  writing speculative registers to a radio interface. Program the AIOC's
  threshold and tail with the vendor tool.

See spec section 22 for the full bench checklist.
