# Hardware notes

What has actually been verified against real hardware, and what has not.
Written from bench sessions rather than datasheets: where the two disagree,
this file records what the hardware did.

## Summary

| path | status |
|---|---|
| Serial PTT (DTR) — AIOC | **verified** on macOS and Linux |
| Serial PTT (RTS) — Digirig Mobile | **verified** on Linux |
| CM108 HID PTT (GPIO 3 output report) | **verified** — AIOC *and* Digirig Lite |
| Digirig Lite (C-Media `0d8c:0012`) | **verified** — auto-detected, keys correctly |
| AIOC CM108 HID PTT | **verified** with the corrected byte offsets |
| CM108 HID COS (`aioc_hardware`) | **not yet verified** — needs a real COS wire |
| AIOC firmware VCOS | **removed** — does not work on the hardware |
| Audio-level COS (`internal_audio`) | **verified**, and what the bench runs |
| USRP backend | **verified** end to end against ASL3 |

**Use `cos.mode: internal_audio` with an AIOC.** The HID COS modes are
implemented and match the device's own HID report descriptor, but no AIOC
tested has ever emitted an input report.

## CM108 HID: what the AIOC actually advertises

Dumped from `/sys/class/hidraw/hidraw0/device/report_descriptor`:

```
USAGE_PAGE 0x0C (Consumer), LOGICAL_MAX 1
  REPORT_SIZE 1, REPORT_COUNT 2, INPUT      2 button bits
  REPORT_COUNT 2, INPUT (relative)          2 more bits
USAGE_PAGE 0x0B (Telephony), COUNT 1, INPUT 1 bit
USAGE_PAGE 0x0C, COUNT 3, INPUT             3 bits
  REPORT_SIZE 8, REPORT_COUNT 3, INPUT      3 bytes
OUTPUT:  REPORT_COUNT 4                     4-byte output report
FEATURE: REPORT_COUNT 6                     6-byte feature report
```

So the input report is **4 bytes**: byte 0 carries eight single-bit flags,
bytes 1-3 are 8-bit values. COS arrives as a **button press in byte 0**, not
as a GPIO level — the AIOC maps VCOS onto one of the four CM108 buttons in
its own "CM108 Button Sources" configuration:

| CM108 button | bit in byte 0 |
|---|---|
| 1 — Vol Up | `0x01` |
| 2 — Vol Down | `0x02` |
| 3 — Playback Mute | `0x04` |
| 4 — Record Mute | `0x08` |

`cos.hid_button` selects which one. It cannot be discovered; it has to match
whatever the AIOC is configured for.

An earlier implementation read **byte 2** — the *output* report's GPIO data
byte — against an *input* report, and treated the bit as a 1-8 GPIO pin.
Wrong byte and wrong bit, both silent.

## Digirig Mobile

Presents as two devices behind an internal Microchip hub:

| device | node | role |
|---|---|---|
| CP2102N UART (`10c4:ea60`) | `/dev/ttyUSB0` | **PTT via RTS**, and CAT |
| C-Media CM108 (`0d8c:0012`) | sound card, `hidraw0` | audio |

So it is `ptt.mode: serial` with `ptt.serial_signal: rts` — not a CM108 GPIO
interface, even though it contains one. Use the `/dev/serial/by-id/...`
path rather than `/dev/ttyUSB0`, which renumbers.

Note the sound half shares `0d8c:0012` with the Digirig Lite and many generic
adapters, so the USB id alone cannot tell those devices apart.

## The CM108 output report: two things that must both be right

Verified against a Digirig Lite, and cross-checked with AllStarLink's
`chan_simpleusb.c`, which is the reference implementation for these devices.

**Byte offsets.** The 4-byte report is:

```
byte 0  unused      (0x00)
byte 1  GPIO data       bit N-1 set = GPIO N driven high
byte 2  GPIO direction  bit N-1 set = GPIO N is an output
byte 3  unused      (0x00)
```

ASL uses `hid_gpio_loc = 1` and `hid_gpio_ctl_loc = 2`, with `hid_io_ptt = 4`
(GPIO 3) and `hid_gpio_ctl = 0x04` to make that pin an output. Direction is
set for **both** key and unkey: releasing PTT must drive the line low rather
than float it.

This project originally had data at byte 2 and a "mask" at byte 3 — off by
one, so every value went into the direction register while the data register
stayed zero. The pin was correctly configured as an output and then never
driven.

**hidapi framing.** `hid.write()` takes the report ID as its first byte,
ahead of the report. These devices declare no report IDs, so it is `0x00`
followed by the four bytes above. Measured on the Digirig Lite:

```
4-byte  [00 00 00 04]      -> -1   (rejected)
5-byte  [00 00 00 00 04]   ->  5   (accepted)
```

So the correct key sequence is `00 00 04 04 00` and unkey is `00 00 00 04 00`.

Both an AIOC and a Digirig Lite key correctly with these bytes, so the layout
is confirmed across two vendors rather than inferred from one.

### Why the AIOC appeared to work with the wrong layout

An early test wrote the 4-byte report to `/dev/hidraw0` directly and the
radio keyed. `hidraw` consumes the first byte of a write as the report
number, which shifted the remaining bytes into the correct registers **by
accident**. Through hidapi, with the report ID supplied properly, the same
struct asserted nothing: the Digirig accepted every write and never lit its
PTT LED. A passing test on one path is not verification of the layout.

## Device paths are backend-specific — do not hardcode them

`ptt.hid_device` and `cos.hid_device` should be left unset so the interface
is auto-detected by USB id. A hand-written path only works on the backend it
was copied from:

| backend | path form |
|---|---|
| hidapi, hidraw build | `/dev/hidrawN` |
| hidapi, libusb build | `1-1.4:1.3` |
| macOS | `DevSrvsID:4295374066` |

The `hidapi` wheel pip installs on Raspberry Pi OS is the **libusb** build,
so `/dev/hidraw0` fails there with "open failed" — and once libusb claims the
interface, writing to the hidraw node raises `EPIPE`. `CM108_DEVICE_IDS`
covers the AIOC plus the C-Media CM108/CM108AH/CM119 ids; anything else still
works with an explicit path.

## VCOS removed; hardware COS kept as a future path (2026-08-31)

`cos.mode: aioc_virtual` is **gone**. The AIOC's firmware VCOS does not work
(evidence below), and it was a corner case even if it had: audio-level COS is
the more consistent path across every interface, and it is what the bench and
the live repeater both run.

`cos.mode: aioc_hardware` **remains, flagged unverified.** It is for a real
COS wire brought into the interface and reported as a CM108 button press --
a RIM-style adapter, or an AIOC with the COS pad soldered. Nothing here has
produced a HID input report yet, so `--validate` warns and the setup wizard
labels it UNVERIFIED rather than offering it as an equal option.

The eventual goal that keeps this path open: talking to an RF interface
directly, bypassing Asterisk, on a node where hardware COS comes through the
interface. The repeater currently running the soak has exactly that (a RIM
Lite USB), but Asterisk is handling COS there today.

Setting the removed mode raises a migration error naming the reason, rather
than a bare "not a permitted value".

## Why VCOS was removed

Not one HID input report was ever observed, across:

- two AIOC units, one of them on the latest firmware
- macOS and Linux (Raspberry Pi Zero 2 W, Debian 12)
- `hidapi` (libusb backend) and raw `/dev/hidraw0` reads
- with and without Asterisk contending for the interface
- VCOS level thresholds from 256 down to 50
- confirmed squelch activity on an attached, receiving radio

The operator then confirmed independently, on a separate serial monitor, that
**the VCOS registers never change state on the device itself.** The fault is
in the AIOC's VCOS, not in the host software: our decoding matches the
descriptor above, and the *output* half of the same interface demonstrably
keys the radio.

### A red herring worth naming

```
hid-generic 0003:1209:7388.0005: No inputs registered, leaving
```

This does **not** mean the device sends no input reports. It means the kernel
could not map the usages to an `/dev/input` event node — every `USAGE` in the
descriptor is `0x00`, undefined. `hidraw` is unaffected. This line was
misread as evidence during the session; the report descriptor is the
authority.

## macOS gates HID input

The AIOC enumerates as a **Consumer Control** device (usage page `0x000c`),
which macOS puts behind Privacy & Security → Input Monitoring. Output reports
are unrestricted, so **PTT works while COS input is silent** — a confusing
combination if you do not know to expect it. Linux `hidraw` applies no such
gate. `--diagnose-aioc` reports this when `cos.mode` is a HID mode.

## RF ingress can drop the AIOC off the USB bus

Observed: transmitting on a badly matched frequency browned out the port and
took the whole device down mid-test.

```
usb 1-1: USB disconnect, device number 4
usb 1-1: device descriptor read/64, error -71   (x4)
usb usb1-port1: attempt power cycle
```

It did not re-enumerate on its own. Consequences worth knowing:

- The serial PTT `unkey()` then fails with `OSError: [Errno 5]`, having
  already asserted PTT. Always drive PTT through `SafePtt` so `max_tx_s`
  bounds a transmission when the interface disappears underneath it.
- `/dev/hidraw0` disappears with the device. Writing to that path afterwards
  **creates a regular file and silently succeeds**, touching no hardware, and
  the stale file then stops udev recreating the node. `_check_device_node()`
  refuses this now.
- Ferrites on the USB cable are the mitigation.

### The Digirig Lite does register an input node

```
input: C-Media Electronics Inc. USB Audio Device as .../input/input9
hid-generic 0003:0D8C:0012.0008: input,hidraw0: USB HID v1.00 Device
```

Unlike the AIOC, whose descriptor has every `USAGE` set to `0x00`, this one
declares real usages (`0xE9` Volume Up, `0xEA` Volume Down, `0xE2` Mute,
Telephony `0x20` Hook Switch). If COS-over-HID is ever to be verified, this
is the more promising device.

## Raspberry Pi Zero 2 W

Software USB port resets **cannot** work: the port power rail is shared, so
there is nothing to cycle. Unbinding and rebinding `dwc_otg` fails with

```
dwc_otg: probe of 3f980000.usb failed with error -14
```

and leaves the controller deregistered — no USB at all until a reboot. After
an `error -71` dropout the only remedies are a physical replug or a reboot.

## Device paths are not stable

macOS `DevSrvsID:...` values change on every re-enumeration; three different
ids were seen for one AIOC within ten minutes. Leave `cos.hid_device` and
`ptt.hid_device` unset so the device is auto-detected by VID:PID (`1209:7388`)
unless more than one interface is present.

## Bus-powered hubs on a Pi Zero 2 W are marginal

The Pi Zero 2 W has a single micro-USB data port, so a USB-C interface such
as the Digirig Lite needs a hub. A bus-powered one reports `bMaxPower:
100mA`, and a CM108 that wedges will not re-enumerate on that rail:

```
usb 1-1.4: device not accepting address 7, error -32
usb 1-1-port4: unable to enumerate USB device
```

The port then stays silent -- no further kernel events at all -- until the
device is physically reconnected. Compare the kernel clock against the last
USB log timestamp to tell "not replugged yet" from "replugged and failing".
A powered hub avoids this.
