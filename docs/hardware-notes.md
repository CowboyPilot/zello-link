# Hardware notes

What has actually been verified against real hardware, and what has not.
Written from bench sessions rather than datasheets: where the two disagree,
this file records what the hardware did.

## Summary

| path | status |
|---|---|
| Serial PTT (DTR) — AIOC | **verified** on macOS and Linux |
| Serial PTT (RTS) — Digirig Mobile | implemented, **no hardware tested** |
| CM108 HID PTT (GPIO 3 output report) | **verified on Linux** — radio keyed |
| CM108 HID COS (`aioc_virtual` / `aioc_hardware`) | **does not work on the AIOC** — see below |
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

## VCOS does not work on the AIOC (2026-08-31)

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
