"""Interactive device selection for --list-audio-devices.

Two things make this fiddly enough to be worth its own module:

1. **Writing back must not destroy the config.** A YAML load/dump round trip
   silently drops every comment, and the example config is mostly comments --
   the calibration notes are the valuable part. So edits are surgical: find
   the one line that sets a key inside the right section and replace its
   value, leaving the rest of the file byte-for-byte identical.

2. **``mode:`` is not unique.** It appears under both ``ptt:`` and ``cos:``,
   so a naive search-and-replace edits the wrong one. The editor here tracks
   which top-level section it is inside.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from typing import Any, Sequence

__all__ = [
    "run_device_setup", "set_config_value", "ensure_section",
    "render_asl_instructions", "SelectionAborted",
]


class SelectionAborted(Exception):
    """Operator declined, or input is not interactive."""


# -- surgical YAML editing -------------------------------------------------
_KEY_RE = re.compile(r"^(?P<indent>\s*)(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*:(?P<rest>.*)$")


def set_config_value(
    text: str, section: str, key: str, value: str, *, insert_if_missing: bool = False
) -> str:
    """Replace ``section.key``'s value, preserving comments and layout.

    Any trailing ``# comment`` on the edited line is kept.

    If the key is absent, raises KeyError unless ``insert_if_missing`` is set,
    in which case it is added directly under the section header using that
    section's own indentation. Insertion is only safe because the section is
    located first -- appending to the end of the file could land the setting
    in whatever block happened to come last.
    """
    lines = text.splitlines(keepends=True)
    current: str | None = None
    out: list[str] = []
    found = False

    for line in lines:
        m = _KEY_RE.match(line.rstrip("\n"))
        if m:
            indent, k = m.group("indent"), m.group("key")
            if not indent:                      # a top-level section header
                current = k
            elif current == section and k == key and not found:
                comment = ""
                rest = m.group("rest")
                hash_at = _comment_start(rest)
                if hash_at is not None:
                    comment = "  " + rest[hash_at:].strip()
                newline = "\n" if line.endswith("\n") else ""
                out.append(f"{indent}{key}: {value}{comment}{newline}")
                found = True
                continue
        out.append(line)

    if not found:
        if not insert_if_missing:
            raise KeyError(f"{section}.{key} not found in the config")
        return _insert_under_section(lines, section, key, value)
    return "".join(out)


def _insert_under_section(
    lines: Sequence[str], section: str, key: str, value: str
) -> str:
    """Add ``key`` immediately below the ``section:`` header."""
    out: list[str] = []
    inserted = False

    for line in lines:
        out.append(line)
        if inserted:
            continue
        m = _KEY_RE.match(line.rstrip("\n"))
        if m and not m.group("indent") and m.group("key") == section:
            indent = _section_indent(lines, section)
            if not line.endswith("\n"):
                out[-1] = line + "\n"
            out.append(f"{indent}{key}: {value}\n")
            inserted = True

    if not inserted:
        raise KeyError(f"section {section!r} not found in the config")
    return "".join(out)


def _section_indent(lines: Sequence[str], section: str) -> str:
    """Indentation used by the keys already inside ``section``."""
    in_section = False
    for line in lines:
        m = _KEY_RE.match(line.rstrip("\n"))
        if not m:
            continue
        if not m.group("indent"):
            if in_section:
                break
            in_section = m.group("key") == section
        elif in_section:
            return m.group("indent")
    return "  "


def ensure_section(text: str, section: str) -> str:
    """Append an empty ``section:`` header if the file has none.

    Needed because a config written for the AIOC backend has no ``usrp:``
    block at all, and set_config_value can only insert into a section that
    already exists.
    """
    for line in text.splitlines():
        m = _KEY_RE.match(line)
        if m and not m.group("indent") and m.group("key") == section:
            return text
    sep = "" if text.endswith("\n") else "\n"
    return f"{text}{sep}\n{section}:\n"


def _comment_start(rest: str) -> int | None:
    """Index of a trailing ``#`` comment, ignoring one inside quotes."""
    in_quote: str | None = None
    for i, ch in enumerate(rest):
        if in_quote:
            if ch == in_quote:
                in_quote = None
        elif ch in "\"'":
            in_quote = ch
        elif ch == "#":
            return i
    return None


def yaml_scalar(value: Any) -> str:
    """Render a value the way it should appear in the config."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    return '"{}"'.format(str(value).replace('"', '\\"'))


# -- prompting -------------------------------------------------------------
@dataclass
class Choice:
    value: Any
    label: str
    note: str = ""


def _prompt_choice(title: str, choices: Sequence[Choice], current: Any) -> Any:
    """Ask the operator to pick one. Empty input keeps the current value."""
    if not choices:
        raise SelectionAborted(f"{title}: nothing to choose from")

    print(f"\n{title}", file=sys.stderr)
    default_idx = None
    for i, c in enumerate(choices, 1):
        marker = " "
        if current is not None and str(c.value) == str(current):
            marker, default_idx = "*", i
        note = f"   {c.note}" if c.note else ""
        print(f"  {marker} [{i}] {c.label}{note}", file=sys.stderr)

    suffix = f" [{default_idx}]" if default_idx else ""
    while True:
        try:
            raw = input(f"Select 1-{len(choices)}{suffix} (q to skip): ").strip()
        except (EOFError, KeyboardInterrupt):
            raise SelectionAborted("input closed") from None

        if not raw:
            if default_idx:
                return choices[default_idx - 1].value
            print("  no current value; please choose.", file=sys.stderr)
            continue
        if raw.lower() in ("q", "quit", "skip"):
            raise SelectionAborted("skipped")
        if raw.isdigit() and 1 <= int(raw) <= len(choices):
            return choices[int(raw) - 1].value
        print(f"  enter a number between 1 and {len(choices)}.", file=sys.stderr)


def _confirm(question: str) -> bool:
    try:
        return input(f"{question} [y/N]: ").strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


# -- device selection ------------------------------------------------------
def _device_value(device: Any, same_direction: Sequence[Any]) -> Any:
    """Choose how to refer to a device in the config.

    Prefers the NAME, which survives an unplug/replug that renumbers indices.
    Falls back to the index when the name is ambiguous within the direction --
    a host with two identical interfaces would otherwise get a config that
    fails to start.
    """
    same_name = [d for d in same_direction if d.name == device.name]
    return device.name if len(same_name) == 1 else device.index


def detect_cos_choices() -> list[Choice]:
    """Offer the COS sources actually available on this host."""
    choices = [
        Choice(
            "internal_audio",
            "internal_audio  - software level detection",
            "always available; calibrate with --cos-monitor",
        )
    ]

    hid_paths: list[str] = []
    hid_error = ""
    try:
        from ..hardware.aioc_hid import find_aioc_hid_path

        hid_paths = find_aioc_hid_path()
    except Exception as e:
        hid_error = str(e).splitlines()[0]

    if hid_paths:
        where = f"{len(hid_paths)} AIOC HID device(s) found"
        choices.append(
            Choice("aioc_virtual", "aioc_virtual    - AIOC firmware VCOS", where)
        )
        choices.append(
            Choice(
                "aioc_hardware",
                "aioc_hardware   - AIOC external hardware input",
                "needs a COS wire from the radio; a 2-pin K1 has none",
            )
        )
    else:
        note = hid_error or "no AIOC HID interface detected"
        choices.append(
            Choice(None, "aioc_virtual / aioc_hardware  (unavailable)", note)
        )

    choices.append(
        Choice("disabled", "disabled        - never report RX", "receive-only bridges")
    )
    return choices


def _prompt_hid_device(current: Any) -> str:
    """Pick the AIOC HID interface that carries COS."""
    try:
        from ..hardware.aioc_hid import find_aioc_hid_path

        paths = find_aioc_hid_path()
    except Exception as e:
        raise SelectionAborted(f"cannot enumerate HID devices: {e}") from None

    if not paths:
        raise SelectionAborted("no AIOC HID device found")

    if len(paths) == 1:
        print(f"\n  Using the only AIOC HID device found: {paths[0]}", file=sys.stderr)
        return paths[0]

    return _prompt_choice(
        "AIOC HID interface  (which one carries COS)",
        [Choice(p, p) for p in paths],
        current,
    )


#: Default private node number for the generated rpt.conf stanza. The
#: shipped rpt.conf documents 1998 as the sample private node.
DEFAULT_ASL_NODE = 1998


def local_addresses() -> list[str]:
    """Every IPv4 address this host could bind, most useful first.

    Enumerates all interfaces rather than just the default route: a bridge
    reached over a VPN binds the tunnel address, which is exactly the one a
    default-route probe misses. The stdlib has no interface-enumeration API,
    hence shelling out, with a probe as a fallback.
    """
    import re
    import socket
    import subprocess

    found: list[str] = []

    def add(addr: str) -> None:
        if addr and addr not in found and not addr.startswith("127."):
            found.append(addr)

    for cmd in (["ip", "-4", "-o", "addr"], ["ifconfig"]):
        try:
            out = subprocess.run(
                cmd, capture_output=True, text=True, timeout=5, check=False
            ).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        if out:
            for m in re.finditer(r"inet\s+(\d+\.\d+\.\d+\.\d+)", out):
                add(m.group(1))
            break

    if not found:
        try:
            probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            probe.connect(("192.0.2.1", 9))      # TEST-NET-1, never routed
            add(probe.getsockname()[0])
            probe.close()
        except OSError:
            pass

    # Loopback last: correct only when Asterisk runs on this same host.
    return found + ["127.0.0.1"]


def _prompt_text(title: str, current: Any, *, hint: str = "") -> str:
    if hint:
        print(f"\n{title}\n  {hint}", file=sys.stderr)
    else:
        print(f"\n{title}", file=sys.stderr)
    suffix = f" [{current}]" if current not in (None, "") else ""
    while True:
        try:
            raw = input(f"Value{suffix}: ").strip()
        except (EOFError, KeyboardInterrupt):
            raise SelectionAborted("input closed") from None
        if raw:
            return raw
        if current not in (None, ""):
            return str(current)
        print("  a value is required.", file=sys.stderr)


def _prompt_port(title: str, current: Any, *, hint: str = "") -> int:
    while True:
        raw = _prompt_text(title, current, hint=hint)
        try:
            port = int(raw)
        except ValueError:
            print("  ports are numbers.", file=sys.stderr)
            continue
        if 1 <= port <= 65535:
            return port
        print("  a port is 1-65535.", file=sys.stderr)


def run_usrp_setup(cfg: Any, config_path: str) -> int:
    """Walk through the chan_usrp settings and write them to the config."""
    print("\n--- AllStarLink (chan_usrp) setup ---", file=sys.stderr)
    print(
        "  The node's rxchannel line reads USRP/<bridge-ip>:<bridge-port>:<asterisk-port>.\n"
        "  Answer these and the exact line is printed for you at the end.",
        file=sys.stderr,
    )

    addrs = local_addresses()
    try:
        choices = [
            Choice(a, a, "loopback: only if Asterisk runs on THIS host"
                   if a.startswith("127.") else "reachable from other hosts")
            for a in addrs
        ]
        choices.append(Choice("", "enter a different address", "typed in by hand"))
        bind_host = _prompt_choice(
            "This bridge's address  (the IP AllStarLink will send audio TO)",
            choices,
            cfg.usrp.bind_host,
        )
        if not bind_host:
            bind_host = _prompt_text(
                "Bridge address",
                cfg.usrp.bind_host,
                hint="Must be an address on THIS host that AllStarLink can reach.",
            )
        bind_port = _prompt_port(
            "This bridge's UDP port  (it listens here; HISPORT in rxchannel)",
            cfg.usrp.bind_port,
            hint="AllStarLink's convention is 34001.",
        )
        asl_host = _prompt_text(
            "AllStarLink host  (where Asterisk runs)",
            cfg.usrp.asl_host,
            hint="An IP or hostname. Use 127.0.0.1 if it is this same machine.",
        )
        asl_port = _prompt_port(
            "Asterisk's UDP port  (it listens here; MYPORT in rxchannel)",
            cfg.usrp.asl_port,
            hint="AllStarLink's convention is 32001.",
        )
    except SelectionAborted as e:
        print(f"\nUSRP setup {e}; nothing written.", file=sys.stderr)
        return 0

    remote = not bind_host.startswith("127.")
    pending: list[tuple[str, str, Any, str]] = [
        ("bridge", "backend", "usrp", "usrp"),
        ("usrp", "bind_host", bind_host, bind_host),
        ("usrp", "bind_port", bind_port, str(bind_port)),
        ("usrp", "asl_host", asl_host, asl_host),
        ("usrp", "asl_port", asl_port, str(asl_port)),
    ]
    if remote:
        # USRP has no authentication; binding off-loopback must be deliberate.
        pending.append(("usrp", "allow_remote_host", True, "true (bind is not loopback)"))

    print("\nAbout to write:", file=sys.stderr)
    for section, key, _v, shown in pending:
        print(f"  {section}.{key}: {shown}", file=sys.stderr)
    if remote:
        print(
            "\n  NOTE: USRP has no authentication or encryption. Binding to "
            f"{bind_host}\n  means anyone who can reach {bind_host}:{bind_port} can key "
            "the radio\n  system. Only do this on a VPN or an isolated segment.",
            file=sys.stderr,
        )

    if not _confirm(f"\nUpdate {config_path}?"):
        print("nothing written.", file=sys.stderr)
        return 0

    rc = _write(config_path, pending, sections=("usrp",))
    if rc == 0:
        print(render_asl_instructions(bind_host, bind_port, asl_port), file=sys.stderr)
    return rc


def render_asl_instructions(
    bind_host: str, bind_port: int, asl_port: int, node: int = DEFAULT_ASL_NODE
) -> str:
    """The AllStarLink side, which the bridge cannot configure itself."""
    return f"""
=======================================================================
 Now configure AllStarLink. Two files, on the Asterisk host.
=======================================================================

1. /etc/asterisk/modules.conf

   chan_usrp is not loaded by default. Under [modules], make sure this
   line is present and NOT commented out:

       load => chan_usrp.so

   ASL3 ships with autoload=no, so an absent or commented line means the
   module never loads and the channel silently will not exist.

2. /etc/asterisk/rpt.conf

   Add a node for the bridge. A node has exactly ONE rxchannel -- adding
   a second line to an existing node does not give it two inputs, one
   silently wins -- so this is its own node, linked to your RF node.

       [{node}]
       rxchannel = USRP/{bind_host}:{bind_port}:{asl_port}
       duplex = 0              ; no telemetry, no hangtime
       hangtime = 0
       tailmessagetime = 0
       nounkeyct = 1           ; no courtesy tone
       telemdefault = 0
       idtime = 0              ; never send periodic IDs
       politeid = 0
       unlinkedct = |          ; suppress every courtesy tone
       linkunkeyct = |
       remotect = |
       idtalkover = |
       idrecording = |

   The telemetry suppression matters: on a talkative node every courtesy
   tone and ID arrives at the bridge as a keyed transmission and is
   relayed to Zello as a spurious over.

   Register it in the [nodes] stanza too:

       {node} = radio@127.0.0.1/{node},NONE

3. Link it to your RF node

   On the RF node's stanza, connect at startup:

       startup_macro = *3{node}

   Then restart Asterisk -- an rxchannel change needs a restart, not a
   reload:

       systemctl restart asterisk

   Verify with:
       asterisk -rx "module show like chan_usrp"
       asterisk -rx "core show channels" | grep -i usrp
=======================================================================
"""


def run_device_setup(cfg: Any, config_path: str) -> int:
    """Prompt for the radio side, then write the config.

    Offers a choice of backend first, because the two are peers: a
    deployment picks one, and walking a USRP user through sound-card
    selection they will never use is just noise.

    Returns a process exit code.
    """
    from ..audio.devices import DeviceError, format_device_table, list_devices

    try:
        devices = list_devices()
    except DeviceError as e:
        print(f"device error: {e}", file=sys.stderr)
        return 3

    print(format_device_table(devices))

    if not sys.stdin.isatty():
        # Piped or redirected: printing the table is the whole job. Blocking
        # on input here would hang a script that just wanted the listing.
        return 0

    # Which radio side is this instance driving? The two are peers: a
    # deployment picks one, so asking up front avoids walking someone
    # through sound-card selection they will never use.
    try:
        backend = _prompt_choice(
            "Radio side  (what is on the far end of this bridge)",
            [
                Choice("aioc", "Audio device  - a radio on a CM108/AIOC interface",
                       f"{len(devices)} device(s) detected"),
                Choice("usrp", "AllStarLink   - a node over chan_usrp (UDP)",
                       "no radio or sound card needed on this host"),
            ],
            cfg.bridge.backend,
        )
    except SelectionAborted as e:
        print(f"\nsetup {e}; nothing written.", file=sys.stderr)
        return 0

    if backend == "usrp":
        return run_usrp_setup(cfg, config_path)

    inputs = [d for d in devices if d.supports("input")]
    outputs = [d for d in devices if d.supports("output")]

    pending: list[tuple[str, str, Any, str]] = [("bridge", "backend", "aioc", "aioc")]

    try:
        if inputs:
            chosen = _prompt_choice(
                "Audio INPUT  (radio receive audio -> Zello)",
                [Choice(d.index, f"{d.name}", f"[{d.index}] {d.hostapi_name}") for d in inputs],
                _current_index(cfg.sound.input_device, inputs),
            )
            dev = next(d for d in inputs if d.index == chosen)
            value = _device_value(dev, inputs)
            pending.append(("sound", "input_device", value, f"{dev.name} [{dev.index}]"))

        if outputs:
            chosen = _prompt_choice(
                "Audio OUTPUT  (Zello -> radio microphone)",
                [Choice(d.index, f"{d.name}", f"[{d.index}] {d.hostapi_name}") for d in outputs],
                _current_index(cfg.sound.output_device, outputs),
            )
            dev = next(d for d in outputs if d.index == chosen)
            value = _device_value(dev, outputs)
            pending.append(("sound", "output_device", value, f"{dev.name} [{dev.index}]"))
    except SelectionAborted as e:
        print(f"\naudio selection {e}; nothing written.", file=sys.stderr)
        return 0

    cos_choices = detect_cos_choices()
    try:
        mode = _prompt_choice(
            "COS source  (how the bridge knows the radio is receiving)",
            [c for c in cos_choices if c.value is not None],
            cfg.cos.mode,
        )
        pending.append(("cos", "mode", mode, str(mode)))

        # The AIOC modes read COS over HID, so they need a device path. The
        # config rejects them without one, so asking here is not optional --
        # otherwise the whole write is refused at the validation step.
        if mode in ("aioc_virtual", "aioc_hardware"):
            hid = _prompt_hid_device(cfg.cos.hid_device)
            pending.append(("cos", "hid_device", hid, str(hid)))
    except SelectionAborted as e:
        print(f"  COS source unchanged ({e}).", file=sys.stderr)
        pending = [p for p in pending if p[0] != "cos"]

    if not pending:
        return 0

    print("\nAbout to write:", file=sys.stderr)
    for section, key, _value, shown in pending:
        print(f"  {section}.{key}: {shown}", file=sys.stderr)

    if not _confirm(f"\nUpdate {config_path}?"):
        print("nothing written.", file=sys.stderr)
        return 0

    return _write(config_path, pending)


def _current_index(selector: Any, devices: Sequence[Any]) -> int | None:
    """Resolve the configured selector to a device index, if it still matches."""
    if selector is None:
        return None
    try:
        from ..audio.devices import resolve_device

        kind = "input" if devices and devices[0].supports("input") else "output"
        return resolve_device(selector, kind, devices=list(devices)).index
    except Exception:
        return None


def _write(
    config_path: str,
    pending: Sequence[tuple[str, str, Any, str]],
    *,
    sections: Sequence[str] = (),
) -> int:
    import os
    import tempfile
    from pathlib import Path

    path = Path(config_path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        print(f"cannot read {path}: {e}", file=sys.stderr)
        return 3

    updated = text
    for section in sections:
        updated = ensure_section(updated, section)
    for section, key, value, _shown in pending:
        try:
            updated = set_config_value(
                updated, section, key, yaml_scalar(value), insert_if_missing=True
            )
        except KeyError as e:
            print(f"  cannot set {section}.{key}: {e}", file=sys.stderr)
            return 2

    # Validate before replacing: a config the bridge cannot load is worse
    # than one the operator has to edit by hand.
    try:
        from ..config import load_config

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
            tmp.write(updated)
            probe = tmp.name
        try:
            load_config(probe)
        finally:
            os.unlink(probe)
    except Exception as e:
        print(f"\nrefusing to write: the result would not load:\n{e}", file=sys.stderr)
        return 2

    backup = path.with_suffix(path.suffix + ".bak")
    try:
        backup.write_text(text, encoding="utf-8")
        path.write_text(updated, encoding="utf-8")
    except OSError as e:
        print(f"cannot write {path}: {e}", file=sys.stderr)
        return 3

    print(f"\nupdated {path}  (previous version saved as {backup.name})", file=sys.stderr)
    print("Check it with:  --validate", file=sys.stderr)
    return 0
