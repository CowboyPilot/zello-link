#!/usr/bin/env bash
#
# Install zello-link and its dependencies on Linux or macOS.
#
#   ./install.sh              full install, including hardware extras
#   ./install.sh --dev        also install test tooling
#   ./install.sh --no-system  skip the system-library check
#
# System libraries (libopus, PortAudio, hidapi) are NOT installed silently:
# that needs root, and a script that sudo's behind your back is not one you
# should run. The exact command is printed for you to review and run.

set -euo pipefail
cd "$(dirname "$0")"

VENV=".venv"
EXTRAS="hardware"
CHECK_SYSTEM=1

for arg in "$@"; do
    case "$arg" in
        --dev)        EXTRAS="hardware,dev" ;;
        --no-system)  CHECK_SYSTEM=0 ;;
        --venv=*)     VENV="${arg#*=}" ;;
        -h|--help)    sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)            echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

say()  { printf '\n\033[1m%s\033[0m\n' "$*"; }
ok()   { printf '  \033[32mok\033[0m    %s\n' "$*"; }
warn() { printf '  \033[33mwarn\033[0m  %s\n' "$*"; }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; }

# -- 1. Python ------------------------------------------------------------
say "Checking Python"

PYTHON=""
for candidate in python3.13 python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
        if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [[ -z "$PYTHON" ]]; then
    bad "need Python 3.11 or newer"
    echo
    echo "  Debian / Raspberry Pi OS:  sudo apt install python3 python3-venv"
    echo "  macOS:                     brew install python@3.12"
    exit 1
fi
ok "$($PYTHON --version) at $(command -v "$PYTHON")"

# -- 2. System libraries --------------------------------------------------
# libopus and PortAudio are C libraries loaded at runtime, not Python wheels.
# The bridge imports them lazily, so --validate and the tests work without
# them -- but audio and the codec do not.
if [[ "$CHECK_SYSTEM" == "1" ]]; then
    say "Checking system libraries"
    MISSING=()

    have_lib() {
        "$PYTHON" - "$1" <<'PY' 2>/dev/null
import ctypes.util, sys
sys.exit(0 if ctypes.util.find_library(sys.argv[1]) else 1)
PY
    }

    for lib in opus portaudio; do
        if have_lib "$lib"; then ok "lib$lib found"; else MISSING+=("$lib"); warn "lib$lib NOT found"; fi
    done
    if have_lib hidapi; then ok "libhidapi found"; else warn "libhidapi not found (only needed for CM108/HID PTT or AIOC COS)"; fi

    if (( ${#MISSING[@]} )); then
        echo
        if command -v apt-get >/dev/null 2>&1; then
            echo "  Install with:"
            echo "    sudo apt install libopus0 libportaudio2 libhidapi-hidraw0"
        elif command -v dnf >/dev/null 2>&1; then
            echo "  Install with:"
            echo "    sudo dnf install opus portaudio hidapi"
        elif command -v brew >/dev/null 2>&1; then
            echo "  Install with:"
            echo "    brew install opus portaudio hidapi"
        else
            echo "  Install libopus, PortAudio and hidapi using your package manager."
        fi
        echo
        warn "continuing; the bridge will not open audio until these are present"
    fi
fi

# -- 3. Virtual environment ----------------------------------------------
say "Creating virtual environment: $VENV"
if [[ -d "$VENV" ]]; then
    ok "reusing existing $VENV"
else
    "$PYTHON" -m venv "$VENV"
    ok "created $VENV"
fi

"$VENV/bin/python" -m pip install --quiet --upgrade pip
ok "pip up to date"

# -- 4. The package -------------------------------------------------------
say "Installing zello-link[$EXTRAS]"
if ! "$VENV/bin/python" -m pip install --quiet -e ".[$EXTRAS]"; then
    bad "pip install failed"
    echo
    echo "  If a hardware extra failed to build, install without it:"
    echo "    $VENV/bin/pip install -e ."
    echo "  Audio/PTT will be unavailable until it is installed."
    exit 1
fi
ok "installed"

# -- 5. Verify ------------------------------------------------------------
say "Verifying"
"$VENV/bin/python" - <<'PY'
import importlib, sys

def check(label, fn):
    try:
        fn()
        print(f"  \033[32mok\033[0m    {label}")
        return True
    except Exception as e:
        print(f"  \033[33mwarn\033[0m  {label}: {e}")
        return False

check("package imports", lambda: importlib.import_module("zello_link"))
check("numpy", lambda: importlib.import_module("numpy"))
check("websockets", lambda: importlib.import_module("websockets"))

def opus():
    from zello_link.zello.opus import load_libopus
    load_libopus()
check("libopus (codec)", opus)

def audio():
    import sounddevice
    sounddevice.query_devices()
check("PortAudio (audio I/O)", audio)

check("pyserial (serial PTT)", lambda: importlib.import_module("serial"))
check("hidapi (HID PTT/COS)", lambda: importlib.import_module("hid"))
PY

say "Done"
cat <<'EOF'
  Next steps:

    1. Copy the example config and edit it:
         cp examples/bridge.yaml my-bridge.yaml

    2. Pick your audio devices interactively (writes them into the config):
         .venv/bin/zello-link --config my-bridge.yaml --list-audio-devices

    3. Check the config without connecting or keying the radio:
         .venv/bin/zello-link --config my-bridge.yaml --validate

    4. Set the receive level, watching the live meter:
         .venv/bin/zello-link --config my-bridge.yaml --cos-monitor

  Secrets belong in the environment, not the config file:
    export ZELLO_AUTH_TOKEN='...'   # from https://developers.zello.com/
    export ZELLO_PASSWORD='...'
EOF
