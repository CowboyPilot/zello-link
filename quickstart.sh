#!/usr/bin/env bash
#
# One-shot setup: install, then offer to build a working config.
#
# Runs ./install.sh (virtualenv + package + dependency check), then asks
# whether to run the setup wizard, which writes a config, a 0600 credentials
# file, and a launch script.
#
#   ./quickstart.sh              install, then offer the wizard
#   ./quickstart.sh --dev        also install the test tooling
#   ./quickstart.sh --no-wizard  install only
#
# Installs nothing as root. If a C library is missing, install.sh prints the
# exact command for your platform and you decide whether to run it.

set -euo pipefail

cd "$(dirname "$0")"

VENV=".venv"
RUN_WIZARD=1
INSTALL_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --no-wizard) RUN_WIZARD=0 ;;
        --venv=*)    VENV="${arg#*=}"; INSTALL_ARGS+=("$arg") ;;
        *)           INSTALL_ARGS+=("$arg") ;;
    esac
done

if [[ ! -x ./install.sh ]]; then
    echo "install.sh not found or not executable -- run this from the checkout" >&2
    exit 1
fi

./install.sh ${INSTALL_ARGS[@]+"${INSTALL_ARGS[@]}"}

PY="$VENV/bin/python"
if [[ ! -x "$PY" ]]; then
    PY="$VENV/Scripts/python.exe"          # Windows layout, e.g. under Git Bash
fi
if [[ ! -x "$PY" ]]; then
    echo "could not find the virtualenv interpreter under $VENV" >&2
    exit 1
fi

if [[ "$RUN_WIZARD" -eq 0 ]]; then
    exit 0
fi

# The wizard is interactive by design; without a terminal, say what to run.
if [[ ! -t 0 ]]; then
    echo
    echo "Not a terminal, so skipping the setup wizard. Run it with:"
    echo "  $PY -m zello_link --setup"
    exit 0
fi

echo
echo "─────────────────────────────────────────────────────────────"
echo " The setup wizard will ask for your Zello channel and account,"
echo " which interface the radio is on, and which directions to carry."
echo " It writes:"
echo "   <name>.yaml     the config"
echo "   <name>.env      your credentials, mode 600"
echo "   run-<name>.sh   a launcher"
echo " Nothing is written until the last question is answered."
echo "─────────────────────────────────────────────────────────────"
echo
read -r -p "Run the setup wizard now? [Y/n]: " reply || reply=""
case "${reply:-y}" in
    [Nn]*)
        echo
        echo "Skipped. Run it later with:"
        echo "  $PY -m zello_link --setup"
        ;;
    *)
        exec "$PY" -m zello_link --setup
        ;;
esac
