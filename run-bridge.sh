#!/usr/bin/env bash
# Run the bridge against the local bench config.
# Flags pass straight through, e.g.:
#   ./run-bridge.sh                  normal run, no meter
#   ./run-bridge.sh --showmonitor    with the live level meter
#   ./run-bridge.sh --validate       check config and exit
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -f bridge-local.env ]]; then
    echo "missing bridge-local.env" >&2
    echo "  cp examples/bridge.env.example bridge-local.env" >&2
    echo "  chmod 600 bridge-local.env    # then fill in your credentials" >&2
    exit 2
fi

if [[ ! -f bridge-local.yaml ]]; then
    echo "missing bridge-local.yaml" >&2
    echo "  cp examples/bridge.yaml bridge-local.yaml" >&2
    echo "  ./run-bridge.sh --list-audio-devices    # then pick your devices" >&2
    exit 2
fi

set -a; source bridge-local.env; set +a

exec .venv/bin/python -m zello_link \
    --config bridge-local.yaml \
    "$@"
