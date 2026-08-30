"""Support `python -m zello_dmr_bridge --config ...`."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
