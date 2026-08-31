"""The radio-side backend contract.

The bridge core knows about *transmissions*, not about sound cards, serial
lines or UDP sockets. A backend translates between the two:

  core -> backend    key(), write_audio(pcm), unkey()
  backend -> core    on_rx_key, on_rx_audio, on_rx_unkey, on_fault

That split is what lets the same arbitration, Opus handling, and Zello client
drive either a physical radio over a CM108 interface or an AllStarLink node
over chan_usrp, with no conditional logic in the controller.

Audio crossing this boundary is always mono int16 at the backend's own
``sample_rate``. Rate conversion is the backend's job, because only it knows
what its far side requires -- chan_usrp is fixed at 8 kHz, a sound card is
whatever the operator configured.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import numpy as np

__all__ = ["RadioBackend", "BackendEvents", "create_backend"]


@dataclass
class BackendEvents:
    """What a backend tells the core.

    All three transmission callbacks are awaited, so a backend must not
    invoke them from a real-time audio callback -- hand the work to the event
    loop first. ``on_fault`` is synchronous and must never raise: it is called
    from error paths, including ones where the hardware has already gone.
    """

    on_rx_key: Callable[[], Awaitable[None]] | None = None
    on_rx_audio: Callable[[np.ndarray], Awaitable[None]] | None = None
    on_rx_unkey: Callable[[], Awaitable[None]] | None = None
    on_fault: Callable[[str], None] | None = None


class RadioBackend(abc.ABC):
    """One radio-side endpoint: a radio, or an AllStarLink node."""

    #: Identifies the backend in logs and metrics.
    name: str = "abstract"

    #: Sample rate this backend expects from, and produces for, the core.
    sample_rate: int = 16000

    def __init__(self) -> None:
        self.events = BackendEvents()

    def set_events(self, events: BackendEvents) -> None:
        """Attach the core's callbacks.

        Separate from construction because the core and the backend each need
        a reference to the other, so one has to be wired up second.
        """
        self.events = events

    # -- lifecycle --------------------------------------------------------
    @abc.abstractmethod
    async def start(self) -> None:
        """Acquire resources. Must leave the transmitter idle."""

    @abc.abstractmethod
    async def stop(self) -> None:
        """Release resources. Must not raise, and must not leave the far side
        keyed -- an explicit unkey first if a transmission is in progress."""

    # -- core -> backend (Zello reaching the radio side) ------------------
    @abc.abstractmethod
    async def key(self) -> None:
        """Begin transmitting. Returns once audio may safely follow.

        For a radio that means PTT is asserted and the transmitter has had
        its attack time; for USRP it is effectively immediate.
        """

    @abc.abstractmethod
    async def write_audio(self, pcm: np.ndarray) -> None:
        """Send one block of mono int16 audio at ``self.sample_rate``."""

    @abc.abstractmethod
    async def unkey(self) -> None:
        """End the transmission, including any tail the far side needs."""

    # -- safety -----------------------------------------------------------
    @abc.abstractmethod
    def fail_safe(self) -> None:
        """Force the radio side idle. Never raises.

        Called from exception handlers and shutdown paths, including ones
        where the underlying device has already disappeared.
        """

    @property
    def keyed(self) -> bool:
        """Whether we are currently transmitting toward the radio side."""
        return False

    def stats(self) -> dict[str, Any]:
        return {}


def create_backend(cfg: Any, **kw: Any) -> RadioBackend:
    """Build the backend named by ``bridge.backend``.

    Imported lazily so a USRP-only host needs no PortAudio, and an AIOC host
    needs nothing from the USRP path.
    """
    which = cfg.bridge.backend
    if which == "usrp":
        from .usrp import UsrpBackend

        return UsrpBackend(cfg, **kw)
    if which == "aioc":
        from .aioc import AiocBackend

        return AiocBackend(cfg, **kw)
    raise ValueError(f"unknown bridge.backend {which!r}")
