"""Every USRP config knob must actually do something.

Four of them did not. jitter_buffer_ms, max_jitter_buffer_ms and
packet_loss_fill were documented in examples/bridge.yaml, validated with
ranges, and read by nothing at all -- an operator could tune them for an
afternoon and change no behaviour whatsoever. inactivity_timeout_ms was worse
than unused: implementing it as written would have fired constantly, because
chan_usrp sends nothing while the node is idle.

Config that silently does nothing is worse than no config, so this fails the
build rather than waiting for someone to discover it in the field.

Scoped to the USRP section deliberately: that is where the rot was, and a
blanket rule across every section would flag fields consumed indirectly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from zello_link.config import UsrpConfig

_ROOT = Path(__file__).resolve().parents[2] / "zello_link"


def _sources() -> dict[Path, str]:
    return {
        p: p.read_text()
        for p in _ROOT.rglob("*.py")
        if p.name != "config.py"
    }


@pytest.mark.parametrize("field", sorted(UsrpConfig.model_fields))
def test_field_is_read_somewhere(field):
    hits = [p.name for p, text in _sources().items() if f"usrp.{field}" in text]
    assert hits, (
        f"usrp.{field} is declared and validated but nothing reads it. "
        "Wire it up, or remove it -- a knob that does nothing is a support "
        "burden and a lie in the example config."
    )


def test_there_is_no_inactivity_timeout():
    """Regression: silence from ASL is normal, so a timeout on it is wrong.

    chan_usrp sends nothing while the node is idle -- measured on a live
    ASL3 node, "usrp show" Write stays 0 at rest. A watchdog on inbound
    silence would fire constantly on a perfectly healthy link, which is the
    same mistake that once disconnected healthy Zello sessions.
    """
    assert "inactivity_timeout_ms" not in UsrpConfig.model_fields
