"""The copy/paste installer, and the README block that advertises it.

A quickstart is the first thing a stranger runs, usually without reading it.
These tests hold it to the properties that matter at that moment: it does not
escalate privileges, it fails loudly rather than half-installing, and the
command printed on the front page is the one that actually exists.
"""

from __future__ import annotations

import re
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "quickstart.sh"
README = ROOT / "README.md"


class TestScriptExists:
    def test_present_and_executable(self):
        assert SCRIPT.exists(), "README tells people to run it"
        assert stat.S_IMODE(SCRIPT.stat().st_mode) & stat.S_IXUSR

    def test_is_valid_bash(self):
        r = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr


class TestSafetyProperties:
    def _text(self) -> str:
        return SCRIPT.read_text()

    def test_never_escalates_privileges(self):
        """C libraries need root; the installer prints the command instead.

        A quickstart that sudo's behind your back is not one anybody should
        paste into a terminal.
        """
        body = re.sub(r"^\s*#.*$", "", self._text(), flags=re.M)
        assert "sudo" not in body

    def test_aborts_on_error(self):
        assert "set -euo pipefail" in self._text()

    def test_does_not_pipe_the_internet_into_a_shell(self):
        text = self._text()
        assert "curl" not in text and "wget" not in text

    def test_runs_from_its_own_directory(self):
        """Pasting the command from anywhere must still work."""
        assert 'cd "$(dirname "$0")"' in self._text()

    def test_refuses_when_not_in_a_checkout(self):
        assert "install.sh not found" in self._text()

    def test_execs_the_wizard_rather_than_forking(self):
        assert re.search(r"exec\s+\"\$PY\"", self._text())


class TestNonInteractive:
    def test_explains_instead_of_hanging(self, tmp_path):
        """Piped into a shell with no tty, it must not block on a read."""
        r = subprocess.run(
            ["bash", str(SCRIPT), "--no-wizard"],
            cwd=ROOT, stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=300,
        )
        assert r.returncode == 0, r.stderr[-500:]

    def test_no_wizard_flag_is_honoured(self):
        assert "--no-wizard" in SCRIPT.read_text()

    def test_says_how_to_run_the_wizard_later(self):
        assert "--setup" in SCRIPT.read_text()


class TestReadmeBlock:
    def _text(self) -> str:
        return README.read_text()

    def test_section_is_present_with_its_subtitle(self):
        text = self._text()
        assert "## Faster Funnier" in text
        assert "One shot install" in text

    def test_appears_before_the_install_section(self):
        """It is the fast path; burying it under the long one defeats it."""
        text = self._text()
        assert text.index("## Faster Funnier") < text.index("## Install")

    def test_appears_after_the_description(self):
        text = self._text()
        assert text.index("## What it does") < text.index("## Faster Funnier")

    def test_advertises_a_script_that_exists(self):
        """Regression guard: renaming the script must break this test."""
        assert SCRIPT.name in self._text()

    def test_clone_url_matches_the_repository(self):
        text = self._text()
        assert "github.com/CowboyPilot/zello-link" in text

    def test_offers_a_one_liner_that_needs_no_git(self):
        """git is absent on plenty of Raspberry Pi and appliance images."""
        text = self._text()
        assert re.search(
            r"wget \S+ https://\S+\.tar\.gz \| tar xz && cd \S+ && \./quickstart\.sh",
            text,
        ), "expected a wget|tar one-liner"

    def test_offers_curl_for_macos(self):
        """macOS ships curl and not wget, so wget alone strands Mac users."""
        assert re.search(
            r"curl \S+ https://\S+\.tar\.gz \| tar xz && cd \S+ && \./quickstart\.sh",
            self._text(),
        )

    def test_still_offers_git_for_those_who_have_it(self):
        assert re.search(
            r"git clone \S+ && cd \S+ && \./quickstart\.sh", self._text()
        )

    def test_tarball_directory_matches_githubs_naming(self):
        """A GitHub branch tarball extracts to <repo>-<branch>, not <repo>."""
        text = self._text()
        assert "zello-link-main" in text

    def test_does_not_pipe_a_script_into_a_shell(self):
        """Piping the archive into tar is fine; piping code into sh is not.

        The distinction is what makes quickstart.sh reviewable before it runs,
        so it must not quietly become `| bash` later.
        """
        text = self._text()
        assert "| bash" not in text and "| sh" not in text

    def test_says_the_archive_is_extracted_before_anything_runs(self):
        text = " ".join(self._text().lower().split())
        assert "extracted before anything runs" in text


class TestReadmeReflectsShippedFeatures:
    """The front page said USRP was "planned" long after it was carrying audio.

    Documentation drift is cheap to create and expensive to find: someone
    evaluating the project reads the README and concludes a working feature
    does not exist. These pin the claims that went stale.
    """

    def _text(self) -> str:
        return README.read_text()

    def test_usrp_is_not_described_as_unreleased(self):
        text = self._text().lower()
        for phrase in ("specified for a future release", "planned second backend"):
            assert phrase not in text, f"README still calls the USRP backend {phrase!r}"

    def test_allstarlink_has_its_own_section(self):
        assert "### AllStarLink (USRP)" in self._text()

    def test_duplex_requirement_is_documented(self):
        """The single setting that silently prevents ASL from ever keying."""
        text = self._text()
        assert "duplex = 3" in text

    def test_loopback_is_recommended_over_the_network(self):
        # Collapse wrapping: prose is hard-wrapped, so phrases span newlines.
        text = " ".join(self._text().lower().split())
        assert "loopback" in text
        assert "no authentication" in text

    def test_verified_hardware_is_listed(self):
        text = self._text()
        for path in ("Digirig Mobile", "Digirig Lite", "CM108 HID PTT"):
            assert path in text

    def test_removed_vcos_is_not_offered(self):
        """It must not read as an available option anywhere on the page."""
        text = self._text()
        if "aioc_virtual" in text:
            assert "removed" in text.lower()

    def test_explains_why_a_virtualenv_is_required(self):
        """It is a PEP 668 constraint, not a style choice, and gets asked about.

        Debian 12 refuses system-wide pip, and its packaged pydantic is 1.10
        against a 2.6+ requirement -- so "can we skip the venv?" has a real
        answer that belongs on the page.
        """
        text = " ".join(README.read_text().split())
        assert "EXTERNALLY-MANAGED" in text
        assert "pydantic" in text
        assert "nothing to activate" in text.lower()
