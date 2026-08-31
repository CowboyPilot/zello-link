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

    def test_command_is_copy_pasteable_as_one_line(self):
        text = self._text()
        m = re.search(r"git clone \S+ && cd \S+ && \./quickstart\.sh", text)
        assert m, "the one-shot command should be a single pasteable line"

    def test_says_it_is_not_curl_pipe_bash(self):
        """The distinction is the reason to prefer this form; say so."""
        assert "curl | bash" in self._text() or "curl` | `bash" in self._text()
