"""The H0 CLI exposes stable flags but cannot make a submission."""

import os
import subprocess
import sys
from pathlib import Path

import ber.cli
import pytest


SRC_ROOT = Path(__file__).resolve().parents[1]


def _run(*args):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_ROOT)
    # -S disables site-packages, including any editable/installed ber distribution.
    return subprocess.run(
        [sys.executable, "-S", "-m", "ber.cli", *args],
        cwd=SRC_ROOT.parent,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_import_and_help():
    assert callable(ber.cli.main)
    assert Path(ber.cli.__file__).resolve() == SRC_ROOT / "ber" / "cli.py"
    result = _run("--help")
    assert result.returncode == 0
    for command in ber.cli.COMMANDS:
        assert command in result.stdout
        stage_help = _run(command, "--help")
        assert stage_help.returncode == 0
        for flag in ("--data-root", "--work-dir", "--output-dir"):
            assert flag in stage_help.stdout


@pytest.mark.parametrize("command", ber.cli.COMMANDS)
def test_stage_flags_and_clear_unimplemented_status(tmp_path, command):
    data, work, output = (tmp_path / part for part in ("data", "work", "output"))
    result = _run(command, "--data-root", str(data), "--work-dir", str(work), "--output-dir", str(output))
    assert result.returncode != 0
    if command == "train":
        assert "H1 pair model API is ready" in result.stderr
    else:
        assert f"{command}: Not implemented in H0" in result.stderr
    assert not work.exists() and not output.exists()


def test_missing_required_paths_fail_usefully():
    result = _run("index", "--data-root", "example")
    assert result.returncode != 0
    assert "--work-dir" in result.stderr and "--output-dir" in result.stderr
