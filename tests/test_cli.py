import subprocess
import sys

import pytest

from goldpass import __version__
from goldpass.cli import ExitCode, main


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"goldpass {__version__}"


def test_command_is_required() -> None:
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == ExitCode.USAGE


def test_compare_needs_two_files() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["compare", "only-one.json"])
    assert exc.value.code == ExitCode.USAGE


def test_module_entry_point() -> None:
    out = subprocess.run(
        [sys.executable, "-m", "goldpass", "--version"], capture_output=True, text=True
    )
    assert out.returncode == 0
    assert out.stdout.startswith("goldpass ")
