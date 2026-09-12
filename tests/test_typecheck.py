"""Some constructions must fail mypy --strict, not only at runtime.

Shells out to the same mypy that gates the package, against fixture files
that are never imported or executed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
FIXTURES = Path(__file__).parent / "typecheck_fixtures"


def _run_mypy_strict(fixture: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--config-file",
            str(REPO_ROOT / "pyproject.toml"),
            str(fixture),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def test_float_price_fails_mypy_strict() -> None:
    result = _run_mypy_strict(FIXTURES / "bad_order_price.py")
    assert result.returncode != 0, (
        f"expected mypy --strict to reject a float Order.price, got a clean pass:\n{result.stdout}"
    )
    assert "error" in result.stdout.lower(), result.stdout
