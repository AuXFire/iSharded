"""iSharded distributed inference prototype."""

from functools import cache
from pathlib import Path
import subprocess

__version__ = "0.3.0"


@cache
def source_commit() -> str | None:
    """The commit of this copy of the code, if it is a git checkout. An installed copy has none."""
    root = Path(__file__).resolve().parents[1]
    if not (root / ".git").exists():
        return None
    try:
        answer = subprocess.run(["git", "-C", str(root), "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return answer.stdout.strip() or None

