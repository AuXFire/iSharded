"""Tests must not pick up the pool this machine really belongs to."""

import os
from pathlib import Path
import tempfile

os.environ.pop("ISHARDED_POOL_TOKEN", None)
os.environ["ISHARDED_POOL_TOKEN_FILE"] = str(Path(tempfile.gettempdir()) / "isharded-tests-have-no-token")
