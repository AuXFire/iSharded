"""Tests must not pick up the pool this machine really belongs to."""

import os
from pathlib import Path
import tempfile

os.environ.pop("SHARDMESH_POOL_TOKEN", None)
os.environ["SHARDMESH_POOL_TOKEN_FILE"] = str(Path(tempfile.gettempdir()) / "shardmesh-tests-have-no-token")
