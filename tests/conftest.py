"""Tests must not pick up the pool this machine really belongs to."""

import importlib.util
import os
from pathlib import Path
import re
import tempfile

import pytest

os.environ.pop("ISHARDED_POOL_TOKEN", None)
os.environ["ISHARDED_POOL_TOKEN_FILE"] = str(Path(tempfile.gettempdir()) / "isharded-tests-have-no-token")


NEEDS_TORCH = re.compile(r"^(import torch|from safetensors\.torch |from test_adapters )", re.M)
HERE = Path(__file__).parent


def pytest_ignore_collect(collection_path: Path) -> bool | None:
    """A file that imports PyTorch at the top cannot load without it."""
    if importlib.util.find_spec("torch") is None and collection_path.suffix == ".py" and NEEDS_TORCH.search(collection_path.read_text(encoding="utf-8")):
        return True
    return None


def asked_for_torch_itself(error: BaseException | None) -> bool:
    """Whether a test's own ``import torch`` failed. iSharded's code asking for it is a real failure."""
    if not isinstance(error, ModuleNotFoundError) or error.name != "torch":
        return False
    frame = error.__traceback__
    while frame.tb_next is not None:
        frame = frame.tb_next
    return Path(frame.tb_frame.f_code.co_filename).parent == HERE


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item: pytest.Item):
    outcome = yield
    if outcome.excinfo is not None and asked_for_torch_itself(outcome.excinfo[1]):
        outcome.force_exception(pytest.skip.Exception("needs PyTorch", _use_item_location=True))
