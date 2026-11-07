"""PyTorch is optional, because a GGUF node never uses it and it is several GiB.

Code for PyTorch models takes ``torch`` from here. Nothing loads until that code
touches it, and a machine without it hears how to add it.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from types import ModuleType
from typing import Any

NO_TORCH = (
    "this node has no PyTorch, and only GGUF models run without it. "
    'To add it, run: pip install "isharded[torch]", or the installer again with --full'
)


class TorchMissing(RuntimeError):
    """PyTorch code ran on a machine that does not have it."""


def has_torch() -> bool:
    """Whether PyTorch is installed, without paying to import it."""
    return importlib.util.find_spec("torch") is not None


def torch_loaded() -> bool:
    return sys.modules.get("torch") is not None


class LazyModule:
    """A module that imports on first use, so that importing its users stays free."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._module: ModuleType | None = None

    def __getattr__(self, attribute: str) -> Any:
        if self._module is None:
            try:
                self._module = importlib.import_module(self._name)
            except ImportError as exc:
                raise TorchMissing(NO_TORCH) from exc
        return getattr(self._module, attribute)


torch = LazyModule("torch")
functional = LazyModule("torch.nn.functional")
torch_safetensors = LazyModule("safetensors.torch")
