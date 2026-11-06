"""List the GPUs a llama.cpp build can see, as JSON.

This runs as its own process, because a broken graphics driver can crash whatever
asks it a question, and that must not be the worker.
"""
from __future__ import annotations

import ctypes as C
import json
import sys
from typing import Any

from .llamacpp import LlamaLibrary

DEVICE_GPU = 1
DEVICE_INTEGRATED = 2


def devices(library: LlamaLibrary) -> list[dict[str, Any]]:
    registry, base = library.ggml, library.ggml_base
    registry.ggml_backend_dev_count.restype = C.c_size_t
    registry.ggml_backend_dev_get.argtypes = [C.c_size_t]
    registry.ggml_backend_dev_get.restype = C.c_void_p
    base.ggml_backend_dev_name.argtypes = [C.c_void_p]
    base.ggml_backend_dev_name.restype = C.c_char_p
    base.ggml_backend_dev_description.argtypes = [C.c_void_p]
    base.ggml_backend_dev_description.restype = C.c_char_p
    base.ggml_backend_dev_type.argtypes = [C.c_void_p]
    base.ggml_backend_dev_type.restype = C.c_int
    base.ggml_backend_dev_memory.argtypes = [C.c_void_p, C.POINTER(C.c_size_t), C.POINTER(C.c_size_t)]
    base.ggml_backend_dev_memory.restype = None
    found = []
    for index in range(registry.ggml_backend_dev_count()):
        device = registry.ggml_backend_dev_get(index)
        kind = base.ggml_backend_dev_type(device)
        if kind not in {DEVICE_GPU, DEVICE_INTEGRATED}:
            continue
        free, total = C.c_size_t(0), C.c_size_t(0)
        base.ggml_backend_dev_memory(device, C.byref(free), C.byref(total))
        found.append(
            {
                "backend": base.ggml_backend_dev_name(device).decode("utf-8", "replace"),
                "name": base.ggml_backend_dev_description(device).decode("utf-8", "replace"),
                "integrated": kind == DEVICE_INTEGRATED,
                "free_bytes": free.value,
                "total_bytes": total.value,
            }
        )
    return found


def main() -> None:
    print(json.dumps(devices(LlamaLibrary(sys.argv[1]))))


if __name__ == "__main__":
    main()
