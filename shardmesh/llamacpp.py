"""Run one stage of a pipeline on llama.cpp's own kernels, through its C API.

Uses a stock release build, loaded with ctypes. The structures below match
llama.h at PINNED_BUILD exactly; another build needs them checked again.

A stage takes hidden states through ``llama_batch.embd``. A stage that is not
the last reads its output from the ``l_out`` tensor of its final layer with the
eval callback, then stops the graph there, skipping the output norm and LM head.
"""

from __future__ import annotations

import ctypes as C
import hashlib
import importlib.util
import os
from pathlib import Path
import platform
import shutil
import sys
import tarfile
import threading
from typing import Any
import zipfile

import httpx
import numpy as np

PINNED_BUILD = "b12119"
RELEASE_URL = f"https://github.com/ggml-org/llama.cpp/releases/download/{PINNED_BUILD}/"
# Release files for each platform and backend, with their published SHA-256.
BUILDS: dict[str, dict[str, list[tuple[str, str]]]] = {
    "windows-x64": {
        "cuda": [
            (f"llama-{PINNED_BUILD}-bin-win-cuda-12.4-x64.zip", "944fea366abc778026d73213277ea104a7e2413b265f42193fcf2da1f5cdd60f"),
            ("cudart-llama-bin-win-cuda-12.4-x64.zip", "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6"),
        ],
        "rocm": [(f"llama-{PINNED_BUILD}-bin-win-rocm-10.0-x64.zip", "f09d7639c4c401e4c715c56ad8ed1c734e0116a9cb3fd6f381aebf8149cf29f8")],
        "vulkan": [(f"llama-{PINNED_BUILD}-bin-win-vulkan-x64.zip", "da6f3e49f5b2383bc79dd9bfcb17205e69df4aaa8ccf72d3db02da72cdb5e89a")],
        "cpu": [(f"llama-{PINNED_BUILD}-bin-win-cpu-x64.zip", "37a6c6c9eae5b7baa2619cba513781fad15ca0f63f7d07362ec83a294cf96f86")],
    },
    "linux-x64": {
        "cpu": [(f"llama-{PINNED_BUILD}-bin-ubuntu-x64.tar.gz", "b766529b1a5082c65b9dc5a5f1a78092c6e107344dc383c53a00b83da47c16a6")],
        "vulkan": [(f"llama-{PINNED_BUILD}-bin-ubuntu-vulkan-x64.tar.gz", "6d4f7e69257f57f0897a6a562d2b31d11055c55748672d8f0955476174e0bc11")],
        "rocm": [(f"llama-{PINNED_BUILD}-bin-ubuntu-rocm-10.0-x64.tar.gz", "42751a517b335aaaacf0ef171bd50af619e02d9d5200d937fdcb46aab708be12")],
        "cuda": [
            (f"llama-{PINNED_BUILD}-bin-ubuntu-cuda-12.8-x64.tar.gz", "d61b6c35f56f3cf74ca8a0ec5bdd8c8dd1b5c304f9e001bf9ddb29f8d4d2a409"),
            (f"cudart-llama-{PINNED_BUILD}-bin-ubuntu-cuda-12.8-x64.tar.gz", "40ddc5e887003a5ac19f1baf0681d9bedb52bf7cf57b9fa896dbaf6439e53b7c"),
        ],
    },
    "linux-arm64": {"cpu": [(f"llama-{PINNED_BUILD}-bin-ubuntu-arm64.tar.gz", "c8329d08022d6d7a029d6babb6479dd8d5a606f0073fae3648d045d7c33c908a")]},
    # The Mac builds carry Metal, so one build serves CPU and GPU there.
    "macos-arm64": {"cpu": [(f"llama-{PINNED_BUILD}-bin-macos-arm64.tar.gz", "51fc160bc6f6a6b53fef3c50d12b27314cfaf8fc37766fc605e0bd30d87a79b8")]},
    "macos-x64": {"cpu": [(f"llama-{PINNED_BUILD}-bin-macos-x64.tar.gz", "79cdaf30cfd10d6ebbb39f65ae95d542bf526730292698056f39443d9d3f7f29")]},
}
# What a machine has to provide itself, and the package that does on Debian and Ubuntu.
SYSTEM_LIBRARIES = {"libgomp": "libgomp1", "libvulkan": "libvulkan1", "libstdc++": "libstdc++6"}


class TensorOverride(C.Structure):
    _fields_ = [("pattern", C.c_char_p), ("buft", C.c_void_p)]


class ModelParams(C.Structure):
    _fields_ = [
        ("devices", C.c_void_p),
        ("tensor_buft_overrides", C.c_void_p),
        ("n_gpu_layers", C.c_int32),
        ("split_mode", C.c_int32),
        ("load_mode", C.c_int32),
        ("lazy_mode", C.c_int32),
        ("main_gpu", C.c_int32),
        ("tensor_split", C.c_void_p),
        ("progress_callback", C.c_void_p),
        ("progress_callback_user_data", C.c_void_p),
        ("kv_overrides", C.c_void_p),
        ("vocab_only", C.c_bool),
        ("check_tensors", C.c_bool),
        ("use_extra_bufts", C.c_bool),
        ("no_host", C.c_bool),
        ("no_alloc", C.c_bool),
        ("load_mtp", C.c_bool),
    ]


LOAD_READ = 0
EVAL_CALLBACK = C.CFUNCTYPE(C.c_bool, C.c_void_p, C.c_bool, C.c_void_p)


class ContextParams(C.Structure):
    _fields_ = [
        ("n_ctx", C.c_uint32),
        ("n_batch", C.c_uint32),
        ("n_ubatch", C.c_uint32),
        ("n_seq_max", C.c_uint32),
        ("n_rs_seq", C.c_uint32),
        ("n_outputs_max", C.c_uint32),
        ("n_outputs_max_per_seq", C.c_uint32),
        ("n_threads", C.c_int32),
        ("n_threads_batch", C.c_int32),
        ("ctx_type", C.c_int32),
        ("rope_scaling_type", C.c_int32),
        ("pooling_type", C.c_int32),
        ("attention_type", C.c_int32),
        ("flash_attn_type", C.c_int32),
        ("rope_freq_base", C.c_float),
        ("rope_freq_scale", C.c_float),
        ("yarn_ext_factor", C.c_float),
        ("yarn_attn_factor", C.c_float),
        ("yarn_beta_fast", C.c_float),
        ("yarn_beta_slow", C.c_float),
        ("yarn_orig_ctx", C.c_uint32),
        ("defrag_thold", C.c_float),
        ("cb_eval", C.c_void_p),
        ("cb_eval_user_data", C.c_void_p),
        ("type_k", C.c_int32),
        ("type_v", C.c_int32),
        ("moe_cache_size", C.c_size_t),
        ("abort_callback", C.c_void_p),
        ("abort_callback_data", C.c_void_p),
        ("embeddings", C.c_bool),
        ("offload_kqv", C.c_bool),
        ("no_perf", C.c_bool),
        ("op_offload", C.c_bool),
        ("swa_full", C.c_bool),
        ("kv_unified", C.c_bool),
        ("samplers", C.c_void_p),
        ("n_samplers", C.c_size_t),
        ("ctx_other", C.c_void_p),
    ]


class Batch(C.Structure):
    _fields_ = [
        ("n_tokens", C.c_int32),
        ("token", C.c_void_p),
        ("embd", C.c_void_p),
        ("pos", C.c_void_p),
        ("n_seq_id", C.c_void_p),
        ("seq_id", C.c_void_p),
        ("logits", C.c_void_p),
    ]


class TypeTraits(C.Structure):
    _fields_ = [
        ("type_name", C.c_char_p),
        ("blck_size", C.c_int64),
        ("blck_size_interleave", C.c_int64),
        ("type_size", C.c_size_t),
        ("is_quantized", C.c_bool),
        ("to_float", C.c_void_p),
        ("from_float_ref", C.c_void_p),
    ]


LOG_CALLBACK = C.CFUNCTYPE(None, C.c_int, C.c_char_p, C.c_void_p)
TO_FLOAT = C.CFUNCTYPE(None, C.c_void_p, C.c_void_p, C.c_int64)
GGML_LOG_ERROR = 4


# Nucleus sampling keeps the head of the distribution, so a full sort of a
# quarter-million logits per row would be wasted work.
NUCLEUS_CANDIDATES = 4096


class LlamaError(RuntimeError):
    pass


class StageLost(LlamaError):
    """A stage process died. The worker is fine and can start another."""

    def __init__(self, message: str, host: Any = None) -> None:
        super().__init__(message)
        self.host = host  # which process, since a reply can report one already replaced


def step_random(seed: int | None, position: int) -> np.random.Generator:
    """The random numbers for one position of a reply.

    A replay must repeat its draws, so they follow from the seed and the position.
    Adding the two would not do: seed 5 at position 9 would share a draw with seed
    6 at position 8, and one unlucky draw would strike every reply with a nearby
    seed, at a position a step earlier each time. Measured doing exactly that.
    """
    return np.random.default_rng(None if seed is None else [seed, position])


def filtered_distribution(logits: np.ndarray, temperature: float, top_p: float, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    """Token ids and probabilities after top-k, temperature, and top-p, in that order."""
    limit = top_k or (NUCLEUS_CANDIDATES if top_p < 1.0 else 0)
    ids = np.argpartition(logits, -limit)[-limit:] if 0 < limit < logits.size else np.arange(logits.size)
    scaled = logits[ids].astype(np.float64) / temperature
    scaled -= scaled.max()
    probabilities = np.exp(scaled)
    probabilities /= probabilities.sum()
    if top_p < 1.0:
        order = np.argsort(probabilities)[::-1]
        ids, probabilities = ids[order], probabilities[order]
        keep = np.cumsum(probabilities) - probabilities <= top_p
        ids, probabilities = ids[keep], probabilities[keep]
        probabilities = probabilities / probabilities.sum()
    return ids, probabilities


def speculative_choice(
    logits_for: Any,
    guesses: list[int],
    temperature: float,
    top_p: float,
    top_k: int,
    generator: np.random.Generator,
) -> list[int]:
    """Decide what a speculative pass produced. ``logits_for(i)`` gives row ``i``.

    Row 0 follows the last real token, and row ``i`` follows guess ``i - 1``. At
    temperature zero a guess stands if it is the model's own choice.

    Sampling must keep the model's distribution, not just its best guess. A guess
    ``g`` is accepted with probability ``q(g)``, and a rejection draws from ``q``
    without ``g``, renormalized. Any token ``x`` then comes out with probability
    exactly ``q(x)``, so the draft changes the speed and never the distribution.
    """
    accepted: list[int] = []
    for row in range(len(guesses) + 1):
        logits = logits_for(row)
        guess = guesses[row] if row < len(guesses) else None
        if temperature == 0:
            choice = int(np.argmax(logits))
            accepted.append(choice)
            if choice != guess:
                return accepted
            continue
        ids, probabilities = filtered_distribution(logits, temperature, top_p, top_k)
        if guess is not None:
            found = np.flatnonzero(ids == guess)
            chance = float(probabilities[found[0]]) if found.size else 0.0
            if generator.random() < chance:
                accepted.append(guess)
                continue
            if found.size:
                ids, probabilities = np.delete(ids, found[0]), np.delete(probabilities, found[0])
            if not ids.size:
                accepted.append(int(np.argmax(logits)))
                return accepted
            probabilities = probabilities / probabilities.sum()
        accepted.append(int(generator.choice(ids, p=probabilities)))
        return accepted
    return accepted


def pick_build(requested_device: str = "auto") -> str:
    """Choose the release build for this machine's accelerator."""
    import torch

    if str(requested_device).startswith("cpu"):
        return "cpu"
    if str(requested_device).startswith("vulkan"):
        return "vulkan"
    if torch.version.hip:
        return "rocm"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def default_build_root() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    return Path(base) / "ShardMesh" / "llama" if base else Path.home() / ".cache" / "shardmesh" / "llama"


def _download(url: str, destination: Path, expected: str, tls: Any) -> None:
    digest = hashlib.sha256()
    partial = destination.with_suffix(".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=httpx.Timeout(60.0, connect=20.0), verify=tls) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            for chunk in response.iter_bytes(1024 * 1024):
                handle.write(chunk)
                digest.update(chunk)
    if digest.hexdigest() != expected:
        partial.unlink(missing_ok=True)
        raise LlamaError(f"{url} failed its hash check")
    partial.replace(destination)


def this_platform() -> str:
    system = "windows" if os.name == "nt" else "macos" if sys.platform == "darwin" else "linux"
    arm = platform.machine().lower() in {"arm64", "aarch64"}
    return f"{system}-{'arm64' if arm else 'x64'}"


def library_file(name: str) -> str:
    """A native library's file name here: llama.dll, libllama.so, or libllama.dylib."""
    if os.name == "nt":
        return f"{name}.dll"
    return f"lib{name}{'.dylib' if sys.platform == 'darwin' else '.so'}"


def build_folder(folder: Path) -> Path | None:
    """Where a build keeps its libraries. Release archives nest them a level down."""
    return next((item.parent for item in sorted(folder.rglob(library_file("llama")))), None) if folder.is_dir() else None


def unpack(archive: Path, folder: Path) -> None:
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(folder)
        return
    with tarfile.open(archive) as bundle:
        bundle.extractall(folder, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))


def ensure_build(kind: str, root: Path | None = None) -> Path:
    """Return a folder holding this backend's build, downloading it if needed."""
    from .weights import tls_context

    choices = BUILDS.get(this_platform())
    if not choices or kind not in choices:
        offered = ", ".join(choices or ()) or "nothing"
        raise LlamaError(f"no {kind} build of llama.cpp for {this_platform()} (it offers {offered}); set --llama-dir to a build")
    folder = (root or default_build_root()) / kind
    found = build_folder(folder)
    if found:
        return found
    folder.mkdir(parents=True, exist_ok=True)
    for name, expected in choices[kind]:
        archive = folder.parent / name
        if not archive.is_file() or hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
            print(f"Downloading {name}", flush=True)
            _download(RELEASE_URL + name, archive, expected, tls_context())
        unpack(archive, folder)
    found = build_folder(folder)
    if not found:
        shutil.rmtree(folder, ignore_errors=True)
        raise LlamaError(f"the {kind} build has no {library_file('llama')}")
    return found


def load_native(path: Path) -> C.CDLL:
    """Load a library, saying which package to install if the machine lacks one."""
    try:
        return C.CDLL(str(path))
    except OSError as error:
        text = str(error)
        package = next((package for library, package in SYSTEM_LIBRARIES.items() if library in text), None)
        advice = f" On Debian or Ubuntu, install it with: sudo apt-get install {package}" if package else ""
        raise LlamaError(f"could not load {path.name}: {text}.{advice}") from error


def rocm_library_paths() -> tuple[str, ...]:
    """Windows ROCm builds leave out hipblas.dll. PyTorch's ROCm wheels carry it."""
    spec = importlib.util.find_spec("_rocm_sdk_libraries")
    if spec is None or not spec.submodule_search_locations:
        return ()
    return tuple(str(Path(entry) / "bin") for entry in spec.submodule_search_locations)


class LlamaLibrary:
    """The native library, loaded once per process."""

    _instance: "LlamaLibrary | None" = None
    _lock = threading.Lock()

    def __init__(self, folder: str, extra_paths: tuple[str, ...] = ()) -> None:
        self.folder = str(Path(folder).resolve())
        search = [self.folder, *extra_paths]
        os.environ["PATH"] = os.pathsep.join([*search, os.environ.get("PATH", "")])
        if hasattr(os, "add_dll_directory"):
            for entry in search:
                os.add_dll_directory(entry)
        self.ggml_base = load_native(Path(self.folder) / library_file("ggml-base"))
        self.ggml = load_native(Path(self.folder) / library_file("ggml"))
        self.llama = load_native(Path(self.folder) / library_file("llama"))
        self._bind()
        self._check_binding()
        self._log = LOG_CALLBACK(self._on_log)
        self.llama.llama_log_set(self._log, None)
        self.ggml.ggml_backend_load_all_from_path(self.folder.encode())
        self.llama.llama_backend_init()

    def _check_binding(self) -> None:
        """Refuse a build whose structures have moved.

        ctypes cannot see llama.h, so a different build would quietly read and
        write the wrong fields. The defaults sit across both structures, and any
        shift scrambles them.
        """
        model = self.llama.llama_model_default_params()
        context = self.llama.llama_context_default_params()
        checks = (
            model.n_gpu_layers == -1,
            model.main_gpu == 0,
            model.use_extra_bufts is True,
            model.load_mtp is False,
            context.n_ctx == 512,
            context.n_batch == 2048,
            context.n_ubatch == 512,
            context.n_seq_max == 1,
            context.yarn_ext_factor == -1.0,
            context.cb_eval is None,
            context.offload_kqv is True,
            context.ctx_other is None,
        )
        if not all(checks):
            raise LlamaError(f"this llama.cpp build does not match the binding for {PINNED_BUILD}; use that build")

    @classmethod
    def load(cls, folder: str, extra_paths: tuple[str, ...] = ()) -> "LlamaLibrary":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(folder, extra_paths)
            return cls._instance

    @staticmethod
    def _on_log(level: int, text: bytes | None, _: Any) -> None:
        if level == GGML_LOG_ERROR and text:
            print("llama.cpp:", text.decode("utf-8", "replace").rstrip(), flush=True)

    def _bind(self) -> None:
        lib, base = self.llama, self.ggml_base
        self.ggml.ggml_backend_load_all_from_path.argtypes = [C.c_char_p]
        self.ggml.ggml_backend_load_all_from_path.restype = None
        lib.llama_log_set.argtypes = [LOG_CALLBACK, C.c_void_p]
        lib.llama_log_set.restype = None
        lib.llama_backend_init.restype = None
        lib.llama_model_default_params.restype = ModelParams
        lib.llama_context_default_params.restype = ContextParams
        lib.llama_model_load_from_file.argtypes = [C.c_char_p, ModelParams]
        lib.llama_model_load_from_file.restype = C.c_void_p
        lib.llama_init_from_model.argtypes = [C.c_void_p, ContextParams]
        lib.llama_init_from_model.restype = C.c_void_p
        lib.llama_model_free.argtypes = [C.c_void_p]
        lib.llama_model_free.restype = None
        lib.llama_free.argtypes = [C.c_void_p]
        lib.llama_free.restype = None
        lib.llama_decode.argtypes = [C.c_void_p, Batch]
        lib.llama_decode.restype = C.c_int32
        lib.llama_get_logits_ith.argtypes = [C.c_void_p, C.c_int32]
        lib.llama_get_logits_ith.restype = C.POINTER(C.c_float)
        lib.llama_get_memory.argtypes = [C.c_void_p]
        lib.llama_get_memory.restype = C.c_void_p
        lib.llama_memory_clear.argtypes = [C.c_void_p, C.c_bool]
        lib.llama_memory_clear.restype = None
        lib.llama_memory_seq_rm.argtypes = [C.c_void_p, C.c_int32, C.c_int32, C.c_int32]
        lib.llama_memory_seq_rm.restype = C.c_bool
        lib.llama_memory_seq_cp.argtypes = [C.c_void_p, C.c_int32, C.c_int32, C.c_int32, C.c_int32]
        lib.llama_memory_seq_cp.restype = None
        lib.llama_state_seq_get_size.argtypes = [C.c_void_p, C.c_int32]
        lib.llama_state_seq_get_size.restype = C.c_size_t
        lib.llama_state_seq_get_data.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int32]
        lib.llama_state_seq_get_data.restype = C.c_size_t
        lib.llama_state_seq_set_data.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int32]
        lib.llama_state_seq_set_data.restype = C.c_size_t
        lib.llama_model_n_embd.argtypes = [C.c_void_p]
        lib.llama_model_n_embd.restype = C.c_int32
        lib.llama_model_n_layer.argtypes = [C.c_void_p]
        lib.llama_model_n_layer.restype = C.c_int32
        lib.llama_model_get_vocab.argtypes = [C.c_void_p]
        lib.llama_model_get_vocab.restype = C.c_void_p
        lib.llama_vocab_n_tokens.argtypes = [C.c_void_p]
        lib.llama_vocab_n_tokens.restype = C.c_int32
        lib.llama_synchronize.argtypes = [C.c_void_p]
        lib.llama_synchronize.restype = None
        base.ggml_get_name.argtypes = [C.c_void_p]
        base.ggml_get_name.restype = C.c_char_p
        base.ggml_nbytes.argtypes = [C.c_void_p]
        base.ggml_nbytes.restype = C.c_size_t
        base.ggml_backend_tensor_get.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.c_size_t]
        base.ggml_backend_tensor_get.restype = None
        base.ggml_get_type_traits.argtypes = [C.c_int]
        base.ggml_get_type_traits.restype = C.POINTER(TypeTraits)
        self.ggml.ggml_backend_dev_by_type.argtypes = [C.c_int]
        self.ggml.ggml_backend_dev_by_type.restype = C.c_void_p
        base.ggml_backend_dev_buffer_type.argtypes = [C.c_void_p]
        base.ggml_backend_dev_buffer_type.restype = C.c_void_p

    def cpu_buffer_type(self) -> int:
        device = self.ggml.ggml_backend_dev_by_type(0)  # GGML_BACKEND_DEVICE_TYPE_CPU
        if not device:
            raise LlamaError("llama.cpp has no CPU backend loaded")
        return self.ggml_base.ggml_backend_dev_buffer_type(device)


class LlamaStage:
    """A loaded stage file with its own KV and recurrent state, for one sequence."""

    def __init__(
        self,
        library: LlamaLibrary,
        path: str,
        *,
        last: bool,
        context_tokens: int = 4096,
        gpu_layers: int = -1,
        threads: int = 4,
        batch_tokens: int = 512,
        cpu_tensors: list[str] | None = None,
        rollback_tokens: int = 0,
        sequences: int = 1,
        unified_cache: bool = False,
        map_file: bool = True,
    ) -> None:
        self.library = library
        self.last = last
        lib = library.llama
        params = lib.llama_model_default_params()
        params.n_gpu_layers = gpu_layers
        if not map_file:
            # A mapped file stays resident in RAM even when its tensors live in
            # VRAM: measured at 15 GiB for a 15 GiB model. Reading it once uses 1 GiB.
            params.load_mode = LOAD_READ
        overrides = None
        if cpu_tensors and gpu_layers:
            buffer_type = library.cpu_buffer_type()
            overrides = (TensorOverride * (len(cpu_tensors) + 1))(
                *[TensorOverride(pattern.encode(), buffer_type) for pattern in cpu_tensors]
            )
            params.tensor_buft_overrides = C.cast(overrides, C.c_void_p).value
        self.model = lib.llama_model_load_from_file(path.encode("utf-8"), params)
        if not self.model:
            raise LlamaError(f"llama.cpp could not load {path}")
        self.hidden_size = lib.llama_model_n_embd(self.model)
        self.layers = lib.llama_model_n_layer(self.model)
        self.vocab_size = lib.llama_vocab_n_tokens(lib.llama_model_get_vocab(self.model))
        self.batch_tokens = batch_tokens
        self._target = f"l_out-{self.layers - 1}".encode()
        self._captured: np.ndarray | None = None
        # llama.cpp can split a pass into several, and this sees each of them.
        self._parts: list[np.ndarray] = []
        self._collecting = False
        self._callback = EVAL_CALLBACK(self._on_eval)

        context = lib.llama_context_default_params()
        context.n_ctx = context_tokens * sequences  # the cache is shared out between sequences
        context.n_batch = batch_tokens
        context.n_ubatch = batch_tokens
        context.n_seq_max = sequences
        # Recurrent state cannot rewind by itself, so keep a snapshot per token.
        context.n_rs_seq = rollback_tokens
        context.n_threads = threads
        context.n_threads_batch = threads
        context.no_perf = True
        context.kv_unified = unified_cache
        if not last:
            context.cb_eval = C.cast(self._callback, C.c_void_p).value
        self.context = lib.llama_init_from_model(self.model, context)
        if not self.context:
            lib.llama_model_free(self.model)
            raise LlamaError("llama.cpp could not create a context")
        self.memory = lib.llama_get_memory(self.context)

    def _on_eval(self, tensor: int, ask: bool, _: Any) -> bool:
        base = self.library.ggml_base
        if ask:
            return base.ggml_get_name(tensor) == self._target
        size = base.ggml_nbytes(tensor)
        captured = np.empty(size // 4, dtype=np.float32)
        base.ggml_backend_tensor_get(tensor, captured.ctypes.data, 0, size)
        self._captured = captured
        if self._collecting:
            self._parts.append(captured)
        return False  # stop here: the output norm and LM head are not needed

    def clear(self, slot: int | None = None) -> None:
        if slot is None:
            self.library.llama.llama_memory_clear(self.memory, True)
        else:
            self.library.llama.llama_memory_seq_rm(self.memory, slot, -1, -1)

    def save_state(self, slot: int = 0) -> np.ndarray:
        """Everything this stage holds for a sequence: attention and recurrent state."""
        lib = self.library.llama
        size = lib.llama_state_seq_get_size(self.context, slot)
        state = np.empty(size, dtype=np.uint8)
        written = lib.llama_state_seq_get_data(self.context, state.ctypes.data, size, slot)
        if not written:
            raise LlamaError("llama.cpp could not save this sequence")
        return state[:written]

    def restore_state(self, state: np.ndarray, slot: int = 0) -> None:
        self.clear(slot)
        if not self.library.llama.llama_state_seq_set_data(self.context, state.ctypes.data, state.size, slot):
            raise LlamaError("llama.cpp could not restore this sequence")

    def truncate(self, keep: int, slot: int = 0) -> None:
        """Forget positions from ``keep`` on. Recurrent models may refuse."""
        if not self.library.llama.llama_memory_seq_rm(self.memory, slot, keep, -1):
            raise LlamaError("this model cannot rewind; clear and replay instead")

    def _decode(
        self,
        count: int,
        position: int,
        want_all: bool,
        hidden: np.ndarray | None = None,
        tokens: np.ndarray | None = None,
        sequence_id: int = 0,
    ) -> None:
        steps = np.arange(position, position + count, dtype=np.int32)
        if tokens is None:
            # Embedding input reads every position section itself.
            positions = np.concatenate([steps, steps, steps, np.zeros(count, dtype=np.int32)])
        else:
            positions = steps
        sequence_counts = np.ones(count, dtype=np.int32)
        sequence = C.c_int32(sequence_id)
        sequence_ids = (C.POINTER(C.c_int32) * count)(*([C.pointer(sequence)] * count))
        outputs = np.ones(count, dtype=np.int8) if want_all else np.zeros(count, dtype=np.int8)
        outputs[-1] = 1
        batch = Batch(
            n_tokens=count,
            token=None if tokens is None else tokens.ctypes.data,
            embd=None if hidden is None else hidden.ctypes.data,
            pos=positions.ctypes.data,
            n_seq_id=sequence_counts.ctypes.data,
            seq_id=C.cast(sequence_ids, C.c_void_p).value,
            logits=outputs.ctypes.data,
        )
        status = self.library.llama.llama_decode(self.context, batch)
        if status != 0:
            raise LlamaError(f"llama_decode failed with status {status}")

    def logits_at(self, index: int) -> np.ndarray:
        pointer = self.library.llama.llama_get_logits_ith(self.context, index)
        if not pointer:
            raise LlamaError(f"no logits for output {index}")
        return np.ctypeslib.as_array(pointer, shape=(self.vocab_size,))

    def decode_tokens(self, tokens: list[int], position: int, *, want_all: bool = False, sequence_id: int = 0) -> None:
        """Run token ids through a whole model, in chunks the batch can hold."""
        ids = np.array(tokens, dtype=np.int32)
        for start in range(0, len(ids), self.batch_tokens):
            chunk = np.ascontiguousarray(ids[start : start + self.batch_tokens])
            self._decode(len(chunk), position + start, want_all, tokens=chunk, sequence_id=sequence_id)

    def verify(
        self,
        hidden: np.ndarray,
        position: int,
        guesses: list[int],
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        seed: int | None = None,
        slot: int = 0,
    ) -> list[int]:
        """Check guesses against this model and return the tokens that came out of it."""
        hidden = np.ascontiguousarray(hidden, dtype=np.float32)
        self._decode(hidden.shape[0], position, True, hidden=hidden, sequence_id=slot)
        return speculative_choice(self.logits_at, guesses, temperature, top_p, top_k, step_random(seed, position))

    def guess(self, commit: list[int], position: int, count: int, slot: int = 0, scratch: int = 1) -> list[int]:
        """Extend a draft model's committed text, then guess ``count`` tokens after it.

        Guesses run on a second sequence that is thrown away afterward, because
        recurrent state cannot rewind across separate single-token passes.
        """
        lib = self.library.llama
        self.decode_tokens(commit, position, sequence_id=slot)
        guess = int(np.argmax(self.logits_at(-1)))
        guesses = [guess]
        if count > 1:
            lib.llama_memory_seq_cp(self.memory, slot, scratch, -1, -1)
            cursor = position + len(commit)
            for _ in range(count - 1):
                self.decode_tokens([guess], cursor, sequence_id=scratch)
                guess = int(np.argmax(self.logits_at(-1)))
                guesses.append(guess)
                cursor += 1
            lib.llama_memory_seq_rm(self.memory, scratch, -1, -1)
        return guesses

    def forward(self, hidden: np.ndarray, position: int, slot: int = 0) -> np.ndarray:
        """Run hidden states [tokens, width]. Returns hidden states, or the last row's logits."""
        hidden = np.ascontiguousarray(hidden, dtype=np.float32)
        pieces = []
        for start in range(0, hidden.shape[0], self.batch_tokens):
            chunk = hidden[start : start + self.batch_tokens]
            self._captured = None
            self._decode(chunk.shape[0], position + start, not self.last, hidden=chunk, sequence_id=slot)
            if not self.last:
                if self._captured is None:
                    raise LlamaError("the stage output tensor was never produced")
                pieces.append(self._captured.reshape(-1, self.hidden_size))
        if self.last:
            return self.logits_at(-1).copy()
        return np.concatenate(pieces) if len(pieces) > 1 else pieces[0]

    def forward_many(self, hidden: np.ndarray, positions: list[int], slots: list[int]) -> np.ndarray:
        """One token for each of several conversations, in a single pass.

        Reading the weights is what a pass costs, so several conversations share
        it for little more than one. Rows must be in slot order, which is the
        order llama.cpp gathers sequences in, and output rows match them.
        """
        count = hidden.shape[0]
        hidden = np.ascontiguousarray(hidden, dtype=np.float32)
        steps = np.array(positions, dtype=np.int32)
        places = np.concatenate([steps, steps, steps, np.zeros(count, dtype=np.int32)])
        sequence_counts = np.ones(count, dtype=np.int32)
        owners = (C.c_int32 * count)(*slots)
        width = C.sizeof(C.c_int32)
        sequence_ids = (C.POINTER(C.c_int32) * count)(
            *[C.cast(C.addressof(owners) + width * index, C.POINTER(C.c_int32)) for index in range(count)]
        )
        outputs = np.ones(count, dtype=np.int8)
        batch = Batch(
            n_tokens=count,
            token=None,
            embd=hidden.ctypes.data,
            pos=places.ctypes.data,
            n_seq_id=sequence_counts.ctypes.data,
            seq_id=C.cast(sequence_ids, C.c_void_p).value,
            logits=outputs.ctypes.data,
        )
        self._parts = []
        self._collecting = True
        try:
            status = self.library.llama.llama_decode(self.context, batch)
        finally:
            self._collecting = False
        if status != 0:
            raise LlamaError(f"llama_decode failed with status {status}")
        if self.last:
            return np.stack([self.logits_at(index).copy() for index in range(count)])
        rows = np.concatenate(self._parts) if self._parts else np.empty(0, dtype=np.float32)
        if rows.size != count * self.hidden_size:
            raise LlamaError("a merged pass did not produce one row per conversation")
        return rows.reshape(count, self.hidden_size)

    def close(self) -> None:
        lib = self.library.llama
        if getattr(self, "context", None):
            lib.llama_free(self.context)
            self.context = None
        if getattr(self, "model", None):
            lib.llama_model_free(self.model)
            self.model = None


class TokenEmbedding:
    """Look up embedding rows from a quantized table without any GPU memory."""

    def __init__(self, library: LlamaLibrary, data: Any, ggml_type: int, width: int) -> None:
        traits = library.ggml_base.ggml_get_type_traits(ggml_type).contents
        self.width = width
        self.row_bytes = width // traits.blck_size * traits.type_size
        self.data = data
        self.to_float = TO_FLOAT(traits.to_float) if traits.to_float else None

    def rows(self, tokens: list[int]) -> np.ndarray:
        result = np.empty((len(tokens), self.width), dtype=np.float32)
        for index, token in enumerate(tokens):
            start = token * self.row_bytes
            row = bytes(self.data[start : start + self.row_bytes])
            if token < 0 or len(row) != self.row_bytes:
                raise ValueError(f"token {token} is outside the embedding table")
            if self.to_float is None:
                result[index] = np.frombuffer(row, dtype=np.float32)
            else:
                self.to_float(C.cast(C.c_char_p(row), C.c_void_p), result[index].ctypes.data, self.width)
        return result
