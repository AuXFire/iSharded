"""Turn an image into embedding rows with llama.cpp's mtmd library, through its C API.

The structures below match mtmd.h and mtmd-helper.h at PINNED_BUILD exactly, and
mtmd.h warns that this API breaks without notice, so another build needs them
checked again. Like the rest of the native code, this runs in a stage process.

mtmd wants the text model for its hyperparameters and vocabulary, and never for its
weights, so the stage file is loaded vocabulary-only. That vocabulary is a stub. The
text around an image is tokenized by ShardMesh, and only the image chunk is used here.
A vocabulary-only model reports a width of 0, which also turns off mtmd's own check
that the projector fits the model, so that check happens at load, in vision.py.
"""

from __future__ import annotations

import ctypes as C
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .llamacpp import GGML_LOG_ERROR, LOG_CALLBACK, PINNED_BUILD, LlamaError, LlamaLibrary, library_file, load_native
from .vision import SECTIONS

CHUNK_IMAGE = 1  # MTMD_INPUT_CHUNK_TYPE_IMAGE
MEDIA_MARKER = b"<__media__>"


class ContextParams(C.Structure):
    _fields_ = [
        ("use_gpu", C.c_bool),
        ("device", C.c_void_p),
        ("print_timings", C.c_bool),
        ("n_threads", C.c_int),
        ("image_marker", C.c_char_p),
        ("media_marker", C.c_char_p),
        ("flash_attn_type", C.c_int32),
        ("warmup", C.c_bool),
        ("image_min_tokens", C.c_int),
        ("image_max_tokens", C.c_int),
        ("cb_eval", C.c_void_p),
        ("cb_eval_user_data", C.c_void_p),
        ("batch_max_tokens", C.c_int32),
        ("progress_callback", C.c_void_p),
        ("progress_callback_user_data", C.c_void_p),
    ]


class InputText(C.Structure):
    _fields_ = [
        ("text", C.c_char_p),
        ("text_len", C.c_size_t),
        ("add_special", C.c_bool),
        ("parse_special", C.c_bool),
    ]


class VideoParams(C.Structure):
    _fields_ = [
        ("fps_target", C.c_float),
        ("ffmpeg_bin_dir", C.c_char_p),
        ("timestamp_interval_ms", C.c_int64),
        ("no_soundtrack", C.c_bool),
    ]


class InitOptions(C.Structure):
    _fields_ = [("video_params", VideoParams)]


class BitmapWrapper(C.Structure):
    _fields_ = [("bitmap", C.c_void_p), ("video_ctx", C.c_void_p)]


class DecoderPosition(C.Structure):
    _fields_ = [("t", C.c_uint32), ("x", C.c_uint32), ("y", C.c_uint32), ("z", C.c_uint32)]


@dataclass
class ImageShape:
    """What an image takes up in a prompt, before anything has been encoded."""

    rows: int  # embedding rows, which is what it costs in context
    span: int  # positions it moves the sequence on by, fewer than rows under M-RoPE
    places: np.ndarray  # [sections, rows], relative to the image's first position


class Projector:
    """An mmproj file, ready to encode images for one model."""

    def __init__(self, library: LlamaLibrary, projector: str, stage: str, width: int, threads: int, use_gpu: bool) -> None:
        self.library = library
        self.width = width
        path = Path(library.folder) / library_file("mtmd")
        if not path.is_file():
            raise LlamaError(f"this llama.cpp build has no {path.name}, which vision needs")
        self.mtmd = load_native(path)
        self._bind()
        self._check_binding()
        self._log = LOG_CALLBACK(self._on_log)
        self.mtmd.mtmd_helper_log_set(self._log, None)
        lib = library.llama
        model_params = lib.llama_model_default_params()
        model_params.vocab_only = True
        self.model = lib.llama_model_load_from_file(stage.encode("utf-8"), model_params)
        if not self.model:
            raise LlamaError(f"llama.cpp could not read {stage}")
        params = self.mtmd.mtmd_context_params_default()
        params.use_gpu = use_gpu
        params.print_timings = False
        params.n_threads = threads
        self.context = self.mtmd.mtmd_init_from_file(projector.encode("utf-8"), self.model, params)
        if not self.context:
            lib.llama_model_free(self.model)
            self.model = None
            raise LlamaError(f"llama.cpp could not load the projector {projector}; it may belong to another model")
        self.options = self.mtmd.mtmd_helper_init_opt_default()
        if not self.mtmd.mtmd_support_vision(self.context):
            self.close()
            raise LlamaError(f"{projector} has no vision encoder")
        if self.mtmd.mtmd_decode_use_non_causal(self.context, None):
            # Every stage would have to switch its attention for these rows.
            self.close()
            raise LlamaError(f"{projector} needs non-causal attention, which split stages do not support yet")

    @staticmethod
    def _on_log(level: int, text: bytes | None, _: Any) -> None:
        if level == GGML_LOG_ERROR and text:
            print("mtmd:", text.decode("utf-8", "replace").rstrip(), flush=True)

    def _bind(self) -> None:
        native = self.mtmd
        native.mtmd_helper_log_set.argtypes = [LOG_CALLBACK, C.c_void_p]
        native.mtmd_helper_log_set.restype = None
        native.mtmd_default_marker.restype = C.c_char_p
        native.mtmd_context_params_default.restype = ContextParams
        native.mtmd_init_from_file.argtypes = [C.c_char_p, C.c_void_p, ContextParams]
        native.mtmd_init_from_file.restype = C.c_void_p
        native.mtmd_free.argtypes = [C.c_void_p]
        native.mtmd_free.restype = None
        native.mtmd_support_vision.argtypes = [C.c_void_p]
        native.mtmd_support_vision.restype = C.c_bool
        native.mtmd_decode_use_non_causal.argtypes = [C.c_void_p, C.c_void_p]
        native.mtmd_decode_use_non_causal.restype = C.c_bool
        native.mtmd_helper_init_opt_default.restype = InitOptions
        native.mtmd_helper_bitmap_init_from_buf.argtypes = [C.c_void_p, C.c_char_p, C.c_size_t, C.c_bool, InitOptions]
        native.mtmd_helper_bitmap_init_from_buf.restype = BitmapWrapper
        native.mtmd_bitmap_free.argtypes = [C.c_void_p]
        native.mtmd_bitmap_free.restype = None
        native.mtmd_input_chunks_init.restype = C.c_void_p
        native.mtmd_input_chunks_free.argtypes = [C.c_void_p]
        native.mtmd_input_chunks_free.restype = None
        native.mtmd_input_chunks_size.argtypes = [C.c_void_p]
        native.mtmd_input_chunks_size.restype = C.c_size_t
        native.mtmd_input_chunks_get.argtypes = [C.c_void_p, C.c_size_t]
        native.mtmd_input_chunks_get.restype = C.c_void_p
        native.mtmd_input_chunk_get_type.argtypes = [C.c_void_p]
        native.mtmd_input_chunk_get_type.restype = C.c_int
        native.mtmd_input_chunk_get_n_tokens.argtypes = [C.c_void_p]
        native.mtmd_input_chunk_get_n_tokens.restype = C.c_size_t
        native.mtmd_input_chunk_get_n_pos.argtypes = [C.c_void_p]
        native.mtmd_input_chunk_get_n_pos.restype = C.c_int32
        native.mtmd_input_chunk_get_tokens_image.argtypes = [C.c_void_p]
        native.mtmd_input_chunk_get_tokens_image.restype = C.c_void_p
        native.mtmd_tokenize.argtypes = [C.c_void_p, C.c_void_p, C.POINTER(InputText), C.POINTER(C.c_void_p), C.c_size_t]
        native.mtmd_tokenize.restype = C.c_int32
        native.mtmd_encode_chunk.argtypes = [C.c_void_p, C.c_void_p]
        native.mtmd_encode_chunk.restype = C.c_int32
        native.mtmd_get_output_embd.argtypes = [C.c_void_p]
        native.mtmd_get_output_embd.restype = C.POINTER(C.c_float)
        native.mtmd_helper_image_get_decoder_pos.argtypes = [C.c_void_p, C.c_int32, C.POINTER(DecoderPosition)]
        native.mtmd_helper_image_get_decoder_pos.restype = None

    def _check_binding(self) -> None:
        """Refuse a build whose structures have moved, as LlamaLibrary does for llama.h."""
        params = self.mtmd.mtmd_context_params_default()
        checks = (
            params.use_gpu is True,
            params.print_timings is True,
            params.n_threads == 4,
            params.image_marker is None,
            params.media_marker == MEDIA_MARKER,
            params.warmup is True,
            params.image_min_tokens == -1,
            params.image_max_tokens == -1,
            params.cb_eval is None,
            params.batch_max_tokens == 1024,
            params.progress_callback_user_data is None,
            self.mtmd.mtmd_default_marker() == MEDIA_MARKER,
        )
        if not all(checks):
            raise LlamaError(f"this mtmd build does not match the binding for {PINNED_BUILD}; use that build")

    def _image_chunk(self, image: bytes) -> tuple[int, int]:
        """Decode an image and find its chunk. Free the first result when done with both."""
        native = self.mtmd
        wrapper = native.mtmd_helper_bitmap_init_from_buf(self.context, image, len(image), False, self.options)
        if not wrapper.bitmap:
            raise LlamaError("this image could not be read; send a PNG, JPEG, GIF, or BMP")
        chunks = native.mtmd_input_chunks_init()
        try:
            text = InputText(MEDIA_MARKER, len(MEDIA_MARKER), False, True)
            bitmaps = (C.c_void_p * 1)(wrapper.bitmap)
            status = native.mtmd_tokenize(self.context, chunks, C.byref(text), bitmaps, 1)
            if status != 0:
                raise LlamaError(f"this image could not be prepared (mtmd_tokenize returned {status})")
            found = [
                chunk
                for chunk in (native.mtmd_input_chunks_get(chunks, index) for index in range(native.mtmd_input_chunks_size(chunks)))
                if native.mtmd_input_chunk_get_type(chunk) == CHUNK_IMAGE
            ]
            if len(found) != 1:
                raise LlamaError(f"this projector split an image into {len(found)} chunks, which is not supported yet")
            return chunks, found[0]
        except LlamaError:
            native.mtmd_input_chunks_free(chunks)
            raise
        finally:
            native.mtmd_bitmap_free(wrapper.bitmap)

    def _shape(self, chunk: int) -> ImageShape:
        native = self.mtmd
        rows = native.mtmd_input_chunk_get_n_tokens(chunk)
        tokens = native.mtmd_input_chunk_get_tokens_image(chunk)
        # Asked at position zero, these are offsets from wherever the image lands.
        positions = (DecoderPosition * rows)()
        native.mtmd_helper_image_get_decoder_pos(tokens, 0, positions)
        raw = np.frombuffer(positions, dtype=np.uint32).reshape(rows, SECTIONS)
        # llama.cpp reads sections as t, y, x, z. See decode_embd_batch::set_position_mrope_2d.
        places = np.ascontiguousarray(raw[:, [0, 2, 1, 3]].T.astype(np.int32))
        return ImageShape(rows, native.mtmd_input_chunk_get_n_pos(chunk), places)

    def measure(self, image: bytes) -> ImageShape:
        """What an image will cost, without paying to encode it."""
        chunks, chunk = self._image_chunk(image)
        try:
            return self._shape(chunk)
        finally:
            self.mtmd.mtmd_input_chunks_free(chunks)

    def encode(self, image: bytes) -> tuple[ImageShape, np.ndarray]:
        """An image's shape and its embedding rows [rows, width], ready for a first stage."""
        native = self.mtmd
        chunks, chunk = self._image_chunk(image)
        try:
            shape = self._shape(chunk)
            status = native.mtmd_encode_chunk(self.context, chunk)
            if status != 0:
                raise LlamaError(f"the projector could not encode this image (status {status})")
            pointer = native.mtmd_get_output_embd(self.context)
            if not pointer:
                raise LlamaError("the projector produced no embeddings")
            return shape, np.ctypeslib.as_array(pointer, shape=(shape.rows, self.width)).copy()
        finally:
            native.mtmd_input_chunks_free(chunks)

    def close(self) -> None:
        if getattr(self, "context", None):
            self.mtmd.mtmd_free(self.context)
            self.context = None
        if getattr(self, "model", None):
            self.library.llama.llama_model_free(self.model)
            self.model = None
