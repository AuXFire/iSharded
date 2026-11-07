"""Images in chat: what a request may carry, and how they sit in a prompt.

An image's rows travel through the prompt as tokens that can never be real ones.
Each is a negative number taken from a digest of the image, so everything that
compares prompts, such as conversation reuse, sees a different image as a different
prompt without being told about images at all.

The stage process imports this, so it must not import torch.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
import hashlib
import struct
from typing import Any

import numpy as np

from .gguf import read_numbers

# All of a request's images together. Base64 in JSON grows by a third, which keeps
# this inside the 16 MiB any node accepts in one request.
MOST_IMAGE_BYTES = 10 * 1024 * 1024
MOST_IMAGES = 16
# Decoded, an image takes three bytes a pixel, whatever it weighed as a file.
MOST_IMAGE_PIXELS = 40_000_000
# What each projector's chat template writes where an image goes. Only projectors
# whose decoder runs as an ordinary split stage belong here.
IMAGE_MARKERS = {"qwen3vl_merger": "<|image_pad|>"}
SECTIONS = 4
FORMATS = "PNG, JPEG, GIF, or BMP"
JPEG_SIZE_MARKERS = {*range(0xC0, 0xD0)} - {0xC4, 0xC8, 0xCC}
JPEG_LONE_MARKERS = {0x01, 0xD8, *range(0xD0, 0xD8)}


class ImageRefused(ValueError):
    """Something a person sent that cannot be used. The text says what to do."""


def jpeg_size(data: bytes) -> tuple[int, int] | None:
    index = 2
    while index + 9 <= len(data):
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        if marker == 0xFF:
            index += 1
        elif marker in JPEG_SIZE_MARKERS:
            height, width = struct.unpack(">HH", data[index + 5 : index + 9])
            return width, height
        elif marker in JPEG_LONE_MARKERS:
            index += 2
        else:
            index += 2 + struct.unpack(">H", data[index + 2 : index + 4])[0]
    return None


def image_size(data: bytes) -> tuple[int, int]:
    """Width and height from a file's header, without decoding it."""
    size: tuple[int, int] | None = None
    if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24:
        size = struct.unpack(">II", data[16:24])
    elif data[:6] in {b"GIF87a", b"GIF89a"} and len(data) >= 10:
        size = struct.unpack("<HH", data[6:10])
    elif data[:2] == b"BM" and len(data) >= 26:
        width, height = struct.unpack("<ii", data[18:26])
        size = abs(width), abs(height)
    elif data[:2] == b"\xff\xd8":
        size = jpeg_size(data)
    if size is None:
        raise ImageRefused(f"this is not an image that can be read here; send a {FORMATS}")
    return size


def check_image(data: bytes) -> None:
    """A few kilobytes can decode to gigabytes, so size is checked before decoding."""
    width, height = image_size(data)
    if not width or not height:
        raise ImageRefused("this image has no pixels")
    if width * height > MOST_IMAGE_PIXELS:
        raise ImageRefused(f"this image is {width} by {height}, and the most an image can hold is {MOST_IMAGE_PIXELS} pixels; send a smaller one")


def read_image_url(url: str) -> bytes:
    """The bytes of a data URL. Anything else is refused, in words."""
    if url[:7].lower() == "http://" or url[:8].lower() == "https://":
        raise ImageRefused("remote image URLs are refused, because fetching one would let a caller make a node request any address; send the image as a data URL")
    head, comma, body = url.partition(",")
    if not comma or not head[:5].lower() == "data:" or ";base64" not in head.lower():
        raise ImageRefused("an image_url must be a base64 data URL, such as data:image/png;base64,...")
    if len(body) * 3 // 4 > MOST_IMAGE_BYTES + 2:
        raise ImageRefused(too_much_image())
    try:
        return base64.b64decode(body, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ImageRefused("this data URL is not valid base64") from exc


def too_much_image() -> str:
    return f"the images in one request can come to {MOST_IMAGE_BYTES // 1024**2} MiB; send smaller ones"


def read_part(part: Any, role: str, parts: list[dict[str, Any]], images: list[bytes]) -> None:
    kind = part.get("type") if isinstance(part, dict) else None
    if kind == "text":
        parts.append({"type": "text", "text": str(part.get("text", ""))})
        return
    if kind != "image_url":
        raise ImageRefused(f"a message part of type {kind!r} is not supported; send text and image_url parts")
    if role != "user":
        raise ImageRefused("only user messages can hold images")
    target = part.get("image_url")
    url = target.get("url") if isinstance(target, dict) else target
    if not isinstance(url, str):
        raise ImageRefused("an image_url part needs a url")
    data = read_image_url(url)
    check_image(data)
    images.append(data)
    parts.append({"type": "image"})


def chat_images(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[bytes]]:
    """Split messages into what a chat template reads and the images in them, in order."""
    plain: list[dict[str, Any]] = []
    images: list[bytes] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            plain.append(message)
            continue
        parts: list[dict[str, Any]] = []
        for part in content:
            read_part(part, str(message.get("role")), parts, images)
        if not any(part["type"] == "image" for part in parts):
            plain.append({**message, "content": "".join(part["text"] for part in parts)})
        else:
            plain.append({**message, "content": parts})
    if len(images) > MOST_IMAGES:
        raise ImageRefused(f"this conversation holds {len(images)} images, and one request can hold {MOST_IMAGES}")
    if sum(map(len, images)) > MOST_IMAGE_BYTES:
        raise ImageRefused(too_much_image())
    return plain, images


def projector_marker(header: Any, hidden_size: int, name: str) -> str:
    """The prompt marker for a projector file, or the reason it cannot be used."""
    if header.architecture != "clip":
        raise ImageRefused(f"{name} is not a projector file; it is a {header.architecture} model")
    if not header.get("clip.has_vision_encoder"):
        raise ImageRefused(f"{name} has no vision encoder")
    kind = header.get("clip.projector_type")
    if kind not in IMAGE_MARKERS:
        raise ImageRefused(f"projectors of type {kind!r} are not supported yet; supported: {', '.join(sorted(IMAGE_MARKERS))}")
    width = header.get("clip.vision.projection_dim")
    if width != hidden_size:
        raise ImageRefused(f"{name} makes rows {width} wide and this model takes {hidden_size}, so it belongs to another model")
    layers = header.keys.get("clip.vision.is_deepstack_layers")
    if layers is not None and any(read_numbers(layers)):
        # These rows feed several decoder layers, and a stage takes one hidden state.
        raise ImageRefused(f"{name} uses deepstack layers, which split stages do not support yet")
    return IMAGE_MARKERS[kind]


def image_id(data: bytes) -> int:
    """What stands for an image's rows in a list of tokens.

    The digest is why a different image can never reuse a cached state. Short of
    62 bits, someone could search for an image that matches another person's.
    """
    return -1 - (int.from_bytes(hashlib.sha256(data).digest()[:8], "big") >> 2)


def expand_prompt(
    tokens: list[int], boundary: int | None, marker: int, images: list[tuple[int, int]]
) -> tuple[list[int], int | None]:
    """Put each image's rows where its marker stands, and move the boundary to match.

    ``images`` is ``(id, rows)`` in order of appearance, and ``boundary`` is a place
    in ``tokens`` that has to keep pointing at the same text.
    """
    expanded: list[int] = []
    moved = boundary
    waiting = iter(images)
    for index, token in enumerate(tokens):
        if token != marker:
            expanded.append(token)
            continue
        identity, rows = next(waiting)
        expanded.extend([identity] * rows)
        if moved is not None and boundary is not None and index < boundary:
            moved += rows - 1
    return expanded, moved


@dataclass
class EncodedImage:
    """One image, ready to go in a prompt."""

    identity: int
    span: int  # positions it moves the sequence on by
    places: np.ndarray  # [SECTIONS, rows], relative to its first position
    vectors: np.ndarray  # [rows, width]


@dataclass
class PromptPart:
    """Some rows of a prompt, as one pass takes them."""

    places: np.ndarray  # [SECTIONS, rows], in the order llama.cpp reads them
    images: np.ndarray  # which of these rows are image rows
    vectors: np.ndarray  # those rows' embeddings
    after: int  # the position that follows this part


@dataclass
class PromptRows:
    """A whole prompt with images in it: what fills each row, and where it sits."""

    places: np.ndarray  # [SECTIONS, rows]
    source: np.ndarray  # per row, an index into ``vectors``, or -1 for a text token
    vectors: np.ndarray
    ends: np.ndarray  # per row, the position that follows it

    def position_at(self, row: int) -> int:
        """Where a sequence stands once it holds this many rows."""
        return int(self.ends[row - 1]) if row else 0

    def part(self, start: int, stop: int) -> PromptPart:
        chosen = self.source[start:stop]
        images = np.flatnonzero(chosen >= 0)
        return PromptPart(
            np.ascontiguousarray(self.places[:, start:stop]), images, self.vectors[chosen[images]], self.position_at(stop)
        )


def lay_out(tokens: list[int], images: list[EncodedImage]) -> PromptRows:
    """Work out every row of a prompt. An image's rows share a stretch of positions.

    Under M-RoPE an image of 24 by 24 rows moves the sequence on by 24, not 576, so
    position and row count part ways from the first image on.
    """
    count = len(tokens)
    places = np.zeros((SECTIONS, count), dtype=np.int32)
    source = np.full(count, -1, dtype=np.int64)
    ends = np.zeros(count, dtype=np.int64)
    is_image = np.asarray(tokens, dtype=np.int64) < 0
    position = row = done = 0

    def add_text(stop: int) -> None:
        nonlocal position, row
        steps = np.arange(position, position + stop - row, dtype=np.int32)
        places[:3, row:stop] = steps
        ends[row:stop] = steps + 1
        position += stop - row
        row = stop

    for image in images:
        rows = image.places.shape[1]
        add_text(row + int(np.argmax(np.append(is_image[row:], True))))
        if tokens[row : row + rows] != [image.identity] * rows:
            raise ValueError("a prompt's image rows do not match the images sent with it")
        places[:, row : row + rows] = image.places
        places[:3, row : row + rows] += position
        source[row : row + rows] = np.arange(done, done + rows)
        position += image.span
        ends[row : row + rows] = position
        row += rows
        done += rows
    if is_image[row:].any():
        raise ValueError("a prompt holds image rows with no image sent for them")
    add_text(count)
    vectors = np.concatenate([image.vectors for image in images]) if images else np.empty((0, 0), dtype=np.float32)
    return PromptRows(places, source, vectors, ends)
