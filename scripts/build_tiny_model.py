"""Build a deterministic, offline GPT-2 model for iSharded smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    words = [
        "<pad>",
        "<bos>",
        "<eos>",
        "<unk>",
        "Once",
        "upon",
        "a",
        "time",
        "small",
        "mesh",
        "of",
        "computers",
        "shared",
        "work",
        ".",
        ",",
        "the",
        "and",
        "Assistant",
        ":",
    ]
    vocabulary = {word: index for index, word in enumerate(words)}
    tokenizer_backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer_backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend,
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    tokenizer.save_pretrained(args.output)

    torch.manual_seed(7)
    config = GPT2Config(
        vocab_size=len(vocabulary),
        n_positions=128,
        n_ctx=128,
        n_embd=32,
        n_layer=2,
        n_head=2,
        bos_token_id=vocabulary["<bos>"],
        eos_token_id=vocabulary["<eos>"],
        pad_token_id=vocabulary["<pad>"],
    )
    model = GPT2LMHeadModel(config).eval()
    model.save_pretrained(args.output, safe_serialization=True)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
