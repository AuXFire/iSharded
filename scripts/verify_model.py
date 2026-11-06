"""Compare a live iSharded greedy completion with an unsharded reference."""

from __future__ import annotations

import argparse
import os

import httpx
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--prompt",
        default="Once upon a time, in a small village surrounded by mountains,",
    )
    parser.add_argument("--tokens", type=int, default=24)
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float32"],
        default="bfloat16",
        help="must match the workers; float32 gives the strictest comparison",
    )
    args = parser.parse_args()

    token = os.getenv("ISHARDED_POOL_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    status = httpx.get(f"{args.url}/admin/status", headers=headers, timeout=10).json()
    active = [worker for worker in status["workers"].values() if worker["active"]]
    cuda_route = bool(active) and all(worker["hardware"]["backend"] == "cuda" for worker in active)
    device = torch.device("cuda" if cuda_route and torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)

    response = httpx.post(
        f"{args.url}/v1/completions",
        headers=headers,
        json={
            "model": args.model,
            "prompt": args.prompt,
            "max_tokens": args.tokens,
            "temperature": 0,
        },
        timeout=300,
    )
    response.raise_for_status()
    distributed = response.json()["choices"][0]["text"]

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=dtype,
    ).to(device).eval()
    ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)
    past = None
    output_ids: list[int] = []
    for _ in range(args.tokens):
        step = ids if past is None else ids[:, -1:]
        with torch.inference_mode():
            result = model(input_ids=step, past_key_values=past, use_cache=True)
        next_token = int(result.logits[:, -1].argmax())
        output_ids.append(next_token)
        ids = torch.cat((ids, torch.tensor([[next_token]], device=device)), dim=1)
        past = result.past_key_values
    reference = tokenizer.decode(output_ids, skip_special_tokens=True)

    print(f"Reference device: {device} / {dtype}")
    print(f"Distributed: {distributed!r}")
    print(f"Reference:   {reference!r}")
    if distributed != reference:
        raise SystemExit("FAIL: distributed output differs from the unsharded reference")
    print("PASS: distributed output exactly matches the unsharded reference")


if __name__ == "__main__":
    main()
