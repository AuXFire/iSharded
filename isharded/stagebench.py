"""isharded bench --stage: one GPU's time for a pass, with CUDA graphs on and off.

Every stage of a route is fed rows of hidden state, the entry included, since it looks its
tokens up on the host. A whole model in llama.cpp's own tools is fed token ids. If rows keep
llama.cpp from replaying its CUDA graphs, each stage pays for launching every kernel again,
and that would explain much of the time a route spends at each boundary. This times one-token
passes of a model on this machine both ways, and as the last stage (logits) and as a stage
before it (the last layer's output, taken as a stage that is not last takes it), each with
graphs on and off.

llama.cpp reads GGML_CUDA_DISABLE_GRAPHS once a process, so each setting runs in a process of
its own, and the settings take turns so that a GPU warming up does not favour one.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any

import numpy as np

from .meter import graphs_environment, open_meter

FEEDS = ("tokens", "rows")
ROLES = ("last", "middle")
WARMUP_PASSES = 16
PROMPT_TOKENS = 16
FIRST_TOKEN = 1000  # any ids inside the vocabulary do: a pass costs the same whatever it holds
CHILD_SECONDS = 600


def word_list(choices: tuple[str, ...]) -> Any:
    def parse(text: str) -> list[str]:
        words = [part.strip() for part in text.split(",") if part.strip()]
        wrong = [word for word in words if word not in choices]
        if wrong or not words:
            raise argparse.ArgumentTypeError(f"choose from {', '.join(choices)}, comma separated")
        return list(dict.fromkeys(words))

    return parse


def graphs_list(text: str) -> list[bool]:
    return [word == "on" for word in word_list(("on", "off"))(text)]


def time_case(library: Any, path: str, role: str, feed: str, passes: int, gpu_layers: int, threads: int, meter: Any) -> dict[str, Any]:
    """Loads the model as one role, and times one-token passes fed one way."""
    from .gguf import read_file_header
    from .llamacpp import LlamaStage, TokenEmbedding

    last = role == "last"
    stage = LlamaStage(library, path, last=last, context_tokens=PROMPT_TOKENS + WARMUP_PASSES + passes + 8, gpu_layers=gpu_layers, threads=threads, native=True)
    try:
        header = read_file_header(path)
        table = header.tensors["token_embd.weight"]
        with open(path, "rb") as handle:
            handle.seek(header.data_start + table.offset)
            embedding = TokenEmbedding(library, handle.read(table.nbytes), table.ggml_type, stage.hidden_size)
        tokens = list(range(FIRST_TOKEN, FIRST_TOKEN + PROMPT_TOKENS + WARMUP_PASSES + passes))
        rows = embedding.rows(tokens) if feed == "rows" else None
        ids = np.array(tokens, dtype=np.int32)

        def one(position: int, count: int = 1) -> None:
            if feed == "rows":
                stage._decode(count, position, not last, hidden=np.ascontiguousarray(rows[position : position + count]))
            else:
                stage._decode(count, position, not last, tokens=np.ascontiguousarray(ids[position : position + count]))
            if last:
                stage.best_token()  # as the last stage reads its logits to pick a token

        one(0, PROMPT_TOKENS)
        for position in range(PROMPT_TOKENS, PROMPT_TOKENS + WARMUP_PASSES):
            one(position)
        times = []
        energy = meter.energy.snapshot() if meter is not None else None
        began = time.perf_counter()
        for position in range(PROMPT_TOKENS + WARMUP_PASSES, PROMPT_TOKENS + WARMUP_PASSES + passes):
            started = time.perf_counter()
            one(position)
            times.append((time.perf_counter() - started) * 1000)
        wall = time.perf_counter() - began
        result: dict[str, Any] = {"role": role, "feed": feed, "ms": times, "engine": stage.engine is not None}
        if meter is not None and energy is not None:
            watts = average_watts(meter, energy)
            if watts is not None:
                result["joules"] = watts * wall
                result["seconds"] = wall
        return result
    finally:
        stage.close()


def average_watts(meter: Any, start: dict[str, Any]) -> float | None:
    """The card's average power over the readings taken since ``start``.

    Readings come every interval and say nothing of the time between the last one and now,
    so the joules counted since ``start`` cover a span that need not match the timed passes.
    Their average power does not depend on that span, and times the passes' own wall it
    gives their energy. A run shorter than one interval waits for the next reading.
    """
    deadline = time.monotonic() + meter.interval_ms / 1000 * 3
    while True:
        now = meter.energy.snapshot()
        seconds = now["seconds"] - start["seconds"]
        if seconds > 0:
            return (now["joules"] - start["joules"]) / seconds
        if time.monotonic() > deadline:
            return None
        time.sleep(meter.interval_ms / 1000 / 4)


def child(arguments: dict[str, Any]) -> dict[str, Any]:
    from .llamacpp import LlamaLibrary

    library = LlamaLibrary.load(arguments["llama_dir"], ())
    meter = None
    if arguments.get("power_ms"):
        found = open_meter(arguments["device"], arguments["power_ms"])
        meter = None if isinstance(found, str) else found
    try:
        cases = [
            time_case(library, arguments["path"], role, feed, arguments["passes"], arguments["gpu_layers"], arguments["threads"], meter)
            for role in arguments["roles"]
            for feed in arguments["feeds"]
        ]
    finally:
        if meter is not None:
            meter.close()
    return {"cases": cases, "power_error": None if meter is not None or not arguments.get("power_ms") else "no power reading here"}


def run_child(arguments: dict[str, Any], graphs: bool) -> dict[str, Any]:
    package_root = str(Path(__file__).resolve().parents[1])
    environment = graphs_environment(graphs)
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, [package_root, environment.get("PYTHONPATH")]))
    done = subprocess.run(
        [sys.executable, "-m", "isharded.stagebench", json.dumps(arguments)], env=environment, capture_output=True, text=True, timeout=CHILD_SECONDS, stdin=subprocess.DEVNULL
    )  # fmt: skip
    lines = [line for line in done.stdout.splitlines() if line.startswith("{")]
    if done.returncode != 0 or not lines:
        tail = (done.stderr or done.stdout).strip().splitlines()[-3:]
        raise RuntimeError(f"the stage process failed: {' / '.join(tail) or done.returncode}")
    return json.loads(lines[-1])


def summarize(runs: list[tuple[bool, dict[str, Any]]], price: float | None) -> list[dict[str, Any]]:
    """One row per setting, role and feed, from every round of it."""
    grouped: dict[tuple[bool, str, str], dict[str, Any]] = {}
    for graphs, result in runs:
        for case in result["cases"]:
            row = grouped.setdefault((graphs, case["role"], case["feed"]), {"rounds": [], "ms": [], "joules": 0.0, "seconds": 0.0, "measured": True, "engine": case["engine"]})
            row["rounds"].append(statistics.median(case["ms"]))
            row["ms"] += case["ms"]
            if "joules" in case:
                row["joules"] += case["joules"]
                row["seconds"] += case["seconds"]
            else:
                row["measured"] = False
    rows = []
    for (graphs, role, feed), row in grouped.items():
        times = sorted(row["ms"])
        median = statistics.median(times)
        out = {
            "cuda_graphs": graphs, "role": role, "feed": feed, "engine": row["engine"],
            "median_ms": round(median, 3),
            "p10_ms": round(times[len(times) // 10], 3),
            "p90_ms": round(times[len(times) * 9 // 10], 3),
            "round_medians_ms": [round(value, 3) for value in row["rounds"]],
            "passes": len(times),
        }  # fmt: skip
        if row["measured"] and row["seconds"] > 0:
            out["joules_per_token"] = round(row["joules"] / len(times), 4)
            out["watts"] = round(row["joules"] / row["seconds"], 1)
        if price is not None:
            out["cost_per_million_tokens"] = round(price / 3600 * median / 1000 * 1e6, 4)
        rows.append(out)
    order = {(graphs, role, feed): index for index, (graphs, role, feed) in enumerate((g, r, f) for g in (True, False) for r in ROLES for f in FEEDS)}
    return sorted(rows, key=lambda row: order[(row["cuda_graphs"], row["role"], row["feed"])])


def stage_report(options: argparse.Namespace, rows: list[dict[str, Any]], device_name: str | None) -> dict[str, Any]:
    from .bench import public_name

    return {
        "kind": "stage",
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": public_name(options.stage),
        "gpu": device_name,
        "options": {"passes": options.passes, "repeat": options.repeat, "gpu_layers": options.gpu_layers, "price_per_hour": options.price_per_hour},
        "rows": rows,
        "notes": graph_notes(rows),
    }


def graph_notes(rows: list[dict[str, Any]]) -> list[str]:
    """What the rows say about the question this asks."""
    found = {(row["cuda_graphs"], row["role"], row["feed"]): row["median_ms"] for row in rows}
    notes = []
    for role in ROLES:
        tokens_on, rows_on, rows_off = found.get((True, role, "tokens")), found.get((True, role, "rows")), found.get((False, role, "rows"))
        if tokens_on and rows_on:
            notes.append(f"As the {role} stage, a pass fed rows took {rows_on - tokens_on:+.3f} ms against one fed tokens, with graphs on.")
        if rows_on and rows_off:
            notes.append(f"As the {role} stage, fed rows, turning graphs off cost {rows_off - rows_on:+.3f} ms a pass.")
    return notes


def stage_table(report: dict[str, Any]) -> tuple[list[str], list[list[str]]]:
    headers = ["CUDA graphs", "Role", "Fed", "Median ms", "p10-p90 ms", "Round medians"]
    extra = [(label, field) for label, field in (("J/token", "joules_per_token"), ("Watts", "watts"), ("Cost per M tokens", "cost_per_million_tokens")) if any(field in row for row in report["rows"])]
    headers += [label for label, _ in extra]
    table = [
        ["on" if row["cuda_graphs"] else "off", row["role"], row["feed"], f"{row['median_ms']:g}", f"{row['p10_ms']:g}-{row['p90_ms']:g}", ", ".join(f"{value:g}" for value in row["round_medians_ms"])]
        + ["-" if row.get(field) is None else f"{row[field]:g}" for _, field in extra]
        for row in report["rows"]
    ]  # fmt: skip
    return headers, table


def stage_facts(report: dict[str, Any]) -> list[str]:
    options = report["options"]
    return [f"Model: {report['model']}", f"GPU: {report['gpu'] or 'not known'}", f"{options['passes']} one-token passes a case, {options['repeat']} rounds, each setting in its own process"]


def stage_text(report: dict[str, Any]) -> str:
    from .bench import columns

    return "\n".join([*stage_facts(report), "", *columns(*stage_table(report)), "", *report["notes"]])


def stage_markdown(report: dict[str, Any]) -> str:
    from .bench import markdown_table

    out = [f"## iSharded stage bench: {report['model']}", "", *(f"- {line}" for line in stage_facts(report)), ""]
    out += markdown_table(*stage_table(report)) + [""] + [f"- {note}" for note in report["notes"]]
    return "\n".join(out).rstrip() + "\n"


def run_stage(args: argparse.Namespace) -> int:
    from .bench import write_files

    llama_dir = args.llama_dir or os.environ.get("ISHARDED_LLAMA_DIR")
    if not llama_dir:
        print("Cannot run the stage bench: give --llama-dir, a llama.cpp build", file=sys.stderr)
        return 1
    if not Path(args.stage).is_file():
        print(f"Cannot run the stage bench: no file at {args.stage}", file=sys.stderr)
        return 1
    arguments = {
        "llama_dir": llama_dir, "path": args.stage, "roles": args.role, "feeds": args.feed, "passes": args.passes,
        "gpu_layers": args.gpu_layers, "threads": args.stage_threads, "power_ms": args.measure_power, "device": args.device,
    }  # fmt: skip
    runs = []
    try:
        for round_number in range(args.repeat):
            for graphs in args.cuda_graphs:
                print(f"Round {round_number + 1}: CUDA graphs {'on' if graphs else 'off'}...", file=sys.stderr, flush=True)
                runs.append((graphs, run_child(arguments, graphs)))
    except (RuntimeError, subprocess.SubprocessError, ValueError) as exc:
        print(f"Cannot run the stage bench: {exc}", file=sys.stderr)
        return 1
    report = stage_report(args, summarize(runs, args.price_per_hour), gpu_name(args.device))
    if args.measure_power and any(result.get("power_error") for _, result in runs):
        report["notes"].append("Power was asked for, but nothing on this machine could read it.")
    print(stage_text(report))
    return write_files(args, report, stage_markdown(report))


def gpu_name(device: str) -> str | None:
    from .memory import nvidia_accelerators

    found = next((item for item in nvidia_accelerators() if item.device == device), None)
    return found.name if found else None


def add_arguments(bench: argparse.ArgumentParser) -> None:
    bench.add_argument("--stage", default=None, metavar="GGUF", help="time one-token passes of this model on this machine, fed tokens and fed rows, with CUDA graphs on and off; asks no node")
    bench.add_argument("--cuda-graphs", type=graphs_list, default=[True, False], help="with --stage: on, off, or on,off")
    bench.add_argument("--feed", type=word_list(FEEDS), default=list(FEEDS), help="with --stage: tokens, rows, or both")
    bench.add_argument("--role", type=word_list(ROLES), default=list(ROLES), help="with --stage: last (logits), middle (last layer's output), or both")
    bench.add_argument("--passes", type=int, default=200, help="with --stage: one-token passes timed in each case")
    bench.add_argument("--gpu-layers", type=int, default=-1, help="with --stage: layers on the GPU; -1 is all")
    bench.add_argument("--stage-threads", type=int, default=4, help="with --stage: CPU threads")
    bench.add_argument("--device", default="cuda:0", help="with --stage: the card whose power --measure-power reads")
    bench.add_argument("--measure-power", type=int, nargs="?", const=200, default=None, metavar="MS", help="with --stage: read the card's power every MS milliseconds (default 200) for joules per token")
    bench.add_argument("--llama-dir", default=None, help="with --stage: a llama.cpp build; default is ISHARDED_LLAMA_DIR")


if __name__ == "__main__":
    print(json.dumps(child(json.loads(sys.argv[1]))), flush=True)
