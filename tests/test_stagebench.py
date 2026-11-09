"""isharded bench --stage: a pass's time on this machine's GPU, CUDA graphs on and off."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from isharded import cli, stagebench

BUILD = os.environ.get("ISHARDED_TEST_LLAMA_DIR", "")
MODEL = os.environ.get("ISHARDED_TEST_GGUF", "")
needs_real_model = pytest.mark.skipif(not (BUILD and MODEL and Path(BUILD).is_dir() and Path(MODEL).is_file()), reason="needs a llama.cpp build and a real GGUF")


def parse(*extra: str):
    return cli.build_parser().parse_args(["bench", *extra])


def test_the_stage_options_default_to_every_case_and_refuse_what_is_not_one(capsys):
    args = parse("--stage", "m.gguf")
    assert (args.cuda_graphs, args.feed, args.role, args.passes, args.gpu_layers, args.measure_power) == ([True, False], ["tokens", "rows"], ["last", "middle"], 200, -1, None)
    assert parse("--stage", "m.gguf", "--cuda-graphs", "off", "--measure-power").cuda_graphs == [False]
    assert parse("--stage", "m.gguf", "--measure-power").measure_power == 200
    for bad in (["--cuda-graphs", "maybe"], ["--feed", "bytes"], ["--role", ""]):
        with pytest.raises(SystemExit):
            parse("--stage", "m.gguf", *bad)


def test_each_setting_runs_in_a_process_of_its_own_with_the_variable_as_llama_cpp_reads_it(monkeypatch):
    seen = []

    def run(command, env, **_):
        seen.append(env.get("GGML_CUDA_DISABLE_GRAPHS"))
        assert command[1:3] == ["-m", "isharded.stagebench"] and json.loads(command[3])["passes"] == 5
        return SimpleNamespace(returncode=0, stdout='log line\n{"cases": []}\n', stderr="")

    monkeypatch.setattr(stagebench.subprocess, "run", run)
    monkeypatch.setenv("GGML_CUDA_DISABLE_GRAPHS", "0")  # which llama.cpp takes as off
    assert stagebench.run_child({"passes": 5}, True) == {"cases": []}
    assert stagebench.run_child({"passes": 5}, False) == {"cases": []}
    assert seen == [None, "1"]


def test_a_failed_stage_process_says_why(monkeypatch):
    monkeypatch.setattr(stagebench.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="loading\nllama.cpp could not load it\n"))
    with pytest.raises(RuntimeError, match="could not load it"):
        stagebench.run_child({}, True)


def case(role, feed, ms, joules=None):
    found = {"role": role, "feed": feed, "ms": ms, "engine": True}
    if joules is not None:
        found |= {"joules": joules, "seconds": sum(ms) / 1000}
    return found


def test_rows_gather_every_round_and_the_notes_answer_the_question():
    runs = [
        (True, {"cases": [case("last", "tokens", [3.0, 3.0, 3.0]), case("last", "rows", [3.5, 3.5, 3.5], joules=1.05)]}),
        (False, {"cases": [case("last", "tokens", [6.0, 6.0, 6.0]), case("last", "rows", [6.5, 6.5, 6.5], joules=1.95)]}),
        (True, {"cases": [case("last", "tokens", [3.0, 3.0, 3.0]), case("last", "rows", [3.5, 3.5, 3.5], joules=1.05)]}),
    ]
    rows = stagebench.summarize(runs, price=0.36)
    assert [(row["cuda_graphs"], row["feed"]) for row in rows] == [(True, "tokens"), (True, "rows"), (False, "tokens"), (False, "rows")]
    on_rows = rows[1]
    assert (on_rows["median_ms"], on_rows["passes"], on_rows["round_medians_ms"]) == (3.5, 6, [3.5, 3.5])
    assert (on_rows["joules_per_token"], on_rows["watts"]) == (0.35, 100.0)
    assert "joules_per_token" not in rows[0], "a case with no power reading has none"
    assert on_rows["cost_per_million_tokens"] == round(0.36 / 3600 * 3.5 / 1000 * 1e6, 4)
    assert stagebench.graph_notes(rows) == [
        "As the last stage, a pass fed rows took +0.500 ms against one fed tokens, with graphs on.",
        "As the last stage, fed rows, turning graphs off cost +3.000 ms a pass.",
    ]


def test_the_command_runs_the_settings_in_turn_and_writes_both_files(monkeypatch, tmp_path, capsys):
    model = tmp_path / "Qwen3-0.6B-Q8_0.gguf"
    model.write_bytes(b"GGUF")
    order = []

    def run_child(arguments, graphs):
        order.append(graphs)
        assert arguments["roles"] == ["last"] and arguments["feeds"] == ["rows"]
        return {"cases": [case("last", "rows", [4.0] if graphs else [6.0])]}

    monkeypatch.setattr(stagebench, "run_child", run_child)
    monkeypatch.setattr(stagebench, "gpu_name", lambda device: "RTX 3060")
    data, page = tmp_path / "out.json", tmp_path / "out.md"
    arguments = ["--stage", str(model), "--llama-dir", str(tmp_path), "--role", "last", "--feed", "rows", "--repeat", "2", "--json", str(data), "--markdown", str(page)]
    args = parse(*arguments)
    assert args.handler(args) == 0
    assert order == [True, False, True, False]
    shown = capsys.readouterr().out
    assert "turning graphs off cost +2.000 ms" in shown and "RTX 3060" in shown
    saved = json.loads(data.read_text(encoding="utf-8"))
    assert saved["model"] == "Qwen3-0.6B-Q8_0.gguf" and str(tmp_path) not in data.read_text(encoding="utf-8")
    assert "| CUDA graphs | Role |" in page.read_text(encoding="utf-8")


def test_the_command_says_what_it_is_missing(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("ISHARDED_LLAMA_DIR", raising=False)
    assert stagebench.run_stage(parse("--stage", str(tmp_path / "none.gguf"))) == 1
    assert "--llama-dir" in capsys.readouterr().err
    assert stagebench.run_stage(parse("--stage", str(tmp_path / "none.gguf"), "--llama-dir", str(tmp_path))) == 1
    assert "no file at" in capsys.readouterr().err


@needs_real_model
def test_a_real_model_is_timed_fed_tokens_and_rows_in_both_roles():
    # In a process of its own, as the bench runs it. Loading llama.cpp into the test process
    # after many other real-model tests aborted the whole run.
    result = stagebench.run_child({"llama_dir": BUILD, "path": MODEL, "roles": ["last", "middle"], "feeds": ["tokens", "rows"], "passes": 6, "gpu_layers": -1, "threads": 4, "power_ms": None, "device": "cuda:0"}, True)
    assert [(item["role"], item["feed"]) for item in result["cases"]] == [("last", "tokens"), ("last", "rows"), ("middle", "tokens"), ("middle", "rows")]
    assert all(len(item["ms"]) == 6 and min(item["ms"]) > 0 for item in result["cases"])


def test_a_cases_energy_is_its_passes_wall_at_the_cards_average_power():
    """Readings come each interval, so joules since the start cover more than the passes did."""
    from isharded.meter import Energy

    meter = SimpleNamespace(interval_ms=100, energy=Energy())
    meter.energy.add(20.0, 0.0)  # idle before the passes
    start = meter.energy.snapshot()
    for at in (0.1, 0.2, 0.3):
        meter.energy.add(80.0, at)
    # 0.3 s of readings: 5 J on the way up from idle, then 8 J and 8 J, so 70 W on average.
    assert stagebench.average_watts(meter, start) == pytest.approx(70.0)
    still = SimpleNamespace(interval_ms=100, energy=Energy())
    still.energy.add(50.0, 0.0)
    assert stagebench.average_watts(still, still.energy.snapshot()) is None, "no reading came"
