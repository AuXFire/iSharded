"""Power, price and CUDA graph settings for measuring what a token costs."""

import asyncio
import subprocess
import sys
from types import SimpleNamespace

import pytest

from isharded import meter
from isharded.meter import Energy, GuessClock, PowerMeter


def test_a_price_is_the_operators_number_and_anything_else_is_none() -> None:
    assert meter.price_per_hour({"ISHARDED_PRICE_PER_HOUR": "0.59"}) == 0.59
    assert meter.price_per_hour({"ISHARDED_PRICE_PER_HOUR": " 0 "}) == 0.0
    assert meter.price_per_hour({}) is None
    assert meter.price_per_hour({"ISHARDED_PRICE_PER_HOUR": "cheap"}) is None
    assert meter.price_per_hour({"ISHARDED_PRICE_PER_HOUR": "-1"}) is None


def test_any_value_of_the_graphs_variable_turns_graphs_off_as_llama_cpp_reads_it() -> None:
    assert meter.cuda_graphs_on({})
    assert not meter.cuda_graphs_on({"GGML_CUDA_DISABLE_GRAPHS": "0"})
    on, off = meter.graphs_environment(True, {"GGML_CUDA_DISABLE_GRAPHS": "1", "A": "b"}), meter.graphs_environment(False, {"A": "b"})
    assert on == {"A": "b"} and off == {"A": "b", "GGML_CUDA_DISABLE_GRAPHS": "1"}


def test_power_is_off_unless_asked_and_never_sampled_faster_than_the_floor() -> None:
    assert meter.sample_ms({}) == 0
    assert meter.sample_ms({"ISHARDED_POWER_SAMPLE_MS": "0"}) == 0
    assert meter.sample_ms({"ISHARDED_POWER_SAMPLE_MS": "500"}) == 500
    assert meter.sample_ms({"ISHARDED_POWER_SAMPLE_MS": "5"}) == meter.SHORTEST_SAMPLE_MS
    assert meter.sample_ms({"ISHARDED_POWER_SAMPLE_MS": "often"}) == 0


@pytest.mark.parametrize(
    ("device", "environment", "card"),
    [
        ("cuda:1", {}, "1"),
        ("cuda:1", {"CUDA_VISIBLE_DEVICES": "3,2"}, "2"),
        ("cuda:0", {"CUDA_VISIBLE_DEVICES": "GPU-abc"}, "GPU-abc"),
        ("cuda:2", {"CUDA_VISIBLE_DEVICES": "3,2"}, None),
        ("cpu", {}, None),
        ("mps", {}, None),
    ],
)
def test_a_worker_device_becomes_the_card_the_vendor_tool_counts(device, environment, card) -> None:
    assert meter.smi_device(device, environment) == card


def test_readings_parse_and_a_card_that_cannot_say_reads_none() -> None:
    assert meter.parse_watts("24.61\n") == 24.61
    assert meter.parse_watts("card0,35.0") == 35.0
    assert meter.parse_watts("[N/A]") is None


def test_energy_adds_each_gap_at_the_average_of_its_two_readings() -> None:
    energy = Energy()
    energy.add(100.0, 10.0)
    energy.add(200.0, 11.0)  # 150 W for a second
    energy.add(200.0, 13.0)  # 200 W for two
    assert energy.snapshot() == {"joules": 550.0, "watts": 200.0, "samples": 3, "seconds": 3.0}


def test_a_counter_of_energy_is_added_as_it_reads() -> None:
    energy = Energy()
    energy.add_joules(0.0, 5.0)
    energy.add_joules(30.0, 7.0)
    assert energy.snapshot() == {"joules": 30.0, "watts": 15.0, "samples": 2, "seconds": 2.0}


def test_a_meter_reports_its_readings_and_why_it_stopped() -> None:
    found = PowerMeter("nvidia-smi", "0", 100, lambda: iter([("watts", 50.0), ("watts", 50.0)])).start()
    found.thread.join(5)
    report = found.report()
    assert report["source"] == "nvidia-smi" and report["samples"] == 2 and report["watts"] == 50.0
    assert report["error"] == "nvidia-smi stopped reporting"


def test_the_nvidia_meter_reads_a_running_tool(tmp_path) -> None:
    """A stand-in for nvidia-smi's loop: one reading a line, as -lms prints them."""
    tool = tmp_path / "fake_smi.py"
    tool.write_text("import sys, time\nfor value in ('10.0', '[N/A]', '30.0'):\n    print(value, flush=True)\n    time.sleep(0.05)\n")
    script = tmp_path / ("fake-smi.cmd" if sys.platform == "win32" else "fake-smi")
    if sys.platform == "win32":
        script.write_text(f'@"{sys.executable}" "{tool}" %*\n')
    else:
        script.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{tool}" "$@"\n')
        script.chmod(0o755)
    found = meter.nvidia_meter("0", 100, str(script)).start()
    found.thread.join(10)
    found.close()
    report = found.report()
    assert report["samples"] == 2 and report["watts"] == 30.0 and report["joules"] > 0


def test_the_rocm_meter_asks_each_interval() -> None:
    answers = iter(["device,Average Graphics Package Power (W)\ncard0,40.0\n", "device,Average Graphics Package Power (W)\ncard0,60.0\n"])
    asked = []

    def run(command, **_):
        asked.append(command)
        try:
            return SimpleNamespace(returncode=0, stdout=next(answers))
        except StopIteration:
            found.close()
            return SimpleNamespace(returncode=1, stdout="")

    found = meter.rocm_meter("1", 100, "rocm-smi", run)
    found.start().thread.join(10)
    assert found.report()["samples"] == 2 and found.report()["watts"] == 60.0
    assert asked[0][:3] == ["rocm-smi", "-d", "1"]


def test_the_cpu_meter_reads_the_package_counter_and_survives_its_wrap(tmp_path) -> None:
    counter, wrap = tmp_path / "energy_uj", tmp_path / "max_energy_range_uj"
    counter.write_text("999000000")
    wrap.write_text("1000000000")
    found = meter.rapl_meter(100, counter, wrap)
    readings = found._read()
    assert next(readings) == ("joules", 0.0)
    counter.write_text("1000000")  # wrapped: 1 J to the top, then 1 J more
    assert next(readings) == ("joules", 2.0)
    found.close()


def test_no_tool_means_no_meter_and_says_why() -> None:
    assert meter.open_meter("cuda:0", 500, {}, which=lambda name: None) == "no nvidia-smi or rocm-smi on this machine"
    assert "no power reading" in meter.open_meter("mps", 500, {}, which=lambda name: None)


def test_steady_guesses_are_the_ones_that_ran_one_row() -> None:
    clock = GuessClock()
    clock.note(4.0, 1)
    clock.note(10.0, 6)
    clock.note(5.0, None)
    assert clock.report() == {"guesses": 3, "ms_total": 19.0, "steady": 1, "steady_ms_total": 4.0}


def test_a_worker_reports_power_only_when_its_node_asked(monkeypatch) -> None:
    from isharded import worker

    runtime = worker.runtime
    monkeypatch.setattr(runtime, "hardware", SimpleNamespace(device="cuda:0"))
    monkeypatch.setattr(runtime, "power", None)
    monkeypatch.setattr(runtime, "power_problem", None)
    opened = []
    monkeypatch.setattr(worker, "open_meter", lambda device, interval: opened.append((device, interval)) or "no tool here")
    monkeypatch.delenv("ISHARDED_POWER_SAMPLE_MS", raising=False)
    assert runtime.power_report() is None and not opened
    monkeypatch.setenv("ISHARDED_POWER_SAMPLE_MS", "250")
    assert runtime.power_report() == {"source": None, "error": "no tool here"}
    assert runtime.power_report() == {"source": None, "error": "no tool here"}
    assert opened == [("cuda:0", 250)], "a meter that cannot start is not tried again on every look"


def test_a_workers_health_carries_the_measuring_fields(monkeypatch) -> None:
    from isharded import worker
    from isharded.protocol import HardwareInfo

    monkeypatch.setattr(worker, "authorized", lambda authorization: True)
    monkeypatch.setattr(worker.runtime, "gguf", None)
    # Whatever this machine computes on, the worker is one on CUDA, which is the only kind with graphs to report.
    monkeypatch.setattr(worker.runtime, "hardware", HardwareInfo(device="cuda:0", backend="cuda", cpu_cores=1, ram_total_bytes=1))
    monkeypatch.setenv("ISHARDED_PRICE_PER_HOUR", "0.59")
    monkeypatch.setenv("GGML_CUDA_DISABLE_GRAPHS", "1")
    monkeypatch.delenv("ISHARDED_POWER_SAMPLE_MS", raising=False)
    telemetry = asyncio.run(worker.health(None))["telemetry"]
    assert telemetry["price_per_hour"] == 0.59 and telemetry["cuda_graphs"] is False
    assert telemetry["power"] is None
    assert telemetry["draft_guess"].keys() == {"guesses", "ms_total", "steady", "steady_ms_total"}


def test_cuda_graphs_are_reported_only_by_a_worker_that_computes_on_cuda(monkeypatch) -> None:
    """A CPU stage has no CUDA graphs to keep, so it must not say they are on."""
    from isharded import worker
    from isharded.protocol import HardwareInfo

    def on(device: str) -> HardwareInfo:
        return HardwareInfo(device=device, backend=device.split(":")[0], cpu_cores=1, ram_total_bytes=1)

    monkeypatch.setattr(worker, "authorized", lambda authorization: True)
    monkeypatch.setattr(worker.runtime, "gguf", None)
    monkeypatch.delenv("GGML_CUDA_DISABLE_GRAPHS", raising=False)
    monkeypatch.setattr(worker.runtime, "hardware", on("cpu"))
    assert asyncio.run(worker.health(None))["telemetry"]["cuda_graphs"] is None
    monkeypatch.setattr(worker.runtime, "hardware", on("cuda:0"))
    assert asyncio.run(worker.health(None))["telemetry"]["cuda_graphs"] is True
    monkeypatch.setenv("GGML_CUDA_DISABLE_GRAPHS", "0")
    assert asyncio.run(worker.health(None))["telemetry"]["cuda_graphs"] is False


def test_a_node_hands_its_price_and_power_setting_to_its_workers(monkeypatch) -> None:
    import argparse

    from isharded import node

    parser = argparse.ArgumentParser()
    node.add_node_arguments(parser)
    args = parser.parse_args(["--price-per-hour", "0.59", "--measure-power", "--advertise-host", "127.0.0.1", "--trusted-network"])
    assert (args.price_per_hour, args.measure_power) == (0.59, 1000)
    assert parser.parse_args(["--measure-power", "250"]).measure_power == 250
    with pytest.raises(SystemExit):
        parser.parse_args(["--price-per-hour", "-1"])
    for name in ("ISHARDED_PRICE_PER_HOUR", "ISHARDED_POWER_SAMPLE_MS"):
        monkeypatch.setenv(name, "")  # so the test's own environment is put back after it
        monkeypatch.delenv(name)
    monkeypatch.setattr(node, "check_exposure", lambda *args: None)
    node._publish_settings(args)
    environment = node._child_environment(args)[0]
    assert environment["ISHARDED_PRICE_PER_HOUR"] == "0.59" and environment["ISHARDED_POWER_SAMPLE_MS"] == "1000"
