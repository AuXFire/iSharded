import argparse
from pathlib import Path

import pytest

from isharded.memory import Accelerator, build_pools, discover_accelerators
from isharded.node import _weight_memory_flags, is_remote_path, local_code_root, run_node
from isharded.worker import parse_ram_budget

GIB = 1024**3


def accelerator(kind: str = "vram", device: str = "cuda:0") -> Accelerator:
    return Accelerator(device, kind, "cuda", "Test GPU", 8 * GIB, 7 * GIB, "12.4")


def pools(tmp_path: Path, accelerators: list[Accelerator], **options) -> dict:
    settings = {
        "accelerator_budget_bytes": 0,
        "compute_dtype": "bfloat16",
        "ram_budget_bytes": None,
        "disk_budget_bytes": 0,
        "cache_dir": tmp_path,
    }
    settings.update(options)
    built = build_pools("host", accelerators, **settings)
    return {pool.pool_id: pool for pool in built}


@pytest.fixture(autouse=True)
def fixed_bandwidth(monkeypatch) -> None:
    monkeypatch.setattr("isharded.memory.measure_bandwidth", lambda device: 100.0)


def test_cpu_only_worker_offers_ram_without_being_told(tmp_path: Path) -> None:
    result = pools(tmp_path, [])
    assert list(result) == ["ram"]
    assert result["ram"].budget_bytes > 0


def test_gpu_worker_stays_gpu_only_unless_given_ram(tmp_path: Path) -> None:
    assert list(pools(tmp_path, [accelerator()])) == ["cuda:0"]


def test_any_node_can_offer_every_memory_type_at_once(tmp_path: Path) -> None:
    result = pools(tmp_path, [accelerator()], ram_budget_bytes=GIB, disk_budget_bytes=GIB)
    assert {pool.kind for pool in result.values()} == {"vram", "ram", "disk"}


def test_each_gpu_is_its_own_pool(tmp_path: Path) -> None:
    result = pools(tmp_path, [accelerator(device="cuda:0"), accelerator(device="cuda:1")])
    assert result["cuda:0"].capacity_group != result["cuda:1"].capacity_group


def test_unified_memory_shares_a_budget_with_ram(tmp_path: Path) -> None:
    result = pools(tmp_path, [accelerator("unified", "mps")], ram_budget_bytes=GIB)
    assert result["mps"].capacity_group == result["ram"].capacity_group


def test_explicit_budget_wins_over_the_automatic_share(tmp_path: Path) -> None:
    result = pools(tmp_path, [accelerator()], accelerator_budget_bytes=2 * GIB)
    assert result["cuda:0"].budget_bytes == 2 * GIB


def test_automatic_budget_leaves_headroom(tmp_path: Path) -> None:
    assert pools(tmp_path, [accelerator()])["cuda:0"].budget_bytes < 7 * GIB


def test_full_precision_covers_vram_and_ram_but_not_mapped_disk(tmp_path: Path) -> None:
    result = pools(
        tmp_path,
        [accelerator()],
        compute_dtype="float32",
        ram_budget_bytes=GIB,
        disk_budget_bytes=GIB,
    )
    assert result["cuda:0"].dtype == "float32"
    assert result["ram"].dtype == "float32"
    assert result["disk"].dtype == "checkpoint"


def test_no_accelerator_is_a_valid_choice() -> None:
    assert discover_accelerators("none") == []
    assert discover_accelerators("cpu") == []


def test_an_unknown_accelerator_lists_what_exists() -> None:
    with pytest.raises(ValueError, match="this host offers"):
        discover_accelerators("cuda:99")


def test_ram_budget_accepts_auto_a_size_or_nothing() -> None:
    assert parse_ram_budget("") is None
    assert parse_ram_budget("2") == 2 * GIB
    assert parse_ram_budget("auto") > 0


def test_node_passes_weight_memory_to_its_worker() -> None:
    args = argparse.Namespace(weights_ram_gb="auto", weights_disk_gb=50.0)
    assert _weight_memory_flags(args) == ["--weights-ram-gb", "auto", "--weights-disk-gb", "50.0"]


def test_launchers_written_before_weight_memory_still_work() -> None:
    assert _weight_memory_flags(argparse.Namespace()) == []


def launched(monkeypatch, **options) -> list[list[str]]:
    """The worker command lines a node would start."""
    commands: list[list[str]] = []

    class Process:
        pid = 1
        returncode = 0

        def poll(self) -> int:
            return 0

        def wait(self, timeout: float) -> int:
            return 0

    def record(command: list[str], env: dict, cwd: Path | None = None) -> Process:
        commands.append(command)
        return Process()

    def stop(seconds: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("isharded.node.subprocess.Popen", record)
    monkeypatch.setattr("isharded.node.time.sleep", stop)
    settings = dict(
        node_id="node",
        orchestrator_url="http://orchestrator:8000",
        bind_host="0.0.0.0",
        advertise_host="node",
        advertise_scheme="http",
        base_port=8200,
        gpu_device="cuda:0",
        gpu_memory_gb=4.0,
        cpu_threads=0,
        cpu_active=False,
        ram_budget_gb=0.0,
        open_pool=True,
    )
    settings.update(options)
    run_node(argparse.Namespace(**settings))
    return [command for command in commands if "isharded.orchestrator" not in command]


def flag(command: list[str], name: str) -> str | None:
    return command[command.index(name) + 1] if name in command else None


def test_gpu_with_weight_memory_is_one_worker_spanning_every_tier(monkeypatch) -> None:
    commands = launched(monkeypatch, cpu_threads=8, weights_ram_gb="auto", weights_disk_gb=50.0)
    assert len(commands) == 1
    assert flag(commands[0], "--device") == "cuda:0"
    assert flag(commands[0], "--weights-ram-gb") == "auto"
    assert flag(commands[0], "--weights-disk-gb") == "50.0"
    assert flag(commands[0], "--cpu-threads") == "8"


def test_gpu_with_cpu_threads_alone_keeps_a_separate_standby_worker(monkeypatch) -> None:
    gpu, cpu = launched(monkeypatch, cpu_threads=4)
    assert flag(gpu, "--device") == "cuda:0"
    assert flag(cpu, "--device") == "cpu"
    assert "--standby" in cpu


def test_full_precision_reaches_every_compute_worker(monkeypatch) -> None:
    commands = launched(monkeypatch, cpu_threads=4, gpu_dtype="float32")
    assert [flag(command, "--dtype") for command in commands] == ["float32", "float32"]


def test_kv_checkpoint_memory_stays_separate_from_weight_memory(monkeypatch) -> None:
    worker, store = launched(monkeypatch, weights_ram_gb="8", ram_budget_gb=2.0)
    assert flag(worker, "--weights-ram-gb") == "8"
    assert flag(store, "--ram-budget-gb") == "2.0"
    assert flag(store, "--roles") == "kv-cache,checkpoint"


def test_a_memory_worker_keeps_copies_where_its_node_was_told_to(monkeypatch) -> None:
    """It is the first choice to hold copies of saved states, so its settings must reach it."""
    _, store = launched(monkeypatch, weights_ram_gb="8", ram_budget_gb=2.0, cache_dir="states-here", state_disk_gb=3.0, state_copy_mb=4.0)
    assert flag(store, "--cache-dir") == "states-here"
    assert flag(store, "--state-disk-gb") == "3.0"
    assert flag(store, "--state-copy-mb") == "4.0"


def test_a_gpu_node_contributes_gpu_cpu_and_ram_as_one_worker(monkeypatch) -> None:
    monkeypatch.setattr("isharded.node._has_accelerator", lambda: True)  # CI machines have no GPU
    commands = launched(monkeypatch, gpu_device="auto", cpu_threads=8, weights_ram_gb="auto", gpu_memory_gb=16.0)
    assert len(commands) == 1
    assert flag(commands[0], "--device") == "auto"


@pytest.mark.parametrize("path", [r"\\\\tsclient\\D\\project", "//server/share/project"])
def test_network_paths_are_recognized(path: str) -> None:
    assert is_remote_path(path)


def test_local_paths_are_left_alone(tmp_path: Path) -> None:
    assert not is_remote_path(tmp_path)
    assert local_code_root(tmp_path) == tmp_path


def test_a_remote_checkout_runs_from_a_local_copy(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "remote"
    (source / "isharded" / "__pycache__").mkdir(parents=True)
    (source / "isharded" / "worker.py").write_text("new")
    (source / "isharded" / "__pycache__" / "worker.pyc").write_text("skip")
    home = tmp_path / "local"
    (home / "iSharded" / "app" / "isharded").mkdir(parents=True)
    (home / "iSharded" / "app" / "isharded" / "stale.py").write_text("old")
    monkeypatch.setenv("LOCALAPPDATA", str(home))
    monkeypatch.setattr("isharded.node.is_remote_path", lambda path: True)

    root = local_code_root(source)

    assert (root / "isharded" / "worker.py").read_text() == "new"
    assert not (root / "isharded" / "stale.py").exists()
    assert not (root / "isharded" / "__pycache__").exists()


def test_every_node_runs_a_control_plane_and_finds_it(monkeypatch) -> None:
    workers = launched(monkeypatch, join="http://seed:8000", control_port=8300, advertise_host="10.0.0.2")
    assert flag(workers[0], "--orchestrator-url") == "http://10.0.0.2:8300,http://seed:8000"


def test_a_large_node_log_starts_over_and_keeps_the_last(tmp_path, monkeypatch) -> None:
    from isharded.node import open_log

    monkeypatch.setattr("isharded.node.LOG_ROTATE_BYTES", 4)
    path = tmp_path / "node.log"
    path.write_bytes(b"123456")
    open_log(str(path)).close()
    assert (tmp_path / "node.log.1").read_bytes() == b"123456"
    assert path.read_bytes() == b""
    assert open_log(None) is None


def test_a_vulkan_gpu_is_read_from_the_probe_and_software_devices_are_left_out(tmp_path) -> None:
    import json
    from types import SimpleNamespace

    from isharded.memory import read_vulkan_devices

    report = [
        {"backend": "Vulkan0", "name": "Radeon RX 7900 XT", "integrated": False, "free_bytes": 18 * 1024**3, "total_bytes": 20 * 1024**3},
        {"backend": "Vulkan1", "name": "Radeon Graphics", "integrated": True, "free_bytes": 1024**3, "total_bytes": 2 * 1024**3},
        {"backend": "CUDA0", "name": "not this build's business", "integrated": False, "free_bytes": 1, "total_bytes": 1},
    ]
    answer = SimpleNamespace(returncode=0, stdout="some driver noise\n" + json.dumps(report))
    first, second = read_vulkan_devices(tmp_path, run=lambda *_, **__: answer)
    assert (first.device, first.kind, first.backend, first.name) == ("vulkan:0", "vram", "vulkan", "Radeon RX 7900 XT")
    assert (second.device, second.kind) == ("vulkan:1", "unified")


def test_a_crashing_driver_means_no_vulkan_gpu_and_no_crash(tmp_path) -> None:
    from types import SimpleNamespace

    from isharded.memory import read_vulkan_devices

    assert read_vulkan_devices(tmp_path, run=lambda *_, **__: SimpleNamespace(returncode=3, stdout="")) == []
    assert read_vulkan_devices(tmp_path, run=lambda *_, **__: SimpleNamespace(returncode=0, stdout="")) == []


def test_asking_for_vulkan_with_none_says_what_to_check(monkeypatch) -> None:
    from isharded import memory

    monkeypatch.setattr(memory, "vulkan_accelerators", lambda: ())
    with pytest.raises(ValueError, match="no Vulkan GPU found"):
        memory.discover_accelerators("vulkan")


def test_a_container_limit_is_the_memory_a_node_can_use(tmp_path: Path, monkeypatch) -> None:
    from types import SimpleNamespace

    from isharded import inventory

    gib = 1024**3
    monkeypatch.setattr(inventory.psutil, "virtual_memory", lambda: SimpleNamespace(total=500 * gib, available=450 * gib))
    assert inventory.ram(tmp_path).total == 500 * gib  # no cgroup files: the host is all there is
    (tmp_path / "memory.max").write_text("max\n")
    assert inventory.ram(tmp_path).total == 500 * gib  # a container with no limit
    (tmp_path / "memory.max").write_text(f"{50 * gib}\n")
    (tmp_path / "memory.current").write_text(f"{30 * gib}\n")
    (tmp_path / "memory.stat").write_text(f"anon 1\ninactive_file {10 * gib}\n")
    limited = inventory.ram(tmp_path)
    assert (limited.total, limited.available) == (50 * gib, 30 * gib)  # page cache comes back on demand
