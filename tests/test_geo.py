"""A node's country: looked up offline, overridden by its owner, and never guessed."""

import argparse
from pathlib import Path
import re
import socket
import time

import pydantic
import pytest

from isharded import geo, node, orchestrator, up, worker
from isharded.protocol import HardwareInfo, WorkerRegistration

ROOT = Path(__file__).resolve().parents[1]
FLAGS = (ROOT / "isharded" / "flags.svg").read_text(encoding="utf-8")
PRIVATE = ["10.1.2.3", "172.16.0.1", "192.168.1.20", "127.0.0.1", "169.254.1.1", "100.64.0.1", "100.100.100.100", "100.127.255.254",
           "::1", "fe80::1", "fc00::1", "fd7a:115c:a1e0::1", "::ffff:10.0.0.1", "::ffff:100.64.0.1", "2001:db8::1", "192.0.2.1"]


@pytest.fixture
def fresh_table():
    geo._table.cache_clear()
    yield
    geo._table.cache_clear()


def registration(**changes):
    hardware = HardwareInfo(device="cpu", backend="cpu", cpu_cores=1, ram_total_bytes=1)
    return WorkerRegistration(worker_id="a-gpu", advertised_url="http://8.8.8.8:8200", hardware=hardware, **changes)


@pytest.mark.parametrize("address, country", [
    ("8.8.8.8", "US"), ("1.1.1.1", "AU"), ("193.0.6.139", "NL"), ("200.3.14.10", "UY"), ("196.216.2.1", "ZA"), ("133.11.0.1", "JP"),
    ("2001:4860:4860::8888", "US"), ("2001:67c:2e8::1", "NL"), ("2001:dc0:2001::1", "AU"), ("2001:13c7:7001::1", "UY"),
    ("::ffff:8.8.8.8", "US"),
])
def test_a_public_address_gives_the_country_holding_it(address, country):
    assert geo.country_of(address) == country


@pytest.mark.parametrize("address", PRIVATE)
def test_an_address_only_this_network_can_reach_gives_nothing(address, fresh_table):
    assert geo.country_of(address) is None
    assert geo._table.cache_info().misses == 0, "a private address should not read the table"


@pytest.mark.parametrize("address", ["240.0.0.1", "224.0.0.1", "::2", "0:0:0:1::", "2c00::1", "3fff::1", "not an address", ""])
def test_space_nobody_holds_gives_nothing(address):
    assert geo.country_of(address) is None


def test_a_name_is_not_resolved_because_that_would_tell_a_server(monkeypatch):
    def refuse(*_):
        raise AssertionError("asked a DNS server")

    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    monkeypatch.setattr(socket, "gethostbyname", refuse)
    assert geo.node_country("http://node-a.example.com:8200") is None


def test_the_advertised_address_is_what_counts():
    assert geo.node_country("https://193.0.6.139:8200") == "NL"
    assert geo.node_country("http://[2001:67c:2e8::1]:8200") == "NL"
    assert geo.node_country("http://100.64.0.7:8200") is None


def test_a_stated_location_wins_and_needs_no_table(fresh_table):
    assert geo.node_country("http://8.8.8.8:8200", "DE") == "DE"
    assert geo.node_country("http://100.64.0.7:8200", "DE") == "DE"
    assert geo._table.cache_info().misses == 0


def test_an_operator_can_show_no_flag_at_all(fresh_table):
    assert geo.country_code(" None ") == "none"
    assert geo.node_country("http://8.8.8.8:8200", "none") is None
    assert geo._table.cache_info().misses == 0


def test_the_table_loads_once_and_quickly(fresh_table):
    began = time.perf_counter()
    assert geo.country_of("8.8.8.8") == "US"
    assert time.perf_counter() - began < 1  # 35 ms when measured, so a miss here means something broke
    geo.country_of("1.1.1.1")
    assert geo._table.cache_info().misses == 1


def test_a_broken_table_costs_a_flag_and_not_a_node(tmp_path, monkeypatch, fresh_table):
    monkeypatch.setattr(geo, "TABLE", tmp_path / "gone.bin")
    assert geo.country_of("8.8.8.8") is None
    (tmp_path / "gone.bin").write_bytes(b"not a table")
    assert geo.country_of("8.8.8.8") is None


def test_a_worker_reports_what_it_advertises_or_what_it_was_told(monkeypatch):
    monkeypatch.setenv("ISHARDED_ADVERTISE_URL", "http://193.0.6.139:8200")
    monkeypatch.delenv("ISHARDED_LOCATION", raising=False)
    assert worker.WorkerRuntime().country == "NL"
    monkeypatch.setenv("ISHARDED_LOCATION", "JP")
    assert worker.WorkerRuntime().country == "JP"


def test_admin_status_carries_the_country():
    record = orchestrator.WorkerRecord(registration=registration(country="DE"), last_seen=time.monotonic(), health={"status": "ok"})
    runtime = orchestrator.Orchestrator()
    runtime.workers["a-gpu"] = record
    assert runtime.status()["workers"]["a-gpu"]["country"] == "DE"


def test_an_older_worker_has_no_country_and_a_bad_one_is_refused():
    assert registration().country is None
    for bad in ("de", "DEU", "<b>", ""):
        with pytest.raises(pydantic.ValidationError):
            registration(country=bad)


def node_parser():
    parser = argparse.ArgumentParser()
    node.add_node_arguments(parser)
    return parser


@pytest.mark.parametrize("build", [node_parser, worker.build_parser], ids=["node", "worker"])
def test_location_is_read_in_either_case_and_must_exist(build, capsys):
    assert build().parse_args(["--location", "de"]).location == "DE"
    for bad in ("ZZ", "DEU", "Germany", ""):
        with pytest.raises(SystemExit):
            build().parse_args(["--location", bad])
    assert "two letter country code" in capsys.readouterr().err


def test_a_node_passes_its_location_to_each_worker():
    args = argparse.Namespace(advertise_scheme="http", advertise_host="10.0.0.1", bind_host="0.0.0.0", base_port=8200, location="DE", no_control=True, join=None, orchestrator_url=None)
    command = node._worker_command(args, "a-gpu", 8200, "cpu", "inference", [])
    assert command[command.index("--location") + 1] == "DE"
    args.location = None
    assert "--location" not in node._worker_command(args, "a-gpu", 8200, "cpu", "inference", [])


def run_up(home, monkeypatch, *flags):
    """Runs isharded up against settings on disk, as far as starting the node."""
    started = []
    monkeypatch.setenv("ISHARDED_UP_HOME", str(home))
    monkeypatch.setattr(up, "run_node", lambda args: started.append(args) or 0)
    monkeypatch.setattr(up, "keep_watch", lambda *_: None)
    parser = argparse.ArgumentParser()
    up.add_arguments(parser.add_subparsers())
    assert up.run_up(parser.parse_args(["up", *flags])) == 0
    return started[0]


def test_a_location_given_once_is_kept_and_a_new_one_replaces_it(tmp_path, monkeypatch):
    up.write_settings(tmp_path, {"name": "first", "address": "100.64.0.1", "port": 8209, "tls": False, "cpu_threads": 4})
    assert run_up(tmp_path, monkeypatch).location is None
    assert run_up(tmp_path, monkeypatch, "--location", "de").location == "DE"
    assert up.read_settings(tmp_path)["location"] == "DE"
    assert run_up(tmp_path, monkeypatch).location == "DE"
    assert run_up(tmp_path, monkeypatch, "--location", "JP").location == "JP"


def test_a_first_run_keeps_its_location(tmp_path):
    parser = argparse.ArgumentParser()
    up.add_arguments(parser.add_subparsers())
    args = parser.parse_args(["up", "--location", "de", "--address", "100.64.0.1"])
    assert up.first_settings(args)["location"] == "DE"


def test_every_country_has_exactly_one_flag_and_every_table_entry_is_a_country():
    assert re.findall(r'<symbol id="flag-([A-Z]{2})"', FLAGS) == sorted(geo.COUNTRIES)
    assert set(geo._table()[2][1:]) <= geo.COUNTRIES


def test_the_flags_are_inline_and_self_contained():
    assert not re.search(r"\burl\(|<image|<use|\bhref=", FLAGS), "a flag must not reach for anything"
    served = orchestrator.page("dashboard.html").body.decode("utf-8")
    assert "isharded:flags" not in served and 'id="flag-DE"' in served
    assert 'id="flag-DE"' not in orchestrator.page("chat.html").body.decode("utf-8")


def test_the_licence_stays_with_the_flags():
    assert "Permission is hereby granted" in FLAGS
    assert re.search(r"Copyright \(c\) \d+ @catamphetamine", FLAGS)


def test_the_data_ships_with_the_package_and_stays_small():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"flags.svg"' in pyproject and '"countries.bin"' in pyproject
    assert (ROOT / "isharded" / "countries.bin").stat().st_size < 1024 * 1024
    assert len(FLAGS) < 160 * 1024
