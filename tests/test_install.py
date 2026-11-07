"""The install scripts, the wheels, and the packages built from them."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_engine  # noqa: E402
import build_wheel  # noqa: E402
from isharded.engine import library_name  # noqa: E402

SHELL = shutil.which("sh")
POWERSHELL = shutil.which("powershell") or shutil.which("pwsh")
needs_shell = pytest.mark.skipif(SHELL is None, reason="needs a POSIX shell")
needs_powershell = pytest.mark.skipif(POWERSHELL is None, reason="needs PowerShell")
INSTALL_SH = (ROOT / "install.sh").read_bytes().replace(b"\r\n", b"\n").decode()
INSTALL_PS1 = (ROOT / "install.ps1").read_text(encoding="utf-8")
SHELL_SCRIPTS = [
    "install.sh", "scripts/ci_local.sh", "packaging/linux/build_deb.sh", "packaging/linux/isharded", "packaging/linux/postinst",
    "packaging/macos/build_pkg.sh", "packaging/macos/iSharded", "packaging/macos/first-run.command", "packaging/macos/postinstall",
]
POWERSHELL_SCRIPTS = ["install.ps1", "scripts/install_node_task.ps1", *(f"packaging/windows/{name}.ps1" for name in ("setup", "firewall", "uninstall"))]
SUMS = "\n".join([
    f"{'a' * 64}  isharded-0.3.0-py3-none-any.whl",
    f"{'b' * 64}  isharded-0.3.0-py3-none-manylinux_2_28_x86_64.whl",
    f"{'c' * 64} *isharded-0.3.0-py3-none-win_amd64.whl",
    f"{'d' * 64}  iSharded-Setup.exe",
]) + "\n"


def in_shell(tmp_path: Path, command: str, **env: str) -> subprocess.CompletedProcess:
    """Runs a command with install.sh's functions loaded. Nothing installs."""
    (tmp_path / "install.sh").write_text(INSTALL_SH, encoding="utf-8", newline="\n")
    return subprocess.run(
        [SHELL, "-c", f". ./install.sh\n{command}"], cwd=tmp_path, capture_output=True, text=True,
        env={**os.environ, "ISHARDED_INSTALL_TEST": "1", **env},
    )


def in_powershell(command: str, **env: str) -> subprocess.CompletedProcess:
    script = str(ROOT / "install.ps1").replace("'", "''")
    return subprocess.run(
        [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", f". '{script}'\n{command}"],
        capture_output=True, text=True, env={**os.environ, "ISHARDED_INSTALL_TEST": "1", **env},
    )


@needs_shell
@pytest.mark.parametrize("name", SHELL_SCRIPTS)
def test_shell_scripts_parse(name: str) -> None:
    source = (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
    assert source.startswith(b"#!/bin/")
    done = subprocess.run([SHELL, "-n"], input=source, capture_output=True)
    assert done.returncode == 0, done.stderr


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="needs shellcheck")
@pytest.mark.parametrize("name", SHELL_SCRIPTS)
def test_shell_scripts_pass_shellcheck(name: str, tmp_path: Path) -> None:
    copy = tmp_path / "script"
    copy.write_bytes((ROOT / name).read_bytes().replace(b"\r\n", b"\n"))
    done = subprocess.run(["shellcheck", "--severity=warning", "--shell=sh" if "ci_local" not in name else "--shell=bash", copy.name], cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout


@needs_powershell
def test_powershell_scripts_parse() -> None:
    names = ",".join(f"'{ROOT / name}'" for name in POWERSHELL_SCRIPTS)
    command = f"foreach ($file in {names}) {{ $bad = $null; [void][Management.Automation.Language.Parser]::ParseFile($file, [ref]$null, [ref]$bad); if ($bad) {{ \"$file $bad\" }} }}"
    done = subprocess.run([POWERSHELL, "-NoProfile", "-Command", command], capture_output=True, text=True)
    assert done.returncode == 0 and not done.stdout.strip(), done.stdout + done.stderr


@needs_shell
@pytest.mark.parametrize(
    "tag, answer",
    [
        ("manylinux_2_28_x86_64", f"{'b' * 64} isharded-0.3.0-py3-none-manylinux_2_28_x86_64.whl"),
        ("win_amd64", f"{'c' * 64} isharded-0.3.0-py3-none-win_amd64.whl"),  # a star marks a binary file in some tools
        ("macosx_11_0_arm64", f"{'a' * 64} isharded-0.3.0-py3-none-any.whl"),
        ("", f"{'a' * 64} isharded-0.3.0-py3-none-any.whl"),
    ],
)
def test_the_shell_picks_this_machines_wheel_or_the_pure_one(tmp_path: Path, tag: str, answer: str) -> None:
    (tmp_path / "sums").write_text(SUMS, encoding="utf-8", newline="\n")
    assert in_shell(tmp_path, f"pick_wheel sums '{tag}'").stdout.strip() == answer


@needs_shell
def test_the_shell_picks_nothing_from_a_release_with_no_wheel_it_can_use(tmp_path: Path) -> None:
    (tmp_path / "sums").write_text(f"{'b' * 64}  isharded-0.3.0-py3-none-manylinux_2_28_x86_64.whl\n", encoding="utf-8", newline="\n")
    assert in_shell(tmp_path, "pick_wheel sums macosx_11_0_arm64").stdout == ""


@needs_shell
@pytest.mark.parametrize(
    "system, machine, glibc, tag",
    [
        ("Linux", "x86_64", "2.35", "manylinux_2_28_x86_64"),
        ("Linux", "aarch64", "2.28", "manylinux_2_28_aarch64"),
        ("Linux", "x86_64", "2.27", ""),  # too old for the engine's wheel
        ("Linux", "x86_64", "", ""),  # musl, which has no glibc
        ("Linux", "riscv64", "2.39", ""),
        ("Darwin", "arm64", "", "macosx_11_0_arm64"),
        ("Darwin", "x86_64", "", ""),
    ],
)
def test_the_shell_knows_which_wheels_run_here(tmp_path: Path, system: str, machine: str, glibc: str, tag: str) -> None:
    assert in_shell(tmp_path, f"wheel_tag {system} {machine} '{glibc}'").stdout.strip() == tag


@needs_shell
def test_the_shell_refuses_a_file_with_the_wrong_hash(tmp_path: Path) -> None:
    (tmp_path / "file").write_bytes(b"made")
    good = hashlib.sha256(b"made").hexdigest()
    assert in_shell(tmp_path, f"check_hash file {good}").returncode == 0
    done = in_shell(tmp_path, f"check_hash file {'0' * 64}")
    assert done.returncode != 0 and "does not match" in done.stderr


def mirror(tmp_path: Path, wheel: bytes, listed: bytes | None = None) -> str:
    """A release folder on disk, which curl reads as it would GitHub."""
    folder = tmp_path / "mirror"
    folder.mkdir()
    name = "isharded-0.3.0-py3-none-any.whl"
    (folder / name).write_bytes(wheel)
    (folder / "SHA256SUMS").write_text(f"{hashlib.sha256(listed or wheel).hexdigest()}  {name}\n", encoding="utf-8", newline="\n")
    return folder.as_uri()


CHOOSE = 'work=$(mktemp -d); choose_package; echo "package=$package"'


@needs_shell
def test_a_release_wheel_is_fetched_and_checked(tmp_path: Path) -> None:
    done = in_shell(tmp_path, CHOOSE, ISHARDED_RELEASE_URL=mirror(tmp_path, b"wheel"))
    assert done.returncode == 0, done.stderr
    assert re.search(r"package=.*isharded-0\.3\.0-py3-none-any\.whl", done.stdout)
    assert "fast path is off" not in done.stdout


@needs_shell
def test_a_wheel_that_fails_its_hash_stops_the_install_with_no_fallback(tmp_path: Path) -> None:
    done = in_shell(tmp_path, CHOOSE, ISHARDED_RELEASE_URL=mirror(tmp_path, b"tampered", listed=b"wheel"))
    assert done.returncode != 0 and "does not match" in done.stderr
    assert "package=" not in done.stdout


@needs_shell
def test_no_release_falls_back_to_the_source_archive_and_says_what_that_costs(tmp_path: Path) -> None:
    done = in_shell(tmp_path, CHOOSE, ISHARDED_RELEASE_URL=(tmp_path / "nothing").as_uri())
    assert done.returncode == 0
    assert "package=https://github.com/AuXFire/iSharded/archive/refs/heads/main.tar.gz" in done.stdout
    assert "fast path is off" in done.stdout and "scripts/build_engine.py" in done.stdout


@needs_shell
def test_a_chosen_source_is_used_as_it_is(tmp_path: Path) -> None:
    done = in_shell(tmp_path, CHOOSE, ISHARDED_SOURCE="https://mirror.example/x.tar.gz", ISHARDED_RELEASE_URL=(tmp_path / "nothing").as_uri())
    assert "package=https://mirror.example/x.tar.gz" in done.stdout and "fast path" not in done.stdout


@needs_shell
@pytest.mark.parametrize("options, says", [("--bad", "Unknown option"), ("--no-start --service", "do not go together")])
def test_the_shell_refuses_options_before_doing_anything(tmp_path: Path, options: str, says: str) -> None:
    done = in_shell(tmp_path, f"main {options}", HOME=str(tmp_path))
    assert done.returncode != 0 and says in done.stderr
    assert not (tmp_path / ".isharded").exists()


@needs_shell
def test_an_upgrade_stops_the_running_node_first(tmp_path: Path) -> None:
    tool = tmp_path / "app" / "bin" / "isharded"
    tool.parent.mkdir(parents=True)
    tool.write_text('#!/bin/sh\necho "called with $1"\nexit 1\n', encoding="utf-8", newline="\n")
    tool.chmod(0o755)
    done = in_shell(tmp_path, "folder=app; stop_old_node; echo went on")
    assert done.stdout.split("\n")[:2] == ["called with stop", "went on"]  # an older install has no stop, and that is no reason to give up


@needs_shell
@pytest.mark.parametrize("system, machine", [("Linux", "x86_64"), ("Linux", "aarch64"), ("Darwin", "x86_64"), ("Darwin", "arm64")])
def test_every_uv_download_has_a_pinned_hash(tmp_path: Path, system: str, machine: str) -> None:
    done = in_shell(tmp_path, f"uv_hash $(uv_asset {system} {machine})")
    assert re.fullmatch(r"[0-9a-f]{64}", done.stdout.strip())


def test_both_scripts_pin_the_same_uv_and_python() -> None:
    uv = re.search(r"^UV_VERSION=(\S+)$", INSTALL_SH, re.M).group(1)
    assert f'$UvVersion = "{uv}"' in INSTALL_PS1
    python = re.search(r"^PYTHON_VERSION=(\S+)$", INSTALL_SH, re.M).group(1)
    assert f'$PythonVersion = "{python}"' in INSTALL_PS1
    assert re.search(r'\$UvHash = "[0-9a-f]{64}"', INSTALL_PS1)


@pytest.mark.parametrize("script", [INSTALL_SH, INSTALL_PS1], ids=["sh", "ps1"])
def test_the_scripts_keep_what_they_promise(script: str) -> None:
    assert "ISHARDED_INSTALL_DIR" in script and "ISHARDED_SOURCE" in script
    assert "UV_SYSTEM_CERTS" in script and "--use-feature=truststore" in script
    assert "--no-start" in script and "--service" in script and "up --background" in script
    assert "download.pytorch.org/whl/" in script


@needs_powershell
def test_powershell_picks_this_machines_wheel_or_the_pure_one(tmp_path: Path) -> None:
    sums = tmp_path / "sums"
    sums.write_text(SUMS, encoding="utf-8")
    pick = f"(Select-Wheel (Get-Content -LiteralPath '{sums}') '{{}}').Name"
    done = in_powershell(f"{pick.format('win_amd64')}\n{pick.format('win_arm64')}\n[bool](Select-Wheel @('x') 'win_amd64')")
    assert done.stdout.split() == ["isharded-0.3.0-py3-none-win_amd64.whl", "isharded-0.3.0-py3-none-any.whl", "False"], done.stderr


@needs_powershell
def test_powershell_refuses_a_file_with_the_wrong_hash(tmp_path: Path) -> None:
    made = tmp_path / "file"
    made.write_bytes(b"made")
    assert in_powershell(f"Assert-Hash '{made}' {hashlib.sha256(b'made').hexdigest().upper()}").returncode == 0
    done = in_powershell(f"Assert-Hash '{made}' {'0' * 64}")
    assert done.returncode != 0 and "does not match" in done.stderr + done.stdout


@needs_powershell
@pytest.mark.parametrize("options, says", [(["--bad"], "Unknown option"), (["--no-start", "--service"], "do not go together")])
def test_powershell_refuses_options_before_doing_anything(options: list[str], says: str) -> None:
    done = in_powershell(f"Install-ISharded @({','.join(repr(item) for item in options)})")
    assert done.returncode != 0 and says in done.stderr + done.stdout


def test_a_wheel_must_hold_exactly_the_engine_it_is_tagged_for(tmp_path: Path) -> None:
    wheel = tmp_path / "x.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("isharded/up.py", "")
        archive.writestr("isharded/native/libisharded_engine.so", "x")
    build_wheel.check_wheel(wheel, "libisharded_engine.so")
    with pytest.raises(SystemExit):
        build_wheel.check_wheel(wheel, "isharded_engine.dll")
    with pytest.raises(SystemExit):
        build_wheel.check_wheel(wheel, None)  # a pure wheel with a Linux engine in it


def test_a_wheel_leaves_a_checkouts_own_engine_out(tmp_path: Path) -> None:
    pure, tagged = tmp_path / "pure", tmp_path / "tagged"
    pure.mkdir()
    tagged.mkdir()
    library = tmp_path / "libisharded_engine.so"
    library.write_bytes(b"x")
    build_wheel.stage_source(pure, None)
    build_wheel.stage_source(tagged, library)
    assert not (pure / "isharded" / "native").exists() and (pure / "isharded" / "up.py").is_file()
    assert [item.name for item in (tagged / "isharded" / "native").iterdir()] == [library.name]


def test_the_build_names_the_library_as_the_package_does() -> None:
    assert build_engine.library_here() == library_name()


def test_a_library_that_needs_a_newer_glibc_is_seen() -> None:
    assert build_engine.newest_glibc(b"\0GLIBC_2.17\0GLIBC_2.28\0GLIBC_2.3.4\0") == (2, 28)
    assert build_engine.newest_glibc(b"\0GLIBC_2.34\0") > build_engine.GLIBC
    assert build_engine.newest_glibc(b"nothing") == (0, 0)


def test_every_target_is_built_by_the_release_and_picked_by_an_installer() -> None:
    release = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    for target, (tag, library) in build_engine.TARGETS.items():
        assert target in release
        assert tag in INSTALL_SH + INSTALL_PS1
        assert library.split(".")[-1] in {"so", "dll", "dylib"}


def test_ci_runs_on_a_mac_and_tests_the_engine_everywhere() -> None:
    tests = (ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    assert tests.count("macos-latest") == 2 and "cargo test" in tests


def test_the_release_publishes_only_what_was_checked() -> None:
    release = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert "SHA256SUMS" in release and 'tags: ["v*"]' in release and "workflow_dispatch" in release
    assert release.count("contents: write") == 1  # the publish job alone can write
    for path in re.findall(r"packaging[/\\][\w./\\-]+\.\w+", release):
        assert (ROOT / path.replace("\\", "/")).is_file(), path


def test_a_release_by_hand_builds_the_tag_it_publishes() -> None:
    release = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert release.count("uses: actions/checkout@v4") == release.count("ref: ${{ inputs.tag || github.ref }}") + 1  # publish checks out $TAG itself


def test_an_upgrade_through_the_windows_installer_drops_the_old_wheel() -> None:
    script = (ROOT / "packaging" / "windows" / "isharded.iss").read_text(encoding="utf-8")
    assert 'Type: files; Name: "{app}\\isharded-*.whl"' in script


def test_a_debian_package_does_not_carry_mktemps_mode() -> None:
    assert 'chmod 755 "$stage"' in (ROOT / "packaging" / "linux" / "build_deb.sh").read_text(encoding="utf-8")


def test_the_windows_installer_asks_for_an_administrator_only_for_the_firewall() -> None:
    script = (ROOT / "packaging" / "windows" / "isharded.iss").read_text(encoding="utf-8")
    assert "PrivilegesRequired=lowest" in script
    assert [line for line in script.splitlines() if "Verb: runas" in line] == [line for line in script.splitlines() if "firewall.ps1" in line and line.startswith("Filename")]
    assert re.search(r"^Name: firewall;.*Flags: unchecked", script, re.M)
    for label in ("Open iSharded", "Stop iSharded", "Uninstall iSharded"):
        assert f"{{group}}\\{label}" in script
    for name in re.findall(r'^Source: "([^"{]+)"', script, re.M):
        assert (ROOT / "packaging" / "windows" / name.replace("\\", "/")).resolve().is_file(), name


def test_the_firewall_rules_stay_off_public_networks() -> None:
    script = (ROOT / "packaging" / "windows" / "firewall.ps1").read_text(encoding="utf-8")
    rules = [line for line in script.splitlines() if " add rule " in line]
    assert len(rules) == 3 and all("profile=private,domain" in rule for rule in rules)


def test_packages_install_the_wheel_they_carry() -> None:
    for name in ("packaging/windows/setup.ps1", "packaging/macos/first-run.command", "packaging/linux/postinst", "packaging/linux/isharded"):
        assert "ISHARDED_SOURCE" in (ROOT / name).read_text(encoding="utf-8"), name


def test_the_cask_is_a_template_the_release_fills_in() -> None:
    cask = (ROOT / "packaging" / "homebrew" / "isharded.rb").read_text(encoding="utf-8")
    assert 'version "__VERSION__"' in cask and 'sha256 "__SHA256__"' in cask and 'pkg "iSharded.pkg"' in cask
