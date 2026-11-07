"""Builds the native engine and puts it where the package looks for it.

    python scripts/build_engine.py
    python scripts/build_engine.py --target aarch64-unknown-linux-gnu

It needs Rust (https://rustup.rs). iSharded runs without the engine, over the slower Python
path, so this is optional. Set CARGO_TARGET_DIR to keep the build somewhere else. A Linux
--target builds with cargo-zigbuild, which links against glibc 2.28 from any machine.
"""
import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
GLIBC = (2, 28)
# Each release target: the wheel tag it ships under, and the file cargo makes.
TARGETS = {
    "x86_64-unknown-linux-gnu": ("manylinux_2_28_x86_64", "libisharded_engine.so"),
    "aarch64-unknown-linux-gnu": ("manylinux_2_28_aarch64", "libisharded_engine.so"),
    "x86_64-pc-windows-msvc": ("win_amd64", "isharded_engine.dll"),
    "aarch64-apple-darwin": ("macosx_11_0_arm64", "libisharded_engine.dylib"),
}


def library_here() -> str:
    # Not isharded.engine.library_name: a release job has none of the package's dependencies.
    return {"win32": "isharded_engine.dll", "darwin": "libisharded_engine.dylib"}.get(sys.platform, "libisharded_engine.so")


def newest_glibc(library: bytes) -> tuple[int, int]:
    """The newest glibc a Linux library asks for, read from its symbol versions."""
    found = [(int(major), int(minor)) for major, minor in re.findall(rb"GLIBC_(\d+)\.(\d+)", library)]
    return max(found, default=(0, 0))


def build(target: str | None = None) -> Path:
    """Runs cargo for this machine, or for a target, and answers the built file."""
    manifest = str(ROOT / "engine" / "Cargo.toml")
    output = Path(os.getenv("CARGO_TARGET_DIR", ROOT / "engine" / "target"))
    if target is None:
        subprocess.run(["cargo", "build", "--release", "--manifest-path", manifest], check=True)
        return output / "release" / library_here()
    linux = "linux" in target
    # zigbuild reads the glibc version from the target's name, and strips it from the folder.
    command = ["cargo", "zigbuild", "--target", f"{target}.{GLIBC[0]}.{GLIBC[1]}"] if linux else ["cargo", "build", "--target", target]
    subprocess.run([*command, "--release", "--manifest-path", manifest], check=True)
    built = output / target / "release" / TARGETS[target][1]
    if linux and newest_glibc(built.read_bytes()) > GLIBC:
        raise SystemExit(f"{built} needs glibc newer than {GLIBC[0]}.{GLIBC[1]}, so its wheel tag would be a lie")
    return built


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", choices=sorted(TARGETS), default=None, help="build for this machine by default")
    args = parser.parse_args()
    built = build(args.target)
    folder = ROOT / "isharded" / "native"
    folder.mkdir(exist_ok=True)
    shutil.copy2(built, folder / built.name)
    print(f"built {folder / built.name}, {built.stat().st_size // 1024} KiB")


if __name__ == "__main__":
    main()
