"""Builds the native engine and puts it where the package looks for it.

    python scripts/build_engine.py

It needs Rust (https://rustup.rs). iSharded runs without the engine, over the slower Python
path, so this is optional. Set CARGO_TARGET_DIR to keep the build somewhere else.
"""
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isharded.engine import library_name  # noqa: E402


def main() -> None:
    subprocess.run(["cargo", "build", "--release", "--manifest-path", str(ROOT / "engine" / "Cargo.toml")], check=True)
    target = Path(os.getenv("CARGO_TARGET_DIR", ROOT / "engine" / "target"))
    built = target / "release" / library_name()
    folder = ROOT / "isharded" / "native"
    folder.mkdir(exist_ok=True)
    shutil.copy2(built, folder / library_name())
    print(f"built {folder / library_name()}, {built.stat().st_size // 1024} KiB")


if __name__ == "__main__":
    main()
