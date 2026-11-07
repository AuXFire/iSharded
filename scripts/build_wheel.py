"""Builds a wheel of iSharded, with the native engine in it or without.

    python scripts/build_wheel.py                                       the pure wheel
    python scripts/build_wheel.py --target x86_64-unknown-linux-gnu     one with the engine

It needs pip and wheel, and for a --target what scripts/build_engine.py needs. --library takes
an engine already built. It builds from a copy, so a checkout's own isharded/native stays out.
"""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_engine import TARGETS, build  # noqa: E402


def stage_source(folder: Path, library: Path | None) -> None:
    """Copies what a wheel holds, with only the engine asked for."""
    shutil.copytree(ROOT / "isharded", folder / "isharded", ignore=shutil.ignore_patterns("__pycache__", "native"))
    for name in ("pyproject.toml", "LICENSE"):
        shutil.copy2(ROOT / name, folder / name)
    if library is not None:
        (folder / "isharded" / "native").mkdir()
        shutil.copy2(library, folder / "isharded" / "native" / library.name)


def check_wheel(wheel: Path, library_name: str | None) -> None:
    """A wheel missing its engine would install and run slowly, with nothing to say why."""
    with zipfile.ZipFile(wheel) as archive:
        engines = [name for name in archive.namelist() if name.startswith("isharded/native/")]
    expected = [f"isharded/native/{library_name}"] if library_name else []
    if engines != expected:
        raise SystemExit(f"{wheel.name} holds {engines}, and should hold {expected}")


def make_wheel(out: Path, library: Path | None, tag: str | None) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as folder:
        source, built = Path(folder, "source"), Path(folder, "wheels")
        source.mkdir()
        stage_source(source, library)
        subprocess.run([sys.executable, "-m", "pip", "wheel", "--quiet", "--no-deps", "--wheel-dir", str(built), str(source)], check=True)
        wheel = next(built.glob("isharded-*.whl"))
        if tag:
            # The engine is loaded by ctypes, so one wheel serves every Python: py3-none, not cp312.
            subprocess.run([sys.executable, "-m", "wheel", "tags", "--remove", "--python-tag", "py3", "--abi-tag", "none", "--platform-tag", tag, str(wheel)], check=True)
            wheel = next(built.glob(f"isharded-*-py3-none-{tag}.whl"))
        check_wheel(wheel, library.name if library else None)
        return Path(shutil.copy2(wheel, out / wheel.name))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", choices=sorted(TARGETS), default=None, help="put the engine in, built for this target")
    parser.add_argument("--library", type=Path, default=None, help="with --target: an engine already built, in place of building one")
    parser.add_argument("--out", type=Path, default=ROOT / "dist", help="where the wheel goes")
    args = parser.parse_args()
    if args.library and not args.target:
        parser.error("--library needs --target, to name the wheel")
    library = args.library or (build(args.target) if args.target else None)
    print(make_wheel(args.out, library, TARGETS[args.target][0] if args.target else None))


if __name__ == "__main__":
    main()
