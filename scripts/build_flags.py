"""Rebuilds isharded/flags.svg, the flags the dashboard draws, from country-flag-icons.

    python scripts/build_flags.py

It needs Node, for npm and svgo. Flags sit in the page itself, so they need no
network and pass any Content-Security-Policy. Each is rounded to a 513th or so of
its width, which nobody can see at the size a table shows it. The drawings are
different sizes, so the number of decimals differs from flag to flag.
"""
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isharded import geo  # noqa: E402

PACKAGE = "country-flag-icons@1.6.20"
SVGO = "svgo@4.1.0"
CONFIG = """export default { multipass: true, floatPrecision: Number(process.env.DECIMALS), plugins: [
  { name: 'preset-default', params: { overrides: { cleanupIds: { minify: true } } } },
  'removeDimensions', 'removeXMLNS' ] };
"""
DRAWING = re.compile(r'<svg viewBox="([^"]+)">(.*)</svg>', re.S)


def decimals(drawing):
    width = float(re.search(r'viewBox="[\d.]+ [\d.]+ ([\d.]+)', drawing.decode("utf-8")).group(1))
    return max(0, math.ceil(math.log10(513 / width)))


def main():
    npm, npx = shutil.which("npm"), shutil.which("npx")
    if not npm or not npx:
        raise SystemExit("This needs Node, for npm and npx.")
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        subprocess.run([npm, "pack", PACKAGE, "--silent"], cwd=folder, check=True)
        with tarfile.open(next(folder.glob("*.tgz"))) as archive:
            licence = archive.extractfile("package/LICENSE").read().decode("utf-8").strip()
            for code in geo.COUNTRIES:
                drawing = archive.extractfile(f"package/3x2/{code}.svg").read()
                group = folder / "large" / str(decimals(drawing))
                group.mkdir(parents=True, exist_ok=True)
                (group / f"{code}.svg").write_bytes(drawing)
        (folder / "svgo.config.mjs").write_text(CONFIG, encoding="utf-8")
        for group in (folder / "large").iterdir():
            command = [npx, "--yes", SVGO, "--config", "svgo.config.mjs", "-f", str(group), "-o", "small", "-q"]
            subprocess.run(command, cwd=folder, check=True, env={**os.environ, "DECIMALS": group.name})
        flags = []
        for code in sorted(geo.COUNTRIES):
            drawing = (folder / "small" / f"{code}.svg").read_text(encoding="utf-8").strip()
            found = DRAWING.fullmatch(drawing)
            if not found or re.search(r"\burl\(|\bid=", drawing):  # ids would clash in one shared sprite
                raise SystemExit(f"{code} does not fit one shared sprite")
            flags.append(f'<symbol id="flag-{code}" viewBox="{found[1]}">{found[2]}</symbol>')
    text = f'<svg xmlns="http://www.w3.org/2000/svg" hidden aria-hidden="true"><!-- {PACKAGE}\n{licence}\n-->\n' + "\n".join(flags) + "\n</svg>\n"
    target = ROOT / "isharded" / "flags.svg"
    target.write_text(text, encoding="utf-8", newline="\n")
    print(f"{len(flags)} flags, {target.stat().st_size} bytes in {target.name}")


if __name__ == "__main__":
    main()
