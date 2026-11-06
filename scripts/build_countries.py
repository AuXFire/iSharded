"""Rebuilds isharded/countries.bin, which says which country holds which addresses.

    python scripts/build_countries.py
    python scripts/build_countries.py --from-dir saved    # saved/arin.txt and so on

The source is the delegation file each of the five regional internet registries
publishes every day. Only blocks given to someone in a country count: free and
reserved space, and blocks marked with a region such as EU, stay blank, so a lookup
there finds nothing and does not guess.

Each block is read back from the finished table before the script says it is done.
"""
import argparse
import ipaddress
from pathlib import Path
import struct
import sys
import urllib.request
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isharded import geo  # noqa: E402

REGISTRIES = {
    "afrinic": "https://ftp.afrinic.net/pub/stats/afrinic/delegated-afrinic-extended-latest",
    "apnic": "https://ftp.apnic.net/stats/apnic/delegated-apnic-extended-latest",
    "arin": "https://ftp.arin.net/pub/stats/arin/delegated-arin-extended-latest",
    "lacnic": "https://ftp.lacnic.net/pub/stats/lacnic/delegated-lacnic-extended-latest",
    "ripencc": "https://ftp.ripe.net/pub/stats/ripencc/delegated-ripencc-extended-latest",
}


def read(registry, folder):
    if folder:
        return (folder / f"{registry}.txt").read_text(encoding="utf-8", errors="replace")
    with urllib.request.urlopen(REGISTRIES[registry], timeout=300) as answer:
        return answer.read().decode("utf-8", errors="replace")


def blocks(text):
    """(first key, key after the last, country) for each block someone in a country holds."""
    for line in text.splitlines():
        parts = line.split("|")
        if len(parts) < 7 or parts[6] not in {"allocated", "assigned"} or parts[1].upper() not in geo.COUNTRIES:
            continue
        if parts[2] == "ipv4":
            first = int(ipaddress.IPv4Address(parts[3]))
            yield first, first + int(parts[4]), parts[1].upper()
        elif parts[2] == "ipv6":
            first = int(ipaddress.IPv6Address(parts[3])) >> 64
            yield first, first + (1 << max(64 - int(parts[4]), 0)), parts[1].upper()


def rows(found):
    """Where each run starts, and its country, blank for a gap. Neighbours in one country join up."""
    table, end, skipped = [], 0, 0
    for first, after, country in sorted(found):
        if first < end:  # two registries claiming a block; none do today
            skipped += 1
            continue
        if first > end:
            table.append((end, ""))
        if not table or table[-1][1] != country:
            table.append((first, country))
        end = after
    table.append((end, ""))
    return table, skipped


def pack(table):
    names = sorted({country for _, country in table if country})
    index = {"": 0, **{name: number + 1 for number, name in enumerate(names)}}
    starts = [start for start, _ in table]
    gaps = struct.pack(f"<{len(starts)}Q", *(after - before for before, after in zip([0, *starts], starts)))
    planes = b"".join(gaps[plane::8] for plane in range(8))
    columns = bytes(index[country] for _, country in table)
    return zlib.compress(struct.pack("<II", len(starts), len(names)) + "".join(names).encode() + planes + columns, 9)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from-dir", type=Path, help="read <registry>.txt from here and download nothing")
    args = parser.parse_args()
    found = [block for registry in REGISTRIES for block in blocks(read(registry, args.from_dir))]
    table, skipped = rows(found)
    geo.TABLE.write_bytes(pack(table))
    geo._table.cache_clear()
    wrong = [block for block in found if geo.holder(block[0]) != block[2] or geo.holder(block[1] - 1) != block[2]]
    if len(wrong) > skipped:
        raise SystemExit(f"{len(wrong)} blocks read back wrong, for example {wrong[0]}")
    print(f"{len(found)} blocks, {len(table)} rows, {geo.TABLE.stat().st_size} bytes in {geo.TABLE.name}")


if __name__ == "__main__":
    main()
