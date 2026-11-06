"""Which country a node is in, for the flag beside it on the dashboard.

Nothing here leaves the machine. The table in the package comes from the regional
internet registries, and scripts/build_countries.py rebuilds it. It is read only
when a public address needs looking up.

The registries record who holds an address, so a cloud's address can name the
cloud's home and not its data center. --location corrects that. It is also the
only way for a node behind a relay, whose address is the relay's.
"""

from __future__ import annotations

import argparse
from array import array
from bisect import bisect_right
from functools import lru_cache
import ipaddress
from itertools import accumulate
import logging
from pathlib import Path
import struct
from urllib.parse import urlsplit
import zlib

TABLE = Path(__file__).with_name("countries.bin")
# ISO 3166-1, and XK for Kosovo, which has no ISO code. Each has a flag in flags.svg.
COUNTRIES = frozenset("""
    AD AE AF AG AI AL AM AO AQ AR AS AT AU AW AX AZ BA BB BD BE BF BG BH BI BJ BL BM BN BO BQ BR BS BT BV BW BY BZ
    CA CC CD CF CG CH CI CK CL CM CN CO CR CU CV CW CX CY CZ DE DJ DK DM DO DZ EC EE EG EH ER ES ET FI FJ FK FM FO
    FR GA GB GD GE GF GG GH GI GL GM GN GP GQ GR GS GT GU GW GY HK HM HN HR HT HU ID IE IL IM IN IO IQ IR IS IT JE
    JM JO JP KE KG KH KI KM KN KP KR KW KY KZ LA LB LC LI LK LR LS LT LU LV LY MA MC MD ME MF MG MH MK ML MM MN MO
    MP MQ MR MS MT MU MV MW MX MY MZ NA NC NE NF NG NI NL NO NP NR NU NZ OM PA PE PF PG PH PK PL PM PN PR PS PT PW
    PY QA RE RO RS RU RW SA SB SC SD SE SG SH SI SJ SK SL SM SN SO SR SS ST SV SX SY SZ TC TD TF TG TH TJ TK TL TM
    TN TO TR TT TV TW TZ UA UG UM US UY UZ VA VC VE VG VI VN VU WF WS XK YE YT ZA ZM ZW
""".split())

logger = logging.getLogger(__name__)


def country_code(text: str) -> str:
    """Reads --location: two letters in either case, and a code that exists."""
    code = text.strip().upper()
    if code not in COUNTRIES:
        raise argparse.ArgumentTypeError(f"{text!r} is not a two letter country code, such as DE")
    return code


@lru_cache(maxsize=1)
def _table() -> tuple[array, bytes, list[str | None]]:
    """Each row starts a run of addresses, and row 0 starts at zero, so every key has a row.

    An IPv4 address is its own key. An IPv6 address is its first 64 bits, which the
    registries never divide further and which sort above every IPv4 key. Starts are
    stored as differences, one byte of each at a time, because that packs smallest.
    """
    data = zlib.decompress(TABLE.read_bytes())
    rows, known = struct.unpack_from("<II", data)
    start = 8 + 2 * known
    names = [None, *(data[at:at + 2].decode() for at in range(8, start, 2))]
    gaps = bytearray(8 * rows)
    for plane in range(8):
        gaps[plane::8] = data[start + plane * rows:start + (plane + 1) * rows]
    return array("Q", accumulate(struct.unpack(f"<{rows}Q", gaps))), data[start + 8 * rows:], names


def holder(key: int) -> str | None:
    """The country in the table for one key, or None where nobody holds it."""
    starts, countries, names = _table()
    return names[countries[bisect_right(starts, key) - 1]]


def country_of(address: str) -> str | None:
    """The country of a public address, or None. Better to show nothing than a guess."""
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:  # a name, and resolving it would tell a DNS server
        return None
    ip = getattr(ip, "ipv4_mapped", None) or ip
    if not ip.is_global:  # private, loopback, link-local, and Tailscale's 100.64.0.0/10
        return None
    key = int(ip) if ip.version == 4 else int(ip) >> 64
    if ip.version == 6 and key < 1 << 32:  # no registry holds these, and the key would read as IPv4
        return None
    try:
        return holder(key)
    except (OSError, zlib.error, struct.error, IndexError) as exc:  # a missing flag must not stop a node
        logger.warning("could not read the country table: %s", exc)
        return None


def node_country(url: str, location: str | None = None) -> str | None:
    """What a node reports: what its owner said, or else what its advertised address says."""
    return location or country_of(urlsplit(url).hostname or "")
