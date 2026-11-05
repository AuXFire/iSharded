"""A pool's credentials: its shared token, and a private authority for node identity.

The authority key signs node certificates, so keep it off the nodes once they have
theirs. Losing it means reissuing every certificate. Nothing here calls out to any
service, so a pool on a closed network can set itself up.
"""

from __future__ import annotations

import datetime
import ipaddress
import os
from pathlib import Path
import secrets
from urllib.parse import urlsplit

LOOPBACK = {"127.0.0.1", "localhost", "::1"}
# Tailscale's address ranges.
OVERLAYS = (ipaddress.ip_network("100.64.0.0/10"), ipaddress.ip_network("fd7a:115c:a1e0::/48"))
CERTIFICATE_DAYS = 825
AUTHORITY_DAYS = 3650


def home() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    return Path(base) / "ShardMesh" if base else Path.home() / ".shardmesh"


def load_token() -> str | None:
    """The pool token, from the environment or the file every node keeps."""
    token = os.getenv("SHARDMESH_POOL_TOKEN")
    if token:
        return token
    path = Path(os.getenv("SHARDMESH_POOL_TOKEN_FILE") or home() / "pool.token")
    if path.is_file():
        return path.read_text(encoding="utf-8").strip() or None
    return None


class ExposedPool(ValueError):
    """A node would be reachable by others with nothing protecting it."""


def _private_link(host: str) -> bool:
    """Whether traffic to this address is already encrypted or stays on this machine."""
    if host in LOOPBACK or host.endswith(".ts.net"):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_loopback or any(address in network for network in OVERLAYS if network.version == address.version)


def check_exposure(
    host: str,
    token: str | None,
    authority: str | None,
    open_pool: bool,
    advertised: str | None = None,
    trusted_network: bool = False,
) -> None:
    """Refuse to serve a network unprotected, unless told to on purpose.

    Credentials are not protection if they cross the network in the clear, since
    every gossip round carries the pool token. Tailscale encrypts its own links.
    """
    if host in LOOPBACK or open_pool:
        return
    if not (token or authority):
        raise ExposedPool(
            f"{host} is reachable by other machines and this pool has no credentials. "
            "Run `shardmesh pool init`, or pass --open if the network itself is trusted."
        )
    if trusted_network or os.getenv("SHARDMESH_TLS_CERT"):
        return
    address = urlsplit(advertised).hostname if advertised and "://" in advertised else advertised
    if not _private_link(address or host):
        raise ExposedPool(
            f"{address or host} would carry the pool's credentials unencrypted. Use TLS (--tls-cert and --tls-key) "
            "or a Tailscale address, or pass --trusted-network if nobody else can read this network."
        )


def _write_private(path: Path, data: bytes) -> None:
    path.write_bytes(data)
    try:
        path.chmod(0o600)
    except OSError:
        pass  # Windows keeps this to the file's inherited permissions


def init_pool(folder: str) -> Path:
    """Create a pool token and authority in a new folder."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    root = Path(folder)
    if (root / "ca.key").exists():
        raise ValueError(f"{root} already holds a pool authority")
    root.mkdir(parents=True, exist_ok=True)
    _write_private(root / "pool.token", (secrets.token_urlsafe(32) + "\n").encode())

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "ShardMesh pool authority")])
    now = datetime.datetime.now(datetime.timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=AUTHORITY_DAYS))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .add_extension(
            x509.KeyUsage(
                digital_signature=False, content_commitment=False, key_encipherment=False, data_encipherment=False,
                key_agreement=False, key_cert_sign=True, crl_sign=True, encipher_only=False, decipher_only=False,
            ),
            critical=True,
        )
        .sign(key, hashes.SHA256())
    )
    _write_private(
        root / "ca.key",
        key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()),
    )
    (root / "ca.pem").write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    return root


def add_node(folder: str, name: str, addresses: list[str]) -> tuple[Path, Path]:
    """Issue a node its certificate. It proves the node's name in both directions."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    root = Path(folder)
    authority = x509.load_pem_x509_certificate((root / "ca.pem").read_bytes())
    signer = serialization.load_pem_private_key((root / "ca.key").read_bytes(), password=None)
    names: list[x509.GeneralName] = []
    for address in addresses:
        try:
            names.append(x509.IPAddress(ipaddress.ip_address(address)))
        except ValueError:
            names.append(x509.DNSName(address))
    key = ec.generate_private_key(ec.SECP256R1())
    now = datetime.datetime.now(datetime.timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)]))
        .issuer_name(authority.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=CERTIFICATE_DAYS))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        # Python 3.13 verifies strictly and refuses certificates without these.
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(signer.public_key()), critical=False)
        .add_extension(x509.SubjectAlternativeName(names), critical=False)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH, ExtendedKeyUsageOID.CLIENT_AUTH]), critical=False)
        .sign(signer, hashes.SHA256())
    )
    certificate_path, key_path = root / f"{name}.pem", root / f"{name}.key"
    certificate_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    _write_private(
        key_path,
        key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()),
    )
    return certificate_path, key_path
