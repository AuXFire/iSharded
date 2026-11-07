#!/bin/sh
# Install iSharded for this user, on Linux or macOS, and start it. Nothing here needs root or Python.
#
#   curl -fsSL https://raw.githubusercontent.com/AuXFire/iSharded/main/install.sh | sh
#   ... | sh -s -- --no-start     only the files
#   ... | sh -s -- --service      also start at login
set -eu

UV_VERSION=0.13.6
REPOSITORY=https://github.com/AuXFire/iSharded
PYTHON_VERSION=3.12

say() { printf '%s\n' "$*"; }
die() { printf '%s\n' "$*" >&2; exit 1; }

# The musl builds of uv are static, so they run on any Linux, old or new.
uv_asset() {
  case "$1 $2" in
    "Linux x86_64") say uv-x86_64-unknown-linux-musl.tar.gz ;;
    "Linux aarch64" | "Linux arm64") say uv-aarch64-unknown-linux-musl.tar.gz ;;
    "Darwin x86_64") say uv-x86_64-apple-darwin.tar.gz ;;
    "Darwin arm64") say uv-aarch64-apple-darwin.tar.gz ;;
    *) return 1 ;;
  esac
}

uv_hash() {
  case "$1" in
    uv-x86_64-unknown-linux-musl.tar.gz) say 5d61f26b6c7213779eda90edc6971ea6b0b796bfb8d49a574c8ad69b59838548 ;;
    uv-aarch64-unknown-linux-musl.tar.gz) say 819692e404d23e5aae6bc44c0e260f1f5492fd1910ab4693b2c650f926dbe800 ;;
    uv-x86_64-apple-darwin.tar.gz) say 63ba17fdda55ad1a230d4b37fc6017ffa10c23688c83818bd3bfdf914c652a02 ;;
    uv-aarch64-apple-darwin.tar.gz) say 60fbd5c8a1ed681845eea28d9d622fcde03c51e91a43465b64a1e17db1ed3472 ;;
    *) return 1 ;;
  esac
}

# glibc's version, or nothing on musl, where a manylinux wheel will not load.
glibc_version() { getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}'; }

# The wheel tag this machine's engine is built for, or nothing: system, machine, glibc version.
wheel_tag() {
  case "$1" in
    Linux)
      major=${3%%.*}
      minor=${3#*.}
      minor=${minor%%.*}
      case "$major$minor" in '' | *[!0-9]*) return 0 ;; esac
      if [ "$major" -lt 2 ] || { [ "$major" -eq 2 ] && [ "$minor" -lt 28 ]; }; then return 0; fi
      case "$2" in
        x86_64) say manylinux_2_28_x86_64 ;;
        aarch64 | arm64) say manylinux_2_28_aarch64 ;;
      esac
      ;;
    Darwin) if [ "$2" = arm64 ]; then say macosx_11_0_arm64; fi ;;
  esac
}

# "hash name" for this machine's wheel in a SHA256SUMS file, or for the pure wheel, or nothing.
pick_wheel() {
  awk -v tag="$2" '
    { sub(/^\*/, "", $2) }
    tag != "" && index($2, "-py3-none-" tag ".whl") { print $1, $2; found = 1; exit }
    $2 ~ /^isharded-.*-py3-none-any\.whl$/ { pure = $1 " " $2 }
    END { if (!found && pure != "") print pure }
  ' "$1"
}

hash_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | awk '{print $1}'
  else die "This needs sha256sum or shasum to check what it downloads."
  fi
}

check_hash() {
  found=$(hash_of "$1")
  [ "$found" = "$2" ] || die "$(basename "$1") does not match its SHA-256, so it was not installed. Expected $2, got $found."
}

fetch() {
  if command -v curl >/dev/null 2>&1; then curl -fsSL -o "$2" "$1"
  elif command -v wget >/dev/null 2>&1; then wget -q -O "$2" "$1"
  else die "This needs curl or wget."
  fi
}

get_uv() {
  uv="$root/uv/$UV_VERSION/uv"
  if [ -x "$uv" ] && "$uv" --version >/dev/null 2>&1; then return 0; fi
  asset=$(uv_asset "$(uname -s)" "$(uname -m)") || die "iSharded has no installer for $(uname -s) on $(uname -m). See the guide for installing by hand."
  say "Fetching uv $UV_VERSION, which provides Python."
  fetch "${ISHARDED_UV_URL:-https://github.com/astral-sh/uv/releases/download/$UV_VERSION}/$asset" "$work/$asset"
  check_hash "$work/$asset" "$(uv_hash "$asset")"
  mkdir -p "$(dirname "$uv")"
  tar -xzf "$work/$asset" -C "$(dirname "$uv")" --strip-components=1
}

# A node still running would keep the old code, and on some systems the old files.
stop_old_node() {
  if [ -x "$folder/bin/isharded" ]; then "$folder/bin/isharded" stop || true; fi
}

make_venv() {
  python="$folder/bin/python"
  if [ -x "$python" ] && "$python" -c '' >/dev/null 2>&1; then return 0; fi
  # --seed keeps pip in the environment, for anyone adding packages by hand.
  "$uv" venv --quiet --seed --clear --managed-python --python "$PYTHON_VERSION" "$folder"
}

install_torch() {
  # PyTorch's default Linux build carries GPU libraries. Without a GPU the CPU build is far smaller.
  if [ "$(uname -s)" = Linux ] && ! command -v nvidia-smi >/dev/null 2>&1 && ! command -v rocminfo >/dev/null 2>&1; then
    say "No GPU tools found, so this takes PyTorch's smaller CPU build."
    "$uv" pip install --python "$python" torch --default-index https://download.pytorch.org/whl/cpu
  fi
}

fallback_note() {
  say "$1 So this installs from the source archive, without the native engine: the fast path is off."
  say "To get it, run this again once a release has a wheel for this machine, or build it from a checkout with: python scripts/build_engine.py"
}

# Sets package to what uv should install.
choose_package() {
  archive="$REPOSITORY/archive/refs/heads/main.tar.gz"
  if [ -n "${ISHARDED_SOURCE:-}" ]; then
    package=$ISHARDED_SOURCE
    return 0
  fi
  releases=${ISHARDED_RELEASE_URL:-$REPOSITORY/releases/latest/download}
  if ! fetch "$releases/SHA256SUMS" "$work/SHA256SUMS" 2>/dev/null; then
    fallback_note "No release is reachable."
    package=$archive
    return 0
  fi
  picked=$(pick_wheel "$work/SHA256SUMS" "$(wheel_tag "$(uname -s)" "$(uname -m)" "$(glibc_version)")")
  if [ -z "$picked" ]; then
    fallback_note "The latest release has no wheel for this machine."
    package=$archive
    return 0
  fi
  name=${picked#* }
  say "Fetching $name."
  fetch "$releases/$name" "$work/$name"
  check_hash "$work/$name" "${picked%% *}"
  package="$work/$name"
}

add_to_path() {
  mkdir -p "$HOME/.local/bin"
  ln -sf "$folder/bin/isharded" "$HOME/.local/bin/isharded"
  case ":$PATH:" in *":$HOME/.local/bin:"*) return 0 ;; esac
  # shellcheck disable=SC2016  # the profile expands this, each time it runs
  line='export PATH="$HOME/.local/bin:$PATH"  # added by the iSharded installer'
  for profile in "$HOME/.profile" "$HOME/.bashrc" "$HOME/.zshrc" "$HOME/.zprofile"; do
    case "$profile" in
      */.profile) ;;  # read by most login shells, so it is made if missing
      */.zprofile) [ "$(uname -s)" = Darwin ] || [ -f "$profile" ] || continue ;;  # zsh is the default on a Mac, and skips .profile
      *) [ -f "$profile" ] || continue ;;
    esac
    grep -qsF "$line" "$profile" || printf '\n%s\n' "$line" >>"$profile"
  done
  say "Added $HOME/.local/bin to your PATH. Open a new terminal for that to take effect."
}

check_system() {
  [ "$(uname -s)" = Linux ] || return 0
  # Debian keeps ldconfig out of a normal user's PATH.
  if ! { ldconfig -p 2>/dev/null || /sbin/ldconfig -p 2>/dev/null; } | grep -q libgomp; then
    say "GGUF models need one system library. On Debian or Ubuntu: sudo apt-get install libgomp1. On Fedora: sudo dnf install libgomp."
  fi
}

# A piped script holds stdin, so a question about joining a mesh has to go to the terminal.
isharded_asks() {
  if (: </dev/tty) 2>/dev/null; then "$folder/bin/isharded" "$@" </dev/tty
  else "$folder/bin/isharded" "$@" </dev/null
  fi
}

start_node() {
  if [ "$service" = yes ]; then
    # This first run settles the mesh, with a person there to answer. The service then takes over.
    isharded_asks up --background --no-browser
    "$folder/bin/isharded" stop
    "$folder/bin/isharded" service install || say "Could not set up the start at login. The node still runs for now."
  fi
  isharded_asks up --background
}

main() {
  start=yes
  service=no
  for option in "$@"; do
    case "$option" in
      --no-start) start=no ;;
      --service) service=yes ;;
      *) die "Unknown option $option. This takes --no-start or --service." ;;
    esac
  done
  [ "$start$service" != noyes ] || die "--no-start and --service do not go together: a service starts the node."

  folder=${ISHARDED_INSTALL_DIR:-$HOME/.isharded/app}
  root=$(dirname "$folder")
  mkdir -p "$root"
  work=$(mktemp -d)
  trap 'rm -rf "$work"' EXIT
  # uv's Python stays beside the install, so removing that folder removes all of it.
  export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$root/python}"
  # Antivirus that inspects TLS breaks uv's bundled certificates. For pip the same fix
  # is --use-feature=truststore. The system's store knows the antivirus's certificate.
  export UV_SYSTEM_CERTS="${UV_SYSTEM_CERTS:-1}"

  stop_old_node
  get_uv
  make_venv
  install_torch
  choose_package
  "$uv" pip install --python "$python" --reinstall-package isharded --refresh-package isharded "$package"
  add_to_path
  check_system
  if ! "$python" -I -c 'import sys; from isharded.engine import find_library; sys.exit(find_library() is None)'; then
    say "This install has no native engine, so the fast path is off. isharded doctor says how to get it."
  fi
  if [ "$start" = no ]; then
    say "Installed. Start with: isharded up"
    return 0
  fi
  start_node
}

# Tests load the functions above without installing anything.
if [ "${ISHARDED_INSTALL_TEST:-}" != 1 ]; then main "$@"; fi
