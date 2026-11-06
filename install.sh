#!/bin/sh
# Install iSharded for this user, on Linux or macOS. Nothing here needs root.
set -e

folder="${ISHARDED_INSTALL_DIR:-$HOME/.isharded/app}"
source="${ISHARDED_SOURCE:-https://github.com/AuXFire/iSharded/archive/refs/heads/main.tar.gz}"

python=$(command -v python3 || true)
if [ -z "$python" ] || ! "$python" -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
  echo "iSharded needs Python 3.10 or newer. Install it, then run this again." >&2
  exit 1
fi
if ! "$python" -m venv "$folder" >/dev/null 2>&1; then
  # Debian and Ubuntu leave venv out of Python. The virtualenv zipapp needs nothing installed.
  rm -rf "$folder"
  mkdir -p "$(dirname "$folder")"
  tool="$(dirname "$folder")/virtualenv.pyz"
  curl -fsSL -o "$tool" https://bootstrap.pypa.io/virtualenv.pyz
  "$python" "$tool" --quiet "$folder"
  rm -f "$tool"
fi
"$folder/bin/python" -m pip install --quiet --upgrade pip

# PyTorch's default Linux build carries GPU libraries. Without a GPU the CPU build is far smaller.
if [ "$(uname)" = "Linux" ] && ! command -v nvidia-smi >/dev/null 2>&1 && ! command -v rocminfo >/dev/null 2>&1; then
  "$folder/bin/python" -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
fi
"$folder/bin/python" -m pip install --quiet "$source"

mkdir -p "$HOME/.local/bin"
ln -sf "$folder/bin/isharded" "$HOME/.local/bin/isharded"
case ":$PATH:" in
  *":$HOME/.local/bin:"*) ;;
  *) echo "Add $HOME/.local/bin to your PATH, or run $folder/bin/isharded directly." ;;
esac
if [ "$(uname)" = "Linux" ] && ! ldconfig -p 2>/dev/null | grep -q libgomp; then
  echo "GGUF models need one system library: sudo apt-get install libgomp1"
fi
echo "Installed. Start with: isharded up"
