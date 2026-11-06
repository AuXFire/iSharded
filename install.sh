#!/bin/sh
# Install ShardMesh for this user, on Linux or macOS. Nothing here needs root.
set -e

folder="${SHARDMESH_INSTALL_DIR:-$HOME/.shardmesh/app}"
source="${SHARDMESH_SOURCE:-https://github.com/AuXFire/shardmesh/archive/refs/heads/main.tar.gz}"

python=$(command -v python3 || true)
if [ -z "$python" ] || ! "$python" -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
  echo "ShardMesh needs Python 3.10 or newer. Install it, then run this again." >&2
  exit 1
fi
if ! "$python" -m venv "$folder"; then
  echo "Python could not make an environment. On Debian or Ubuntu: sudo apt-get install python3-venv" >&2
  exit 1
fi
"$folder/bin/python" -m pip install --quiet --upgrade pip

# PyTorch's default Linux build carries GPU libraries. Without a GPU the CPU build is far smaller.
if [ "$(uname)" = "Linux" ] && ! command -v nvidia-smi >/dev/null 2>&1 && ! command -v rocminfo >/dev/null 2>&1; then
  "$folder/bin/python" -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
fi
"$folder/bin/python" -m pip install --quiet "$source"

mkdir -p "$HOME/.local/bin"
ln -sf "$folder/bin/shardmesh" "$HOME/.local/bin/shardmesh"
case ":$PATH:" in
  *":$HOME/.local/bin:"*) ;;
  *) echo "Add $HOME/.local/bin to your PATH, or run $folder/bin/shardmesh directly." ;;
esac
if [ "$(uname)" = "Linux" ] && ! ldconfig -p 2>/dev/null | grep -q libgomp; then
  echo "GGUF models need one system library: sudo apt-get install libgomp1"
fi
echo "Installed. Start with: shardmesh up"
