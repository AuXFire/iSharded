#!/bin/bash
# Run what .github/workflows/tests.yml runs, from a clean clone, without GitHub.
#
#   bash scripts/ci_local.sh https://github.com/you/isharded.git
#
# A clean clone matters: it fails if a test needs a file that only your machine has.
set -euo pipefail
source_repo="${1:?usage: ci_local.sh <repository path or URL>}"
python_bin="${PYTHON:-python3}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

git clone --quiet "$source_repo" "$work/repo"
"$python_bin" -m venv "$work/venv" 2>/dev/null || "$python_bin" -m virtualenv --quiet "$work/venv"
pip="$work/venv/bin/python -m pip"
$pip install --quiet --upgrade pip
$pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
$pip install --quiet -e "$work/repo[test,pool]"
cd "$work/repo"
"$work/venv/bin/python" --version
"$work/venv/bin/python" -m pytest tests -q --no-header --tb=short
