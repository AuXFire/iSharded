#!/bin/sh
# Runs the same install as install.sh, with the wheel this package carries.
cd "$(dirname "$0")" || exit 1
ISHARDED_SOURCE="$(pwd)/$(ls isharded-*.whl)"
export ISHARDED_SOURCE
sh ./install.sh || echo "iSharded did not finish installing. See the message above."
