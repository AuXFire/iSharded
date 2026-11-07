#!/bin/sh
# Builds iSharded.pkg with pkgbuild, which only macOS has.
#   sh packaging/macos/build_pkg.sh dist/isharded-0.3.0-py3-none-macosx_11_0_arm64.whl 0.3.0 dist
# It is unsigned until the project has a certificate. See "Unsigned packages" in docs/GUIDE.md.
set -eu

wheel=$1
version=$2
out=$3
here=$(cd "$(dirname "$0")" && pwd)
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT

app="$stage/root/iSharded.app/Contents"
mkdir -p "$app/MacOS" "$app/Resources" "$stage/scripts"
sed "s/__VERSION__/$version/g" "$here/Info.plist" >"$app/Info.plist"
cp "$here/iSharded" "$app/MacOS/iSharded"
cp "$here/first-run.command" "$here/../../install.sh" "$wheel" "$app/Resources/"
cp "$here/postinstall" "$stage/scripts/postinstall"
chmod 755 "$app/MacOS/iSharded" "$app/Resources/first-run.command" "$stage/scripts/postinstall"

mkdir -p "$out"
pkgbuild --root "$stage/root" --scripts "$stage/scripts" --identifier ai.isharded.app --version "$version" --install-location /Applications "$out/iSharded.pkg"
