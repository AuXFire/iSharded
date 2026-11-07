#!/bin/sh
# Builds a .deb with dpkg-deb.
#   sh packaging/linux/build_deb.sh dist/isharded-0.3.0-py3-none-manylinux_2_28_x86_64.whl amd64 0.3.0 dist
set -eu

wheel=$1
architecture=$2
version=$3
out=$4
here=$(cd "$(dirname "$0")" && pwd)
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
chmod 755 "$stage"  # mktemp's 700 would go into the package as the mode of its ./ entry

mkdir -p "$stage/DEBIAN" "$stage/usr/bin" "$stage/usr/lib/isharded" "$stage/usr/share/applications"
sed -e "s/__VERSION__/$version/" -e "s/__ARCHITECTURE__/$architecture/" "$here/control" >"$stage/DEBIAN/control"
install -m 755 "$here/postinst" "$stage/DEBIAN/postinst"
install -m 755 "$here/isharded" "$stage/usr/bin/isharded"
install -m 644 "$here/../../install.sh" "$wheel" "$stage/usr/lib/isharded/"
install -m 644 "$here/isharded.desktop" "$stage/usr/share/applications/isharded.desktop"

mkdir -p "$out"
dpkg-deb --root-owner-group --build "$stage" "$out/isharded_${version}_$architecture.deb"
