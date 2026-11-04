#!/bin/sh
# assemble_rootfs.sh ROOTFS
# Runs in the build stage of packaging/edge_agent/container/Containerfile. Copies into ROOTFS only what the agent runs
# on: the Python runtime (no pip, no ensurepip, no idle/tkinter), the agent's virtual environment, the shared libraries
# they load (found with ldd), the CA bundle, and passwd/group entries for the non-root user. No shell, no coreutils,
# no package manager: the final stage is FROM scratch with ROOTFS as its only layer.
# Also writes ROOTFS/usr/share/blurredbox-edge/os-packages.tsv: the Debian packages the copied files came from (SBOM).
set -eu
R="$1"
UID_=10001
# byte-compile in place first (unchecked-hash: valid on a read-only root, no timestamps; paths stay the runtime ones;
# a fixed hash seed so the same sources give the same bytes; one process (-j 1), in compileall's sorted order: with
# parallel workers which files a worker compiled before decides marshal's reference flags, so a .pyc could differ
# between two builds, as one sklearn module did on 2026-10-31)
PYTHONHASHSEED=0 /usr/local/bin/python3.12 -m compileall -f -q -j 1 --invalidation-mode unchecked-hash \
  /usr/local/lib/python3.12 /opt/blurredbox-edge >/dev/null || true   # what does not compile is compiled at import
mkdir -p "$R"
cd "$R"
mkdir -p usr/local/bin usr/local/lib usr/lib usr/lib64 opt etc tmp run var/tmp var/lib/blurredbox-edge \
  etc/blurredbox-edge usr/share/blurredbox-edge
ln -s usr/lib lib
ln -s usr/lib64 lib64
chmod 1777 tmp var/tmp

# Python runtime
cp -a /usr/local/bin/python3.12 /usr/local/bin/python3 /usr/local/bin/python usr/local/bin/
cp -a /usr/local/lib/libpython3.12.so* usr/local/lib/
cp -a /usr/local/lib/python3.12 usr/local/lib/
for d in site-packages/pip site-packages/pip-* ensurepip idlelib tkinter turtledemo lib2to3 pydoc_data \
         lib-dynload/_tkinter*; do
  rm -rf usr/local/lib/python3.12/$d
done
# the agent, without the libraries' own test suites (never imported by the agent; ~90 MB, pyarrow's included)
cp -a /opt/blurredbox-edge opt/
find opt/blurredbox-edge/venv/lib/python3.12/site-packages -type d -name tests -prune \
  \( -path '*/numpy/*' -o -path '*/scipy/*' -o -path '*/pandas/*' -o -path '*/sklearn/*' -o -path '*/pyarrow/*' \) \
  -exec rm -rf {} +

# shared libraries: everything python and the extension modules link against, plus libgcc_s (dlopened by glibc)
LIST=/tmp/libs.txt
: > "$LIST"
for f in usr/local/bin/python3.12 $(find usr/local/lib opt -name '*.so' -o -name '*.so.*'); do
  ldd "$f" 2>/dev/null | awk '/=> \//{print $3} /^[[:space:]]*\/[^ ]*ld-linux/{print $1}' >> "$LIST" || true
done
echo /lib/x86_64-linux-gnu/libgcc_s.so.1 >> "$LIST"
echo /lib64/ld-linux-x86-64.so.2 >> "$LIST"
sort -u "$LIST" | while read -r lib; do
  case "$lib" in /usr/local/*|/opt/*|"$R"/*) continue ;; esac   # already copied (runtime, venv, $ORIGIN rpaths)
  real="$(readlink -f "$lib")"
  dir="$(readlink -f "$(dirname "$lib")")"
  mkdir -p ".$(dirname "$real")" ".$dir"
  [ -e ".$real" ] || cp -p "$real" ".$real"
  [ "$dir/$(basename "$lib")" = "$real" ] || [ -e ".$dir/$(basename "$lib")" ] || ln -s "$real" ".$dir/$(basename "$lib")"
  echo "$real" >> /tmp/owned.txt
done

# TLS trust: OpenSSL's default paths (/usr/lib/ssl -> /etc/ssl) and the system CA bundle
cp -a /usr/lib/ssl usr/lib/
mkdir -p etc/ssl/certs
cp -p /etc/ssl/openssl.cnf etc/ssl/
cp -p /etc/ssl/certs/ca-certificates.crt etc/ssl/certs/
echo /etc/ssl/certs/ca-certificates.crt >> /tmp/owned.txt
[ ! -d /usr/share/zoneinfo ] || { cp -a /usr/share/zoneinfo usr/share/; echo /usr/share/zoneinfo/UTC >> /tmp/owned.txt; }
cp -p /etc/nsswitch.conf /etc/os-release etc/ 2>/dev/null || cp -p /etc/nsswitch.conf etc/
[ -e usr/lib/os-release ] || cp -p /usr/lib/os-release usr/lib/ 2>/dev/null || true

# users: root (owns everything) and the agent's own unprivileged user
printf 'root:x:0:0:root:/root:/nonexistent\nblurredbox-edge:x:%s:%s:BlurredBox edge agent:/var/lib/blurredbox-edge:/nonexistent\n' \
  "$UID_" "$UID_" > etc/passwd
printf 'root:x:0:\nblurredbox-edge:x:%s:\n' "$UID_" > etc/group
chown "$UID_:$UID_" var/lib/blurredbox-edge
chmod 0700 var/lib/blurredbox-edge

# the Debian packages the copied files belong to (for the SBOM)
sort -u /tmp/owned.txt | while read -r f; do dpkg -S "$f" 2>/dev/null || dpkg -S "${f#/usr}" 2>/dev/null || true; done \
  | cut -d: -f1 | tr ',' '\n' | sed 's/^ *//' | sort -u \
  | xargs -r dpkg-query -W -f='${Package}\t${Version}\t${Architecture}\n' > usr/share/blurredbox-edge/os-packages.tsv

echo "rootfs: $(du -sh . | cut -f1)"
