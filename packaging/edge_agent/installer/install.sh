#!/bin/bash
# BlurredBox edge agent: one-command installer for Linux (systemd).
#
#   sudo ./install.sh                         install, then the guided setup (asks for what it needs)
#   sudo ./install.sh --non-interactive --console-url URL --tenant-id T --install-id I --model local=BBID \
#        --output-kind probability --source-jsonl /var/log/model.jsonl --pairing-code=CODE \
#        --client-cert client.pem --client-key client.key       (secrets: AGD_HMAC_SECRET and BLURREDBOX_API_KEY in the
#                                                                environment, or --hmac-secret-file / --api-key-file)
#   sudo ./install.sh --no-setup              install files only; run `blurredbox-edge setup` later
#   sudo ./install.sh --uninstall             remove the program and the service; config, state and the user stay
#
# Offline by default: pip runs with --no-index against the bundled wheelhouse. Re-running is safe: an existing
# config, secrets, spool and state are never touched; a new version is installed beside the old one and swapped in
# only if it works (and the old one is put back if `--check` fails after the swap).
#
# Upgrade and a quiet source: a source the agent has caught up with (nothing new after hours, after a stop) passes the
# check: it re-reads the newest rows, or reports CAUGHT_UP (exit 0) when the state holds windows built from it.
# Upgrade and an empty source: if `--check` with the new version finds the source readable but with no rows and no
# window built from it yet (NO_ROWS_YET, an install that never received a row), the installer re-checks every 10 s for up to --no-rows-wait seconds (default 300, one default
# window), with the service stopped; the agent resumes from its read offset, so no row is lost. If a row arrives the
# upgrade goes ahead; if none does, the old version is put back (exit 1). --no-rollback-on-no-rows keeps the new
# version at once instead (it starts and waits for data like a fresh install).
#
# Install options (everything else is handed to `blurredbox-edge setup`, see its --help):
#   --kit-dir DIR      the kit's edge/ directory (default: the directory above this script)
#   --wheel F  --requirements F  --wheelhouse DIR  --partner-pair F  --config-template F   override the kit paths
#   --python PATH      Python 3.12 interpreter (default: python3.12, then python3)
#   --with-postgres / --with-parquet / --with-kafka   also install the optional input libraries (from the wheelhouse)
#   --online           let pip take the third-party pins from the package index (the SDK still comes from the wheelhouse)
#   --no-rows-wait N   on an upgrade, seconds to wait for a first row before rolling back on NO_ROWS_YET (default 300)
#   --no-rollback-on-no-rows   on an upgrade, keep the new version when the source has no rows yet
#   --no-setup  --uninstall  -h
set -euo pipefail

APP=/opt/blurredbox-edge CONF=/etc/blurredbox-edge DATA=/var/lib/blurredbox-edge
SVC=blurredbox-edge USER_=blurredbox-edge UNIT=/etc/systemd/system/blurredbox-edge.service BIN=/usr/local/bin/blurredbox-edge
# kitrun6 R-01: RHEL/Rocky sudo (secure_path) does not search /usr/local/bin; the command is linked in /usr/bin too.
BIN2=/usr/bin/blurredbox-edge
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KIT="$(dirname "$HERE")"
PY="" WHEEL="" REQ="" WHEELHOUSE="" PAIR="" TEMPLATE="" EXTRAS="" ONLINE=0 SETUP=1 UNINSTALL=0 FWD=()
NO_ROWS_WAIT=300 ROLLBACK_NO_ROWS=1

die() { echo "install.sh: $*" >&2; exit "${2:-1}"; }
usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
  case "$1" in
    --kit-dir) KIT="$2"; shift 2;;
    --wheel) WHEEL="$2"; shift 2;;
    --requirements) REQ="$2"; shift 2;;
    --wheelhouse) WHEELHOUSE="$2"; shift 2;;
    --partner-pair) PAIR="$2"; shift 2;;
    --config-template) TEMPLATE="$2"; shift 2;;
    --python) PY="$2"; shift 2;;
    --with-postgres) EXTRAS="$EXTRAS postgres"; shift;;
    --with-parquet) EXTRAS="$EXTRAS parquet"; shift;;
    --with-kafka) EXTRAS="$EXTRAS kafka"; shift;;
    --online) ONLINE=1; shift;;
    --no-rows-wait) NO_ROWS_WAIT="${2:-}"; shift 2;;
    --no-rollback-on-no-rows) ROLLBACK_NO_ROWS=0; shift;;
    --no-setup) SETUP=0; shift;;
    --uninstall) UNINSTALL=1; shift;;
    -h|--help) usage; exit 0;;
    *) FWD+=("$1"); shift;;
  esac
done

case "$NO_ROWS_WAIT" in ''|*[!0-9]*) die "--no-rows-wait wants a whole number of seconds" 2;; esac
[ "$(id -u)" -eq 0 ] || die "run as root (sudo ./install.sh)" 2
command -v systemctl >/dev/null || die "systemd is required" 2
exec 9>/run/blurredbox-edge-install.lock
flock -n 9 || die "another install.sh is running" 2

# ---------------------------------------------------------------- uninstall
if [ "$UNINSTALL" = 1 ]; then
  systemctl disable --now "$SVC" 2>/dev/null || true       # a stop keeps the open window in the state
  rm -f "$UNIT" "$BIN"
  [ -L "$BIN2" ] && [ "$(readlink "$BIN2")" = "$APP/bin/blurredbox-edge" ] && rm -f "$BIN2"
  systemctl daemon-reload
  rm -rf "$APP"
  echo "Removed the program ($APP), the service, $BIN and $BIN2."
  echo "Left untouched, as yours: $CONF (config, secrets, certificates), $DATA (state, spool) and the user '$USER_'."
  echo "To delete them too (this loses queued windows): rm -rf $CONF $DATA && userdel $USER_"
  exit 0
fi

# ---------------------------------------------------------------- locate inputs
KIT_WHEELS=("$KIT"/blurredbox_edge_agent-*.whl)
[ -n "$WHEEL" ]      || WHEEL="${KIT_WHEELS[0]}"
[ -n "$REQ" ]        || REQ="$KIT/requirements-edge.txt"
[ -n "$WHEELHOUSE" ] || WHEELHOUSE="$KIT/wheelhouse"
[ -n "$PAIR" ]       || PAIR="$KIT/partner_pair.py"
[ -n "$TEMPLATE" ]   || TEMPLATE="$KIT/agent.json.template"
[ -f "$WHEEL" ] || die "edge wheel not found ($WHEEL); pass --wheel or --kit-dir" 2
[ -f "$REQ" ]   || die "requirements-edge.txt not found ($REQ)" 2
[ -f "$PAIR" ]  || die "partner_pair.py not found ($PAIR)" 2
if [ "$ONLINE" = 0 ] && [ ! -d "$WHEELHOUSE" ]; then
  die "no wheelhouse at $WHEELHOUSE; this kit cannot install offline. Pass --wheelhouse DIR, or --online if this host has a package index" 2
fi

if [ -z "$PY" ]; then
  for c in python3.12 python3; do command -v "$c" >/dev/null && { PY="$(command -v "$c")"; break; }; done
fi
[ -n "$PY" ] || die "Python 3.12 not found (install your distribution's python3.12 package, or pass --python)" 2
"$PY" -I -c 'import sys; sys.exit(0 if sys.version_info[:2]==(3,12) else 1)' \
  || die "$PY is Python $("$PY" -I -c 'import sys;print("%d.%d"%sys.version_info[:2])'); the agent is pinned to 3.12 (install python3.12 and pass --python)" 2
"$PY" -I -c 'import venv, ensurepip' 2>/dev/null || die "$PY lacks venv/ensurepip (install python3.12-venv / python3.12-pip)" 2

if [ -f "$WHEELHOUSE/SHA256SUMS" ]; then
  (cd "$WHEELHOUSE" && sha256sum --quiet -c SHA256SUMS) || die "wheelhouse checksum mismatch; the kit is damaged" 2
fi
if [ -f "$KIT/SHA256SUMS" ]; then
  (cd "$KIT" && grep -F "  $(basename "$WHEEL")" SHA256SUMS | sha256sum --quiet -c -) || die "edge wheel checksum mismatch" 2
fi

# ---------------------------------------------------------------- user and directories
getent group "$USER_" >/dev/null || groupadd --system "$USER_"
id "$USER_" >/dev/null 2>&1 || useradd --system --gid "$USER_" --no-create-home --home-dir "$DATA" \
    --shell /usr/sbin/nologin "$USER_"
install -d -m 0755 -o root -g root "$APP" "$APP/lib"
install -d -m 0750 -o root -g "$USER_" "$CONF"
install -d -m 0750 -o "$USER_" -g "$USER_" "$DATA"
# hunt H1: $DATA is the service user's, so root never follows a link in it; the service user makes its own state dir.
[ -L "$DATA/state" ] && die "$DATA/state is a symbolic link; refusing to follow it as root (remove it and re-run)" 2
runuser -u "$USER_" -- install -d -m 0700 "$DATA/state"

# ---------------------------------------------------------------- venv: build beside the old one, swap on success
WAS_ACTIVE=0; systemctl is-active --quiet "$SVC" 2>/dev/null && WAS_ACTIVE=1
OLD_VERSION="$(cat "$APP/VERSION" 2>/dev/null || true)"
NEW_VERSION="$(basename "$WHEEL" | sed -E 's/^blurredbox_edge_agent-([^-]+)-.*/\1/')"
rm -rf "$APP/venv.new" "$APP/venv.prev"
"$PY" -I -m venv "$APP/venv.new"
PIP=("$APP/venv.new/bin/python" -I -m pip install --disable-pip-version-check --no-input --quiet)
KIT_ONLY=(--no-index --find-links "$WHEELHOUSE")
if [ "$ONLINE" = 0 ]; then PIP+=("${KIT_ONLY[@]}")
else
  # hunt H4: the private SDK only ever from this kit, never from an index (its name is not ours on PyPI). pip then
  # keeps the installed SDK for the edge wheel's requirement and takes only the third-party pins from the index.
  "${PIP[@]}" "${KIT_ONLY[@]}" --no-deps blurredbox-agd \
    || { rm -rf "$APP/venv.new"; die "no blurredbox_agd wheel for this Python in $WHEELHOUSE; --online still installs the SDK from the kit (pass --wheelhouse DIR)" 2; }
fi
REQS=(-r "$REQ")
for x in $EXTRAS; do
  f="$(dirname "$REQ")/requirements-edge-$x.txt"; [ -f "$f" ] || f="$WHEELHOUSE/requirements-edge-$x.txt"
  if [ ! -f "$f" ] && [ "$x" = kafka ]; then
    die "requirements-edge-kafka.txt is not in this kit: it was built without the Kafka input (--without-kafka); ask BlurredBox for a kit with it" 2
  fi
  [ -f "$f" ] || die "requirements-edge-$x.txt not found next to $REQ" 2
  REQS+=(-r "$f")
done
"${PIP[@]}" "${REQS[@]}" blurredbox-agd "$WHEEL" || { rm -rf "$APP/venv.new"; die "pip install failed (offline wheelhouse incomplete for this Python/platform?); the existing install, if any, is unchanged"; }
"$APP/venv.new/bin/python" -I -m edge_agent --help >/dev/null || { rm -rf "$APP/venv.new"; die "installed agent does not start; existing install unchanged"; }

if [ -d "$APP/venv" ] && [ "$WAS_ACTIVE" = 1 ]; then systemctl stop "$SVC"; fi   # keeps the open window in the state
[ -d "$APP/venv" ] && mv "$APP/venv" "$APP/venv.prev"
mv "$APP/venv.new" "$APP/venv"
# The venv was built at venv.new and moved; only `python -m` is used, never the venv's own console scripts.
install -d -m 0755 "$APP/bin"
install -m 0755 "$HERE/blurredbox-edge" "$APP/bin/blurredbox-edge"
ln -sf "$APP/bin/blurredbox-edge" "$BIN"
if [ ! -e "$BIN2" ] || [ -L "$BIN2" ]; then ln -sfn "$APP/bin/blurredbox-edge" "$BIN2"
else echo "install.sh: $BIN2 exists and is not ours; left alone (use the full path $BIN)." >&2; fi
install -m 0644 "$PAIR" "$APP/lib/partner_pair.py"
echo "$NEW_VERSION" > "$APP/VERSION"
install -m 0644 "$HERE/blurredbox-edge.service" "$UNIT"
systemctl daemon-reload

# ---------------------------------------------------------------- config template (only when there is no config)
if [ ! -e "$CONF/config.json" ]; then
  if [ -f "$TEMPLATE" ] && "$PY" -I -c 'import json,sys; json.load(open(sys.argv[1]))' "$TEMPLATE" 2>/dev/null; then
    "$PY" -I - "$TEMPLATE" "$CONF/config.json" <<'PYEOF'
import json, sys
c = json.load(open(sys.argv[1]))
# the kit template names the old layout; this installer owns these
c["state_dir"] = "/var/lib/blurredbox-edge/state"
c.pop("client_cert", None); c.pop("ca_file", None)
json.dump(c, open(sys.argv[2], "w"), indent=2); open(sys.argv[2], "a").write("\n")
PYEOF
  else
    cat > "$CONF/config.json" <<'JEOF'
{
  "tenant_id": "<TENANT_ID>",
  "install_id": "<INSTALL_ID>",
  "models": {"<YOUR_MODEL_NAME>": "<BLURREDBOX_MODEL_ID>"},
  "secret_env": "AGD_HMAC_SECRET",
  "endpoints": ["<CONSOLE_URL>"],
  "state_dir": "/var/lib/blurredbox-edge/state",
  "cadence_seconds": 300,
  "output_kind": "<probability|signed>",
  "source": {"type": "jsonl", "path": "<PATH_TO_MODEL_LOG.jsonl>"}
}
JEOF
  fi
  chown "root:$USER_" "$CONF/config.json"; chmod 0640 "$CONF/config.json"     # the agent only reads it
  echo "Wrote $CONF/config.json from the template."
fi

# ---------------------------------------------------------------- upgrade verification / rollback
configured() { ! grep -q '<[A-Z_|a-z]*>' "$CONF/config.json" 2>/dev/null && [ -f "$CONF/agent.env" ]; }
rollback() {
  echo "install.sh: $1; putting back $OLD_VERSION." >&2
  rm -rf "$APP/venv.bad"; mv "$APP/venv" "$APP/venv.bad"; mv "$APP/venv.prev" "$APP/venv"
  echo "$OLD_VERSION" > "$APP/VERSION"; rm -rf "$APP/venv.bad"
  [ "$WAS_ACTIVE" = 1 ] && systemctl start "$SVC"
  exit 1
}
check_new() { CHECK_OUT="$("$BIN" check 2>&1)"; }
no_rows() { grep -q NO_ROWS_YET <<<"$CHECK_OUT"; }
if [ -d "$APP/venv.prev" ]; then
  # kitrun6 R-03: a re-run of the same version is not an upgrade: the old venv is the same code, so a failing check
  # cannot be fixed by putting it back. It skips the check and the rollback wait; `setup` below checks the config it
  # writes (and keeps the previous working config if the new one fails).
  if configured && [ "$OLD_VERSION" != "$NEW_VERSION" ]; then
    CHECK_OUT="" ; check_new && CHECK_RC=0 || CHECK_RC=$?
    if [ "$CHECK_RC" != 0 ] && no_rows; then
      if [ "$ROLLBACK_NO_ROWS" = 0 ]; then
        echo "install.sh: the source has no rows yet (NO_ROWS_YET); keeping $NEW_VERSION (--no-rollback-on-no-rows)." >&2
        CHECK_RC=0
      else
        echo "install.sh: the source has no rows yet (NO_ROWS_YET) with $NEW_VERSION; waiting up to ${NO_ROWS_WAIT} s for a first row before putting back $OLD_VERSION." >&2
        deadline=$((SECONDS + NO_ROWS_WAIT))
        while [ "$CHECK_RC" != 0 ] && no_rows && [ "$SECONDS" -lt "$deadline" ]; do
          left=$((deadline - SECONDS)); sleep $(( left < 10 ? left : 10 ))
          check_new && CHECK_RC=0 || CHECK_RC=$?
        done
        if [ "$CHECK_RC" != 0 ] && no_rows; then
          rollback "no rows in the source within ${NO_ROWS_WAIT} s of the upgrade to $NEW_VERSION (NO_ROWS_YET)"
        fi
      fi
    fi
    [ "$CHECK_RC" = 0 ] || rollback "--check fails with the new version ($NEW_VERSION)"
  fi
  if [ "$WAS_ACTIVE" = 1 ]; then systemctl start "$SVC"; fi
  rm -rf "$APP/venv.prev"
  if [ "$OLD_VERSION" = "$NEW_VERSION" ]; then echo "Reinstalled $NEW_VERSION; config, secrets, spool and state kept."
  else echo "Upgraded ${OLD_VERSION:-unknown} -> $NEW_VERSION; config, secrets, spool and state kept."; fi
else
  echo "Installed edge agent $NEW_VERSION in $APP."
fi

# ---------------------------------------------------------------- guided setup
echo "The command is $BIN2 (also $BIN); sudo finds it on RHEL/Rocky too."
if [ "$SETUP" = 1 ]; then
  if [ ${#FWD[@]} -gt 0 ] || [ -t 0 ]; then
    exec "$BIN" setup "${FWD[@]}"
  elif configured; then
    echo "Already configured; to change settings: sudo $BIN2 setup <flags>"
  else
    echo "Not configured yet. Run: sudo $BIN2 setup   (or re-run install.sh with the setup flags)"
  fi
else
  echo "Next: sudo $BIN2 setup"
fi
