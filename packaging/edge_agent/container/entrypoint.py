"""Entry point of the edge agent container image (packaging/edge_agent/container/Containerfile).

    (no arguments) | run         run the agent, as the systemd unit does (`python -m edge_agent --config ...`)
    --check | check              the source check; exit 0 for SOURCE_COMPATIBLE or CAUGHT_UP (the image's HEALTHCHECK).
                                 Loads agent.env when present (database / broker credentials named by *_env)
    --status | status            the local counts-only state report
    replay-rejected [flags]      `edge_agent replay-rejected` (dry run unless --apply)
    pair --model ID:CODE ...     pair this install with each model once (the kit's partner_pair.py, as `blurredbox-edge
                                 setup` runs it): base URL, install id and ca_file from config.json, secrets from
                                 agent.env. With the state volume mounted, the window cadence the server holds for the
                                 models is saved there and the run uses it in place of config.json's cadence_seconds
    version                      the installed agent version

Same files, same paths, as the Linux install (packaging/edge_agent/installer/):
  /etc/blurredbox-edge/config.json    the agent configuration (BBE_CONFIG to change the path), mounted read-only
  /etc/blurredbox-edge/agent.env      the two secrets, `NAME=VALUE` lines, mode 0600 or 0400 (or a podman/docker secret
                                      at /run/secrets/agent.env; BBE_ENV_FILE to change the path)
  /etc/blurredbox-edge/certs/         client.pem + client.key named by `client_cert` in config.json, or
  BBE_P12 + BBE_P12_PASS_FILE         paths of a .p12 and its passphrase file: split in memory onto the /tmp tmpfs
  /var/lib/blurredbox-edge            the only writable volume: state, spool, spool/rejected/, read offsets

Secrets are read from files only. A secret given in the container environment (`-e AGD_HMAC_SECRET=...`) is refused,
because it would be visible in `podman inspect` / `docker inspect`; the BBE_* variables hold paths, never values.
Nothing secret is printed. Logs go to stdout and stderr, as they go to the journal under systemd.

Exit codes: those of the agent (`--check`: 0 SOURCE_COMPATIBLE / CAUGHT_UP, 2 NO_ROWS_YET / NOT_READY), and 2 for a container
setup problem (missing config, a secret in the environment, an unreadable or group/world-readable secret file, a state
directory that is not a writable volume); the message names it.
"""
from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
from pathlib import Path

CONFIG = Path(os.environ.get("BBE_CONFIG", "/etc/blurredbox-edge/config.json"))
ENV_FILES = (Path("/run/secrets/agent.env"), Path("/etc/blurredbox-edge/agent.env"))
STATE_ROOT = Path("/var/lib/blurredbox-edge")
PAIR_TOOL = Path("/opt/blurredbox-edge/lib/partner_pair.py")
# kitrun6 C-01: the server's window cadence per model, written by `pair` into the state volume (config.json is mounted
# read-only), applied by the run as `blurredbox-edge setup` applies it on a Linux install.
SERVER_CADENCE = "server-edge-configuration.json"
# Variables that hold a secret value. Never accepted from the container environment.
SECRET_VARS = ("AGD_HMAC_SECRET", "BLURREDBOX_API_KEY", "AGD_API_KEY")


class SetupError(Exception):
    """A container setup problem; the message names it and never holds a secret value."""


def parse_env_file(text: str) -> dict[str, str]:
    """`NAME=VALUE` lines (the agent.env the Linux installer writes): comments, blanks, `export ` and one pair of
    surrounding quotes are allowed. Anything else is refused, naming the line number only."""
    out: dict[str, str] = {}
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        name, sep, value = line.partition("=")
        name = name.strip()
        if not sep or not name or not (name[0].isalpha() or name[0] == "_") \
                or not all(c.isalnum() or c == "_" for c in name):
            raise SetupError(f"agent.env line {n} is not NAME=VALUE")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        out[name] = value
    return out


def _private_file(path: Path, what: str) -> str:
    try:
        st = path.stat()
        text = path.read_text()
    except OSError as exc:
        raise SetupError(f"cannot read {what} {path}: {exc.strerror or exc}") from None
    if stat.S_IMODE(st.st_mode) & 0o077:
        raise SetupError(f"{what} {path} is readable by group or others (mode {stat.S_IMODE(st.st_mode):04o}); "
                         "make it 0600 or 0400 (podman: --secret ...,mode=0400)")
    return text


def env_file_path(environ: dict[str, str]) -> Path | None:
    if environ.get("BBE_ENV_FILE"):
        return Path(environ["BBE_ENV_FILE"])
    return next((p for p in ENV_FILES if p.exists()), None)


def load_secrets(environ: dict[str, str], secret_var: str) -> dict[str, str]:
    """The values of agent.env, to be put into this process's environment only (never the container's)."""
    path = env_file_path(environ)
    if path is None:
        raise SetupError("no agent.env: mount it at /etc/blurredbox-edge/agent.env (mode 0600) or as the secret "
                         "/run/secrets/agent.env")
    values = parse_env_file(_private_file(path, "agent.env"))
    for need in (secret_var, "BLURREDBOX_API_KEY"):
        if not values.get(need):
            raise SetupError(f"agent.env {path} has no {need}")
    return values


def load_optional_env(environ: dict[str, str]) -> dict[str, str]:
    """agent.env for `check`: its values when the file is there (same parsing and 0600/0400 rule as the run), nothing
    when it is not. The HMAC secret and the API key are not required: the check sends nothing."""
    path = env_file_path(environ)
    if path is None:
        return {}
    if not path.exists() and environ.get("BBE_ENV_FILE"):
        raise SetupError(f"BBE_ENV_FILE names {path}, which does not exist")
    return parse_env_file(_private_file(path, "agent.env"))


def refuse_secret_environment(environ: dict[str, str], secret_var: str) -> None:
    given = [v for v in dict.fromkeys((*SECRET_VARS, secret_var)) if v in environ]
    if given:
        raise SetupError("secret values are read from files only; remove " + ", ".join(given)
                         + " from the container environment (-e / environment:) and put them in agent.env")


def split_p12(p12: Path, pass_file: Path, out_dir: Path) -> tuple[str, str]:
    """The .p12 split into client.pem (certificate) and client.key (0600) in a private directory, as
    `blurredbox-edge setup --p12` does with openssl, here with the `cryptography` package the agent already ships."""
    from cryptography.hazmat.primitives.serialization import (
        Encoding, NoEncryption, PrivateFormat, pkcs12)
    passphrase = _private_file(pass_file, "the .p12 passphrase file").rstrip("\r\n").encode()
    try:
        key, cert, _ca = pkcs12.load_key_and_certificates(p12.read_bytes(), passphrase or None)
    except OSError as exc:
        raise SetupError(f"cannot read the .p12 {p12}: {exc.strerror or exc}") from None
    except ValueError:
        raise SetupError(f"cannot open the .p12 {p12}: wrong passphrase or not a PKCS#12 file") from None
    if key is None or cert is None:
        raise SetupError(f"the .p12 {p12} holds no client certificate and key")
    out_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    pem, keyf = out_dir / "client.pem", out_dir / "client.key"
    pem.write_bytes(cert.public_bytes(Encoding.PEM))
    fd = os.open(keyf, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()))
    return str(pem), str(keyf)


def load_config(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except OSError as exc:
        raise SetupError(f"cannot read the config {path}: {exc.strerror or exc}; mount it read-only at "
                         "/etc/blurredbox-edge/config.json (or set BBE_CONFIG)") from None
    except ValueError as exc:
        raise SetupError(f"the config {path} is not valid JSON: {exc}") from None


def check_state_dir(cfg: dict) -> None:
    """The state directory must be on the writable volume: on a read-only root it is the only place the agent can
    write its spool, offsets and baselines (`state_dir` is /var/lib/blurredbox-edge/state, as in the Linux install)."""
    state = Path(cfg.get("state_dir") or "")
    if not str(state) or not (state == STATE_ROOT or STATE_ROOT in state.parents):
        raise SetupError(f"state_dir {str(state)!r} is not under {STATE_ROOT}, the image's only writable volume; "
                         f"set \"state_dir\": \"{STATE_ROOT}/state\" in config.json")
    try:
        state.mkdir(mode=0o700, parents=True, exist_ok=True)
        probe = tempfile.NamedTemporaryFile(dir=state, prefix=".write-probe-", delete=True)
        probe.close()
    except OSError as exc:
        raise SetupError(f"state_dir {state} is not writable ({exc.strerror or exc}): mount a volume at "
                         f"{STATE_ROOT} owned by the container user (uid {os.getuid()})") from None


def server_cadence_path(cfg: dict) -> Path | None:
    """Where `pair` saves the server's cadence: the state directory, when it is on a writable mounted volume."""
    if not STATE_ROOT.is_dir() or not os.access(STATE_ROOT, os.W_OK):
        return None
    try:
        check_state_dir(cfg)
    except SetupError:
        return None
    return Path(cfg["state_dir"]) / SERVER_CADENCE


def server_cadence_seconds(cfg: dict, held: dict) -> int | None:
    """The one cadence (seconds) the server holds for this config's models; None if it holds none. Several different
    cadences are refused, as `blurredbox-edge setup` refuses them: one agent sends at one cadence."""
    mins = {bb: (held.get(bb) or {}).get("cadence_minutes") for bb in (cfg.get("models") or {}).values()}
    known = {bb: float(c) for bb, c in mins.items()
             if isinstance(c, (int, float)) and not isinstance(c, bool) and c > 0}
    secs = {max(1, int(round(c * 60))) for c in known.values()}
    if len(secs) > 1:
        raise SetupError("the models have different window cadences on the server ("
                         + ", ".join(f"{bb}: {c:g} min" for bb, c in sorted(known.items()))
                         + "); one agent sends at one cadence: run one container per cadence")
    return secs.pop() if secs else None


def apply_server_cadence(cfg: dict) -> bool:
    """Put the server's cadence (saved by `pair`) into cfg; True when it changed cfg."""
    path = Path(cfg.get("state_dir") or "") / SERVER_CADENCE
    try:
        held = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    secs = server_cadence_seconds(cfg, held if isinstance(held, dict) else {})
    if secs is None or cfg.get("cadence_seconds") == secs:
        return False
    print(f"blurredbox-edge container: cadence: one window every {secs} s (the model's cadence on the server, saved "
          f"at pairing; config.json says {cfg.get('cadence_seconds', 300)})", file=sys.stderr, flush=True)
    cfg["cadence_seconds"] = secs
    return True


def report_pair_cadence(agent_argv: list[str]) -> None:
    """After a successful `pair`: say which cadence the run will use, or how to set it."""
    try:
        cfg = load_config(CONFIG)
        out = Path(agent_argv[agent_argv.index("--edge-config-out") + 1]) if "--edge-config-out" in agent_argv else None
        held = json.loads(out.read_text()) if out is not None and out.exists() else None
    except (SetupError, OSError, ValueError, IndexError):
        return
    if held is None:
        print("blurredbox-edge container: the state volume was not mounted on `pair`, so the server's window cadence was "
              "not saved; mount it (-v <volume>:/var/lib/blurredbox-edge) and pair again, or set cadence_seconds in "
              "config.json to the model's cadence in the console", file=sys.stderr)
        return
    try:
        secs = server_cadence_seconds(cfg, held)
    except SetupError as exc:
        print(f"blurredbox-edge container: {exc}", file=sys.stderr)
        return
    if secs is not None:
        # the image declares the volume, so a `pair` without -v writes to an anonymous volume that --rm discards
        print(f"cadence: one window every {secs} s (the model's cadence on the server). It was saved in the state "
              "volume: the agent uses it when it runs with the same volume (-v <volume>:/var/lib/blurredbox-edge); "
              f"if pair ran without that volume, set \"cadence_seconds\": {secs} in config.json")


def prepare(argv: list[str], environ: dict[str, str]) -> tuple[list[str], dict[str, str]]:
    """The edge_agent argv to run and the variables to add to this process's environment."""
    cmd, rest = (argv[0], argv[1:]) if argv else ("run", [])
    cmd = {"--check": "check", "--status": "status", "--help": "help", "-h": "help"}.get(cmd, cmd)
    if cmd == "help":
        return ["help"], {}
    if cmd == "version":
        return ["version"], {}
    if cmd not in ("run", "check", "status", "replay-rejected", "pair"):
        raise SetupError(f"unknown command {cmd!r}: run, --check, --status, replay-rejected, pair, version")
    cfg = load_config(CONFIG)
    secret_var = cfg.get("secret_env") or "AGD_HMAC_SECRET"
    refuse_secret_environment(environ, secret_var)
    if cmd == "status":
        return ["--config", str(CONFIG), "--status", *rest], {}
    if cmd == "check":            # reads a sample of the source only: no HMAC secret or API key needed, nothing written
        # but agent.env is loaded when present: a Postgres/Kafka source names its password / DSN by `password_env` /
        # `dsn_env`, and those live there, as the Linux CLI passes them (`blurredbox-edge check`, read_envfile).
        return ["--config", str(CONFIG), "--check", *rest], load_optional_env(environ)
    if cmd == "pair":             # three signed calls; writes only the server's cadence (state volume, when mounted)
        add = load_secrets(environ, secret_var)
        if not cfg.get("endpoints") or not cfg.get("install_id"):
            raise SetupError("config.json needs endpoints and install_id before pairing")
        from edge_agent.config import require_https_or_loopback
        try:
            require_https_or_loopback(cfg["endpoints"][0])
        except ValueError as exc:
            raise SetupError(str(exc)) from None
        extra = []
        if cfg.get("ca_file") and "--ca-file" not in rest:              # kitrun6 C-02
            extra += ["--ca-file", str(cfg["ca_file"])]
        if "--edge-config-out" not in rest:
            out = server_cadence_path(cfg)
            if out is not None:
                extra += ["--edge-config-out", str(out)]
        return ["pair", str(PAIR_TOOL), "--base-url", cfg["endpoints"][0], "--install-id", cfg["install_id"],
                *rest, *extra], add
    check_state_dir(cfg)
    add = load_secrets(environ, secret_var)
    config_path = CONFIG
    changed = apply_server_cadence(cfg)
    if environ.get("BBE_P12") or changed:
        if environ.get("BBE_P12") and not environ.get("BBE_P12_PASS_FILE"):
            raise SetupError("BBE_P12 needs BBE_P12_PASS_FILE (the path of the passphrase file, never the passphrase)")
        work = Path(tempfile.mkdtemp(prefix="bbe-", dir=environ.get("BBE_TMP", "/tmp")))
        if environ.get("BBE_P12"):
            cfg["client_cert"] = list(split_p12(Path(environ["BBE_P12"]), Path(environ["BBE_P12_PASS_FILE"]),
                                                work / "certs"))
        config_path = work / "config.json"
        fd = os.open(config_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as fh:
            json.dump(cfg, fh, indent=2)
    if cmd == "replay-rejected":
        return ["replay-rejected", "--config", str(config_path), *rest], add
    return ["--config", str(config_path), *rest], add


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    try:
        agent_argv, add = prepare(argv, dict(os.environ))
    except SetupError as exc:
        print(f"blurredbox-edge container: {exc}", file=sys.stderr)
        return 2
    if agent_argv == ["help"]:
        print(__doc__)
        return 0
    if agent_argv == ["version"]:
        from importlib.metadata import version
        print(version("blurredbox-edge-agent"))
        return 0
    os.environ.update(add)                       # this process only: not in the container's config or /proc/1/environ
    if agent_argv[0] == "pair":
        import runpy
        sys.argv = agent_argv[1:]
        try:
            runpy.run_path(agent_argv[1], run_name="__main__")
            code = 0
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        if code == 0:
            report_pair_cadence(agent_argv)
        return code
    from edge_agent.__main__ import main as agent_main
    sys.argv = ["edge_agent", *agent_argv]
    running = agent_argv[:1] == ["--config"] and not {"--check", "--status"} & set(agent_argv)
    if running:                                  # the start/stop lines systemd writes to the journal for the unit
        from importlib.metadata import version
        print(f"blurredbox-edge container: starting edge agent {version('blurredbox-edge-agent')} "
              f"(config {CONFIG}, state on {STATE_ROOT})", flush=True)
    code = 0
    try:
        agent_main()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    if running:
        print(f"blurredbox-edge container: edge agent stopped (exit {code})", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
