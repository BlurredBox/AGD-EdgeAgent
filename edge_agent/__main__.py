"""`python -m edge_agent --config agent.json` -- read the configured input, run forever.

`python -m edge_agent replay-rejected --config agent.json [--apply]` lists (or resends) windows the
backend refused during an outage; see edge_agent/replay_rejected.py.

`python -m edge_agent incident mark|list|delete ...` and `python -m edge_agent erase [--yes] --config ...` are
the partner's signed calls to the incident and erasure routes; see edge_agent/partner_api.py.

agent.json: EdgeConfig fields (models, endpoints, state_dir, ...) plus
{"source": {...}} (a JSONL log, a PostgreSQL table, or a CSV/Parquet folder; see
edge_agent/sources.py). The secret may be given as "secret_env".

Optional, shadow-only (G52): "proxy_source" (an outcome-shaped source plus "proxy_name" and
"delay_seconds": early partial outcomes, counted apart from full labels; edge_agent/proxy.py) and
"upstream_source" (deploy / job / freshness status read from the partner side; edge_agent/upstream.py).
"""
from __future__ import annotations

import argparse
import json
import math
import os

from edge_agent.adapters import JsonlTailAdapter
from edge_agent.agent import EdgeAgent
from edge_agent.config import EdgeConfig
from edge_agent.mapping import ConfigError
from edge_agent.sources import build_proxy_sources, build_sources
from edge_agent.upstream import build_upstream_sources
from corpus.onboarding import collection_contract, RequiredSpec, required_from_bundle


CAUGHT_UP_MESSAGE = ("caught up: the source has no row left to read now, and this install has already built windows "
                     "from it. A quiet source is healthy; nothing to fix.")
# Verdicts that pass (exit 0): the installer's upgrade check, `blurredbox-edge check` and the image HEALTHCHECK.
PASSING_VERDICTS = ("SOURCE_COMPATIBLE", "CAUGHT_UP")
NO_ROWS_YET_MESSAGE = ("no rows yet: the source is readable but the model has not written a prediction to it. "
                       "Waiting for data; the config is not at fault. Re-run the check after the first row is written.")


def check_source(cfg: EdgeConfig, path: str | None = None, limit: int = 1000, *,
                 required: RequiredSpec | None = None, adapters: list | None = None,
                 outcome_adapters: list | None = None) -> dict:
    """Inspect a bounded sample of the configured input without writing offsets, starting
    services or sending data. `path` is the original JSONL shorthand; `adapters` takes any
    adapter that implements `sample()`. The report holds counts only."""
    if adapters is None:
        adapters = [JsonlTailAdapter(path, cfg.state_dir / "source.offset")]
    records, bad = [], 0
    try:
        for adapter in adapters:
            s = adapter.sample(limit)
            records += s.records
            bad += s.bad
    except OSError as exc:
        return _unreadable("source", exc)
    rows, valid = len(records) + bad, len(records)
    unknown_models = sum(r.model_id not in cfg.models for r in records)
    without_features = sum(not r.features for r in records)
    output_mismatch = sum(not math.isfinite(r.output) or (
        cfg.output_kind == "probability" and not 0 <= r.output <= 1) for r in records)
    contract = collection_contract(cfg.output_kind, required)
    outcome_check = None
    if outcome_adapters:
        try:
            ob = [a.sample(limit) for a in outcome_adapters]
        except OSError as exc:
            return _unreadable("outcome_source", exc)
        n_ok, n_bad = sum(len(x.records) for x in ob), sum(x.bad for x in ob)
        outcome_check = {"rows_checked": n_ok + n_bad, "valid_shape": n_ok, "invalid_shape": n_bad,
                         "unknown_model_rows": sum(r.model_id not in cfg.models
                                                   for x in ob for r in x.records)}
    outcome_ok = outcome_check is None or not (outcome_check["invalid_shape"]
                                               or outcome_check["unknown_model_rows"])
    compatible = (rows > 0 and not (bad or unknown_models or without_features or output_mismatch)
                  and outcome_ok
                  and any(path["output_compatible"] for path in contract["paths"].values()))
    # Rehearsal D-10: a readable source with no rows yet is day 1, not a broken config. Its own verdict
    # (still not ready: nothing was checked) so the installer can say "waiting for data".
    no_rows_yet = (rows == 0 and outcome_ok
                   and any(path["output_compatible"] for path in contract["paths"].values()))
    # Pre-tag review (edgefix2): the sources sample past their committed position and, once caught up, re-read the
    # newest rows instead; when even that finds none (log rotated, files cleaned up, topic retention, a purged table),
    # a state that already holds windows built from this source means "caught up", not day 1. Run 1 of a fresh
    # install has no such state and keeps NO_ROWS_YET.
    caught_up = no_rows_yet and _earlier_windows(cfg)
    verdict = ("SOURCE_COMPATIBLE" if compatible else "CAUGHT_UP" if caught_up
               else "NO_ROWS_YET" if no_rows_yet else "NOT_READY")
    report = {"contract": contract, "verdict": verdict,
              "source_check": {"rows_checked": rows, "valid_shape": valid, "invalid_shape": bad,
                               "unknown_model_rows": unknown_models, "rows_without_features": without_features,
                               "output_incompatible_rows": output_mismatch},
              "scoring_verified": False,
              "note": "local sample inspection only; no network beyond the configured source, "
                      "no raw data output or source offset changes"}
    if outcome_check is not None:
        report["outcome_source_check"] = outcome_check
    if caught_up:
        report["message"] = CAUGHT_UP_MESSAGE
    elif no_rows_yet:
        report["message"] = NO_ROWS_YET_MESSAGE
    return report


def _earlier_windows(cfg: EdgeConfig) -> bool:
    """True when the local state shows windows this install built from its source: a window queued for a
    configured model (emitted.json) or rows kept over a stop (open_windows.json). Read-only."""
    from edge_agent.agent import OPEN_WINDOWS_FILE
    state = cfg.state_dir
    for name in ("emitted.json", OPEN_WINDOWS_FILE):
        try:
            data = json.loads((state / name).read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        models = data.get("models") if name == OPEN_WINDOWS_FILE else data
        if isinstance(models, dict) and any(k in cfg.models and models[k] for k in models):
            return True
    return False


def _unreadable(where: str, exc: OSError) -> dict:
    """A configured file or folder that cannot be opened (missing, no permission) is a NOT_READY verdict with
    the reason, not a traceback (day1pack 2026-10-30: a missing outcome file crashed --check)."""
    what = "the outcome source" if where == "outcome_source" else "the prediction source"
    name = f" {exc.filename}" if getattr(exc, "filename", None) else ""
    reason = f"{where}: cannot read{name}: {exc.strerror or exc}"
    return {"verdict": "NOT_READY", f"{where}_error": reason, "scoring_verified": False,
            "message": f"{what} cannot be read ({reason}). Check the path in the config and that the service "
                       "user can read it" + ("; or remove the outcome_source block if you do not give outcomes."
                                             if where == "outcome_source" else ".")}


def load_edge_config(raw: dict, *, need_secret: bool, path: str | None = None) -> EdgeConfig:
    """EdgeConfig from agent.json fields (the source blocks already taken out, or ignored here).
    `secret_env` names the variable holding the HMAC secret; without `need_secret` it is not read.
    `path` is the config file: an inline `secret` in a group- or world-readable file is refused (F11)."""
    raw = {k: v for k, v in raw.items()
           if k not in ("source", "outcome_source", "proxy_source", "upstream_source")}
    if "secret_env" in raw:
        name = raw.pop("secret_env")
        if need_secret and name not in os.environ:
            raise ValueError(f"secret_env names the environment variable {name!r}, which is not set")
        raw["secret"] = os.environ[name] if need_secret else ""
    elif need_secret and path and raw.get("secret") and os.stat(path).st_mode & 0o077:
        raise ValueError(f"{path} holds the signing secret and is readable by other users; run: chmod 600 {path}")
    elif not need_secret:
        raw.setdefault("secret", "")
    if need_secret and not raw.get("secret"):
        raise ValueError("secret is empty: set secret_env to a variable holding the HMAC secret")
    for k in ("feature_bounds",):
        if k in raw:
            raw[k] = {a: tuple(b) for a, b in raw[k].items()}
    if raw.get("output_bounds"):
        raw["output_bounds"] = tuple(raw["output_bounds"])
    return EdgeConfig(**raw)


def main() -> None:
    import sys
    if sys.argv[1:2] == ["replay-rejected"]:     # operator command with its own flags (edge_agent/replay_rejected.py)
        from edge_agent.replay_rejected import main as replay_rejected
        raise SystemExit(replay_rejected(sys.argv[2:]))
    if sys.argv[1:2] in (["incident"], ["erase"]):   # signed partner calls (edge_agent/partner_api.py)
        from edge_agent.partner_api import main as partner_api
        raise SystemExit(partner_api(sys.argv[1:]))
    ap = argparse.ArgumentParser(prog="edge_agent")
    ap.add_argument("--config", required=True)
    ap.add_argument("--check", action="store_true", help="Inspect the local source and collection contract without sending data")
    ap.add_argument("--status", action="store_true",
                    help="Print the local state report (read-only; no network, no secret needed)")
    ap.add_argument("--bundle", help="Operator-trusted forecaster directory; requires --bundle-sha256")
    ap.add_argument("--bundle-sha256")
    ap.add_argument("--paired-bundle", help="Operator-trusted learned-path forecaster directory")
    ap.add_argument("--paired-bundle-sha256")
    args = ap.parse_args()
    raw = json.load(open(args.config))
    if args.status:
        from edge_agent.status import read_status
        report = read_status(raw["state_dir"])
        sys.stdout.write(json.dumps(report, sort_keys=True, default=str) + "\n")
        raise SystemExit(0 if report["state_dir_exists"] else 2)
    src = raw.pop("source")
    outcome_source = raw.pop("outcome_source", None)
    proxy_source = raw.pop("proxy_source", None)            # G52: early proxy labels, kept apart from outcomes
    upstream_source = raw.pop("upstream_source", None)      # G52: deploy / job / freshness status, read-only
    required = None
    if args.paired_bundle and not args.bundle:
        ap.error("--paired-bundle requires --bundle")
    if args.bundle:
        live = _trusted_bundle(args.bundle, args.bundle_sha256)
        paired = _trusted_bundle(args.paired_bundle, args.paired_bundle_sha256) if args.paired_bundle else None
        required = required_from_bundle(live, paired)
    try:
        cfg = load_edge_config(raw, need_secret=not args.check, path=args.config)
    except KeyError as exc:
        ap.error(f"environment variable {exc.args[0]} named by secret_env is not set")
    except (ValueError, TypeError) as exc:
        ap.error(f"invalid config: {exc}")
    if sys.platform == "win32" and not args.check:
        sys.stderr.write("warning: Windows is not a supported platform for the edge agent; state-file permissions "
                         "are not restricted and a service stop may not flush the open window\n")
    if not args.check:
        from edge_agent.statelock import StateLockError, acquire
        try:
            _state_lock = acquire(cfg.state_dir)          # held until the process exits
        except StateLockError as exc:
            ap.error(str(exc))
    try:
        taken: set[str] = set()
        adapters = build_sources(src, cfg.state_dir, "prediction", taken=taken)
        outcomes = (build_sources(outcome_source, cfg.state_dir, "outcome", taken=taken)
                    if outcome_source else [])
        proxies = build_proxy_sources(proxy_source, cfg.state_dir, taken=taken) if proxy_source else []
        upstream = build_upstream_sources(upstream_source, cfg.state_dir) if upstream_source else []
        if args.check:
            report = check_source(cfg, required=required, adapters=adapters, outcome_adapters=outcomes)
    except ConfigError as exc:
        # Config problems are reported by key name; --check prints them as a verdict, a live run exits.
        if not args.check:
            ap.error(str(exc))
        sys.stdout.write(json.dumps({"verdict": "NOT_READY", "config_error": str(exc),
                                     "scoring_verified": False}, sort_keys=True) + "\n")
        raise SystemExit(2) from None
    if args.check:
        sys.stdout.write(json.dumps(report, sort_keys=True) + "\n")
        if report["verdict"] not in PASSING_VERDICTS:
            raise SystemExit(2)
        return
    from edge_agent.featureids import FeatureIdSaltError
    try:
        agent = EdgeAgent(cfg, adapters, required_spec=required, outcome_sources=outcomes,
                          proxy_sources=proxies, upstream_sources=upstream)
    except FeatureIdSaltError as exc:
        ap.error(str(exc))
    agent.run_forever()


def _trusted_bundle(directory: str, digest: str | None):
    """Deserialize only operator-pinned artifact bytes, never a customer-selected artifact."""
    import hashlib
    import hmac
    import pickle
    from pathlib import Path
    from corpus.forecaster import ForecasterBundle

    if digest is None or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("a lowercase SHA-256 digest is required for each operator-trusted bundle")
    raw = (Path(directory) / "forecaster.pkl").read_bytes()
    if not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), digest):
        raise ValueError("operator-trusted bundle digest mismatch")
    bundle = pickle.loads(raw)
    if not isinstance(bundle, ForecasterBundle):
        raise ValueError("operator-trusted artifact is not a ForecasterBundle")
    return bundle


if __name__ == "__main__":
    main()
