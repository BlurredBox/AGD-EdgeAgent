"""`python -m edge_agent --config agent.json` -- tail a JSONL log, run forever.

agent.json: EdgeConfig fields (models, endpoints, state_dir, ...) plus
{"source": {"path": "...jsonl"}}. The secret may be given as "secret_env".
"""
from __future__ import annotations

import argparse
import json
import math
import os

from edge_agent.adapters import JsonlTailAdapter, JsonlOutcomeAdapter
from edge_agent.agent import EdgeAgent
from edge_agent.config import EdgeConfig
from edge_agent.records import Record
from corpus.onboarding import collection_contract, RequiredSpec, required_from_bundle


def check_source(cfg: EdgeConfig, path: str, limit: int = 1000, *, required: RequiredSpec | None = None) -> dict:
    """Inspect a local JSONL sample without writing offsets, starting services or sending data."""
    rows = valid = bad = unknown_models = without_features = output_mismatch = 0
    with open(path, encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            rows += 1
            try:
                d = json.loads(line)
                rec = Record.make(d["ts"], d["model_id"], d["instrument"], d["features"], d["output"],
                                  prediction_id=d.get("prediction_id"), outcome=d.get("outcome"),
                                  outcome_observed_at=d.get("outcome_observed_at"))
                valid += 1
                unknown_models += rec.model_id not in cfg.models
                without_features += not bool(rec.features)
                output_mismatch += not math.isfinite(rec.output) or (
                    cfg.output_kind == "probability" and not 0 <= rec.output <= 1)
            except (ValueError, KeyError, TypeError, AttributeError, OverflowError):
                bad += 1
            if rows >= limit:
                break
    contract = collection_contract(cfg.output_kind, required)
    compatible = (rows > 0 and not (bad or unknown_models or without_features or output_mismatch)
                  and any(path["output_compatible"] for path in contract["paths"].values()))
    return {"contract": contract, "verdict": "SOURCE_COMPATIBLE" if compatible else "NOT_READY",
            "source_check": {"rows_checked": rows, "valid_shape": valid, "invalid_shape": bad,
                             "unknown_model_rows": unknown_models, "rows_without_features": without_features,
                             "output_incompatible_rows": output_mismatch},
            "scoring_verified": False,
            "note": "local sample inspection only; no network, raw data output or source offset changes"}


def main() -> None:
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
        import sys
        from edge_agent.status import read_status
        report = read_status(raw["state_dir"])
        sys.stdout.write(json.dumps(report, sort_keys=True, default=str) + "\n")
        raise SystemExit(0 if report["state_dir_exists"] else 2)
    src = raw.pop("source")
    outcome_source = raw.pop("outcome_source", None)
    required = None
    if args.paired_bundle and not args.bundle:
        ap.error("--paired-bundle requires --bundle")
    if args.bundle:
        live = _trusted_bundle(args.bundle, args.bundle_sha256)
        paired = _trusted_bundle(args.paired_bundle, args.paired_bundle_sha256) if args.paired_bundle else None
        required = required_from_bundle(live, paired)
    if "secret_env" in raw:
        name = raw.pop("secret_env")
        raw["secret"] = "" if args.check else os.environ[name]
    elif args.check:
        raw.setdefault("secret", "")
    for k in ("feature_bounds",):
        if k in raw:
            raw[k] = {a: tuple(b) for a, b in raw[k].items()}
    if raw.get("output_bounds"):
        raw["output_bounds"] = tuple(raw["output_bounds"])
    cfg = EdgeConfig(**raw)
    if args.check:
        import sys
        report = check_source(cfg, src["path"], required=required)
        sys.stdout.write(json.dumps(report, sort_keys=True) + "\n")
        if report["verdict"] != "SOURCE_COMPATIBLE":
            raise SystemExit(2)
        return
    adapter = JsonlTailAdapter(src["path"], cfg.state_dir / "source.offset")
    outcomes = ([JsonlOutcomeAdapter(outcome_source["path"], cfg.state_dir / "outcome.offset")]
                if outcome_source else [])
    EdgeAgent(cfg, [adapter], required_spec=required, outcome_sources=outcomes).run_forever()


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
