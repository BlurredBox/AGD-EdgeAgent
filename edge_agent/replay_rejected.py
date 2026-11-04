"""`python -m edge_agent replay-rejected --config agent.json [--apply]` -- resend windows from `rejected/`.

Why: before the server fix of 2026-10-30 (#289/#295/#297) the api answered a signed window 401 while
its Redis or Postgres was unreachable (a stack swap, an outage). The live sender treats a 401 as final
and moves the window to `<state_dir>/spool/rejected/`, so those windows were never stored
(docs/measurements/deployfix-lab-2026-10-30/README.md, item 4). This command lists them and, with
`--apply`, resends the ones that plausibly failed because of an outage.

Rules (the same as a live send, `edge_agent/transport.py`):

* Dry run by default: nothing is sent and no file is changed.
* The body is resent byte for byte through the SDK's `IngestTransport` (HMAC-signed, a fresh nonce per
  attempt, the configured endpoints, TLS and client certificate). The server keys a window by
  (tenant, model, `window_id`) and a payload digest: a window it already holds is answered with its
  stored result (`replayed_response`) and nothing is stored or alerted twice.
* Oldest first per model. A window is held back while an older window of the same model is still
  queued in the live spool (sending it first would make the queued one out of order). A transport-class
  failure (no endpoint reachable, 5xx, 408/425/429) stops the run, as it stops a live drain.
* Eligible by default: a recorded answer of 401 or 503 from every endpoint (what an auth-backend outage
  produced). Files written before edge 0.1.2 carry no answer ("unknown") and are eligible only with
  `--include-unknown`. Any other status (400, 403, 413, 422 schema or order refusals, ...) is never
  resent unless named with `--force-status`. A window older than the offline-queue limit
  (`spool_max_age_seconds`) is not resent: a live send would drop it too.
* A window the server accepts (or already held) moves to `spool/replayed/` with a receipt in the file
  and a line in `spool/replayed/receipts.jsonl`. A refused one stays in `rejected/` with the attempt
  recorded under `replay_attempts`; its latest refusal is the status the next run judges it by.

The server only accepts a window that closes after the last window it accepted from this model
(SRV-7, `out_of_order_window`). A window rejected in the middle of a stream, with newer windows
accepted since, is therefore refused 422 by the current server, recorded, and left in `rejected/`.

Exit codes: 0 done (or nothing to do), 1 at least one resend was refused or the link failed,
2 bad input.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from edge_agent.config import EdgeConfig
from edge_agent.transport import REPLAYED_DIR, DurableSpool, Sender, answers_of, is_deterministic

OUTAGE_STATUSES = frozenset({401, 503})
RECEIPTS = "receipts.jsonl"


@dataclass
class Candidate:
    path: Path
    model: str | None
    window_id: str | None
    end: datetime | None
    statuses: list[int | None] = field(default_factory=list)
    reason: str = ""
    rejected_at: str | None = None
    source: str = "unknown"          # "rejection" | "replay_attempt" | "unknown" | "unreadable"
    eligible: bool = False
    why: str = ""
    entry: dict[str, Any] | None = None

    def row(self) -> dict[str, Any]:
        return {"file": self.path.name, "model": self.model, "window_id": self.window_id,
                "window_end": self.end.isoformat() if self.end else None,
                "status": "unknown" if self.source in ("unknown", "unreadable") else self.statuses,
                "reason": self.reason, "rejected_at": self.rejected_at, "judged_by": self.source,
                "eligible": self.eligible, "why": self.why}


def _parse_time(text: str) -> datetime:
    t = datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def _latest_answer(entry: dict[str, Any]) -> tuple[str, list[dict[str, Any]], str | None]:
    attempts = entry.get("replay_attempts") or []
    if attempts:
        return "replay_attempt", attempts[-1].get("answers") or [], attempts[-1].get("at")
    rej = entry.get("rejection")
    if isinstance(rej, dict):
        return "rejection", rej.get("answers") or [], rej.get("at")
    return "unknown", [], None


def load_candidate(path: Path) -> Candidate:
    try:
        entry = json.loads(path.read_text())
        body = entry["body"]
        end = _parse_time(entry["end"])
    except (OSError, ValueError, KeyError, TypeError):
        return Candidate(path, None, None, None, source="unreadable", why="file is not a readable spooled window")
    source, answers, at = _latest_answer(entry)
    return Candidate(path, body.get("monitored_model_id"), body.get("window_id"), end,
                     statuses=[a.get("status") for a in answers],
                     reason=" | ".join(str(a.get("reason") or "") for a in answers)[:300],
                     rejected_at=at, source=source, entry=entry)


def judge(c: Candidate, *, include_unknown: bool, force_status: set[int], now: datetime,
          max_age_seconds: float) -> None:
    """Set `eligible` and `why` (the filters have already been applied)."""
    if c.source == "unreadable":
        c.eligible, c.why = False, c.why
        return
    if (now - c.end).total_seconds() > max_age_seconds:
        c.eligible, c.why = False, "older than the offline-queue limit (spool_max_age_seconds); a live send drops it"
        return
    if c.source == "unknown":
        c.eligible = include_unknown
        c.why = ("no recorded answer (written before edge 0.1.2); --include-unknown" if include_unknown
                 else "no recorded answer (written before edge 0.1.2); pass --include-unknown to resend")
        return
    allowed = OUTAGE_STATUSES | force_status
    if c.statuses and all(s in allowed for s in c.statuses):
        forced = [s for s in c.statuses if s not in OUTAGE_STATUSES]
        c.eligible = True
        c.why = f"forced (--force-status {','.join(map(str, sorted(set(forced))))})" if forced \
            else "outage-class answer (401/503)"
        return
    shown = ",".join(str(s) for s in c.statuses) or "none"
    c.eligible = False
    c.why = f"answer {shown} is not outage-class; resend only with --force-status"


def collect(cfg: EdgeConfig, *, since: datetime | None = None, until: datetime | None = None,
            models: list[str] | None = None, include_unknown: bool = False,
            force_status: set[int] | None = None, now: datetime | None = None) -> list[Candidate]:
    """Every file in `rejected/` that passes the filters, judged, oldest first per model."""
    spool = DurableSpool(cfg.state_dir / "spool")
    now = now or datetime.now(timezone.utc)
    wanted = None
    if models:
        wanted = {m for m in models} | {cfg.models[m] for m in models if m in cfg.models}
    out = []
    for path in sorted(spool.rejected.glob("*.json")):
        c = load_candidate(path)
        if c.source != "unreadable":
            if wanted is not None and c.model not in wanted:
                continue
            if (since and c.end < since) or (until and c.end > until):
                continue
        elif wanted is not None or since or until:
            continue
        judge(c, include_unknown=include_unknown, force_status=force_status or set(), now=now,
              max_age_seconds=cfg.spool_max_age_seconds)
        out.append(c)
    far = datetime.max.replace(tzinfo=timezone.utc)
    return sorted(out, key=lambda c: (c.model or "", c.end or far, c.path.name))


def _write_atomic(path: Path, entry: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        json.dump(entry, fh, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _free_name(directory: Path, name: str) -> Path:
    dest, n = directory / name, 1
    while dest.exists():
        dest = directory / f"{Path(name).stem}.{n}.json"
        n += 1
    return dest


def _older_queued(spool: DurableSpool, model: str, end: datetime) -> bool:
    for path in spool.pending():
        try:
            e = spool.read(path)
            if e["body"]["monitored_model_id"] == model and _parse_time(e["end"]) <= end:
                return True
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return False


@dataclass
class ReplayReport:
    listed: list[dict[str, Any]] = field(default_factory=list)
    replayed: int = 0
    already_stored: int = 0
    refused: int = 0
    held_back: int = 0
    link_failed: bool = False
    applied: bool = False

    def summary(self) -> dict[str, Any]:
        return {"applied": self.applied, "listed": len(self.listed),
                "eligible": sum(r["eligible"] for r in self.listed), "replayed": self.replayed,
                "already_stored": self.already_stored, "refused": self.refused,
                "held_back": self.held_back, "link_failed": self.link_failed}


def replay(cfg: EdgeConfig, candidates: list[Candidate], *, post: Callable | None = None,
           clock: Callable[[], datetime] | None = None) -> ReplayReport:
    """Resend the eligible candidates (already sorted oldest first per model)."""
    from blurredbox_agd.ingest import SIGNAL_PATH, IngestError
    from edge_agent.tripwires import LocalAlerter

    clock = clock or (lambda: datetime.now(timezone.utc))
    spool = DurableSpool(cfg.state_dir / "spool")
    replayed_dir = spool.root / REPLAYED_DIR
    replayed_dir.mkdir(parents=True, exist_ok=True)
    sender = Sender(cfg, spool, LocalAlerter(cfg.state_dir / "alerts.jsonl", cfg.alert_cooldown_seconds),
                    post=post, clock=clock)
    report = ReplayReport(applied=True)
    try:
        for c in candidates:
            row = c.row()
            report.listed.append(row)
            if not c.eligible or report.link_failed:
                continue
            if _older_queued(spool, c.model, c.end):
                row["outcome"] = "held_back: an older window of this model is still queued; let the agent send it first"
                report.held_back += 1
                continue
            at = clock().isoformat()
            try:
                result = sender.transport.send(SIGNAL_PATH, c.entry["body"])
            except IngestError as exc:
                answers = answers_of(exc)
                c.entry.setdefault("replay_attempts", []).append({"at": at, "answers": answers})
                _write_atomic(c.path, c.entry)
                said = (",".join(str(a["status"]) for a in answers) + " "
                        + " | ".join(a["reason"] for a in answers)[:200])
                if is_deterministic(exc):
                    report.refused += 1
                    row["outcome"] = f"refused: {said}; left in rejected/"
                    continue
                report.link_failed = True          # keep order: nothing behind it may overtake it
                row["outcome"] = f"link failed ({said}); stopped, left in rejected/"
                continue
            stored_before = bool(isinstance(result, dict) and result.get("replayed_response"))
            receipt = {"at": at, "result": "already_stored" if stored_before else "accepted",
                       "window_id": c.window_id, "model": c.model, "window_end": c.end.isoformat(),
                       "from": c.path.name, "judged_by": c.source, "status_before": c.statuses or "unknown"}
            c.entry["replay_receipt"] = receipt
            dest = _free_name(replayed_dir, c.path.name)
            _write_atomic(dest, c.entry)
            c.path.unlink(missing_ok=True)
            with (replayed_dir / RECEIPTS).open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({**receipt, "file": dest.name}, sort_keys=True) + "\n")
            row["outcome"] = receipt["result"]
            if stored_before:
                report.already_stored += 1
            else:
                report.replayed += 1
    finally:
        sender.close()
    return report


def _print_table(rows: list[dict[str, Any]], out) -> None:
    for r in rows:
        status = r["status"] if r["status"] == "unknown" else ",".join(str(s) for s in r["status"]) or "-"
        line = (f"{r['window_end'] or '?':<32} {r['model'] or '?':<24} status={status:<8} "
                f"{'ELIGIBLE' if r['eligible'] else 'skip    '} {r['why']}")
        if r.get("reason"):
            line += f" | reason: {r['reason'][:120]}"
        if r.get("outcome"):
            line += f" -> {r['outcome']}"
        out.write(line + "\n")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="edge_agent replay-rejected",
                                 description="List (default) or resend (--apply) windows in the spool's rejected/ directory.")
    ap.add_argument("--config", required=True, help="the agent's config file (agent.json / config.json)")
    ap.add_argument("--apply", action="store_true", help="resend the eligible windows (default: dry run)")
    ap.add_argument("--include-unknown", action="store_true",
                    help="also resend files with no recorded answer (written before edge 0.1.2)")
    ap.add_argument("--force-status", type=int, action="append", default=[], metavar="HTTP",
                    help="also resend windows refused with this HTTP status (repeatable), e.g. 403")
    ap.add_argument("--since", type=_parse_time, help="only windows closing at or after this time (ISO 8601, UTC default)")
    ap.add_argument("--until", type=_parse_time, help="only windows closing at or before this time")
    ap.add_argument("--model", action="append", default=[],
                    help="only this model (your local name or the BlurredBox model id; repeatable)")
    ap.add_argument("--json", action="store_true", help="print one JSON object instead of a table")
    return ap


def main(argv: list[str] | None = None, *, post: Callable | None = None,
         clock: Callable[[], datetime] | None = None, out=None) -> int:
    from edge_agent.__main__ import load_edge_config

    out = out or sys.stdout
    ap = build_parser()
    a = ap.parse_args(argv)
    try:
        raw = json.loads(Path(a.config).read_text())
        cfg = load_edge_config(raw, need_secret=a.apply)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        ap.exit(2, f"edge_agent replay-rejected: cannot load the config: {type(exc).__name__}: {exc}\n")
    now = (clock or (lambda: datetime.now(timezone.utc)))()
    cands = collect(cfg, since=a.since, until=a.until, models=a.model, include_unknown=a.include_unknown,
                    force_status=set(a.force_status), now=now)
    if a.apply:
        report = replay(cfg, cands, post=post, clock=clock)
    else:
        report = ReplayReport(listed=[c.row() for c in cands])
    summary = report.summary()
    if a.json:
        out.write(json.dumps({"summary": summary, "windows": report.listed}, sort_keys=True, default=str) + "\n")
    else:
        _print_table(report.listed, out)
        out.write(("applied: " if a.apply else "dry run (nothing sent; add --apply to resend): ")
                  + json.dumps(summary, sort_keys=True) + "\n")
    return 1 if (report.refused or report.link_failed) else 0
