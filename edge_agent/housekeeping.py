"""Bounded local state (audit F24): trim `spool/rejected/` and remove stale `*.tmp` files left by a crash.

`rejected/` holds windows the backend refused for good (for `replay-rejected`); it is capped by age and count so it
cannot grow without limit. Only files under the spool are touched, never a queued window (`*.json` outside `rejected/`).
"""
from __future__ import annotations

import time
from pathlib import Path

REJECTED_MAX_FILES = 1000
REJECTED_MAX_AGE_SECONDS = 90 * 86400
TMP_MAX_AGE_SECONDS = 3600


def trim_spool(spool_root: str | Path, *, now: float | None = None) -> dict[str, int]:
    root = Path(spool_root)
    now = time.time() if now is None else now
    removed = {"rejected": 0, "tmp": 0}
    rejected = root / "rejected"
    files = sorted((p for p in rejected.glob("*.json") if p.is_file()), key=lambda p: p.stat().st_mtime)
    for i, p in enumerate(files):
        too_many = len(files) - i > REJECTED_MAX_FILES
        if too_many or now - p.stat().st_mtime > REJECTED_MAX_AGE_SECONDS:
            p.unlink(missing_ok=True)
            removed["rejected"] += 1
    for p in root.rglob("*.tmp"):
        try:
            if now - p.stat().st_mtime > TMP_MAX_AGE_SECONDS:
                p.unlink(missing_ok=True)
                removed["tmp"] += 1
        except OSError:
            continue
    return removed
