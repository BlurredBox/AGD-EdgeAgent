"""Stable feature ids; names stay on the partner's disk.

Feature names can be sensitive, so the wire carries ids. An id is
`f_` + 12 hex of HMAC-SHA256(per-install random salt, name): stable across
restarts, not derivable from the name without the local salt (so a dictionary
of common names cannot be run against what we receive), and it does not leak
column order or count the way a sequential id would.

The id -> name map lives in `<state_dir>/feature_ids.json` (mode 0600) for the
partner's own UI to map alerts back. It is never sent.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time
from pathlib import Path


class FeatureIdMap:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._salt: bytes
        self._by_name: dict[str, str] = {}
        self._by_id: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        """Read the salt. An id is HMAC(salt, name), so the salt is what must never change: a file that
        exists but cannot be parsed is kept aside (`.corrupt-<time>`), the backup copy (`.bak`) is tried, and only if
        both fail is a new salt made, flagged in `salt_regenerated` so the agent can alert."""
        self.salt_regenerated = False
        damaged = False
        for candidate in (self.path, self.path.with_suffix(".bak")):
            try:
                text = candidate.read_text()
            except FileNotFoundError:
                continue                       # nothing there yet
            # Any other OSError (EACCES, EIO, ...) propagates: an unreadable disk must stop the agent, not mint
            # a new salt, which would change every feature id.
            try:
                d = json.loads(text)
                self._salt = bytes.fromhex(d["salt"])
                ids = d.get("ids")
                self._by_id = dict(ids) if isinstance(ids, dict) else {}
                self._by_name = {n: i for i, n in self._by_id.items()}
            except (ValueError, KeyError, TypeError, AttributeError):
                damaged = True
                continue
            if candidate is not self.path:
                self._save()
            return
        if damaged:
            self.salt_regenerated = True
            if self.path.exists():
                try:
                    os.replace(self.path, self.path.with_name(f"{self.path.stem}.corrupt-{int(time.time())}"))
                except OSError:
                    pass
        self._salt = secrets.token_bytes(32)
        self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"salt": self._salt.hex(), "ids": self._by_id}, sort_keys=True)
        for dest in (self.path.with_suffix(".bak"), self.path):
            tmp = dest.with_suffix(".tmp")
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, dest)

    def id_for(self, name: str) -> str:
        known = self._by_name.get(name)
        if known is not None:
            return known
        digest = hmac.new(self._salt, name.encode("utf-8"), hashlib.sha256).hexdigest()
        n = 12
        fid = f"f_{digest[:n]}"
        while fid in self._by_id and self._by_id[fid] != name:   # collision: lengthen
            n += 2
            fid = f"f_{digest[:n]}"
        self._by_name[name], self._by_id[fid] = fid, name
        self._save()
        return fid

    def name_for(self, fid: str) -> str | None:
        """Local reverse lookup, for the partner's UI only."""
        return self._by_id.get(fid)

    def rekey(self, mapping: dict | None) -> dict | None:
        return None if mapping is None else {self.id_for(k): v for k, v in mapping.items()}
