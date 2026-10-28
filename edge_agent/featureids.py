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
from pathlib import Path


class FeatureIdMap:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._salt: bytes
        self._by_name: dict[str, str] = {}
        self._by_id: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        try:
            d = json.loads(self.path.read_text())
            self._salt = bytes.fromhex(d["salt"])
            self._by_id = dict(d["ids"])
            self._by_name = {n: i for i, n in self._by_id.items()}
        except (OSError, ValueError, KeyError):
            self._salt = secrets.token_bytes(32)
            self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as fh:
            json.dump({"salt": self._salt.hex(), "ids": self._by_id}, fh, sort_keys=True)
        os.replace(tmp, self.path)

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
