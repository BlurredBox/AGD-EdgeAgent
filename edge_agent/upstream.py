"""Upstream change events (G52): read job / deploy / freshness status from the PARTNER side, read-only.

Two sources, one event shape (the server's `UpstreamEvent`, mirrored below and pinned equal by a test):

    {"type": "file",    "path": "/var/lib/agd/upstream-events.jsonl"}
    {"type": "airflow", "base_url": "https://airflow.internal:8080", "dag_ids": ["features_daily"],
     "token_env": "EDGE_AIRFLOW_TOKEN"}                       (or username_env + password_env)

`file`: a JSONL file the partner's cron / orchestrator / deploy hook appends one event per line to; the
agent only reads it. `airflow`: the stable REST API, `GET /api/v1/dags/{dag}/dagRuns` only, never a write.
Secrets are named (`*_env`), never written in the config, as for the Kafka source.

Events are checked HERE with the server's rules and a bad one is dropped and counted, so one malformed
line can never make the server refuse (and the agent discard) a whole batch. Nothing but the event
fields leaves the host: no log lines, no DAG parameters, no row content. Shadow-only: nothing pages.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import ssl
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from edge_agent.mapping import ConfigError

UPSTREAM_PATH = "/api/v1/ingest/signal/upstream-events"
MAX_EVENTS_PER_POST = 100
KIND_STATUSES: dict[str, frozenset[str]] = {
    "deploy": frozenset({"started", "success", "failure", "rollback"}),
    "schema_migration": frozenset({"started", "success", "failure"}),
    "job_run": frozenset({"running", "success", "failure", "late"}),
    "feature_freshness": frozenset({"fresh", "stale"}),
}
_FIELDS = {"event_id", "kind", "status", "subject", "occurred_at", "monitored_model_id", "freshness_lag_s"}
_EVENT_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_SUBJECT = re.compile(r"^[A-Za-z0-9._:/@+-]{1,64}$")
_NAME = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
_FUTURE = timedelta(minutes=5)
_MAX_AGE = timedelta(days=30)
_SECRET_WORDS = ("token", "password", "secret", "passwd", "credential", "key")


def clean_event(raw: Any, now: datetime) -> dict[str, Any] | None:
    """The event in wire shape, or None when it breaks any rule (the caller counts it)."""
    if not isinstance(raw, dict) or set(raw) - _FIELDS:
        return None
    try:
        kind, status = raw["kind"], raw["status"]
        if kind not in KIND_STATUSES or status not in KIND_STATUSES[kind]:
            return None
        if not (isinstance(raw["event_id"], str) and _EVENT_ID.match(raw["event_id"])):
            return None
        if not (isinstance(raw["subject"], str) and _SUBJECT.match(raw["subject"])):
            return None
        at = raw["occurred_at"]
        at = datetime.fromisoformat(at.replace("Z", "+00:00")) if isinstance(at, str) else None
        if at is None or at.tzinfo is None or at > now + _FUTURE or at < now - _MAX_AGE:
            return None
        out: dict[str, Any] = {"event_id": raw["event_id"], "kind": kind, "status": status,
                               "subject": raw["subject"], "occurred_at": at.astimezone(timezone.utc).isoformat()}
        model = raw.get("monitored_model_id")
        if model is not None:
            if not (isinstance(model, str) and 1 <= len(model) <= 128):
                return None
            out["monitored_model_id"] = model
        lag = raw.get("freshness_lag_s")
        if lag is not None:
            if (isinstance(lag, bool) or not isinstance(lag, (int, float)) or not 0 <= lag < 1e8
                    or kind not in ("job_run", "feature_freshness")):
                return None
            out["freshness_lag_s"] = float(lag)
        return out
    except (KeyError, ValueError, TypeError):
        return None


def _check_no_inline_secret(spec: Mapping[str, Any], where: str) -> None:
    for k in spec:
        low = k.lower()
        if any(w in low for w in _SECRET_WORDS) and not low.endswith("_env"):
            raise ConfigError(f"{where}.{k}: use {k}_env (the name of an environment variable)")


def _env_name(spec: Mapping[str, Any], key: str, where: str) -> str:
    v = spec.get(key)
    if not isinstance(v, str) or not re.match(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$", v):
        raise ConfigError(f"{where}.{key} must be the name of an environment variable")
    return v


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """A 3xx is an error: never replay Authorization to another host."""
    def redirect_request(self, *a, **k):
        return None


class _Base:
    kind = "upstream"

    def __init__(self, name: str, state_dir: Path) -> None:
        self.name = name
        self.dropped = 0          # events that broke a rule (counted, never logged with content)
        self.read = 0
        self._state = Path(state_dir) / "sources" / f"{name}.upstream.json"

    def _load(self) -> dict[str, Any]:
        try:
            d = json.loads(self._state.read_text())
            return d if isinstance(d, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save(self, state: dict[str, Any]) -> None:
        self._state.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._state.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as fh:
            json.dump(state, fh, sort_keys=True)
        os.replace(tmp, self._state)

    def commit(self) -> None: ...

    def describe(self) -> dict[str, Any]:
        return {"type": self.type_name, "name": self.name, "events_read": self.read, "events_dropped": self.dropped}

    type_name = "upstream"


class FileStatusSource(_Base):
    type_name = "file"

    def __init__(self, spec: Mapping[str, Any], state_dir: str | Path, *, name: str,
                 clock: Callable[[], datetime] | None = None) -> None:
        extra = sorted(set(spec) - {"type", "name", "path"})
        if extra:
            raise ConfigError(f"upstream_source '{name}': unknown key(s) {extra}")
        path = spec.get("path")
        if not isinstance(path, str) or not path:
            raise ConfigError(f"upstream_source '{name}'.path is required for a file source")
        super().__init__(name, Path(state_dir))
        self.path = Path(path)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._offset = int(self._load().get("offset", 0))
        self._pending = self._offset

    def poll(self, max_events: int = 500) -> list[dict[str, Any]]:
        try:
            size = self.path.stat().st_size
        except OSError:
            return []
        if size < self._offset:               # truncated or rotated: start again, the server dedupes
            self._offset = self._pending = 0
        out: list[dict[str, Any]] = []
        now = self._clock()
        with open(self.path, "rb") as fh:
            fh.seek(self._offset)
            while len(out) < max_events:
                line = fh.readline()
                if not line or not line.endswith(b"\n"):     # a half-written last line waits
                    break
                self._pending = fh.tell()
                if not line.strip():
                    continue
                try:
                    ev = clean_event(json.loads(line), now)
                except ValueError:
                    ev = None
                self.read += 1
                if ev is None:
                    self.dropped += 1
                else:
                    out.append(ev)
        return out

    def commit(self) -> None:
        """Called after the events are durably in the outbox."""
        if self._pending != self._offset:
            self._offset = self._pending
            self._save({"offset": self._offset})


class AirflowSource(_Base):
    """Airflow stable REST API, read-only GET. `fetch(url, headers) -> parsed JSON` is injectable."""
    type_name = "airflow"
    _KEYS = {"type", "name", "base_url", "dag_ids", "token_env", "username_env", "password_env", "ca_file",
             "runs_per_poll", "poll_interval_seconds", "late_after_seconds", "timeout_seconds"}

    def __init__(self, spec: Mapping[str, Any], state_dir: str | Path, *, name: str,
                 env: Mapping[str, str] | None = None, clock: Callable[[], datetime] | None = None,
                 fetch: Callable[[str, dict[str, str]], Any] | None = None) -> None:
        where = f"upstream_source '{name}'"
        _check_no_inline_secret(spec, where)
        extra = sorted(set(spec) - self._KEYS)
        if extra:
            raise ConfigError(f"{where}: unknown key(s) {extra}")
        base = spec.get("base_url")
        if not isinstance(base, str) or not base.startswith(("https://", "http://127.0.0.1", "http://localhost")):
            raise ConfigError(f"{where}.base_url must be an https:// URL (plain http only for localhost)")
        dags = spec.get("dag_ids")
        if (not isinstance(dags, list) or not dags or len(dags) > 50
                or not all(isinstance(d, str) and re.match(r"^[A-Za-z0-9_.-]{1,250}$", d) for d in dags)):
            raise ConfigError(f"{where}.dag_ids must be a list of 1-50 DAG ids")
        self._token_env = _env_name(spec, "token_env", where) if "token_env" in spec else None
        self._user_env = _env_name(spec, "username_env", where) if "username_env" in spec else None
        self._pass_env = _env_name(spec, "password_env", where) if "password_env" in spec else None
        if (self._user_env is None) != (self._pass_env is None):
            raise ConfigError(f"{where}: username_env and password_env go together")
        if self._token_env is None and self._user_env is None:
            raise ConfigError(f"{where}: give token_env or username_env + password_env")
        super().__init__(name, Path(state_dir))
        self.base, self.dags = base.rstrip("/"), list(dags)
        self.runs = int(spec.get("runs_per_poll", 10))
        self.interval = float(spec.get("poll_interval_seconds", 60))
        self.late_after = float(spec["late_after_seconds"]) if "late_after_seconds" in spec else None
        self.timeout = float(spec.get("timeout_seconds", 10))
        if not (1 <= self.runs <= 100 and self.interval >= 5 and self.timeout > 0):
            raise ConfigError(f"{where}: runs_per_poll 1-100, poll_interval_seconds >= 5, timeout_seconds > 0")
        self._ca = spec.get("ca_file")
        self._env = os.environ if env is None else env
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._fetch = fetch or self._http_get
        self._seen: list[str] = list(self._load().get("seen", []))
        self._last_poll = 0.0
        self.errors = 0

    def _headers(self) -> dict[str, str]:
        if self._token_env:
            return {"Authorization": f"Bearer {self._env[self._token_env]}"}
        raw = f"{self._env[self._user_env]}:{self._env[self._pass_env]}".encode()
        return {"Authorization": "Basic " + base64.b64encode(raw).decode()}

    def _http_get(self, url: str, headers: dict[str, str]) -> Any:
        ctx = ssl.create_default_context(cafile=self._ca) if url.startswith("https") else None
        req = urllib.request.Request(url, headers={**headers, "Accept": "application/json"}, method="GET")
        opener = urllib.request.build_opener(_NoRedirect, urllib.request.HTTPSHandler(context=ctx))
        with opener.open(req, timeout=self.timeout) as resp:
            return json.loads(resp.read(2_000_000))

    @staticmethod
    def _when(v: Any) -> datetime | None:
        try:
            return datetime.fromisoformat(v.replace("Z", "+00:00")) if isinstance(v, str) else None
        except ValueError:
            return None

    def poll(self, max_events: int = 500) -> list[dict[str, Any]]:
        now = self._clock()
        if self._last_poll and (now.timestamp() - self._last_poll) < self.interval:
            return []
        self._last_poll = now.timestamp()
        out: list[dict[str, Any]] = []
        for dag in self.dags:
            url = (f"{self.base}/api/v1/dags/{urllib.parse.quote(dag, safe='')}/dagRuns"
                   f"?order_by=-start_date&limit={self.runs}")
            try:
                body = self._fetch(url, self._headers())
            except (OSError, ValueError, KeyError, urllib.error.URLError):
                self.errors += 1                     # unreachable / unauthorised: counted, retried next poll
                continue
            for run in (body.get("dag_runs") or []) if isinstance(body, dict) else []:
                out.extend(self._events_of(dag, run, now))
        return out[:max_events]

    def _events_of(self, dag: str, run: Any, now: datetime) -> list[dict[str, Any]]:
        if not isinstance(run, dict):
            self.dropped += 1
            return []
        state = run.get("state")
        started, ended = self._when(run.get("start_date")), self._when(run.get("end_date"))
        status, at = {"success": ("success", ended), "failed": ("failure", ended)}.get(state, (None, None))
        if state == "running" and started is not None:
            status, at = "running", started
            if self.late_after is not None and (now - started).total_seconds() > self.late_after:
                status = "late"
        if status is None or at is None:
            return []
        run_id = str(run.get("dag_run_id", ""))
        digest = hashlib.sha256(f"{dag}\x00{run_id}".encode()).hexdigest()[:20]
        raw = {"event_id": f"af:{digest}:{status}", "kind": "job_run", "status": status,
               "subject": re.sub(r"[^A-Za-z0-9._:/@+-]", "_", dag)[:64], "occurred_at": at.isoformat()}
        key = raw["event_id"]
        if key in self._seen:
            return []
        self.read += 1
        ev = clean_event(raw, now)
        if ev is None:
            self.dropped += 1
            return []
        self._seen.append(key)
        self._seen = self._seen[-2000:]
        return [ev]

    def commit(self) -> None:
        self._save({"seen": self._seen})

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "fetch_errors": self.errors, "dags": len(self.dags)}


def build_upstream_sources(raw: Any, state_dir: str | Path, *, taken: set[str] | None = None,
                           env: Mapping[str, str] | None = None,
                           clock: Callable[[], datetime] | None = None) -> list[_Base]:
    """Validate every `upstream_source` block and build its source. Raises ConfigError; never touches the
    network."""
    items = raw if isinstance(raw, list) else [raw]
    if not items or not all(isinstance(i, dict) for i in items):
        raise ConfigError("upstream_source must be an object (or a nonempty list of objects)")
    taken = set() if taken is None else taken
    out: list[_Base] = []
    for i, spec in enumerate(items):
        typ = spec.get("type", "file")
        name = spec.get("name", f"upstream{i}" if len(items) > 1 else "upstream")
        if not isinstance(name, str) or not _NAME.match(name):
            raise ConfigError(f"upstream_source[{i}].name must be 1-40 letters, digits, '_' or '-'")
        if name in taken:
            raise ConfigError(f"upstream_source[{i}].name '{name}' is used twice")
        taken.add(name)
        if typ == "file":
            out.append(FileStatusSource(spec, state_dir, name=name, clock=clock))
        elif typ == "airflow":
            out.append(AirflowSource(spec, state_dir, name=name, env=env, clock=clock))
        else:
            raise ConfigError(f"upstream_source[{i}].type must be 'file' or 'airflow', got {typ!r}")
    return out
