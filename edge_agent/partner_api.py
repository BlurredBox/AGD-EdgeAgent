"""`edge_agent incident mark|list|delete` and `edge_agent erase`: the partner's own calls to the
stewardship routes, signed the way the agent signs its windows.

Why: the partner docs say the partner marks incidents through `POST /api/v1/incidents` and can erase
its telemetry with `DELETE /api/v1/telemetry-data`. Both routes need the client certificate (mTLS, checked
by nginx) and an HMAC-signed request with the tenant's API key, and no console screen or tool made that
request (day-1 dry read, 2026-10-30). This module reuses what the installed agent already has: the config
(endpoints, client certificate, `secret_env`) and the API key in its environment.

* Signing is the SDK's `_signing.sign_request` (the same bytes the ingest transport signs: canonical JSON,
  a fresh timestamp and nonce per request) plus `Authorization: Bearer <BLURREDBOX_API_KEY | AGD_API_KEY>`.
* TLS is the SDK's `_tls_context` with the configured `client_cert` and `ca_file`.
* Endpoints are tried in order; the next one is tried only when the previous could not be reached. An HTTP
  answer of any status is final (a mark is not repeated on a second endpoint after a refusal).
* `incident mark` lists the marks first and creates none if one with the same model, start, end and class
  exists (it prints that id and exits 0); after a lost answer it lists again before the next endpoint, so a
  retry never makes a second mark (founder decision 2026-10-31).
* Nothing secret is printed. Exit codes: 0 done, 1 refused or unreachable, 2 bad input or not confirmed.

`erase` withdraws model-improvement consent and erases this tenant's predictive telemetry on the server
(windows, shadow records, incident marks; see `src/api/tenant_erasure.py` for what is kept and why). It
asks for confirmation unless `--yes` is given; without a terminal and without `--yes` it refuses.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

INCIDENTS_PATH = "/api/v1/incidents"
ERASE_PATH = "/api/v1/telemetry-data"
CONFIRM_WORD = "ERASE"

# (method, base url, path, body bytes, headers) -> (status code, decoded JSON or {})
Request = Callable[[str, str, str, bytes, Mapping[str, str]], "tuple[int, Any]"]


class Unreachable(RuntimeError):
    pass


class OutcomeUnknown(Unreachable):
    """A mark's POST got no answer and the marks could not be listed after it: it may have been stored."""


def _parse_time(text: str) -> datetime:
    """ISO 8601; a time without an offset is UTC."""
    try:
        t = datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not an ISO 8601 time: {text!r}") from exc
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def _uuid(text: str) -> str:
    import uuid
    try:
        return str(uuid.UUID(text))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not an incident id: {text!r}") from exc


def _httpx_request(timeout: float, verify=None) -> Request:
    import httpx

    def request(method: str, url: str, path: str, body: bytes, headers: Mapping[str, str]):
        with httpx.Client(base_url=url, timeout=timeout, **({} if verify is None else {"verify": verify})) as c:
            r = c.request(method, path, content=body, headers=dict(headers))
        try:
            data = r.json()
        except ValueError:
            data = {}
        return r.status_code, data
    return request


class SignedClient:
    """One signed request to the configured endpoints (in order, next only when unreachable)."""

    def __init__(self, cfg, *, request: Request | None = None, api_key: str | None = None) -> None:
        from blurredbox_agd.ingest import _tls_context
        self.endpoints = [e.rstrip("/") for e in cfg.endpoints]
        if not self.endpoints:
            raise ValueError("the config names no endpoints")
        self._secret = cfg.secret
        self._api_key = (api_key or os.environ.get("BLURREDBOX_API_KEY") or os.environ.get("AGD_API_KEY") or "").strip()
        self._request = request or _httpx_request(cfg.timeout_seconds, _tls_context(cfg.client_cert, cfg.ca_file))

    def call_one(self, endpoint: str, method: str, path: str, body: Any = None) -> tuple[int, Any]:
        """One signed request to one endpoint; raises when it cannot be reached."""
        from blurredbox_agd._signing import sign_request
        headers, raw = sign_request(self._secret, body)          # fresh timestamp + nonce per attempt
        if self._api_key:
            headers = {**headers, "Authorization": f"Bearer {self._api_key}"}
        return self._request(method, endpoint, path, raw, headers)

    def call(self, method: str, path: str, body: Any = None) -> tuple[int, Any]:
        errors = []
        for endpoint in self.endpoints:
            try:
                return self.call_one(endpoint, method, path, body)
            except Exception as exc:  # noqa: BLE001 -- reported per endpoint, then the next one
                errors.append(f"{endpoint}: {type(exc).__name__}: {exc}")
        raise Unreachable("no endpoint could be reached: " + "; ".join(errors))


class Refused(RuntimeError):
    def __init__(self, code: int, data: Any) -> None:
        super().__init__(f"HTTP {code}")
        self.code, self.data = code, data


def _same_time(a: Any, b: str) -> bool:
    try:
        return _parse_time(str(a)) == _parse_time(b)
    except (argparse.ArgumentTypeError, TypeError):
        return False


def find_existing(client: SignedClient, body: dict[str, Any]) -> str | None:
    """The id of a mark with the same model, start, end and class as `body`, else None (founder decision
    2026-10-31: a mark is never created twice). Raises `Refused` if the list is refused, `Unreachable`."""
    code, data = client.call("GET", INCIDENTS_PATH)
    if code != 200:
        raise Refused(code, data)
    for r in (data or {}).get("incidents", []):
        if (r.get("monitored_model_id") == body["monitored_model_id"]
                and (r.get("failure_class") or None) == body.get("failure_class")
                and _same_time(r.get("failed_from"), body["failed_from"])
                and _same_time(r.get("failed_to"), body["failed_to"])):
            return str(r.get("id"))
    return None


def mark_once(client: SignedClient, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """Check the existing marks first, then POST. When an endpoint gives no answer (the mark may have been
    stored and the answer lost), the marks are listed again before the next endpoint is tried, so a lost
    answer and a retry never make a second mark. Returns (status, {"id", "existing"})."""
    existing = find_existing(client, body)
    if existing:
        return 200, {"id": existing, "existing": True}
    errors = []
    for endpoint in client.endpoints:
        try:
            code, data = client.call_one(endpoint, "POST", INCIDENTS_PATH, body)
        except Exception as exc:  # noqa: BLE001 -- no answer: maybe stored; check before the next endpoint
            errors.append(f"{endpoint}: {type(exc).__name__}: {exc}")
            try:
                existing = find_existing(client, body)
            except Unreachable as list_exc:
                # Pre-tag review L4: the POST may have been stored and the list cannot tell us. POSTing to the
                # next endpoint could make a second mark; stop and say so.
                raise OutcomeUnknown(
                    "the mark may have been stored (no answer from " + endpoint + ") and the marks could not "
                    "be listed to check: outcome unknown. Run `incident list` before marking again. "
                    + "; ".join(errors) + f"; list: {list_exc}") from exc
            if existing:
                return 200, {"id": existing, "existing": True}
            continue
        return code, ({**data, "existing": False} if code == 201 and isinstance(data, dict) else data)
    raise Unreachable("no endpoint could be reached: " + "; ".join(errors))


def resolve_model(cfg, name: str | None) -> str:
    """The BlurredBox model id for `name` (a local name or the id itself); with one model configured,
    `name` may be left out."""
    if name is None:
        ids = sorted(set(cfg.models.values()))
        if len(ids) == 1:
            return ids[0]
        raise ValueError("more than one model is configured; name one with --model "
                         f"(your local name or the BlurredBox id): {', '.join(sorted(cfg.models))}")
    if name in cfg.models:
        return cfg.models[name]
    if name in cfg.models.values():
        return name
    raise ValueError(f"model {name!r} is not in the config (configured: {', '.join(sorted(cfg.models))})")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="edge_agent", description="Signed partner calls: incident marks and erasure.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    inc = sub.add_parser("incident", help="mark, list or delete incident marks")
    isub = inc.add_subparsers(dest="action", required=True)
    m = isub.add_parser("mark", help="mark that a model failed between two times")
    m.add_argument("--start", required=True, type=_parse_time, help="when the failure started (ISO 8601, UTC default)")
    m.add_argument("--end", required=True, type=_parse_time, help="when it ended (ISO 8601, UTC default)")
    m.add_argument("--model", help="your local model name or the BlurredBox model id (optional with one model)")
    m.add_argument("--class", dest="failure_class", help="optional short failure class, e.g. data_feed")
    m.add_argument("--note", help="optional free text (at most 2000 characters; stored on the server)")
    ls = isub.add_parser("list", help="list this tenant's incident marks")
    d = isub.add_parser("delete", help="delete one mark (to amend: delete it and mark again)")
    d.add_argument("incident_id", type=_uuid, help="the id `incident mark` or `incident list` printed")
    e = sub.add_parser("erase", help="withdraw consent and erase this tenant's telemetry on the server")
    e.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    for p in (m, ls, d, e):
        p.add_argument("--config", required=True, help="the agent's config file (agent.json / config.json)")
        p.add_argument("--json", action="store_true", help="print the server's JSON answer")
    return ap


def _confirm(inp, out) -> bool:
    if not getattr(inp, "isatty", lambda: False)():
        return False
    out.write("This erases ALL of this tenant's predictive telemetry on BlurredBox (windows, shadow records,\n"
              "incident marks) and withdraws model-improvement consent. It cannot be undone.\n"
              f"Type {CONFIRM_WORD} to continue: ")
    out.flush()
    return inp.readline().strip() == CONFIRM_WORD


def main(argv: list[str] | None = None, *, request: Request | None = None,
         out=None, inp=None) -> int:
    from edge_agent.__main__ import load_edge_config

    out = out or sys.stdout
    inp = inp or sys.stdin
    ap = build_parser()
    a = ap.parse_args(argv)
    try:
        cfg = load_edge_config(json.loads(Path(a.config).read_text()), need_secret=True, path=a.config)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        # KeyError: the variable `secret_env` names is not set; its NAME is shown, never a value.
        ap.exit(2, f"edge_agent {a.cmd}: cannot load the config: {type(exc).__name__}: {exc}\n")

    if a.cmd == "erase":
        if not a.yes and not _confirm(inp, out):
            out.write("not confirmed; nothing was sent (pass --yes to skip the question)\n")
            return 2
        method, path, body, ok = "DELETE", ERASE_PATH, None, {200}
    elif a.action == "mark":
        try:
            model = resolve_model(cfg, a.model)
        except ValueError as exc:
            ap.exit(2, f"edge_agent incident mark: {exc}\n")
        if a.end < a.start:
            ap.exit(2, "edge_agent incident mark: --end is before --start\n")
        body = {"monitored_model_id": model, "failed_from": a.start.isoformat(), "failed_to": a.end.isoformat()}
        if a.failure_class:
            body["failure_class"] = a.failure_class
        if a.note:
            body["notes"] = a.note
        method, path, ok = "POST", INCIDENTS_PATH, {201}
    elif a.action == "list":
        method, path, body, ok = "GET", INCIDENTS_PATH, None, {200}
    else:
        method, path, body, ok = "DELETE", f"{INCIDENTS_PATH}/{a.incident_id}", None, {200}

    try:
        client = SignedClient(cfg, request=request)
        if method == "POST":
            code, data = mark_once(client, body)
            ok = {200, 201}
        else:
            code, data = client.call(method, path, body)
    except Unreachable as exc:
        out.write(f"edge_agent: {exc}\n")
        return 1
    except Refused as exc:
        code, data = exc.code, exc.data
    if code not in ok:
        hint = " (is the API key set? BLURREDBOX_API_KEY in agent.env)" if code == 401 else ""
        out.write(f"refused: HTTP {code}{hint}: {json.dumps(data)[:300]}\n")
        return 1
    if a.json:
        out.write(json.dumps(data, sort_keys=True, default=str) + "\n")
    elif a.cmd == "erase":
        out.write(f"erased: {json.dumps(data.get('deleted', {}), sort_keys=True)}; consent is now off\n")
    elif a.action == "mark":
        what = "already marked (nothing new created)" if data.get("existing") else "marked"
        out.write(f"{what}: incident {data.get('id')} for {body['monitored_model_id']} "
                  f"from {body['failed_from']} to {body['failed_to']}\n")
    elif a.action == "list":
        rows = data.get("incidents", [])
        for r in rows:
            out.write(f"{r.get('id')}  {r.get('monitored_model_id')}  {r.get('failed_from')} -> "
                      f"{r.get('failed_to')}  {r.get('failure_class') or ''}  {r.get('notes') or ''}\n")
        out.write(f"{len(rows)} mark(s)\n")
    else:
        out.write(f"deleted: {data.get('deleted', 0)}\n")
    return 0
