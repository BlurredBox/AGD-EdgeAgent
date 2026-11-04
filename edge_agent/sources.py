"""Build input adapters from the `source` / `outcome_source` config blocks.

`source` and `outcome_source` are each one object or a list of objects. An object's
`type` is one of:

    jsonl     {"path": "/var/log/model.jsonl"}                        (default; the original tail)
    postgres  a table or view, read-only                              (edge_agent/source_postgres.py)
    folder    a directory of CSV / Parquet files                      (edge_agent/source_folder.py)
    kafka     topics read through one consumer group, JSON or Avro   (edge_agent/source_kafka.py)
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

from edge_agent.adapters import JsonlOutcomeAdapter, JsonlTailAdapter, SourceAdapter
from edge_agent.checkpoint import SOURCES_DIR
from edge_agent.mapping import ConfigError, check_keys

TYPES = ("jsonl", "postgres", "folder", "kafka")
_NAME = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


def _specs(raw: Any, where: str) -> list[dict[str, Any]]:
    items = raw if isinstance(raw, list) else [raw]
    if not items or not all(isinstance(i, dict) for i in items):
        raise ConfigError(f"{where} must be an object (or a nonempty list of objects)")
    return items


def build_sources(raw: Any, state_dir: str | Path, kind: str, *, taken: set[str] | None = None,
                  env: Mapping[str, str] | None = None) -> list[SourceAdapter]:
    """Validate every block and construct its adapter. Raises ConfigError, never touches
    the network or writes state (adapters read their own saved position only)."""
    where = "source" if kind == "prediction" else "outcome_source"
    taken = set() if taken is None else taken
    specs = _specs(raw, where)
    multiple = len(specs) > 1
    base = "predictions" if kind == "prediction" else "outcomes"
    out: list[SourceAdapter] = []
    for i, spec in enumerate(specs):
        label = f"{where}[{i}]" if multiple else where
        typ = spec.get("type", "jsonl")
        if typ not in TYPES:
            raise ConfigError(f"{label}.type must be one of {list(TYPES)}, got {typ!r}")
        name = spec.get("name", f"{base}{i}" if multiple else base)
        if not isinstance(name, str) or not _NAME.match(name):
            raise ConfigError(f"{label}.name must be 1-40 letters, digits, '_' or '-'")
        if name in taken:
            raise ConfigError(f"{label}.name '{name}' is used twice; every source needs its own name")
        taken.add(name)
        if typ == "jsonl":
            out.append(_jsonl(spec, state_dir, kind, name, multiple, label))
        elif typ == "postgres":
            from edge_agent.source_postgres import PostgresAdapter
            out.append(PostgresAdapter({**spec, "name": name}, state_dir, kind=kind, env=env))
        elif typ == "kafka":
            from edge_agent.source_kafka import KafkaAdapter
            out.append(KafkaAdapter({**spec, "name": name}, state_dir, kind=kind, env=env))
        else:
            from edge_agent.source_folder import FolderAdapter
            out.append(FolderAdapter({**spec, "name": name}, state_dir, kind=kind))
    return out


def build_proxy_sources(raw: Any, state_dir: str | Path, *, taken: set[str] | None = None,
                        env: Mapping[str, str] | None = None) -> list[SourceAdapter]:
    """G52 `proxy_source` blocks: an outcome-shaped source (same columns, same adapters) plus the proxy's
    `proxy_name` and `delay_seconds`. The adapters come back tagged `.proxy_name` / `.proxy_delay_s`; the
    agent joins them in its own store, never with the full outcomes."""
    from edge_agent.proxy import split_proxy_specs
    taken = set() if taken is None else taken
    out: list[SourceAdapter] = []
    for spec, proxy_name, delay in split_proxy_specs(raw):
        (adapter,) = build_sources(spec, state_dir, "outcome", taken=taken, env=env)
        adapter.proxy_name, adapter.proxy_delay_s = proxy_name, delay
        out.append(adapter)
    return out


def _jsonl(spec: dict, state_dir: str | Path, kind: str, name: str, multiple: bool, label: str):
    check_keys(spec, {"type", "name", "path"}, label)
    path = spec.get("path")
    if not isinstance(path, str) or not path:
        raise ConfigError(f"{label}.path is required for a jsonl source")
    sd = Path(state_dir)
    # Single unnamed sources keep their original offset file so an upgrade resumes in place.
    legacy = not multiple and "name" not in spec
    offset = sd / ("source.offset" if kind == "prediction" else "outcome.offset") if legacy \
        else sd / SOURCES_DIR / f"{name}.offset"
    cls = JsonlTailAdapter if kind == "prediction" else JsonlOutcomeAdapter
    return cls(path, offset)
