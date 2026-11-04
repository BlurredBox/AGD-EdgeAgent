"""CSV / Parquet folder input (predictions or outcomes).

    "source": {"type": "folder", "path": "/data/model-exports",
               "patterns": ["*.csv", "*.parquet"], "min_age_seconds": 30,
               "columns": {"ts": "scored_at", "model_id": "model", "instrument": "ticker",
                           "output": "p", "features": {"prefix": "f_"}}}

The agent never writes to, renames or deletes the partner's files. It remembers, per
file, how far it has read, keyed by device+inode (so a rename or rotation is the same
file, and a new file that reuses a name is a new file), checks the first bytes so a
recycled inode is not mistaken for the old file, and marks a file done once it has
been read to the end and has stopped changing.

A file is *partial* (ignored until it is not) while its name ends in an
`ignore_suffixes` entry or starts with an `ignore_prefixes` entry (writers usually
write `x.csv.tmp` and rename), or while it was modified less than `min_age_seconds`
ago. A CSV row that has no line ending yet is never read; a Parquet file whose footer
is not written yet cannot be opened and is retried. Memory is bounded by `batch_size`
rows (CSV) or one row group (Parquet).
"""
from __future__ import annotations

import copy
import csv
import fnmatch
import hashlib
import os
import time
from pathlib import Path
from typing import Any, Mapping

import structlog

from edge_agent.adapters import Sample
from edge_agent.checkpoint import CheckpointedAdapter, read_json
from edge_agent.mapping import ColumnMapping, ConfigError, check_keys, epoch_of, int_in, num_in
from edge_agent.records import Record

log = structlog.get_logger("edge_agent.source")

ALLOWED = {"type", "name", "path", "patterns", "format", "recursive", "ignore_suffixes",
           "ignore_prefixes", "min_age_seconds", "scan_interval_seconds", "batch_size",
           "delimiter", "encoding", "null_values", "unterminated_line_after_seconds",
           "columns", "constants", "max_uncommitted_rows"}
DEFAULT_IGNORE_SUFFIXES = (".tmp", ".part", ".partial", ".crdownload", ".swp", ".filepart", "~")
DEFAULT_NULLS = ("", "NULL", "null", "NA", "N/A")
FP_BYTES = 4096


class _BadFile(Exception):
    """A file that can never be read as configured (missing mapped columns)."""
    def __init__(self, missing: list[str]) -> None:
        super().__init__(f"missing columns {missing}")
        self.missing = missing


class _NotReady(Exception):
    """The file is still being written (no header line yet / no Parquet footer yet)."""


def _pyarrow():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:       # optional extra
        raise ConfigError("reading Parquet needs the optional dependency pyarrow: "
                          "pip install -r requirements-edge-parquet.txt") from exc
    return pq


class FolderAdapter(CheckpointedAdapter):
    def __init__(self, spec: Mapping[str, Any], state_dir: str | Path, *, kind: str = "prediction",
                 name: str | None = None, clock=time.time) -> None:
        where = "source" if kind == "prediction" else "outcome_source"
        spec = dict(spec)
        check_keys(spec, ALLOWED, where)
        self.kind = kind
        name = name or spec.get("name") or ("predictions" if kind == "prediction" else "outcomes")
        super().__init__(Path(state_dir), name, clock=clock,
                         max_uncommitted_rows=int_in(spec, "max_uncommitted_rows", 5_000_000,
                                                     1_000, 1_000_000_000, where))
        path = spec.get("path")
        if not isinstance(path, str) or not path:
            raise ConfigError(f"{where}.path is required: the directory to watch")
        self.root = Path(path)
        pats = spec.get("patterns", ["*.csv", "*.parquet"])
        if (not isinstance(pats, list) or not pats
                or not all(isinstance(p, str) and p and "/" not in p for p in pats)):
            raise ConfigError(f"{where}.patterns must be a nonempty list of file-name patterns such as \"*.csv\"")
        self.patterns = tuple(pats)
        self.format = spec.get("format", "auto")
        if self.format not in ("auto", "csv", "parquet"):
            raise ConfigError(f"{where}.format must be 'auto', 'csv' or 'parquet'")
        for key, default in (("ignore_suffixes", DEFAULT_IGNORE_SUFFIXES), ("ignore_prefixes", (".", "~"))):
            v = spec.get(key, list(default))
            if not isinstance(v, list) or not all(isinstance(x, str) and x for x in v):
                raise ConfigError(f"{where}.{key} must be a list of nonempty strings")
            setattr(self, key, tuple(v))
        self.recursive = spec.get("recursive", False)
        if not isinstance(self.recursive, bool):
            raise ConfigError(f"{where}.recursive must be true or false")
        self.min_age = num_in(spec, "min_age_seconds", 30.0, 0.0, 30 * 86400.0, where)
        self.scan_interval = num_in(spec, "scan_interval_seconds", 5.0, 0.0, 3600.0, where)
        # how long after a row is written it can first be read here: the file must be min_age old, then a scan must run
        # (the agent holds its clock close of a window by this plus one window; kitrun6 C-03)
        self.arrival_delay_seconds = self.min_age + self.scan_interval
        self.batch_size = int_in(spec, "batch_size", 5_000, 1, 100_000, where)
        self.delimiter = spec.get("delimiter", ",")
        if not isinstance(self.delimiter, str) or len(self.delimiter) != 1:
            raise ConfigError(f"{where}.delimiter must be one character")
        self.encoding = spec.get("encoding", "utf-8-sig")
        try:
            "".encode(self.encoding)
        except (LookupError, TypeError):
            raise ConfigError(f"{where}.encoding is not a known text encoding") from None
        nulls = spec.get("null_values", list(DEFAULT_NULLS))
        if not isinstance(nulls, list) or not all(isinstance(x, str) for x in nulls):
            raise ConfigError(f"{where}.null_values must be a list of strings")
        tail = spec.get("unterminated_line_after_seconds", 300.0)
        self.tail_after = None if tail is None else num_in(spec, "unterminated_line_after_seconds",
                                                           300.0, 1.0, 30 * 86400.0, where)
        self.mapping = ColumnMapping(spec.get("columns"), spec.get("constants"), kind=kind,
                                     null_values=tuple(nulls))
        if self.format == "parquet" or any(p.lower().endswith((".parquet", ".pq")) for p in self.patterns):
            _pyarrow()
        saved = read_json(self.state_file) or {}
        self._committed: dict[str, dict[str, Any]] = dict(saved.get("files", {}))
        self._live = copy.deepcopy(self._committed)
        self._last_scan = float("-inf")
        self._more = False
        self.last_scan: dict[str, Any] = {}
        self._rejected_logged: set[str] = set()

    # -- state ---------------------------------------------------------------------------
    def _apply(self, delta: dict[str, Any]) -> None:
        for key, entry in delta.items():
            if entry is None:
                self._committed.pop(key, None)
            else:
                self._committed[key] = entry

    def _state(self) -> dict[str, Any]:
        return {"v": 1, "type": "folder", "files": self._committed}

    # -- listing -------------------------------------------------------------------------
    def _kind_of(self, name: str) -> str:
        if self.format != "auto":
            return self.format
        return "parquet" if name.lower().endswith((".parquet", ".pq")) else "csv"

    def _list(self) -> tuple[list[tuple[Path, os.stat_result]], set[str], int]:
        """(eligible files oldest first, keys of every matching file, count still too fresh)."""
        found, seen, waiting = [], set(), 0
        now = self._clock()
        walker = os.walk(self.root) if self.recursive else [(str(self.root), [], os.listdir(self.root))]
        for dirpath, _, names in walker:
            for fname in names:
                if not any(fnmatch.fnmatch(fname, p) for p in self.patterns):
                    continue
                if fname.endswith(self.ignore_suffixes) or fname.startswith(self.ignore_prefixes):
                    waiting += 1
                    continue
                path = Path(dirpath) / fname
                try:
                    st = path.stat()
                except OSError:
                    continue
                if not path.is_file():
                    continue
                seen.add(f"{st.st_dev}:{st.st_ino}")
                if now - st.st_mtime < self.min_age or st.st_size == 0:
                    waiting += 1
                    continue
                found.append((path, st))
        found.sort(key=lambda it: (it[1].st_mtime_ns, str(it[0])))
        return found, seen, waiting

    # -- identity ------------------------------------------------------------------------
    @staticmethod
    def _fp(path: Path, n: int) -> str:
        with open(path, "rb") as fh:
            return hashlib.sha1(fh.read(n)).hexdigest()

    def _new_entry(self, path: Path, st: os.stat_result) -> dict[str, Any]:
        n = min(st.st_size, FP_BYTES)
        return {"name": path.name, "offset": 0, "size": st.st_size, "mtime_ns": st.st_mtime_ns,
                "fp_len": n, "fp": self._fp(path, n), "done": False}

    def _same_file(self, path: Path, e: Mapping[str, Any], st: os.stat_result) -> bool:
        """False for a truncated file or a recycled inode: both restart from the beginning."""
        if st.st_size < e["size"] or st.st_size < e["fp_len"]:
            return False
        try:
            return self._fp(path, e["fp_len"]) == e["fp"]
        except OSError:
            return False

    # -- reading -------------------------------------------------------------------------
    def _csv_page(self, path: Path, e: dict[str, Any], st: os.stat_result, n: int):
        needed = set(self.mapping.fixed_columns())
        with open(path, "rb") as fh:
            fst = os.fstat(fh.fileno())
            head = fh.readline()
            if not head.endswith(b"\n"):
                raise _NotReady
            header = [h.strip() for h in next(csv.reader([head.decode(self.encoding)],
                                                         delimiter=self.delimiter))]
            if self.mapping.features and self.mapping.features[0] in ("list", "json"):
                missing = sorted(needed - set(header))
                if missing:
                    raise _BadFile(missing)
            else:
                missing = sorted(set(self.mapping.cols.values()) - set(header))
                if missing:
                    raise _BadFile(missing)
            offset = max(e["offset"], len(head))
            fh.seek(offset)
            cur = {"consumed": offset, "trunc": False}
            tail_ok = (self.tail_after is not None and self._clock() - fst.st_mtime >= self.tail_after)

            def lines():
                while True:
                    line = fh.readline()
                    if not line:
                        return
                    if not line.endswith(b"\n") and not tail_ok:
                        cur["trunc"] = True
                        return
                    cur["consumed"] += len(line)
                    yield line.decode(self.encoding)

            recs, bad, rows, newest, good = [], 0, 0, None, offset
            for row in csv.reader(lines(), delimiter=self.delimiter):
                if cur["trunc"]:
                    break                        # a quoted field cut off mid-row: wait for the rest
                good = cur["consumed"]
                if not row:
                    continue
                rows += 1
                if len(row) != len(header):
                    bad += 1
                else:
                    cells = dict(zip(header, row))
                    t = epoch_of(self._raw_time(cells))
                    newest = t if t is not None and (newest is None or t > newest) else newest
                    try:
                        rec = self.mapping.to_record(cells)
                    except (ValueError, KeyError, TypeError, AttributeError, OverflowError):
                        bad += 1
                    else:
                        if rec is not None:
                            recs.append(rec)
                if rows >= n:
                    break
        e = dict(e, offset=good, size=fst.st_size, mtime_ns=fst.st_mtime_ns,
                 done=good == fst.st_size, name=path.name)
        return recs, bad, rows, newest, e

    def _raw_time(self, cells: Mapping[str, Any]) -> Any:
        return cells.get(self.mapping.time_column)

    def _parquet_page(self, path: Path, e: dict[str, Any], st: os.stat_result, n: int):
        pq = _pyarrow()
        try:
            pf = pq.ParquetFile(str(path))
        except Exception as exc:                 # noqa: BLE001 -- no footer yet: still being written
            raise _NotReady from exc
        names = pf.schema_arrow.names
        feats = self.mapping.features or (None, None)
        fixed = (self.mapping.fixed_columns() if feats[0] in ("list", "json", None)
                 else list(self.mapping.cols.values()))
        missing = sorted(set(fixed) - set(names))
        if missing:
            raise _BadFile(missing)
        if feats[0] == "prefix":
            cols = list(dict.fromkeys(fixed + [c for c in names if c.startswith(feats[1])]))
        elif feats[0] == "rest":
            cols = None
        else:
            cols = fixed
        md = pf.metadata
        offset, rows_out, group_start = e["offset"], [], 0
        for g in range(md.num_row_groups):
            gn = md.row_group(g).num_rows
            if group_start + gn <= offset:
                group_start += gn
                continue
            pos = group_start
            for batch in pf.iter_batches(batch_size=min(max(n, 1), 65_536), row_groups=[g], columns=cols):
                if pos + batch.num_rows <= offset:
                    pos += batch.num_rows
                    continue
                start = max(0, offset - pos)
                chunk = batch.slice(start, n - len(rows_out)).to_pylist()
                rows_out += chunk
                pos += start + len(chunk)
                offset = pos
                if len(rows_out) >= n:
                    break
            group_start += gn
            if len(rows_out) >= n:
                break
        end = e["offset"] + len(rows_out)
        recs, bad, newest = [], 0, None
        for row in rows_out:
            t = epoch_of(row.get(self.mapping.time_column))
            newest = t if t is not None and (newest is None or t > newest) else newest
            try:
                rec = self.mapping.to_record(row)
            except (ValueError, KeyError, TypeError, AttributeError, OverflowError):
                bad += 1
                continue
            if rec is not None:
                recs.append(rec)
        e = dict(e, offset=end, size=st.st_size, mtime_ns=st.st_mtime_ns,
                 done=end >= md.num_rows, name=path.name)
        return recs, bad, len(rows_out), newest, e

    # -- the scan ------------------------------------------------------------------------
    def _advance(self, files: dict[str, dict[str, Any]], max_records: int):
        """Read up to `max_records` rows from eligible files, mutating `files` (a cursor)
        and returning [(records, bad, rows, newest_ts, delta)]."""
        eligible, seen, waiting = self._list()
        batches, total, more = [], 0, False
        unread = rejected = 0
        for path, st in eligible:
            key = f"{st.st_dev}:{st.st_ino}"
            e = files.get(key)
            if e is not None and e["done"] and e["size"] == st.st_size and e["mtime_ns"] == st.st_mtime_ns:
                if e["name"] != path.name:
                    e = dict(e, name=path.name)
                    files[key] = e
                    batches.append(([], 0, 0, None, {key: dict(e)}))
                continue
            if e is not None and not self._same_file(path, e, st):
                e = None
            if e is None:
                e = self._new_entry(path, st)
            reader = self._parquet_page if self._kind_of(path.name) == "parquet" else self._csv_page
            while True:
                if total >= max_records:
                    more = True
                    break
                n = min(self.batch_size, max_records - total)
                try:
                    recs, bad, rows, newest, e2 = reader(path, e, st, n)
                except _NotReady:
                    waiting += 1
                    break
                except _BadFile as exc:
                    rejected += 1
                    if key not in self._rejected_logged:
                        self._rejected_logged.add(key)
                        log.error("source_file_missing_columns", source=self.name, columns=exc.missing)
                    break
                except (OSError, ValueError) as exc:
                    self._fail(exc)
                    break
                files[key] = e2
                if rows or e2 != e:
                    batches.append((recs, bad, rows, newest, {key: copy.deepcopy(e2)}))
                total += rows
                progressed = e2["offset"] != e["offset"]
                e = e2
                if not progressed or rows < n:
                    break
            if not e["done"]:
                unread += max(0, e["size"] - e["offset"]) if self._kind_of(path.name) == "csv" else 1
        gone = {k: None for k in files if k not in seen}
        if gone:
            for k in gone:
                files.pop(k)
            batches.append(([], 0, 0, None, gone))
        self.last_scan = {"at": self._clock(), "eligible": len(eligible), "waiting": waiting,
                          "rejected": rejected, "unread_units": unread}
        self._more = more
        return batches

    def poll(self, max_records: int = 10_000) -> list[Record]:
        out: list[Record] = []
        now = self._clock()
        if self.blocked or (not self._more and now - self._last_scan < self.scan_interval):
            return out
        self._last_scan = now
        max_records = min(max_records, self.max_uncommitted_rows - self._staged_rows)
        try:
            for recs, bad, rows, newest, delta in self._advance(self._live, max_records):
                self.bad_rows += bad
                self._stage(newest, delta, rows)
                out.extend(recs)
            self._ok()
        except Exception as exc:                  # noqa: BLE001 -- an unreadable directory must not stop the agent
            self._fail(exc)
            log.warning("folder_source_unavailable", source=self.name, error_type=type(exc).__name__)
        return out

    def sample(self, limit: int = 1000) -> Sample:
        """Read up to `limit` rows from the committed position; writes and stages nothing.

        A source the agent has caught up with has nothing past its committed position (every night, after a stop):
        then the rows are read from the start of the files still there, ignoring the position, so a quiet healthy
        source is not reported as having no rows. A fresh install (nothing committed) reads from the start anyway."""
        try:
            batches = self._advance(copy.deepcopy(self._committed), limit)
            if self._committed and not any(b[0] or b[1] for b in batches):
                batches = self._advance({}, limit)
        except OSError as exc:
            raise ConfigError(f"cannot list the source directory ({type(exc).__name__})") from None
        recs = [r for b in batches for r in b[0]]
        bad = sum(b[1] for b in batches)
        if not recs and not bad and self.last_scan.get("rejected"):
            raise ConfigError("the files found do not contain the mapped columns "
                              f"{sorted(self.mapping.fixed_columns())}")
        return Sample(recs, bad)

    def describe(self) -> dict[str, Any]:
        files = self._committed
        return {"type": "folder", "name": self.name, "kind": self.kind,
                "files_tracked": len(files), "files_done": sum(1 for f in files.values() if f["done"]),
                "files_waiting": self.last_scan.get("waiting"),
                "files_rejected": self.last_scan.get("rejected"),
                "unread_units": self.last_scan.get("unread_units"),
                "rows_delivered": self.rows_delivered, "rows_committed": self.rows_committed,
                "rows_uncommitted": self._staged_rows, "bad_rows": self.bad_rows,
                "blocked": self.blocked, "last_error": self.last_error}
