#!/usr/bin/env python3
"""Mirror the edge agent from a BlurredBox release tag into this repository.

    python3 scripts/release_from_agd.py --agd /path/to/AGD-checkout --tag v0.0.2-rc6

Reads the tag with `git archive` (the working tree of the checkout is never used), so what lands here is exactly
what the tag contains. Replaced wholesale: edge_agent/, packaging/edge_agent/, tests/edge_agent/,
contracts/signal_vector/, and the corpus/ modules that edge_agent imports (found by following `corpus` imports from
edge_agent/ and tests/edge_agent/). pyproject.toml and requirements-edge.txt come from packaging/edge_agent/, with the
licence field pointed at LICENSE and the pytest settings kept. Files that belong to this repository (README.md,
LICENSE, SECURITY.md, scripts/, .github/, docs/) are never touched. Review `git diff`, run the tests, commit.
"""
from __future__ import annotations

import argparse
import ast
import io
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MIRRORED_DIRS = ["edge_agent", "packaging/edge_agent", "tests/edge_agent", "contracts/signal_vector"]
PYTEST_BLOCK = '\n[tool.pytest.ini_options]\ntestpaths = ["tests"]\npythonpath = ["."]\n'


def corpus_modules_used(roots: list[Path], corpus_dir: Path) -> set[str]:
    """Names of corpus/<name>.py reachable from the .py files under `roots` (transitively)."""
    available = {p.stem for p in corpus_dir.glob("*.py")}
    seen: set[str] = set()

    def imports(path: Path) -> set[str]:
        found: set[str] = set()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                parts = node.module.split(".")
                if parts[0] == "corpus":
                    if len(parts) > 1:
                        found.add(parts[1])
                    else:
                        found.update(a.name for a in node.names)
            elif isinstance(node, ast.Import):
                for a in node.names:
                    parts = a.name.split(".")
                    if parts[0] == "corpus" and len(parts) > 1:
                        found.add(parts[1])
        return found & available

    queue = [f for root in roots for f in root.rglob("*.py")]
    while queue:
        for name in imports(queue.pop()) - seen:
            seen.add(name)
            queue.append(corpus_dir / f"{name}.py")
    return seen | {"__init__"}


# Generic patterns live here. Specific internal names (hosts, people, address prefixes) must not be written in this
# public file: the release operator sets EDGE_LEAK_PATTERNS locally (newline-separated regexes) and CI reads it from
# a repository secret. Unset: a warning, not a failure.
GENERIC_PATTERNS = [re.compile(p) for p in (r"/home/", r"/Users/")]
IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
# documentation ranges (RFC 5737), loopback and the unspecified address are not leaks
SAFE_IPV4 = re.compile(r"^(127\.|0\.0\.0\.0$|192\.0\.2\.|198\.51\.100\.|203\.0\.113\.)")
LEAK_EXEMPT = {"SECURITY.md"}            # the published disclosure contact
LEAK_SCAN = ["edge_agent", "packaging", "tests", "contracts", "corpus", "scripts", "README.md", "pyproject.toml",
             "requirements-edge.txt", ".github"]


def leak_patterns() -> list[re.Pattern[str]]:
    extra = os.environ.get("EDGE_LEAK_PATTERNS", "")
    if not extra.strip():
        print("warning: EDGE_LEAK_PATTERNS is not set; only the generic leak checks ran", file=sys.stderr)
    return GENERIC_PATTERNS + [re.compile(line.strip()) for line in extra.splitlines() if line.strip()]


def find_leaks(root: Path) -> list[str]:
    """Internal hostnames, paths, addresses and names that must not be in the public tree."""
    patterns = leak_patterns()
    hits: list[str] = []
    files = [f for name in LEAK_SCAN for f in ([root / name] if (root / name).is_file() else (root / name).rglob("*"))]
    for f in sorted(files):
        rel = f.relative_to(root).as_posix()
        if not f.is_file() or rel in LEAK_EXEMPT or f.suffix in {".pkl", ".png"} or rel == "scripts/release_from_agd.py":
            continue
        try:
            text = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for n, line in enumerate(text.splitlines(), 1):
            bad = [p.pattern if p in GENERIC_PATTERNS else "configured pattern" for p in patterns if p.search(line)]
            bad += [ip for ip in IPV4.findall(line) if not SAFE_IPV4.match(ip)]
            if bad:
                hits.append(f"{rel}:{n}: {', '.join(bad)}")
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agd", help="path to a git checkout holding the tag")
    ap.add_argument("--tag")
    ap.add_argument("--dry-run", action="store_true", help="print the file set and change nothing")
    ap.add_argument("--check-leaks", action="store_true",
                    help="only scan this tree for internal addresses, paths and names; exit 1 if any")
    a = ap.parse_args()
    if a.check_leaks:
        hits = find_leaks(REPO)
        print("\n".join(hits) or "no leaks found")
        return 1 if hits else 0
    if not a.agd or not a.tag:
        ap.error("--agd and --tag are required")
    if not re.fullmatch(r"v\d+\.\d+\.\d+(-[A-Za-z0-9.]+)?", a.tag):
        ap.error("--tag must look like v0.0.2-rc6")
    commit = subprocess.run(["git", "-C", a.agd, "rev-parse", "--verify", f"{a.tag}^{{commit}}"],
                            capture_output=True, text=True)
    if commit.returncode:
        sys.exit(f"tag {a.tag} not found in {a.agd}")
    sha = commit.stdout.strip()
    paths = [*MIRRORED_DIRS, "corpus", "packaging/edge_agent/pyproject.toml", "packaging/edge_agent/requirements-edge.txt"]
    raw = subprocess.run(["git", "-C", a.agd, "archive", sha, *paths], capture_output=True, check=True).stdout
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp)
        with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
            tar.extractall(src, filter="data")
        used = corpus_modules_used([src / "edge_agent", src / "tests/edge_agent"], src / "corpus")
        print(f"corpus modules: {', '.join(sorted(used - {'__init__'}))}")
        if a.dry_run:
            return 0
        for rel in MIRRORED_DIRS:
            shutil.rmtree(REPO / rel, ignore_errors=True)
            shutil.copytree(src / rel, REPO / rel, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.rmtree(REPO / "corpus", ignore_errors=True)
        (REPO / "corpus").mkdir()
        for name in sorted(used):
            shutil.copy2(src / "corpus" / f"{name}.py", REPO / "corpus" / f"{name}.py")
        shutil.copy2(src / "packaging/edge_agent/requirements-edge.txt", REPO / "requirements-edge.txt")
        text = (src / "packaging/edge_agent/pyproject.toml").read_text(encoding="utf-8")
        text = re.sub(r'^license = \{.*\}$', 'license = {file = "LICENSE"}', text, flags=re.M)
        if "[tool.pytest.ini_options]" not in text:
            text = text.rstrip("\n") + "\n" + PYTEST_BLOCK
        (REPO / "pyproject.toml").write_text(text, encoding="utf-8")
    hits = find_leaks(REPO)
    if hits:
        print("LEAK CHECK FAILED: the mirrored tree contains internal strings; fix them in the source repository, "
              "or revert this sync:\n" + "\n".join(hits), file=sys.stderr)
        return 1
    print(f"mirrored {a.tag} ({sha[:9]}). Next: run the tests, update the README status line, commit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
