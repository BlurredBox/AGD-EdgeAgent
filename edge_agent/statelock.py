"""One agent per state directory (audit F23).

Two agents sharing `source.offset`, the spool and the sqlite join split rows and overwrite windows. The
running agent holds an exclusive advisory lock on `<state_dir>/agent.lock` for its whole life; a second
live run exits at once. Read-only commands (`--check`, `--status`) do not take it. The lock is released by
the kernel when the process dies, so a crash never leaves a stale lock. Where `fcntl` is absent (Windows)
`msvcrt.locking` is used.
"""
from __future__ import annotations

import errno
import os
from pathlib import Path

LOCK_FILE = "agent.lock"


class StateLockError(RuntimeError):
    """The lock could not be taken for a reason other than another agent holding it (no lock support on this
    filesystem, out of locks, I/O error)."""


class StateDirBusy(StateLockError):
    """Another agent process holds this state directory."""


def acquire(state_dir: str | Path):
    """Take the lock; returns the open file (keep it referenced until exit). Raises StateDirBusy."""
    d = Path(state_dir)
    d.mkdir(parents=True, exist_ok=True)
    fd = os.open(d / LOCK_FILE, os.O_RDWR | os.O_CREAT, 0o600)
    fh = os.fdopen(fd, "r+")
    try:
        try:
            import fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except ImportError:
            import msvcrt
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError as exc:
        fh.close()
        if exc.errno not in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK):
            raise StateLockError(f"cannot lock {d / LOCK_FILE}: {exc.strerror or exc} (errno {exc.errno}); the state "
                                 "directory must be on a filesystem that supports file locks") from exc
        raise StateDirBusy(f"another edge agent is already running on {d} (lock {LOCK_FILE} is held); "
                           "stop it first, or use --check / --status, which do not need the lock") from exc
    fh.seek(0)
    fh.truncate()
    fh.write(str(os.getpid()))
    fh.flush()
    return fh
