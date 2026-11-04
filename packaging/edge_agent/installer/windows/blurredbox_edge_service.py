"""Windows service host for the BlurredBox edge agent (pywin32). NOT TESTED on real Windows.

Runs `python -m edge_agent --config <config.json>` as a child process and restarts it if it dies; on service
stop it sends CTRL_BREAK so the agent stops cleanly and keeps its open window in the state (as SIGTERM does on Linux), then waits.

    python blurredbox_edge_service.py install|remove|start|stop      (install.ps1 does this)
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("ProgramData", r"C:\ProgramData")) / "BlurredBoxEdge"
APP, CONF, DATA = ROOT / "app", ROOT / "conf", ROOT / "data"
# Math-library threads capped at 2 (G60): same output, 15-60x faster under a CPU limit
# (docs/measurements/edge-classifier-cost-2026-10-29.md). Defaults only: the service's environment and agent.env win.
THREAD_CAP_DEFAULTS = {"OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}


def load_env(path: Path) -> dict[str, str]:
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, _, v = line.partition("=")
                out[k.strip()] = v.strip()
    return out


def child_env(base: dict[str, str] | None = None, env_file: Path | None = None) -> dict[str, str]:
    """The agent's environment: thread-cap defaults, then the service's environment, then agent.env."""
    base = dict(os.environ) if base is None else base
    return {**THREAD_CAP_DEFAULTS, **base, **load_env(CONF / "agent.env" if env_file is None else env_file)}


def run_agent_forever(stop_requested) -> None:
    py = APP / "venv" / "Scripts" / "python.exe"
    cmd = [str(py), "-m", "edge_agent", "--config", str(CONF / "config.json")]
    while not stop_requested():
        env = child_env()
        child = subprocess.Popen(cmd, env=env, cwd=str(DATA), creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        while child.poll() is None:
            if stop_requested():
                child.send_signal(signal.CTRL_BREAK_EVENT)
                try:
                    child.wait(60)
                except subprocess.TimeoutExpired:
                    child.kill()
                return
            time.sleep(1)
        if not stop_requested():
            time.sleep(10)          # Restart=on-failure, RestartSec


try:
    import servicemanager
    import win32event
    import win32service
    import win32serviceutil

    class EdgeService(win32serviceutil.ServiceFramework):
        _svc_name_ = "blurredbox-edge"
        _svc_display_name_ = "BlurredBox edge agent"
        _svc_description_ = "Computes per-window statistics locally and sends only those to BlurredBox."

        def __init__(self, args):
            super().__init__(args)
            self._stop = win32event.CreateEvent(None, 0, 0, None)

        def SvcStop(self):
            self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
            win32event.SetEvent(self._stop)

        def SvcDoRun(self):
            servicemanager.LogInfoMsg("blurredbox-edge starting")
            run_agent_forever(lambda: win32event.WaitForSingleObject(self._stop, 0) == win32event.WAIT_OBJECT_0)

    if __name__ == "__main__":
        if len(sys.argv) == 1:
            servicemanager.Initialize()
            servicemanager.PrepareToHostSingle(EdgeService)
            servicemanager.StartServiceCtrlDispatcher()
        else:
            win32serviceutil.HandleCommandLine(EdgeService)
except ImportError:      # not on Windows / pywin32 absent: module still imports for tests
    pass
