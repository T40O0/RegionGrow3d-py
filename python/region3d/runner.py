"""Persistent active-run tracking for the Streamlit UI.

Streamlit's `st.session_state` is tied to the WebSocket connection — closing
the browser or letting the laptop sleep tears the session down and the next
visit starts from a blank state. The analysis subprocess, however, keeps
running because it is just a child of the long-lived Streamlit process.

To keep the UI usable across session boundaries we persist the running-run
information to disk:

  <out_root>/.active_run.json   manifest of the in-flight subprocess
  <out_root>/<susname>/_run.log driver stdout/stderr stream (line-buffered)
  <out_root>/.last_completed.json   pointer to the most recent successful run

The Streamlit script reads these files on every rerun, so a fresh browser
session reconnects to the same running subprocess (by PID) and shows the same
log tail / progress as the original session.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Optional, List, Dict, Any


# Filenames placed under the `out_root` directory (NOT under <susname>/).
ACTIVE_MANIFEST = '.active_run.json'
LAST_COMPLETED = '.last_completed.json'


# =============================================================================
#  Manifest read / write
# =============================================================================

def write_manifest(out_root: Path, data: Dict[str, Any]) -> None:
    """Atomically write the active-run manifest to <out_root>/.active_run.json."""
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    tmp = out_root / (ACTIVE_MANIFEST + '.tmp')
    tmp.write_text(json.dumps(data, indent=2), encoding='utf-8')
    tmp.replace(out_root / ACTIVE_MANIFEST)


def read_manifest(out_root: Path) -> Optional[Dict[str, Any]]:
    p = Path(out_root) / ACTIVE_MANIFEST
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return None


def clear_manifest(out_root: Path) -> None:
    p = Path(out_root) / ACTIVE_MANIFEST
    if p.exists():
        try:
            p.unlink()
        except Exception:
            pass


def write_last_completed(out_root: Path, data: Dict[str, Any]) -> None:
    """Record the most recent completed run so idle UI can show its results."""
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / LAST_COMPLETED).write_text(
        json.dumps(data, indent=2), encoding='utf-8')


def read_last_completed(out_root: Path) -> Optional[Dict[str, Any]]:
    p = Path(out_root) / LAST_COMPLETED
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return None


# =============================================================================
#  Process utilities (psutil preferred, with Windows/POSIX fallback)
# =============================================================================

def proc_create_time(pid: Optional[int]) -> Optional[float]:
    """Return the process creation time (epoch seconds), or None if unknown.

    Used to pin a PID to the *specific* process we launched so a recycled PID
    (the OS reassigns a dead run's PID to an unrelated process) is not mistaken
    for our run. Requires psutil; returns None when psutil is unavailable or the
    process is gone.
    """
    if not pid:
        return None
    try:
        import psutil
        return psutil.Process(int(pid)).create_time()
    except Exception:
        return None


def _identity_ok(pid: int, create_time: Optional[float]) -> bool:
    """True if `pid`'s creation time matches `create_time` (within 2 s).

    When create_time is None (caller didn't record it) or psutil is missing we
    can't verify identity, so we don't block on it — bare existence is the
    best available signal.
    """
    if create_time is None:
        return True
    try:
        import psutil
    except ImportError:
        return True
    try:
        return abs(psutil.Process(pid).create_time() - float(create_time)) < 2.0
    except Exception:
        return False


def _is_zombie(pid: int) -> bool:
    """True if `pid` is a zombie (defunct): it has exited but its parent has not
    reaped it, so it lingers in the process table holding nothing but its PID.

    Both `os.kill(pid, 0)` and `psutil.pid_exists()` report a zombie as existing,
    which made the UI hang on "Computing…" forever after a run finished: the
    detached orchestrator is reparented to PID 1 — the container's Streamlit
    process — which never wait()s for it, so its zombie never goes away.

    Works without psutil (the container image ships without it) by reading
    /proc directly. On Windows there are no zombies and no /proc, so this is
    always False there.
    """
    try:
        import psutil
        try:
            return psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False
    except ImportError:
        pass
    try:
        with open(f'/proc/{pid}/stat', 'rb') as fh:
            # "<pid> (<comm>) <state> ..." — comm can itself contain spaces and
            # parentheses, so the state is the first field after the LAST ')'.
            return fh.read().rsplit(b')', 1)[1].split()[0] == b'Z'
    except (OSError, IndexError):
        return False


def pid_alive(pid: Optional[int], create_time: Optional[float] = None) -> bool:
    """Whether `pid` is running AND (if create_time is given) is the same
    process we launched — guards against PID reuse reconnecting to a stranger.
    """
    if not pid:
        return False
    if _is_zombie(int(pid)):
        return False
    try:
        import psutil
        if not psutil.pid_exists(int(pid)):
            return False
        return _identity_ok(int(pid), create_time)
    except ImportError:
        pass
    if os.name == 'nt':
        try:
            out = subprocess.check_output(
                ['tasklist', '/FI', f'PID eq {int(pid)}', '/NH', '/FO', 'CSV'],
                stderr=subprocess.DEVNULL).decode('utf-8', errors='ignore')
            return f'"{int(pid)}"' in out
        except Exception:
            return False
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but is owned by another user — it IS alive.
        return True


def kill_pid(pid: Optional[int], timeout: float = 5.0,
             create_time: Optional[float] = None) -> None:
    """Terminate `pid` and its child tree. If create_time is given, refuse to
    kill unless the PID's creation time matches — never kill a recycled PID
    that now belongs to an unrelated process.
    """
    if not pid:
        return
    pid = int(pid)
    if not _identity_ok(pid, create_time):
        return
    try:
        import psutil
        try:
            p = psutil.Process(pid)
            # Kill the process tree so child subprocesses (numba JIT helpers,
            # rasterio threads, etc.) don't outlive the driver.
            for child in p.children(recursive=True):
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            p.terminate()
            try:
                p.wait(timeout=timeout)
            except psutil.TimeoutExpired:
                for child in p.children(recursive=True):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                p.kill()
        except psutil.NoSuchProcess:
            pass
        return
    except ImportError:
        pass
    if os.name == 'nt':
        subprocess.run(['taskkill', '/F', '/T', '/PID', str(pid)],
                       stderr=subprocess.DEVNULL,
                       stdout=subprocess.DEVNULL)
    else:
        import signal
        try:
            os.kill(pid, signal.SIGTERM)
            t0 = time.time()
            while time.time() - t0 < timeout and pid_alive(pid):
                time.sleep(0.2)
            if pid_alive(pid):
                os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


# =============================================================================
#  Log file tailing
# =============================================================================

def tail_log(path: Path, max_lines: int = 25, max_bytes: int = 256 * 1024
             ) -> List[str]:
    """Return the last `max_lines` non-empty lines of a text log file.

    Reads at most `max_bytes` from the file tail to keep the cost bounded for
    very long-running jobs whose log grows into the hundreds of MB.
    """
    p = Path(path)
    if not p.exists():
        return []
    try:
        sz = p.stat().st_size
        with open(p, 'rb') as f:
            if sz > max_bytes:
                f.seek(sz - max_bytes)
                # discard the (likely partial) first line
                f.readline()
            data = f.read().decode('utf-8', errors='replace')
        lines = data.splitlines()
        return lines[-max_lines:]
    except Exception:
        return []
