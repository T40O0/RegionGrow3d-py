"""Windows-only speedup for scipy's Qhull wrappers (Delaunay / ConvexHull).

Why this exists
---------------
`scipy._lib.messagestream.MessageStream` captures Qhull's C-level diagnostic
output by handing it a real file. On Linux it uses an in-memory stream
(`open_memstream`), but Windows has no `open_memstream`, so scipy falls back to
creating a fresh on-disk temp file via `tempfile.mkstemp` on EVERY Qhull call
and deleting it on close. With Windows Defender real-time scanning, each
create/close/delete of that throwaway file costs on the order of ~3 ms.

`alpha_shape_boundary` calls `Delaunay` once per cluster and again on every
region-grow cycle, so on the Noto 5 m run (tens of thousands of clusters) this
temp-file churn dominates wall-clock — measured at ~87 % of Delaunay time here,
i.e. Delaunay is ~8x faster once it is removed.

The fix
-------
Inside a scoped context, redirect `tempfile.mkstemp` to return repeated `dup()`s
of ONE persistent temp file (truncated per call) instead of creating a new file
each time, and make `os.remove`/`os.unlink` skip that file. The captured Qhull
messages are only ever read back on the Qhull *error* path (to build an
exception message); on the normal path they are discarded, so reusing the buffer
is numerically inert. Byte-identical Delaunay output has been verified against
the unpatched path over a battery of random and integer-grid point sets.

Scope & safety
--------------
- Enabled on Windows only; a no-op elsewhere (Linux already uses memory).
- Installed only for the duration of the `with fast_qhull_tempfile():` block that
  wraps the Delaunay/ConvexHull calls in `boundary.py` — no global side effects.
- The only other `tempfile` users in this project call `mkdtemp` (GRASS path),
  which is untouched.
- One persistent temp file is created lazily per process and removed at exit.
- region-grow runs single-threaded per process (parallelism is across processes
  via ``driver.py --run-index``), so no two threads share the reused buffer; an
  RLock guards the swap defensively regardless.
"""
from __future__ import annotations

import atexit
import os
import sys
import tempfile
import threading

_ENABLED = sys.platform == "win32"

_orig_mkstemp = tempfile.mkstemp
_orig_remove = os.remove
_orig_unlink = os.unlink

_lock = threading.RLock()
_depth = 0          # re-entrancy counter (nested with-blocks reuse one install)
_pfd = None         # persistent file descriptor
_ppath_str = None   # persistent file path, canonical str form


def _ensure_file():
    """Lazily create the single reusable temp file (once per process)."""
    global _pfd, _ppath_str
    if _pfd is None:
        fd, path = _orig_mkstemp(prefix="rg3d_qhull_")
        _pfd, _ppath_str = fd, os.fsdecode(path)
        atexit.register(_cleanup)
    return _pfd, _ppath_str


def _cleanup():
    global _pfd, _ppath_str
    fd, path = _pfd, _ppath_str
    _pfd = _ppath_str = None
    if fd is not None:
        try:
            os.close(fd)
        except OSError:
            pass
    if path is not None:
        try:
            _orig_remove(path)
        except OSError:
            pass


def _fast_mkstemp(suffix=None, prefix=None, dir=None, text=False):
    """Hand scipy a fresh fd onto the one reused, truncated temp file.

    scipy's MessageStream calls mkstemp with bytes arguments and expects a bytes
    path back; other callers expect str. Return the persistent path in whichever
    type the caller asked for (mirroring tempfile's own str/bytes inference).
    """
    fd, path_str = _ensure_file()
    os.lseek(fd, 0, os.SEEK_SET)
    try:
        os.ftruncate(fd, 0)
    except OSError:
        pass
    wants_bytes = any(isinstance(a, bytes) for a in (suffix, prefix, dir))
    path = os.fsencode(path_str) if wants_bytes else path_str
    return os.dup(fd), path


def _is_pooled(path) -> bool:
    if _ppath_str is None or path is None:
        return False
    try:
        return os.fsdecode(path) == _ppath_str
    except (TypeError, ValueError):
        return False


def _skip_remove(path, *args, **kwargs):
    if path is None or _is_pooled(path):
        return  # keep the reused buffer alive (None: nothing to remove)
    return _orig_remove(path, *args, **kwargs)


class fast_qhull_tempfile:
    """Context manager that installs the reuse patch around a Qhull call.

    No-op on non-Windows platforms. Re-entrant: nested blocks share one install
    and only the outermost restores the originals.
    """

    def __enter__(self):
        global _depth
        if not _ENABLED:
            return self
        _lock.acquire()
        if _depth == 0:
            tempfile.mkstemp = _fast_mkstemp
            os.remove = _skip_remove
            os.unlink = _skip_remove
        _depth += 1
        return self

    def __exit__(self, *exc):
        global _depth
        if not _ENABLED:
            return False
        _depth -= 1
        if _depth == 0:
            tempfile.mkstemp = _orig_mkstemp
            os.remove = _orig_remove
            os.unlink = _orig_unlink
        _lock.release()
        return False
