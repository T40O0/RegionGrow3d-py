"""MATLAB-compatible bwmorph implementations.

MATLAB's bwmorph applies a 3x3 lookup-table operation per iteration. The 9-bit
neighborhood index uses the column-major weighting:

    1   8   64
    2  16  128
    4  32  256

That is, weight = 2**(col*3 + row) for cell (row, col) in the 3x3 patch.

Each operation has a fixed 512-entry LUT. We compute the LUTs from documented
predicates so that the implementation matches MATLAB's behavior precisely.

Boundary handling: cells outside the image are treated as 0 (MATLAB default).
"""
from __future__ import annotations

import numpy as np


# ---- Neighborhood encoding ----------------------------------------------------
# Weight matrix matching MATLAB column-major neighborhood encoding.
_WEIGHTS = np.array([[1, 8, 64],
                     [2, 16, 128],
                     [4, 32, 256]], dtype=np.int32)


def _decode_index(idx: int) -> np.ndarray:
    """Decode 9-bit MATLAB neighborhood index into a 3x3 boolean array."""
    nbh = np.zeros((3, 3), dtype=bool)
    for r in range(3):
        for c in range(3):
            w = _WEIGHTS[r, c]
            nbh[r, c] = bool(idx & w)
    return nbh


def _make_lut(predicate) -> np.ndarray:
    """Build a 512-entry boolean LUT from a predicate(nbh: 3x3 bool) -> bool."""
    lut = np.zeros(512, dtype=bool)
    for i in range(512):
        nbh = _decode_index(i)
        lut[i] = predicate(nbh)
    return lut


# ---- LUT predicates -----------------------------------------------------------

def _clean_pred(nbh):
    """Remove isolated 1-pixels: center=1 and all 8 neighbors=0 -> 0."""
    center = nbh[1, 1]
    if not center:
        return False
    n_ones = nbh.sum() - 1  # exclude center
    if n_ones == 0:
        return False  # remove
    return True  # keep


def _fill_pred(nbh):
    """Fill isolated 0-pixels: center=0 and all 8 neighbors=1 -> 1."""
    center = nbh[1, 1]
    if center:
        return True
    n_ones = nbh.sum()  # center is 0
    return n_ones == 8


def _majority_pred(nbh):
    """Set to 1 iff 5 or more of the 9 cells are 1."""
    return nbh.sum() >= 5


def _spur_pred(nbh):
    """Remove pixels with at most one 8-connected neighbor (single-pass).

    A 1-pixel with 0 or 1 nonzero neighbors is treated as a spur and removed.
    Background pixels are kept as-is.
    """
    center = nbh[1, 1]
    if not center:
        return False
    n_ones = nbh.sum() - 1
    if n_ones <= 1:
        return False  # remove
    return True


# 8-neighbor adjacency for the bridge connected-component test.
# Positions in a 3x3 patch (excluding center), labelled 0..7:
#   0 1 2
#   3 . 4
#   5 6 7
_NEIGH_POS = [(0, 0), (0, 1), (0, 2),
              (1, 0),         (1, 2),
              (2, 0), (2, 1), (2, 2)]
_ADJ = [[] for _ in range(8)]
for _i in range(8):
    for _j in range(8):
        if _i == _j:
            continue
        _r1, _c1 = _NEIGH_POS[_i]
        _r2, _c2 = _NEIGH_POS[_j]
        if max(abs(_r1 - _r2), abs(_c1 - _c2)) == 1:
            _ADJ[_i].append(_j)


def _count_components_8(neigh_bits):
    """Count connected components among the 8 neighbors using 8-connectivity."""
    seen = [False] * 8
    n_components = 0
    for s in range(8):
        if not neigh_bits[s] or seen[s]:
            continue
        n_components += 1
        stack = [s]
        while stack:
            v = stack.pop()
            if seen[v]:
                continue
            seen[v] = True
            for u in _ADJ[v]:
                if neigh_bits[u] and not seen[u]:
                    stack.append(u)
    return n_components


def _bridge_pred(nbh):
    """Set center=1 iff center=0 and 8-neighbors split into >=2 connected components."""
    center = nbh[1, 1]
    if center:
        return True
    bits = [
        nbh[0, 0], nbh[0, 1], nbh[0, 2],
        nbh[1, 0],            nbh[1, 2],
        nbh[2, 0], nbh[2, 1], nbh[2, 2],
    ]
    return _count_components_8(bits) >= 2


def _diag_pred(nbh):
    """MATLAB `bwmorph(BW, 'diag')`: diagonal fill to break 8-connectivity of bg.

    A 0-pixel becomes 1 if it has two 4-cardinal neighbours that are both 1 and
    the diagonal between them is 0, i.e. one of these L-shaped corners:

        . 1 .       . 1 .       . . .       . . .
        1 0 .       . 0 1       1 0 .       . 0 1
        . . .       . . .       . 1 .       . 1 .
    """
    center = nbh[1, 1]
    if center:
        return True
    N = bool(nbh[0, 1]); S = bool(nbh[2, 1])
    E = bool(nbh[1, 2]); W = bool(nbh[1, 0])
    NW = bool(nbh[0, 0]); NE = bool(nbh[0, 2])
    SW = bool(nbh[2, 0]); SE = bool(nbh[2, 2])
    return ((N and W and not NW)
            or (N and E and not NE)
            or (S and W and not SW)
            or (S and E and not SE))


def _zhang_suen_neighbours(nbh):
    """Return (P2..P9) = neighbours of the centre in clockwise order from N.

    Zhang-Suen labels pixels as P1=center, P2=N, P3=NE, P4=E, P5=SE, P6=S,
    P7=SW, P8=W, P9=NW. The 3x3 patch indexing in (row, col) is:
        nbh[0,1] = N, nbh[0,2] = NE, nbh[1,2] = E, nbh[2,2] = SE,
        nbh[2,1] = S, nbh[2,0] = SW, nbh[1,0] = W, nbh[0,0] = NW.
    """
    return (bool(nbh[0, 1]),  # P2 = N
            bool(nbh[0, 2]),  # P3 = NE
            bool(nbh[1, 2]),  # P4 = E
            bool(nbh[2, 2]),  # P5 = SE
            bool(nbh[2, 1]),  # P6 = S
            bool(nbh[2, 0]),  # P7 = SW
            bool(nbh[1, 0]),  # P8 = W
            bool(nbh[0, 0]))  # P9 = NW


def _skel1_pred(nbh):
    """First sub-iteration of MATLAB-style skeletonisation (Zhang-Suen Pass A).

    A foreground pixel P is REMOVED in pass A iff all of:
        (a) 2 <= B(P) <= 6                         (not an endpoint, not full)
        (b) A(P) == 1     (exactly one 0->1 transition in clockwise neighbours)
        (c) P2 * P4 * P6 == 0     (at least one of N, E, S is background)
        (d) P4 * P6 * P8 == 0     (at least one of E, S, W is background)

    The LUT entry is "the centre AFTER one pass" — True (keep), False (remove).
    Background centres stay background.
    """
    center = bool(nbh[1, 1])
    if not center:
        return False
    P2, P3, P4, P5, P6, P7, P8, P9 = _zhang_suen_neighbours(nbh)
    seq = (P2, P3, P4, P5, P6, P7, P8, P9)
    B = sum(seq)
    if B < 2 or B > 6:
        return True
    A = sum(1 for k in range(8) if (not seq[k]) and seq[(k + 1) % 8])
    if A != 1:
        return True
    if P2 and P4 and P6:
        return True
    if P4 and P6 and P8:
        return True
    return False  # remove


def _skel2_pred(nbh):
    """Second sub-iteration of MATLAB-style skeletonisation (Zhang-Suen Pass B).

    Same as pass A except the corner conditions check the OTHER diagonal:
        (c') P2 * P4 * P8 == 0     (at least one of N, E, W is background)
        (d') P2 * P6 * P8 == 0     (at least one of N, S, W is background)
    """
    center = bool(nbh[1, 1])
    if not center:
        return False
    P2, P3, P4, P5, P6, P7, P8, P9 = _zhang_suen_neighbours(nbh)
    seq = (P2, P3, P4, P5, P6, P7, P8, P9)
    B = sum(seq)
    if B < 2 or B > 6:
        return True
    A = sum(1 for k in range(8) if (not seq[k]) and seq[(k + 1) % 8])
    if A != 1:
        return True
    if P2 and P4 and P8:
        return True
    if P2 and P6 and P8:
        return True
    return False  # remove


# Build LUTs at import time (cheap: 512 iterations each)
LUT_CLEAN = _make_lut(_clean_pred)
LUT_FILL = _make_lut(_fill_pred)
LUT_MAJORITY = _make_lut(_majority_pred)
LUT_SPUR = _make_lut(_spur_pred)
LUT_BRIDGE = _make_lut(_bridge_pred)
LUT_DIAG = _make_lut(_diag_pred)
LUT_SKEL1 = _make_lut(_skel1_pred)
LUT_SKEL2 = _make_lut(_skel2_pred)
# Backward-compat alias: anyone importing the old single LUT gets pass A.
LUT_SKEL = LUT_SKEL1


_LUTS = {
    'clean': LUT_CLEAN,
    'fill': LUT_FILL,
    'majority': LUT_MAJORITY,
    'spur': LUT_SPUR,
    'bridge': LUT_BRIDGE,
    'diag': LUT_DIAG,
    # 'skel' is special-cased in bwmorph(): two LUTs alternated per iteration.
}


def _apply_lut(bw: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Apply a 512-entry LUT to a binary image using a column-major encoded index."""
    bw = np.asarray(bw, dtype=bool)
    # Compute index image via convolution-like sum. We build it manually using padding.
    pad = np.zeros((bw.shape[0] + 2, bw.shape[1] + 2), dtype=np.int32)
    pad[1:-1, 1:-1] = bw.astype(np.int32)
    idx = np.zeros(bw.shape, dtype=np.int32)
    for r in range(3):
        for c in range(3):
            w = int(_WEIGHTS[r, c])
            idx += pad[r:r + bw.shape[0], c:c + bw.shape[1]] * w
    return lut[idx]


def bwmorph(bw, op: str, n: int = 1):
    """MATLAB-compatible bwmorph.

    Supported ops: 'clean', 'fill', 'majority', 'spur', 'bridge', 'diag', 'skel'.

    For 'skel', each iteration applies LUT_SKEL1 then LUT_SKEL2 (Zhang-Suen
    parallel thinning). The loop halts as soon as both sub-passes are
    idempotent, so the user can pass `n=Inf` (or any large integer) to mirror
    MATLAB's `bwmorph(BW, 'skel', Inf)` to convergence.
    """
    if op == 'skel':
        if n is None or n <= 0:
            return np.asarray(bw, dtype=bool).copy()
        if n == np.inf:
            n_iter = 10**9
        else:
            n_iter = int(n)
        out = np.asarray(bw, dtype=bool)
        for _ in range(n_iter):
            after_a = _apply_lut(out, LUT_SKEL1)
            after_b = _apply_lut(after_a, LUT_SKEL2)
            if np.array_equal(after_b, out):
                return after_b
            out = after_b
        return out

    if op not in _LUTS:
        raise ValueError(f"Unsupported bwmorph op: {op!r}")
    if n is None or n <= 0:
        return np.asarray(bw, dtype=bool).copy()
    lut = _LUTS[op]
    out = np.asarray(bw, dtype=bool)
    for _ in range(int(n)):
        new = _apply_lut(out, lut)
        if np.array_equal(new, out):
            return new
        out = new
    return out


def bwareaopen(bw, min_size: int):
    """MATLAB bwareaopen with default 8-connectivity: drop components smaller than min_size."""
    from scipy.ndimage import label

    bw = np.asarray(bw, dtype=bool)
    structure = np.ones((3, 3), dtype=bool)
    lbl, n = label(bw, structure=structure)
    if n == 0:
        return bw.copy()
    counts = np.bincount(lbl.ravel(), minlength=n + 1)
    keep = counts >= min_size
    keep[0] = False  # background
    return keep[lbl]
