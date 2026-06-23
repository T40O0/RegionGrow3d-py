"""Compare two slope-unit (label) rasters on the SAME grid.

Used to verify the Python port against GRASS r.slopeunits / r.watershed hbasin:
both must be exported to GeoTIFF on the identical grid (same DEM, same region).

Metrics (label ids need NOT match between the two — these are partition-
similarity measures, not per-id equality):

  * n_units on each side
  * Adjusted Rand Index (ARI)        1 = identical partition, 0 = random
  * Variation of Information (VI)     0 = identical, larger = more different
  * mean best-match IoU (A->B, B->A)  how well each unit overlaps its best
                                      counterpart
  * boundary F-score (1-px, 4-conn)   agreement of the unit-boundary lines

Usage:
  python compare_partitions.py A.tif B.tif [--mask valid.tif]
"""
from __future__ import annotations
import argparse, sys
import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


def _read(path):
    import rasterio
    with rasterio.open(path) as s:
        return s.read(1)


def _boundaries(lbl):
    b = np.zeros(lbl.shape, bool)
    b[:-1, :] |= lbl[:-1, :] != lbl[1:, :]
    b[1:, :] |= lbl[:-1, :] != lbl[1:, :]
    b[:, :-1] |= lbl[:, :-1] != lbl[:, 1:]
    b[:, 1:] |= lbl[:, :-1] != lbl[:, 1:]
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('a'); ap.add_argument('b')
    ap.add_argument('--mask', default='')
    ap.add_argument('--dilate', type=int, default=1,
                    help='boundary match tolerance in cells')
    args = ap.parse_args()

    A = _read(args.a).astype(np.int64)
    B = _read(args.b).astype(np.int64)
    if A.shape != B.shape:
        raise SystemExit(f"grid mismatch {A.shape} vs {B.shape}")
    valid = np.isfinite(A) & (A > 0) & (B > 0)
    if args.mask:
        valid &= _read(args.mask).astype(bool)
    a = A[valid]; b = B[valid]
    na = np.unique(a).size; nb = np.unique(b).size
    print(f"n_units  A={na}  B={nb}  (valid cells={a.size:,})")

    # contingency table via sparse pairing
    ai = np.unique(a, return_inverse=True)[1]
    bi = np.unique(b, return_inverse=True)[1]
    n = a.size
    pair = ai.astype(np.int64) * nb + bi
    cont = np.bincount(pair, minlength=na * nb).reshape(na, nb).astype(np.float64)
    sa = cont.sum(1); sb = cont.sum(0)

    # Adjusted Rand Index
    from math import comb
    def C2(x):
        return x * (x - 1) / 2.0
    sum_c = C2(cont).sum()
    sum_a = C2(sa).sum(); sum_b = C2(sb).sum()
    tot = C2(np.array([n], dtype=np.float64))[0]
    exp = sum_a * sum_b / tot
    ari = (sum_c - exp) / (0.5 * (sum_a + sum_b) - exp) if (0.5*(sum_a+sum_b)-exp) else 1.0
    print(f"Adjusted Rand Index : {ari:.4f}   (1=identical)")

    # Variation of Information (bits)
    pa = sa / n; pb = sb / n; pij = cont / n
    Ha = -np.sum(pa[pa > 0] * np.log2(pa[pa > 0]))
    Hb = -np.sum(pb[pb > 0] * np.log2(pb[pb > 0]))
    nz = pij > 0
    MI = np.sum(pij[nz] * np.log2(pij[nz] / (pa[:, None] * pb[None, :])[nz]))
    VI = Ha + Hb - 2 * MI
    print(f"Variation of Info   : {VI:.4f} bits (0=identical)")

    # mean best-match IoU
    inter = cont
    union_ab = sa[:, None] + sb[None, :] - inter
    iou = np.where(union_ab > 0, inter / union_ab, 0.0)
    best_a = iou.max(1); best_b = iou.max(0)
    print(f"mean best IoU A->B  : {np.average(best_a, weights=sa):.4f}")
    print(f"mean best IoU B->A  : {np.average(best_b, weights=sb):.4f}")

    # boundary F-score with tolerance
    from scipy import ndimage as ndi
    ba = _boundaries(A) & valid
    bb = _boundaries(B) & valid
    d = max(int(args.dilate), 0)
    if d:
        ba_d = ndi.binary_dilation(ba, iterations=d)
        bb_d = ndi.binary_dilation(bb, iterations=d)
    else:
        ba_d, bb_d = ba, bb
    prec = (ba & bb_d).sum() / max(ba.sum(), 1)
    rec = (bb & ba_d).sum() / max(bb.sum(), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    print(f"boundary F (tol={d}px) : {f1:.4f}  (P={prec:.3f} R={rec:.3f})")


if __name__ == '__main__':
    main()
