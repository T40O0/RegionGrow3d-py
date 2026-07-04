"""Parallel susceptibility runner (phi fan-out) via driver.py --run-index.

Runs the N phi runs of a soil_strength_mode=1 job as up to MAXCONC concurrent
processes (each banks its own contribution), then combines them with
`driver.py --aggregate`. The final map is a commutative weighted sum, so the
result is bit-identical to the serial loop. Resume-safe: a run whose contribution
already exists is skipped.

Two entry modes:
  * CLI (no args): uses the hard-coded a20k/test-999 parameter set below.
  * GUI (--gui-args <json>): the Streamlit UI writes a JSON describing the
    driver command it built; this process fans it out and streams combined
    progress to stdout (which the UI has redirected to the run's _run.log).

Platform-agnostic: derives repo root from __file__, uses sys.executable, so the
same file runs natively (Windows) and inside the Docker/Linux container.
Env overrides: SUS_MAXCONC, SUS_STAGGER (seconds).
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

MAXCONC = int(os.environ.get("SUS_MAXCONC", "2"))
# Stagger new launches so two runs never hit their heavy pre-compute peak
# (DEM load + derivatives + rotation forces, ~24 GB RSS) at the same time —
# that is what OOM-killed the pool at STAGGER=60. By ~240 s the earlier run is
# in the steady cluster loop (~21 GB), so the peaks don't overlap.
STAGGER = int(os.environ.get("SUS_STAGGER", "240"))
POLL = 10
_LS = re.compile(r"LS Cluster (\d+)/(\d+)")


def _now():
    return f"{datetime.now():%Y-%m-%d %H:%M:%S}"


class Pool:
    def __init__(self, py, drv, base_common, out_root, susname,
                 n_runs, maxconc=MAXCONC, stagger=STAGGER, extra_log=None):
        self.py = py
        self.drv = drv
        self.base_common = list(base_common)   # driver args, NO --out_dir/--run-index
        self.out_root = Path(out_root)
        self.susname = susname
        self.n_runs = n_runs
        self.maxconc = maxconc
        self.stagger = stagger
        self.par = self.out_root / "_par"
        self.final_contribs = self.out_root / susname / "contribs"
        self.extra_log = Path(extra_log) if extra_log else None
        self._last_emit_key = None
        self.par.mkdir(parents=True, exist_ok=True)
        self.final_contribs.mkdir(parents=True, exist_ok=True)

    # -- logging: to stdout (UI reads it) and optionally a side file --------
    def log(self, msg):
        line = f"{_now()}  {msg}"
        print(line, flush=True)
        if self.extra_log:
            try:
                with open(self.extra_log, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except OSError:
                pass

    def _cname(self, n):
        return f"contrib_run{n:02d}.npz"

    def _sname(self, n):
        return f"summary_run{n:02d}.json"

    def per_run_contrib(self, n):
        return self.par / f"r{n}" / self.susname / "contribs" / self._cname(n)

    def collected_contrib(self, n):
        return self.final_contribs / self._cname(n)

    def collect(self, n):
        src = self.per_run_contrib(n)
        if src.exists():
            shutil.copy2(src, self.collected_contrib(n))
            ssum = src.parent / self._sname(n)
            if ssum.exists():
                shutil.copy2(ssum, self.final_contribs / self._sname(n))
            return True
        return False

    def _run_status(self, n):
        """(fraction, human phase) for run n from its per-run log. During the
        pre-compute phase there is no LS Cluster line yet, so report the current
        phase (last log line) instead of a flat 0 % — otherwise the UI looks
        frozen for the first few minutes."""
        p = self.out_root / f"_run{n}.log"
        try:
            tail = p.read_text(encoding="utf-8", errors="ignore")[-4000:]
        except OSError:
            return 0.0, "starting"
        last = None
        for m in _LS.finditer(tail):
            last = m
        if last:
            done, tot = int(last.group(1)), int(last.group(2))
            frac = min(1.0, done / tot) if tot else 0.0
            return frac, f"cluster {done}/{tot}"
        lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
        return 0.0, (lines[-1][:48] if lines else "starting")

    def emit_progress(self, completed, running):
        frac = float(len(completed))
        parts = []
        for n in sorted(running):
            f, phase = self._run_status(n)
            frac += f
            parts.append(f"run{n}: {phase}")
        human = (f"[parallel] {len(completed)}/{self.n_runs} done | "
                 + (" | ".join(parts) if parts else "launching..."))
        # Only emit when something actually changed — during the multi-minute
        # pre-compute phase the summary is identical every poll, which would
        # otherwise spam the log with the same line (and a redundant 0.000/10).
        key = (human, round(frac, 3))
        if key == self._last_emit_key:
            return
        self._last_emit_key = key
        self.log(human)
        # Machine-readable line the UI's progress parser consumes (the UI hides
        # it from the human log view):
        self.log(f"PARALLEL progress: {frac:.3f}/{self.n_runs}")

    def run(self):
        t0 = time.time()
        self.log(f"===== PARALLEL SUS START (MAXCONC={self.maxconc} "
                 f"STAGGER={self.stagger}s N={self.n_runs}) =====")
        pending = list(range(self.n_runs))
        running = {}     # n -> (Popen, (out_fh, err_fh))
        completed = set()
        last_launch = 0.0
        last_emit = 0.0

        while pending or running:
            while (len(running) < self.maxconc and pending
                   and (not running or time.time() - last_launch >= self.stagger)):
                n = pending.pop(0)
                if self.collected_contrib(n).exists() or self.per_run_contrib(n).exists():
                    self.collect(n)
                    completed.add(n)
                    self.log(f"run {n} SKIP (contrib already exists)")
                    continue
                out_dir = self.par / f"r{n}"
                out_dir.mkdir(parents=True, exist_ok=True)
                rlog = open(self.out_root / f"_run{n}.log", "w", encoding="utf-8")
                relog = open(self.out_root / f"_run{n}.err.log", "w", encoding="utf-8")
                args = ([self.py, "-u", str(self.drv)] + self.base_common
                        + ["--out_dir", str(out_dir), "--run-index", str(n)])
                p = subprocess.Popen(args, stdout=rlog, stderr=relog,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                running[n] = (p, (rlog, relog))
                last_launch = time.time()
                self.log(f"run {n} START pid={p.pid} -> _run{n}.log")
                break

            time.sleep(POLL)

            for n in list(running.keys()):
                p, handles = running[n]
                rc = p.poll()
                if rc is not None:
                    for h in handles:
                        try:
                            h.close()
                        except Exception:
                            pass
                    if self.collect(n):
                        completed.add(n)
                        self.log(f"run {n} DONE rc={rc} -> contrib collected")
                    else:
                        self.log(f"run {n} FAILED rc={rc} (no contrib) — see _run{n}.err.log")
                    del running[n]

            if time.time() - last_emit >= POLL:
                self.emit_progress(completed, running)
                last_emit = time.time()

        cnt = len(list(self.final_contribs.glob("contrib_run*.npz")))
        self.log(f"all runs finished; contribs present = {cnt}/{self.n_runs}")
        if cnt >= self.n_runs:
            self.log("===== AGGREGATE START =====")
            aargs = [self.py, "-u", str(self.drv),
                     "--DEM_path", self._dem_path(),
                     "--test_no", self._arg_val("--test_no", "999"),
                     "--susname_override", self.susname,
                     "--aggregate", "1", "--out_dir", str(self.out_root)]
            subprocess.run(aargs, stdout=sys.stdout, stderr=subprocess.STDOUT)
            if (self.out_root / self.susname / f"sus_{self.susname}_python.tif").exists():
                self.log(f"===== AGGREGATE OK -> {self.susname}/sus_{self.susname}_python.tif =====")
            else:
                self.log("===== AGGREGATE finished but raster not found =====")

            # Auto-generate the tension/compression map (per-cell net force q,
            # prob-weighted; compression positive / tension negative). Reuses the
            # same driver inputs (base_common) — a cheap post-process, no
            # region-grow. All runs are done here, so memory is free.
            self.log("===== TENSION/COMPRESSION START =====")
            tcargs = ([self.py, "-u", str(self.drv)] + self.base_common
                      + ["--out_dir", str(self.out_root), "--tension_compression", "1"])
            subprocess.run(tcargs, stdout=sys.stdout, stderr=subprocess.STDOUT)
            if (self.out_root / self.susname
                    / f"net_force_prob_{self.susname}.tif").exists():
                self.log(f"===== TENSION/COMPRESSION OK -> "
                         f"{self.susname}/net_force_prob_{self.susname}.tif =====")
            else:
                self.log("===== TENSION/COMPRESSION finished but raster not found =====")
        else:
            self.log(f"===== AGGREGATE SKIPPED ({cnt}/{self.n_runs}); re-launch to finish =====")
        self.log(f"PARALLEL progress: {self.n_runs:.3f}/{self.n_runs}")
        self.log(f"Total elapsed: {(time.time()-t0)/60.0:.2f} min")

    def _arg_val(self, flag, default=None):
        try:
            i = self.base_common.index(flag)
            return self.base_common[i + 1]
        except (ValueError, IndexError):
            return default

    def _dem_path(self):
        return self._arg_val("--DEM_path", str(REPO / "lib/DEM/dem_afterEQ_5m_crop.tif"))


# ---- helpers to split a full driver cmd into (py, drv, base_common) --------
def _split_driver_cmd(cmd):
    """cmd = [py, '-u', driver.py, ...driver args...]. Return
    (py, drv, base_common) with --out_dir<val> and --run-index<val> removed."""
    py = cmd[0]
    # locate driver.py token
    di = next(i for i, t in enumerate(cmd) if str(t).endswith("driver.py"))
    drv = cmd[di]
    rest = cmd[di + 1:]
    base = []
    skip = False
    for i, tok in enumerate(rest):
        if skip:
            skip = False
            continue
        if tok in ("--out_dir", "--run-index"):
            skip = True          # drop the flag and its value
            continue
        base.append(tok)
    return py, drv, base


# ---- hard-coded CLI parameter set (a20k / test 999) ------------------------
_CLI_COMMON = [
    "--DEM_path", str(REPO / "lib/DEM/dem_afterEQ_5m_crop.tif"),
    "--test_no", "999", "--susname_override", "00999",
    "--soil_moisture_mode", "1", "--soil_depth_mode", "1", "--soil_strength_mode", "1",
    "--nogrow_mode", "1", "--seismic_mode", "off",
    "--soil_depth_source", "mat", "--nogrow_source", "mat",
    "--mw", "1.0", "--Gs", "2.65", "--gam_w", "9.8", "--gam_dry", "16.0",
    "--gam_sat", "20.0", "--S_roots", "0.0",
    "--soil_depth_uniform", "2.0", "--soil_depth_endtime", "5000.0",
    "--phi_uniform", "25.0", "--coh_uniform", "2.0", "--uniform_PGA", "0.3",
    "--pseudo_scaling", "1.0",
    "--ridge_acc_thresh", "5.0", "--valley_acc_thresh", "100.0",
    "--save_intermediates", "1",
    "--soil_depth_mat", str(REPO / "lib/soil_depth/dem_afterEQ_5m_crop_soil_depth_python.mat"),
    "--no_grow_mat", str(REPO / "lib/no_grow/dem_afterEQ_5m_crop_no_grow_slopeunits_grass_cv005_a20k.mat"),
    "--shear_strength_mat", str(REPO / "lib/soil_strength/shear_strength.mat"),
]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gui-args", help="JSON file describing the driver command "
                                       "built by the Streamlit UI")
    args = ap.parse_args(argv)

    if args.gui_args:
        spec = json.loads(Path(args.gui_args).read_text(encoding="utf-8"))
        py, drv, base = _split_driver_cmd(spec["driver_cmd"])
        out_root = spec["out_root"]
        susname = spec["susname"]
        n_runs = int(spec.get("n_runs", 10))
        maxconc = int(spec.get("maxconc", MAXCONC))
        stagger = int(spec.get("stagger", STAGGER))
        pool = Pool(py, drv, base, out_root, susname, n_runs,
                    maxconc=maxconc, stagger=stagger)
    else:
        out_root = REPO / "python" / "output_webui"
        pool = Pool(sys.executable, REPO / "python" / "driver.py", _CLI_COMMON,
                    out_root, "00999", 10,
                    extra_log=out_root / "_sus_parallel.log")
    pool.run()


if __name__ == "__main__":
    sys.exit(main())
