"""In-GRASS exec target: import a DEM, run r.slopeunits (create+clean, or the
morphometric optimizer), and export the slope-unit raster as an Int32 GeoTIFF
(0 = nodata).

Invoked by ``grass_slopeunits.run()`` via ``grass -c <dem> <loc> --exec
<grass_python> grass_runner_exec.py <args...>``.

Two modes:
  * default      → r.slopeunits.create (MFD) then r.slopeunits.clean.
  * --optimize   → r.slopeunits.optimize searches cvmin/areamin ranges to
                   maximise the morphometric objective F = V·I (Alvioli 2016),
                   over a basin derived from the DEM footprint. NO landslide
                   inventory is used (parameter-free). Slow.

r.slopeunits.create/clean/optimize are preferred as installed addons (baked
into the Docker image). For create, an optional standalone script may be passed
as a fallback on hosts where the addon is unavailable.
"""
import argparse
import os
import subprocess
import sys
import tempfile

import grass.script as gs


def _f(x):
    return str(float(x))


def _i(x):
    return str(int(round(float(x))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dem", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--thresh", required=True)
    p.add_argument("--areamin", required=True)
    p.add_argument("--cvmin", default="0.3")
    p.add_argument("--rf", default="2")
    p.add_argument("--maxiter", default="50")
    p.add_argument("--rsu_script", default="")
    p.add_argument("--optimize", action="store_true")
    p.add_argument("--cvmin_range", default="0.05,0.25")
    p.add_argument("--areamin_range", default="50000,200000")
    p.add_argument("--epsilonx", default="0.01")
    p.add_argument("--epsilony", default="50000")
    a = p.parse_args()

    gs.run_command("r.in.gdal", input=a.dem, output="dem", overwrite=True,
                   quiet=True)
    gs.run_command("g.region", raster="dem", quiet=True)

    thresh, areamin, cvmin = _f(a.thresh), _f(a.areamin), _f(a.cvmin)
    rf, maxiter = _i(a.rf), _i(a.maxiter)
    cleansize = _i(a.areamin)

    if a.optimize:
        # Basin = DEM footprint (one area polygon) — required by metrics.
        # GRASS vector names must be SQL-compliant (start with a letter).
        gs.mapcalc("rsuones = if(!isnull(dem), 1, null())", overwrite=True)
        gs.run_command("r.to.vect", input="rsuones", output="rsubasin",
                       type="area", overwrite=True, quiet=True)
        outdir = tempfile.mkdtemp(prefix="rsu_opt_")
        # cleansize for the SEARCH must sit within the areamin range, else
        # clean wipes every candidate map and the metric can't be computed.
        amin_lo = float(a.areamin_range.split(",")[0])
        gs.run_command(
            "r.slopeunits.optimize", demmap="dem", basin="rsubasin",
            slumap="sluopt", slumapclean="sluoptc",
            thresh=thresh, rf=rf, maxiteration=maxiter,
            cleansize=str(int(round(amin_lo))),
            cvmin=a.cvmin_range, areamin=a.areamin_range,
            epsilonx=_f(a.epsilonx), epsilony=_f(a.epsilony), outdir=outdir)
        # optimize writes opt.txt with the optimal cvmin (x_opt) and areamin
        # (y_opt) but does NOT emit the final map -> rebuild it with create+clean.
        cvmin_opt, areamin_opt = cvmin, areamin
        for line in open(os.path.join(outdir, "opt.txt")):
            if line.startswith("x_opt:"):
                cvmin_opt = line.split(":", 1)[1].strip()
            elif line.startswith("y_opt:"):
                areamin_opt = line.split(":", 1)[1].strip()
        print(f"SLOPEUNITS_OPTIMIZED cvmin={cvmin_opt} areamin={areamin_opt}",
              flush=True)
        gs.run_command("r.slopeunits.create", demmap="dem", slumap="slu",
                       thresh=thresh, areamin=areamin_opt, cvmin=cvmin_opt,
                       rf=rf, maxiteration=maxiter, overwrite=True)
        gs.run_command("r.slopeunits.clean", demmap="dem", slumap="slu",
                       slumapclean="slu_clean",
                       cleansize=str(int(round(float(areamin_opt)))),
                       flags="m", overwrite=True)
        final_map = "slu_clean"
    else:
        params = dict(demmap="dem", slumap="slu", thresh=thresh,
                      areamin=areamin, cvmin=cvmin, rf=rf, maxiteration=maxiter,
                      overwrite=True)
        try:
            gs.run_command("r.slopeunits.create", **params)
            print("RSU via installed module", flush=True)
        except Exception as exc:
            if not a.rsu_script:
                raise
            print(f"installed module unavailable ({exc}); running script",
                  flush=True)
            cmd = [sys.executable, a.rsu_script, "demmap=dem", "slumap=slu",
                   f"thresh={thresh}", f"areamin={areamin}", f"cvmin={cvmin}",
                   f"rf={rf}", f"maxiteration={maxiter}", "--overwrite"]
            r = subprocess.run(cmd, env=os.environ.copy())
            if r.returncode != 0:
                sys.exit(r.returncode)
        # create alone does NOT enforce a minimum area; clean removes
        # sub-cleansize units (cleansize in m^2; integer).
        final_map = "slu"
        try:
            gs.run_command("r.slopeunits.clean", demmap="dem", slumap="slu",
                           slumapclean="slu_clean", cleansize=cleansize,
                           flags="m", overwrite=True)
            final_map = "slu_clean"
            print(f"SLOPEUNITS_CLEANED via r.slopeunits.clean "
                  f"(cleansize={cleansize} m^2)", flush=True)
        except Exception as exc:
            print(f"r.slopeunits.clean unavailable ({exc}); "
                  "exporting uncleaned units", flush=True)

    gs.run_command("r.out.gdal", input=final_map, output=a.out, type="Int32",
                   overwrite=True, quiet=True, createopt="COMPRESS=LZW",
                   nodata=0)
    s = gs.read_command("r.stats", input=final_map, flags="n", quiet=True)
    n = len([l for l in s.splitlines() if l.strip()])
    print(f"SLOPEUNITS_DONE units={n}", flush=True)


if __name__ == "__main__":
    main()
