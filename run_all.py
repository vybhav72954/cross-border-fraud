"""
One reproducibility entrypoint — chains the whole bake-off in documented order:

    build -> baseline -> gnn(ring) -> ssm(temporal) -> ssm(velocity)
          -> ssm(selective) -> gnn(snapshot) -> category -> geo
          -> integration -> robustness

Each step is the corresponding numbered ``scripts/NN_*.py`` script (the §6
"Reproduce" list in RESULTS.md), run as a SEPARATE subprocess: every step loads
the 1.3M-row dataset
and several import torch/PyG, so a fresh process per step keeps memory bounded and
avoids cross-step import/state bleed. The same interpreter that launched this
driver is reused for the children (``sys.executable``), so running it with the
bundled venv propagates torch/PyG to every step:

    .venv\\Scripts\\python.exe run_all.py            # default chain, in order
    .venv\\Scripts\\python.exe run_all.py --skip-build  # reuse existing parquet
    .venv\\Scripts\\python.exe run_all.py gnn temporal  # a named subset, in order
    .venv\\Scripts\\python.exe run_all.py sequence ablations  # opt-in EXTRAS
    .venv\\Scripts\\python.exe run_all.py g3                  # G3 cross-dataset rep

The default (no-args) run is the 12-step Sparkov chain above. Four numbered
scripts are EXTRAS and run only when named explicitly: ``sequence``
(13_sequence_zoo) and ``ablations`` (14_ssm_ablations) are CPU-heavy 150-card
subsample showcases; ``external`` (12_external_validity) needs IEEE-CIS in
data/raw/ieee/; ``g3`` (15_g3_replication) needs PaySim/BankSim in data/raw/. They are kept out of the default reproducibility run by design,
but exposing them here lets the same driver exercise ``sequence.py`` (the only
track-G module the default chain never touches) for before/after validation.

Steps stop at the first failure unless ``--keep-going`` is passed. A timing/verdict
summary prints at the end.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

# name -> (script, *args). Order here IS the pipeline order. The first block is
# the default chain; the EXTRAS block below it is opt-in only (see EXTRAS).
STEPS: dict[str, list[str]] = {
    "build": ["scripts/00_build_dataset.py"],
    "baseline": ["scripts/01_glm_baseline.py"],
    "benchmark": ["scripts/02_model_benchmark.py"],
    "gnn": ["scripts/03_gnn_ring.py"],
    "temporal": ["scripts/04_ssm_temporal.py"],
    "velocity": ["scripts/05_ssm_velocity.py"],
    "selective": ["scripts/06_ssm_selective.py"],
    "snapshot": ["scripts/07_gnn_snapshot.py"],
    "category": ["scripts/08_category_headroom.py"],
    "geo": ["scripts/09_geo_control.py"],
    "integration": ["scripts/10_production_integration.py"],
    "robustness": ["scripts/11_robustness.py"],
    # --- EXTRAS: never part of the default run; selectable only by name ---
    "external": ["scripts/12_external_validity.py"],
    "sequence": ["scripts/13_sequence_zoo.py"],
    "ablations": ["scripts/14_ssm_ablations.py"],
    "g3": ["scripts/15_g3_replication.py"],
}
# Excluded from the default (no-args) run: the external-data fold (needs IEEE-CIS
# in data/raw/ieee/), the CPU-heavy 150-card sequence-zoo/ablation showcases, and
# the G3 cross-dataset replication (needs PaySim/BankSim in data/raw/).
EXTRAS: frozenset[str] = frozenset({"external", "sequence", "ablations", "g3"})
ROOT = Path(__file__).resolve().parent


def run_step(name: str) -> tuple[bool, float]:
    cmd = [sys.executable, *[str(ROOT / a) if a.endswith(".py") else a
                             for a in STEPS[name]]]
    print(f"\n{'=' * 70}\n>> {name}: {' '.join(STEPS[name])}\n{'=' * 70}", flush=True)
    t0 = time.perf_counter()
    rc = subprocess.run(cmd, cwd=ROOT).returncode
    return rc == 0, time.perf_counter() - t0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("steps", nargs="*", choices=list(STEPS),
                   help="subset of steps to run, in pipeline order (default: the "
                        "12-step chain; EXTRAS external/sequence/ablations/g3 "
                        "run only when named explicitly)")
    p.add_argument("--skip-build", action="store_true",
                   help="skip the dataset build (reuse existing parquet)")
    p.add_argument("--keep-going", action="store_true",
                   help="run remaining steps even after one fails")
    args = p.parse_args()

    default_steps = [s for s in STEPS if s not in EXTRAS]
    selected = [s for s in STEPS if s in args.steps] if args.steps else default_steps
    if args.skip_build and "build" in selected:
        selected.remove("build")

    results: list[tuple[str, bool, float]] = []
    for name in selected:
        ok, dt = run_step(name)
        results.append((name, ok, dt))
        if not ok and not args.keep_going:
            print(f"\n!! step '{name}' failed (exit != 0); stopping. "
                  f"Use --keep-going to continue.", flush=True)
            break

    print(f"\n{'=' * 70}\nPipeline summary\n{'=' * 70}")
    for name, ok, dt in results:
        print(f"  {('OK ' if ok else 'FAIL'):4s} {name:12s} {dt:8.1f}s")
    total = sum(dt for _, _, dt in results)
    n_fail = sum(not ok for _, ok, _ in results)
    print(f"  {'':4s} {'TOTAL':12s} {total:8.1f}s  ({n_fail} failed)")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
