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

    .venv\\Scripts\\python.exe run_all.py            # full pipeline, in order
    .venv\\Scripts\\python.exe run_all.py --skip-build  # reuse existing parquet
    .venv\\Scripts\\python.exe run_all.py gnn temporal  # a named subset, in order

Steps stop at the first failure unless ``--keep-going`` is passed. A timing/verdict
summary prints at the end.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

# name -> (script, *args). Order here IS the pipeline order.
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
}
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
                   help="subset of steps to run, in pipeline order (default: all)")
    p.add_argument("--skip-build", action="store_true",
                   help="skip the dataset build (reuse existing parquet)")
    p.add_argument("--keep-going", action="store_true",
                   help="run remaining steps even after one fails")
    args = p.parse_args()

    selected = [s for s in STEPS if s in args.steps] if args.steps else list(STEPS)
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
