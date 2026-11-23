"""Run the experiments for one synthetic dataset (scenario x replica) and save raw predictions.

Usage:
    python scripts/run_all.py --curve hyperbolic --seed 2026       # scenario A, replica 1
    python scripts/run_all.py --curve exponential --seed 2026      # scenario B, replica 1
    python scripts/run_all.py --all                                # 2 scenarios x 3 replicas
    python scripts/run_all.py --quick                              # one seed, fast check
Raw predictions go to results/raw/<curve>_s<seed>/.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from settlement.data import generate_dataset, load_dataset  # noqa: E402
from settlement.protocol import run_e1, run_e2, run_e3_ablation, run_e4  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
REPLICAS = (2026, 2027, 2028)
CURVES = ("hyperbolic", "exponential")


def run_one(curve, seed, quick=False, data=None, full=True, log=print):
    torch.set_num_threads(1)
    np.random.seed(0)
    if data:
        df, tag = load_dataset(data), "real"
    else:
        df = generate_dataset(seed, curve=curve)
        tag = f"{curve}_s{seed}"
        df.to_csv(os.path.join(ROOT, "data", "synthetic", f"{tag}.csv"), index=False)
    out = os.path.join(ROOT, "results", "raw", tag)
    os.makedirs(out, exist_ok=True)
    seeds = (0,) if quick else (0, 1, 2)
    t0 = time.time()
    preds, fails, hampel, info = run_e1(df, seeds=seeds, log=log)
    preds.to_csv(os.path.join(out, "e1_predictions.csv.gz"), index=False)
    fails.to_csv(os.path.join(out, "e1_fit_status.csv"), index=False)
    hampel.to_csv(os.path.join(out, "e1_hampel_flags.csv"), index=False)
    info.to_json(os.path.join(out, "e1_model_info.json"), orient="records")
    if full:   # transfer, ablation and data-volume experiments
        run_e2(df, seeds=(0,), log=log).to_csv(os.path.join(out, "e2_predictions.csv.gz"), index=False)
        run_e3_ablation(df, seeds=(0,), log=log).to_csv(os.path.join(out, "e3_predictions.csv.gz"), index=False)
        run_e4(df, log=log).to_csv(os.path.join(out, "e4_predictions.csv.gz"), index=False)
    meta = dict(dataset=tag, curve=curve, generator_seed=seed, model_seeds=seeds, full=full,
                minutes=round((time.time() - t0) / 60, 1))
    json.dump(meta, open(os.path.join(out, "run_meta.json"), "w"), indent=2)
    log(f"done {meta}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve", default="hyperbolic", choices=CURVES)
    ap.add_argument("--seed", type=int, default=2026, help="seed of the synthetic generator")
    ap.add_argument("--all", action="store_true", help="2 scenarios x 3 replicas")
    ap.add_argument("--e1-only", action="store_true", help="skip E2, E3 and E4")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--data", default=None, help="optional real CSV with the same schema")
    args = ap.parse_args()
    jobs = [(c, s) for c in CURVES for s in REPLICAS] if args.all else [(args.curve, args.seed)]
    for curve, seed in jobs:
        run_one(curve, seed, quick=args.quick, data=args.data, full=not args.e1_only,
                log=lambda m, tag=f"{curve}_s{seed}": print(f"[{tag}] {m}", flush=True))


if __name__ == "__main__":
    main()
