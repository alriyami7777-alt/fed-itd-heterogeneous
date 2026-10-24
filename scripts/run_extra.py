"""Additional experiments added after the freeze (protocol change log, 14 Oct 2026, entry h).
Usage: python run_extra.py <job>
  eqsteps      FedAvg, equal weights, every client does exactly S=100 local minibatch steps per round
  eqsteps_in   as eqsteps, with the input layer kept local (C2-b)
  fedper_in_ft local input layer (C2-b), then each client fine-tunes all layers locally (<=10 epochs, patience 3)
  eqsteps_local control: each client alone with the same S-step schedule (isolates the step budget)
  lowdata      CERT r4.2 keeps only f of its training rows (f = 1, 5, 10, 25 %, stratified by label, seed-specific);
               compared: r4.2 local-only MLP on the subset vs federated with full CERT r5.2
               (FedAvg size, FedAvg equal, local input layer). Validation and test sets unchanged.
Same MLP, splits, seeds, scalers and selection rule as protocol v1.0.
"""
from __future__ import annotations

import os as _os, sys as _sys  # repository layout: modules live in ../src
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from fl import finetune, run_fl
from run_e1_e2 import class_weights, predict, prep, train_mlp
from run_fl import ALL, FEDS

SEEDS = (0, 1, 2, 3, 4)
from config import RESULTS as OUT
STEPS = 100


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def rec(rows, job, regime, fed, seed, n, y, p, extra):
    pr = float(average_precision_score(y, p))
    rows.append(dict(job=job, regime=regime, federation=fed, seed=seed, client=n, n_test=len(y),
                     pos_test=int(y.sum()), base_rate=float(y.mean()), pr_auc=pr, lift=pr / y.mean(), **extra))
    log(f"{job} {fed} seed{seed} {n}: PR-AUC={pr:.4f} {extra}")


def subsample(c, frac, seed):
    rng = np.random.default_rng(10_000 + seed)
    y = c["ytr"]
    keep = []
    for lab in (0, 1):
        idx = np.flatnonzero(y == lab)
        k = max(1, int(round(frac * len(idx))))
        keep.append(rng.choice(idx, k, replace=False))
    keep = np.sort(np.concatenate(keep))
    d = dict(c)
    d["Xtr"], d["ytr"] = c["Xtr"][keep], y[keep]
    return d


def main(job):
    rows = []
    csv = f"{OUT}/fl_{job}.csv"
    done = set()
    if job == "lowdata" and os.path.exists(csv):  # resume after a workspace restart
        old = pd.read_csv(csv)
        cnt = old.groupby(["seed", "frac"]).size()
        done = {k for k, v in cnt.items() if v == 7}
        rows = old[[ (s, f) in done for s, f in zip(old.seed, old.frac)]].to_dict("records")
    for seed in SEEDS:
        P = prep(seed)
        if job == "lowdata":
            for frac in (0.01, 0.05, 0.10, 0.25):
                if (seed, frac) in done:
                    continue
                c1 = subsample(P["cert42"], frac, seed)
                ex = dict(frac=frac, n_train=len(c1["ytr"]), pos_train=int(c1["ytr"].sum()))
                m, v, ep = train_mlp(c1["Xtr"], c1["ytr"], class_weights(c1["ytr"]), [(c1["Xva"], c1["yva"])], seed)
                rec(rows, job, "local", f"r4.2@{frac}", seed, "cert42", c1["yte"], predict(m, c1["Xte"]), ex)
                C = {"cert42": c1, "cert52": P["cert52"]}
                for reg, kw in (("fedavg_size", dict(weighting="size")), ("fedavg_equal", dict(weighting="equal")),
                                ("local_input", dict(weighting="size", mode="fedper_in"))):
                    ms, br, bv = run_fl(C, seed, **kw)
                    for n in C:
                        rec(rows, job, reg, f"r4.2@{frac}", seed, n, C[n]["yte"], predict(ms[n], C[n]["Xte"]),
                            dict(ex, best_round=br))
                pd.DataFrame(rows).to_csv(csv, index=False)
            continue
        if job == "eqsteps_local":
            # control for eqsteps: each client alone under the same schedule (S steps per round, same stopping)
            for n in ALL:
                ms, br, bv = run_fl({n: P[n]}, seed, "equal", steps=STEPS)
                rec(rows, job, f"local_S{STEPS}", n, seed, n, P[n]["yte"], predict(ms[n], P[n]["Xte"]), dict(best_round=br, val=bv))
            pd.DataFrame(rows).to_csv(csv, index=False)
            continue
        for fed, members in FEDS.items():
            C = {n: P[n] for n in members}
            if job == "eqsteps":
                ms, br, bv = run_fl(C, seed, "equal", steps=STEPS)
                regime = f"eqsteps_S{STEPS}"
            elif job == "eqsteps_in":
                ms, br, bv = run_fl(C, seed, "equal", mode="fedper_in", steps=STEPS)
                regime = f"eqsteps_S{STEPS}_local_input"
            elif job == "fedper_in_ft":
                g, br, bv = run_fl(C, seed, "size", mode="fedper_in")
                ms = {n: finetune(g[n], C[n], seed) for n in members}
                regime = "local_input_finetune"
            else:
                raise ValueError(job)
            for n in members:
                rec(rows, job, regime, fed, seed, n, P[n]["yte"], predict(ms[n], P[n]["Xte"]), dict(best_round=br, val=bv))
            pd.DataFrame(rows).to_csv(csv, index=False)
    log(f"done {job}")


if __name__ == "__main__":
    main(sys.argv[1])
