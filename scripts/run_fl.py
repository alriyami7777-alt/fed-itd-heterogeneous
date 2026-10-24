"""E3–E7 federated experiments. Usage: python run_fl.py <job>
Jobs:
  fedavg_size   R2 FedAvg, size-weighted, all 7 federations (E3)
  fedavg_equal  R2 FedAvg, equal weights, all 7 federations (E3)
  fedper        R4 FedPer (local output layer), all 7 federations (E4)
  finetune      R4 FedAvg(size) + local fine-tuning, all 7 federations (E4)
  fedprox_tune  pick mu on validation (seed 0, all four clients) -> results/fedprox_mu.json
  fedprox       R3 FedProx with the tuned mu, all 7 federations (E4)
  fedper_in     C2 variant (b): local input layer, all 7 federations (E5)
  loo           R6 leave-one-client-out FedAvg(size): train on 3, test the 4th unseen (E7)
Each row: per-client test PR-AUC of the selected model.
"""
from __future__ import annotations

import os as _os, sys as _sys  # repository layout: modules live in ../src
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from fl import finetune, run_fl
from run_e1_e2 import predict, prep

SEEDS = (0, 1, 2, 3, 4)
FEDS = {
    "C1+C2": ("cert42", "cert52"),
    "C1+C3": ("cert42", "spedia"),
    "C2+C3": ("cert52", "spedia"),
    "C3+C4": ("spedia", "lanl"),
    "C1+C4": ("cert42", "lanl"),
    "C1+C2+C3": ("cert42", "cert52", "spedia"),
    "C1+C2+C3+C4": ("cert42", "cert52", "spedia", "lanl"),
}
ALL = ("cert42", "cert52", "spedia", "lanl")
from config import RESULTS as OUT


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def rec(rows, job, regime, fed, seed, n, y, p, extra):
    pr = float(average_precision_score(y, p))
    rows.append(dict(job=job, regime=regime, federation=fed, seed=seed, client=n, n_test=len(y),
                     pos_test=int(y.sum()), base_rate=float(y.mean()), pr_auc=pr, lift=pr / y.mean(), **extra))
    log(f"{job} {fed} seed{seed} {n}: PR-AUC={pr:.4f} {extra}")


def main(job):
    rows = []
    csv = f"{OUT}/fl_{job}.csv"
    if job == "fedprox_tune":
        P = prep(0)
        res = {}
        for mu in (0.001, 0.01, 0.1):
            _, br, bv = run_fl({n: P[n] for n in ALL}, 0, "size", mu=mu)
            res[mu] = bv
            log(f"mu={mu} best_val={bv:.4f} round={br}")
        best_mu = max(res, key=res.get)
        json.dump({"val_by_mu": {str(k): v for k, v in res.items()}, "mu": best_mu},
                  open(f"{OUT}/fedprox_mu.json", "w"), indent=1)
        log(f"chosen mu={best_mu}")
        return
    for seed in SEEDS:
        P = prep(seed)
        if job == "zeroshot_mlp":
            # R5 with the same MLP: train on one source (its own train/val), score every other client's test set
            from run_e1_e2 import class_weights, train_mlp
            for src in ALL:
                c = P[src]
                m, v, ep = train_mlp(c["Xtr"], c["ytr"], class_weights(c["ytr"]), [(c["Xva"], c["yva"])], seed)
                for tgt in ALL:
                    if tgt != src:
                        rec(rows, job, "R5_zeroshot_mlp", src, seed, tgt, P[tgt]["yte"], predict(m, P[tgt]["Xte"]),
                            dict(source=src, best_epoch=ep))
                pd.DataFrame(rows).to_csv(csv, index=False)
            continue
        if job == "loo":
            for held in ALL:
                tr = [n for n in ALL if n != held]
                models, br, bv = run_fl({n: P[n] for n in tr}, seed, "size")
                # unseen client: no fine-tuning; scaled with its own training scaler (no labels used)
                m = models[tr[0]]
                rec(rows, job, "R6_loo", "-".join(tr), seed, held, P[held]["yte"], predict(m, P[held]["Xte"]),
                    dict(best_round=br, val=bv, held_out=held))
                pd.DataFrame(rows).to_csv(csv, index=False)
            continue
        for fed, members in FEDS.items():
            C = {n: P[n] for n in members}
            if job in ("fedavg_size", "fedavg_equal"):
                w = job.split("_")[1]
                models, br, bv = run_fl(C, seed, w)
                regime = f"R2_fedavg_{w}"
            elif job == "fedper":
                models, br, bv = run_fl(C, seed, "size", mode="fedper")
                regime = "R4_fedper"
            elif job == "fedper_in":
                models, br, bv = run_fl(C, seed, "size", mode="fedper_in")
                regime = "C2b_local_input"
            elif job == "fedprox":
                mu = json.load(open(f"{OUT}/fedprox_mu.json"))["mu"]
                models, br, bv = run_fl(C, seed, "size", mu=mu)
                regime = f"R3_fedprox_mu{mu}"
            elif job == "finetune":
                g, br, bv = run_fl(C, seed, "size")
                models = {n: finetune(g[n], C[n], seed) for n in members}
                regime = "R4_fedavg_finetune"
            else:
                raise ValueError(job)
            for n in members:
                rec(rows, job, regime, fed, seed, n, P[n]["yte"], predict(models[n], P[n]["Xte"]),
                    dict(best_round=br, val=bv))
            pd.DataFrame(rows).to_csv(csv, index=False)
    log(f"done {job}")


if __name__ == "__main__":
    main(sys.argv[1])
