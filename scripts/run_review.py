"""Runs requested after the independent review (change log 2026-10-14, entries j–n).

Same training code, settings and selection rules as run_e1_e2.py / run_fl.py (protocol v1.0); this script
only adds: more seeds, saved per-row predictions, cost logging, and the single-client sanity check.
Every training call reseeds itself, so a (job, federation, client, seed) result does not depend on run order;
seeds 0-4 must reproduce the stored 5-seed scores (checked by check_reproduction.py).

Usage: python run_review.py <job> <seeds>     e.g.  python run_review.py fedavg_size 0-19
Jobs: mlp_local fedavg_size fedavg_equal fedper_in finetune fedprox fedper mlp_pooled loo zeroshot_mlp sanity
Outputs (under config.RESULTS):
  review/<job>_s<seed>.csv            one row per scored client (PR-AUC, lift, cost columns)
  preds/<job>/<federation>/s<seed>_<client>.npz   float32 'val' and 'test' scores (row order = keys file)
  preds/keys/<client>_<split>.npz     user/day keys and labels for val and test rows
"""
from __future__ import annotations

import os as _os, sys as _sys  # repository layout: modules live in ../src
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import fl
from config import RESULTS
from fl import finetune, run_fl
from run_e1_e2 import CLIENTS, class_weights, predict, prep, train_mlp
from run_fl import ALL, FEDS
from fitd_common import PROTOCOL

REV = os.path.join(RESULTS, "review")
PRED = os.path.join(RESULTS, "preds")
os.makedirs(REV, exist_ok=True)
os.makedirs(os.path.join(PRED, "keys"), exist_ok=True)
D_IN = 42
MLP_PARAMS = D_IN * 128 + 128 + 128 * 64 + 64 + 64 + 1


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def split_id(n, seed):
    return "temporal" if PROTOCOL[n] == "temporal" else f"s{seed}"


def save_keys(P, seed):
    for n in CLIENTS:
        f = os.path.join(PRED, "keys", f"{n}_{split_id(n, seed)}.npz")
        if not os.path.exists(f):
            c = P[n]
            np.savez_compressed(f, val_user=c["kva_user"], val_day=c["kva_day"], yva=c["yva"].astype(np.int8),
                                test_user=c["kte_user"], test_day=c["kte_day"], yte=c["yte"].astype(np.int8))


def save_pred(job, fed, seed, n, val, test):
    d = os.path.join(PRED, job, fed)
    os.makedirs(d, exist_ok=True)
    np.savez(os.path.join(d, f"s{seed}_{n}.npz"), val=np.asarray(val, np.float32), test=np.asarray(test, np.float32))


def rec(rows, job, regime, fed, seed, n, P, model, extra, target=None):
    """Score `model` on client `target` (default n) and save its val/test predictions."""
    t = target or n
    pv, pt = predict(model, P[t]["Xva"]), predict(model, P[t]["Xte"])
    y = P[t]["yte"]
    pr = float(average_precision_score(y, pt))
    save_pred(job, fed, seed, t, pv, pt)
    rows.append(dict(job=job, regime=regime, federation=fed, seed=seed, client=t, n_test=len(y), pos_test=int(y.sum()),
                     base_rate=float(y.mean()), pr_auc=pr, lift=pr / y.mean(), **extra))
    log(f"{job} {fed} seed{seed} {t}: PR-AUC={pr:.4f}")


def local_cost(ep, wall):
    return dict(best_round=ep, stop_round=min(100, ep + 10), params_shared=0, params_local=MLP_PARAMS,
                bytes_per_round=0, wall_s=wall)


def fl_cost(wall):
    return dict(best_round=fl.LAST["best_round"], stop_round=fl.LAST["stop_round"],
                params_shared=fl.LAST["params_shared"], params_local=fl.LAST["params_local"],
                bytes_per_round=fl.LAST["bytes_per_round"], wall_s=wall)


def run_seed(job, seed):
    P = prep(seed)
    save_keys(P, seed)
    rows = []
    T = time.perf_counter
    if job == "mlp_local":
        for n in CLIENTS:
            c = P[n]
            t0 = T()
            m, v, ep = train_mlp(c["Xtr"], c["ytr"], class_weights(c["ytr"]), [(c["Xva"], c["yva"])], seed)
            rec(rows, job, "R0_local", n, seed, n, P, m, dict(val=v, **local_cost(ep, T() - t0)))
    elif job == "mlp_pooled":
        Xtr = np.concatenate([P[n]["Xtr"] for n in CLIENTS])
        ytr = np.concatenate([P[n]["ytr"] for n in CLIENTS])
        wtr = np.concatenate([class_weights(P[n]["ytr"]) for n in CLIENTS])
        t0 = T()
        m, v, ep = train_mlp(Xtr, ytr, wtr, [(P[n]["Xva"], P[n]["yva"]) for n in CLIENTS], seed)
        w = T() - t0
        for n in CLIENTS:
            rec(rows, job, "R1_pooled", "pooled", seed, n, P, m, dict(val=v, **local_cost(ep, w)))
    elif job == "zeroshot_mlp":
        for src in ALL:
            c = P[src]
            t0 = T()
            m, v, ep = train_mlp(c["Xtr"], c["ytr"], class_weights(c["ytr"]), [(c["Xva"], c["yva"])], seed)
            w = T() - t0
            for tgt in ALL:
                if tgt != src:
                    rec(rows, job, "R5_zeroshot_mlp", src, seed, src, P, m, dict(source=src, **local_cost(ep, w)), target=tgt)
    elif job == "loo":
        for held in ALL:
            tr = [n for n in ALL if n != held]
            t0 = T()
            models, br, bv = run_fl({n: P[n] for n in tr}, seed, "size")
            rec(rows, job, "R6_loo", "-".join(tr), seed, tr[0], P, models[tr[0]],
                dict(val=bv, held_out=held, **fl_cost(T() - t0)), target=held)
    elif job == "sanity":
        # protocol §6: FedAvg with a single client vs the local-only MLP (same data, same seed)
        for n in CLIENTS:
            c = P[n]
            t0 = T()
            m0, v0, ep0 = train_mlp(c["Xtr"], c["ytr"], class_weights(c["ytr"]), [(c["Xva"], c["yva"])], seed)
            rec(rows, job, "R0_local_recomputed", "local", seed, n, P, m0, dict(val=v0, **local_cost(ep0, T() - t0)))
            for mode, ml in (("fedavg1_matched_streams", True), ("fedavg1_fl_seeding", False)):
                t0 = T()
                ms, br, bv = run_fl({n: c}, seed, "size", match_local=ml)
                rec(rows, job, mode, mode, seed, n, P, ms[n], dict(val=bv, **fl_cost(T() - t0)))
    elif job == "fednova_sanity":
        # D-1 sanity check: FedNova with one client must reduce to that client's own training
        for n in CLIENTS:
            c = P[n]
            t0 = T()
            m0, v0, ep0 = train_mlp(c["Xtr"], c["ytr"], class_weights(c["ytr"]), [(c["Xva"], c["yva"])], seed)
            rec(rows, job, "R0_local_recomputed", "local", seed, n, P, m0, dict(val=v0, **local_cost(ep0, T() - t0)))
            t0 = T()
            ms, br, bv = run_fl({n: c}, seed, "size", match_local=True, agg="fednova")
            rec(rows, job, "fednova1_matched_streams", "fednova1_matched_streams", seed, n, P, ms[n],
                dict(val=bv, tau=fl.LAST["tau"][n], **fl_cost(T() - t0)))
    else:
        # D-2: "c2c4_<regime>" runs only the CERT r5.2 + LANL federation for an existing regime;
        # FedNova jobs (D-1) run the 7 protocol federations plus C2+C4.
        c2c4 = job.startswith("c2c4_")
        base = job[5:] if c2c4 else job
        feds = ({"C2+C4": ("cert52", "lanl")} if c2c4 else
                dict(FEDS, **{"C2+C4": ("cert52", "lanl")}) if base.startswith("fednova") else FEDS)
        mu = json.load(open(os.path.join(RESULTS, "fedprox_mu.json")))["mu"] if base == "fedprox" else 0.0
        for fed, members in feds.items():
            C = {n: P[n] for n in members}
            t0 = T()
            if base in ("fedavg_size", "fedavg_equal"):
                models, br, bv = run_fl(C, seed, base.split("_")[1])
                regime = f"R2_{base}"
            elif base in ("fednova_size", "fednova_equal", "fednova_sgd_size", "fednova_sgd_equal"):
                models, br, bv = run_fl(C, seed, base.split("_")[-1], agg="fednova",
                                        optimizer="sgd" if "_sgd_" in base else "adam")
                regime = f"D1_{base}"
            elif base == "fedper":
                models, br, bv = run_fl(C, seed, "size", mode="fedper")
                regime = "R4_fedper"
            elif base == "fedper_in":
                models, br, bv = run_fl(C, seed, "size", mode="fedper_in")
                regime = "C2b_local_input"
            elif base == "fedprox":
                models, br, bv = run_fl(C, seed, "size", mu=mu)
                regime = f"R3_fedprox_mu{mu}"
            elif base == "finetune":
                g, br, bv = run_fl(C, seed, "size")
                regime = "R4_fedavg_finetune"
            else:
                raise ValueError(job)
            cost = fl_cost(T() - t0)
            if "tau" in fl.LAST and base.startswith("fednova"):
                cost.update(tau_eff=fl.LAST["tau_eff"])
            for n in members:
                ex = dict(val=bv, **cost)
                if "tau" in fl.LAST and base.startswith("fednova"):
                    ex["tau"] = fl.LAST["tau"][n]
                if base == "finetune":
                    t1 = T()
                    m = finetune(g[n], C[n], seed)
                    ex.update(ft_epochs_run=fl.LAST["ft_epochs_run"], ft_best_epoch=fl.LAST["ft_best_epoch"],
                              ft_wall_s=T() - t1)
                else:
                    m = models[n]
                rec(rows, job, regime, fed, seed, n, P, m, ex)
    return rows


def main(job, seeds):
    meta = os.path.join(REV, "machine.json")
    if not os.path.exists(meta):
        json.dump(dict(platform=platform.platform(), processor=platform.processor(), cores=os.cpu_count(),
                       python=platform.python_version(), torch_threads=2), open(meta, "w"), indent=1)
    for seed in seeds:
        out = os.path.join(REV, f"{job}_s{seed}.csv")
        if os.path.exists(out):
            log(f"skip {job} seed{seed} (done)")
            continue
        rows = run_seed(job, seed)
        pd.DataFrame(rows).to_csv(out + ".tmp", index=False)
        os.replace(out + ".tmp", out)
    log(f"done {job}")


if __name__ == "__main__":
    a, b = (sys.argv[2].split("-") + [None])[:2]
    seeds = list(range(int(a), int(b or a) + 1))
    main(sys.argv[1], seeds)
