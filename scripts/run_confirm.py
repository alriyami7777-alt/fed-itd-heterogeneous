"""Confirmatory runs with designs fixed in advance (change log 2026-10-14, entries C-1 and C-2; requested after
the second independent audit). Fresh seeds 5-19 only. Same MLP, splits, T, E,
learning rate and batch size as protocol v1.0.

C-1  Extra local steps for SPEDIA only.
     SPEDIA makes S = 100 minibatch steps per round WITHOUT replacement (cycling through freshly shuffled
     epochs; with 155 rows that is 100 passes per round); every other client keeps one epoch per round.
     Aggregation weights: size and equal. Personalised selection: each client reports the round with its own
     best validation PR-AUC; training stops when SPEDIA's own validation PR-AUC has not improved for 10 rounds.
     Same-schedule control: SPEDIA alone, S = 100 steps per round, selected and stopped on its own validation.
     Primary outcome: SPEDIA federated minus the same-schedule control, in each of its 5 federations.
C-2  Low-data CERT r4.2 with the right baselines.
     Fractions 1, 5, 10, 25 % of CERT r4.2 training rows (label-stratified, seed-specific; same draw rule as the
     exploratory run). The scaler is fitted on the subsampled training rows only. Validation and test sets are
     unchanged (full validation set, 33,851 rows, stated as a limitation).
     Baselines: (a) CERT r4.2 local MLP on the same subsample; (b) zero-shot CERT r5.2 -> CERT r4.2 (R5): the
     CERT r5.2 local MLP applied to CERT r4.2 test rows scaled with the subsample scaler.
     Federated: FedAvg size and equal, CERT r4.2 (subsample) + CERT r5.2 (full).
     Primary outcome: federated minus the better of baselines (a) and (b).

Usage: python run_confirm.py c1|c2 5-19     Output: results/review/confirm_<job>_s<seed>.csv (+ predictions)
"""
from __future__ import annotations

import os as _os, sys as _sys  # repository layout: modules live in ../src
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import fl
from fitd_common import LABEL, feature_cols, load, signed_log1p, three_way
from fl import run_fl
from run_e1_e2 import class_weights, predict, prep, train_mlp
from run_review import REV, fl_cost, local_cost, log, rec

SPEDIA_FEDS = {"C1+C3": ("cert42", "spedia"), "C2+C3": ("cert52", "spedia"), "C3+C4": ("spedia", "lanl"),
               "C1+C2+C3": ("cert42", "cert52", "spedia"), "C1+C2+C3+C4": ("cert42", "cert52", "spedia", "lanl")}
S = 100
FRACS = (0.01, 0.05, 0.10, 0.25)


def c1(seed):
    P = prep(seed)
    rows = []
    T = time.perf_counter
    kw = dict(client_steps={"spedia": S}, select="own", stop_on="spedia")
    t0 = T()
    ms, br, bv = run_fl({"spedia": P["spedia"]}, seed, "size", **kw)
    rec(rows, "confirm_c1", "spedia_alone_S100", "spedia_alone", seed, "spedia", P, ms["spedia"],
        dict(weighting="-", own_round=fl.LAST["own_round"]["spedia"], own_val=fl.LAST["own_val"]["spedia"],
             **fl_cost(T() - t0)))
    for w in ("size", "equal"):
        for fed, members in SPEDIA_FEDS.items():
            t0 = T()
            ms, br, bv = run_fl({n: P[n] for n in members}, seed, w, **kw)
            cost = fl_cost(T() - t0)
            for n in members:
                rec(rows, "confirm_c1", f"spedia_S100_{w}", f"{fed}_{w}", seed, n, P, ms[n],
                    dict(weighting=w, own_round=fl.LAST["own_round"][n], own_val=fl.LAST["own_val"][n], **cost))
    return rows


def lowdata_client(seed, frac):
    """CERT r4.2 with a label-stratified fraction of its training rows; scaler fitted on that subsample only."""
    d = load("cert42")
    cols = feature_cols(d)
    _, tr, va, te = three_way(d, "cert42", seed)
    y = tr[LABEL].to_numpy()
    rng = np.random.default_rng(10_000 + seed)          # same draw rule as the exploratory run
    keep = []
    for lab in (0, 1):
        idx = np.flatnonzero(y == lab)
        keep.append(rng.choice(idx, max(1, int(round(frac * len(idx)))), replace=False))
    keep = np.sort(np.concatenate(keep))
    sub = tr.iloc[keep]
    sc = StandardScaler().fit(signed_log1p(sub[cols].to_numpy(np.float64)))
    f = lambda x: sc.transform(signed_log1p(x[cols].to_numpy(np.float64))).astype(np.float32)
    return dict(Xtr=f(sub), ytr=sub[LABEL].to_numpy(), Xva=f(va), yva=va[LABEL].to_numpy(),
                Xte=f(te), yte=te[LABEL].to_numpy())


def c2(seed):
    P = prep(seed)
    rows = []
    T = time.perf_counter
    c52 = P["cert52"]
    t0 = T()
    m52, v52, ep52 = train_mlp(c52["Xtr"], c52["ytr"], class_weights(c52["ytr"]), [(c52["Xva"], c52["yva"])], seed)
    w52 = T() - t0
    for frac in FRACS:
        c = lowdata_client(seed, frac)
        Q = {"cert42": c, "cert52": c52}
        ex = dict(frac=frac, n_train=len(c["ytr"]), pos_train=int(c["ytr"].sum()))
        tag = f"r42_{frac}"
        t0 = T()
        m, v, ep = train_mlp(c["Xtr"], c["ytr"], class_weights(c["ytr"]), [(c["Xva"], c["yva"])], seed)
        rec(rows, "confirm_c2", "local_subsample", f"{tag}_local", seed, "cert42", Q, m, dict(ex, **local_cost(ep, T() - t0)))
        rec(rows, "confirm_c2", "zeroshot_r52", f"{tag}_zeroshot", seed, "cert52", Q, m52,
            dict(ex, **local_cost(ep52, w52)), target="cert42")
        for w in ("size", "equal"):
            t0 = T()
            ms, br, bv = run_fl(Q, seed, w)
            cost = fl_cost(T() - t0)
            for n in Q:
                rec(rows, "confirm_c2", f"fedavg_{w}", f"{tag}_fedavg_{w}", seed, n, Q, ms[n], dict(ex, val=bv, **cost))
    return rows


def main(job, seeds):
    for seed in seeds:
        out = os.path.join(REV, f"confirm_{job}_s{seed}.csv")
        if os.path.exists(out):
            log(f"skip confirm_{job} seed{seed} (done)")
            continue
        rows = c1(seed) if job == "c1" else c2(seed)
        pd.DataFrame(rows).to_csv(out + ".tmp", index=False)
        os.replace(out + ".tmp", out)
    log(f"done confirm_{job}")


if __name__ == "__main__":
    a, b = (sys.argv[2].split("-") + [None])[:2]
    main(sys.argv[1], list(range(int(a), int(b or a) + 1)))
