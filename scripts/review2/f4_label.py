"""F-4 (review plan of 11 Nov 2026, M3c): change only the label meaning of member A (m1) in the E-2 four-member CERT r5.2 split.
Partition, users, features, scaling and sizes are exactly E-2 'R' (run_r2.e2_members('R', seed)); only A's labels
change. Scenario of each positive user-day from the CERT r5.2 answer key (r5.2-<scenario>/ files; dates of the
listed malicious events) -> data/answers_r52/f4_r52_answer_days.csv. Families (change log 2026-11-11):
  L1 device  = scenarios 1, 2 (removable-device exfiltration);  L2 e-mail = scenario 4;
  scenario 3 (keylogger via thumb drive + mass e-mail from supervisor's machine) is mixed -> in neither family,
  its days become negative in L1 and L2 (kept positive in R and L3).
  L3 rate control: all scenarios, A's positive days randomly relabelled negative (rng 43,000 + seed) down to L1's
  positive-day count in EACH split (train, val, test) of the same seed.
Rules: local (every member), FedAvg size, FedAvg equal, private first layer (fedper_in) - as run_r2.e2.
Usage: python3 f4_label.py <variant R|L1|L2|L3> <seeds a-b>   |   python3 f4_label.py check"""
import os as _os, sys as _sys  # repository layout: modules live in ../../src and ../
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_os.path.join(_HERE, "..", "..", "src"), _os.path.join(_HERE, "..")]
from config import ROOT as _ROOT
_os.chdir(_ROOT)  # relative paths below (results/, data/) are relative to FITD_ROOT
import os
import sys

import numpy as np
import pandas as pd

import run_r2 as R
from run_r2 import USER, LABEL

ANS = pd.read_csv("data/answers_r52/f4_r52_answer_days.csv", header=None, names=["scen", "user", "date"])
ANS["day"] = pd.to_datetime(ANS.date, format="%m/%d/%Y")
SCEN = ANS.set_index(["user", "day"]).scen
FAM = {"L1": {1, 2}, "L2": {4}}
KEYS = "results/preds/keys/f4"


def members(variant, seed):
    M, steps, info = R.e2_members("R", seed)
    if variant == "R":
        return M, steps
    d, cols, (tr, va, te), key, mem, rng = R.e2_partition(seed)
    A = set(mem["m1"])
    parts = [x[x[USER].isin(A)] for x in (tr, va, te)]
    sc = [pd.Series(list(zip(p[USER], pd.to_datetime(p["day"])))).map(SCEN.to_dict()).to_numpy() for p in parts]
    ys = [M["m1"][k].copy() for k in ("ytr", "yva", "yte")]
    for y, s in zip(ys, sc):
        # every positive day has a scenario; 2 answer-key days (CBW1826 2010-10-14, JCC0016 2011-03-22) are not
        # positive in the shared label cache (one has no feature row) - the cache labels are kept, as in R
        assert not np.isnan(s[y == 1].astype(float)).any()
    if variant in FAM:
        ys = [np.where(np.isin(s.astype(float), list(FAM[variant])), y, 0) for y, s in zip(ys, sc)]
    elif variant == "L3":
        r = np.random.default_rng(43_000 + seed)
        for y, s in zip(ys, sc):
            n1 = int(np.isin(s.astype(float), list(FAM["L1"])).sum())
            pos = np.flatnonzero(y == 1)
            drop = r.choice(pos, len(pos) - n1, replace=False) if len(pos) > n1 else []
            y[drop] = 0
    else:
        raise ValueError(variant)
    c = M["m1"]
    c["ytr"], c["yva"], c["yte"] = ys
    pu = lambda p, y: int(pd.Series(y, index=p[USER].values).groupby(level=0).max().sum())
    c["desc"] = dict(c["desc"], variant=variant, rate_tr=float(ys[0].mean()), rate_all=float(np.concatenate(ys).mean()),
                     pos_tr=int(ys[0].sum()), pos_va=int(ys[1].sum()), pos_te=int(ys[2].sum()),
                     posusers_tr=pu(parts[0], ys[0]), posusers_va=pu(parts[1], ys[1]), posusers_te=pu(parts[2], ys[2]),
                     n_pos_users=len({u for p, y in zip(parts, ys) for u in p[USER].values[y == 1]}))
    return M, steps


def save_keys(seed):
    os.makedirs(KEYS, exist_ok=True)
    f = f"{KEYS}/s{seed}.npz"
    if os.path.exists(f):
        return
    d, cols, (tr, va, te), key, mem, rng = R.e2_partition(seed)
    out = {}
    for m, us in mem.items():
        p = te[te[USER].isin(set(us))]
        out[f"{m}_test_user"] = p[USER].astype(str).to_numpy()
    np.savez(f, **out)


def check():
    rows = []
    for s in range(20):
        for v in ("R", "L1", "L2", "L3"):
            M, _ = members(v, s)
            rows.append(dict(seed=s, variant=v, **{k: M["m1"]["desc"][k] for k in
                        ("n_users", "pos_tr", "pos_va", "pos_te", "posusers_tr", "posusers_va", "posusers_te", "rate_tr")}))
    D = pd.DataFrame(rows)
    D.to_csv("results/review2/f4_feasibility.csv", index=False)
    print(D.groupby("variant")[["pos_tr", "pos_va", "pos_te", "posusers_tr", "posusers_va", "posusers_te"]].agg(["min", "mean", "max"]).round(1).to_string())


def run(variant, seeds):
    job = f"f4_{variant}"
    for seed in seeds:
        out = os.path.join(R.R2, f"{job}_s{seed}.csv")
        if os.path.exists(out):
            continue
        save_keys(seed)
        M, steps = members(variant, seed)
        rows = []
        for m in M:
            c = M[m]
            m_, v, ep = R.train_mlp(c["Xtr"], c["ytr"], R.class_weights(c["ytr"]), [(c["Xva"], c["yva"])], seed)
            R.rec(rows, job, "local", f"{variant}_local", seed, m, M, m_, dict(val=v, best_round=ep, **c["desc"]))
        for rule in ("fedavg_size", "fedavg_equal", "fedper_in"):
            ms, regime, bv, cost, ext = R.run_rule(rule, M, seed)
            for m in M:
                R.rec(rows, job, regime, f"{variant}_{rule}", seed, m, M, ms[m], dict(val=bv, **cost, **M[m]["desc"]))
        pd.DataFrame(rows).to_csv(out + ".tmp", index=False)
        os.replace(out + ".tmp", out)


if __name__ == "__main__":
    if sys.argv[1] == "check":
        check()
    else:
        a, b = (sys.argv[2].split("-") + [None])[:2]
        run(sys.argv[1], range(int(a), int(b or a) + 1))
