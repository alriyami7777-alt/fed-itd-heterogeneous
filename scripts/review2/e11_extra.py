"""E-11 extras (writing team, 21 Oct 20:30). No training except the cheap TWOS tree runs (e11_trees, seeds 0-4).
For every TWOS cell (local MLP, RF, XGBoost, the 3 rules in every group with TWOS, pooled, R5, R6), from saved test
predictions (results/preds/r2_e11_main/<federation>__<regime>/, r2_e11_trees/{RF,XGB}/):
  1. TWOS user-level cluster bootstrap of test PR-AUC, as spedia_cluster_boot.py: resample TWOS test users with
     replacement (10,000 draws, seed 20261015), PR-AUC averaged over the available seeds per draw; also the paired
     difference vs the local MLP (same draws, same seeds). Seeds are not resampled: test-set (user) uncertainty only.
  2. Per-seed TWOS PR-AUC for every cell, to see whether one or two seeds drive the gain.
  3. Trees next to the MLP: RF and XGBoost R0 (train + validation, protocol settings).
  4. Positive users in the TWOS test set, and how many of the test positives each model ranks in its top 10% of
     test user-days (k = ceil(0.10 * n_test), stable sort; mean over seeds, and per seed).
Outputs: results/review2/e11_twos_cells.csv, e11_twos_per_seed.csv
Usage: python e11_extra.py [seeds, default 0-4]"""
import os as _os, sys as _sys  # repository layout: modules live in ../../src and ../
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_os.path.join(_HERE, "..", "..", "src"), _os.path.join(_HERE, "..")]
from config import ROOT as _ROOT
_os.chdir(_ROOT)  # relative paths below (results/, data/) are relative to FITD_ROOT
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from run_r2 import e11_prep
from spedia_cluster_boot import ap_fast

PRED = "results/preds"
B, SEED = 10_000, 20261015
lo, hi = map(int, (sys.argv[1] if len(sys.argv) > 1 else "0-4").split("-"))
SEEDS = list(range(lo, hi + 1))
P, cnt = e11_prep(0, "day")
yte = P["twos"]["yte"].astype(int)
users = P["twos"]["kte_user"]
uniq = np.unique(users)
rows_of = {u: np.flatnonzero(users == u) for u in uniq}
n = len(yte)
k = int(np.ceil(0.10 * n))
pos_users = sorted(set(users[yte == 1]))
print(f"TWOS test: {n} user-days, {yte.sum()} positive user-days, {len(uniq)} users, {len(pos_users)} positive users; "
      f"top 10% = {k} user-days")

cells = {}
for f in glob.glob(os.path.join(PRED, "r2_e11_main", "*", "s*_twos.npz")):
    cell = os.path.basename(os.path.dirname(f))
    s = int(os.path.basename(f).split("_")[0][1:])
    if s in SEEDS:
        cells.setdefault(cell, {})[s] = np.load(f)["test"].astype(float)
for kind in ("RF", "XGB"):
    for f in glob.glob(os.path.join(PRED, "r2_e11_trees", kind, "s*_twos.npz")):
        s = int(os.path.basename(f).split("_")[0][1:])
        if s in SEEDS:
            cells.setdefault(f"twos__R0_{kind}", {})[s] = np.load(f)["test"].astype(float)
assert "twos__R0_local" in cells, sorted(cells)
rng = np.random.default_rng(SEED)
draws = [np.concatenate([rows_of[u] for u in rng.choice(uniq, len(uniq), replace=True)]) for _ in range(B)]


def boot(by_seed):
    ss = sorted(by_seed)
    M = np.empty(B)
    for b, idx in enumerate(draws):
        yb = yte[idx]
        M[b] = np.nanmean([ap_fast(yb, by_seed[s][idx]) if yb.sum() else np.nan for s in ss])
    return M


def topk(p):
    o = np.argsort(-p, kind="mergesort")[:k]
    return int(yte[o].sum())


base = cells["twos__R0_local"]
Mb = boot(base)
out, per = [], []
for cell, by_seed in sorted(cells.items()):
    ss = sorted(by_seed)
    M = Mb if cell == "twos__R0_local" else boot(by_seed)
    pr = {s: average_precision_score(yte, by_seed[s]) for s in ss}
    tk = {s: topk(by_seed[s]) for s in ss}
    fed, regime = cell.split("__")
    r = dict(federation=fed, regime=regime, n_seeds=len(ss), pr_auc=np.mean(list(pr.values())),
             user_lo=np.nanpercentile(M, 2.5), user_hi=np.nanpercentile(M, 97.5),
             top10pct_hits_mean=np.mean(list(tk.values())), top10pct_hits_per_seed=" ".join(str(tk[s]) for s in ss),
             test_pos=int(yte.sum()), k=k)
    if cell != "twos__R0_local" and ss == sorted(base):
        D = M - Mb
        r.update(diff_vs_local=r["pr_auc"] - np.mean([average_precision_score(yte, base[s]) for s in ss]),
                 diff_user_lo=np.nanpercentile(D, 2.5), diff_user_hi=np.nanpercentile(D, 97.5))
    out.append(r)
    for s in ss:
        per.append(dict(federation=fed, regime=regime, seed=s, pr_auc=pr[s], top10pct_hits=tk[s]))
O = pd.DataFrame(out)
O.to_csv("results/review2/e11_twos_cells.csv", index=False)
W = pd.DataFrame(per).pivot_table(index=["federation", "regime"], columns="seed", values="pr_auc")
W.to_csv("results/review2/e11_twos_per_seed.csv")
pd.set_option("display.width", 250)
print(O.round(3).to_string(index=False))
print(W.round(3).to_string())
