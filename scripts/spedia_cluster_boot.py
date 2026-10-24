"""D-3 (1): SPEDIA test-set uncertainty by a user-level cluster bootstrap (change log 2026-10-15 D-3).

SPEDIA's test set has only 76 user-days from 15 users, so PR-AUC is uncertain even with many seeds.
For each draw b (10,000 draws, seed 20261015) the test USERS are resampled with replacement (a user's days stay
together). For every regime and federation that includes SPEDIA, PR-AUC is computed on the resampled rows for
each available seed and averaged over seeds. The same user draw is used for every regime, so the paired
difference (regime - local) is also reported. Seeds are not resampled here: this interval reflects test-set
(user) uncertainty only, and complements the seed bootstrap in review_stats.py.
Output: results/review/spedia_cluster_boot.csv; SPEDIA test predictions packed in results/review/spedia_test_preds.npz
"""
from __future__ import annotations

import os as _os, sys as _sys  # repository layout: modules live in ../src
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from config import RESULTS

PRED = os.path.join(RESULTS, "preds")
B, SEED = 10_000, 20261015


def ap_fast(y, s):
    """Average precision identical to sklearn's average_precision_score (ties handled at distinct scores)."""
    o = np.argsort(-s, kind="mergesort")
    y, s = y[o], s[o]
    distinct = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tp = np.cumsum(y)[distinct]
    fp = (distinct + 1) - tp
    P = y.sum()
    if P == 0:
        return np.nan
    prec = tp / (tp + fp)
    rec = tp / P
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def main():
    z = np.load(os.path.join(PRED, "keys", "spedia_temporal.npz"), allow_pickle=True)  # our own key file
    yte, users = z["yte"].astype(int), z["test_user"]
    uniq = np.unique(users)
    rows_of = {u: np.flatnonzero(users == u) for u in uniq}
    # collect predictions: (job, federation) -> {seed: test scores}
    cells = {}
    for f in glob.glob(os.path.join(PRED, "*", "*", "s*_spedia.npz")):
        job, fed = f.split(os.sep)[-3], f.split(os.sep)[-2]
        if job in ("keys", "zeroshot_mlp", "loo", "sanity", "fednova_sanity") or job.startswith("confirm"):
            continue
        seed = int(os.path.basename(f).split("_")[0][1:])
        cells.setdefault((job, fed), {})[seed] = np.load(f)["test"].astype(float)
    assert ("mlp_local", "spedia") in cells, "local predictions missing"
    # sanity: fast AP equals sklearn
    s0 = next(iter(cells[("mlp_local", "spedia")].values()))
    assert abs(ap_fast(yte, s0) - average_precision_score(yte, s0)) < 1e-12
    rng = np.random.default_rng(SEED)
    draws = [np.concatenate([rows_of[u] for u in rng.choice(uniq, len(uniq), replace=True)]) for _ in range(B)]
    stat = {}
    for key, by_seed in cells.items():
        seeds = sorted(by_seed)
        M = np.empty(B)
        for b, idx in enumerate(draws):
            yb = yte[idx]
            M[b] = np.nanmean([ap_fast(yb, by_seed[s][idx]) for s in seeds])
        point = np.mean([average_precision_score(yte, by_seed[s]) for s in seeds])
        stat[key] = (M, point, seeds)
    L, R0 = [], stat[("mlp_local", "spedia")]
    for (job, fed), (M, point, seeds) in sorted(stat.items()):
        r = dict(job=job, federation=fed, n_seeds=len(seeds), pr_auc=point,
                 user_lo=np.nanpercentile(M, 2.5), user_hi=np.nanpercentile(M, 97.5))
        if job != "mlp_local":
            if seeds == R0[2]:   # paired difference only when the regime has the same seeds as local
                D = M - R0[0]
                r.update(diff_vs_local=point - R0[1], diff_lo=np.nanpercentile(D, 2.5), diff_hi=np.nanpercentile(D, 97.5))
        L.append(r)
    out = pd.DataFrame(L)
    out.to_csv(os.path.join(RESULTS, "review", "spedia_cluster_boot.csv"), index=False)
    np.savez_compressed(os.path.join(RESULTS, "review", "spedia_test_preds.npz"), y=yte, user=users,
                        **{f"{j}__{f}__s{s}": v for (j, f), d in cells.items() for s, v in d.items()})
    print(f"{len(uniq)} SPEDIA test users, {len(yte)} user-days; {B:,} user-level draws")
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
