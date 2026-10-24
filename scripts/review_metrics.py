"""Secondary metrics from saved predictions (change log 2026-10-14 (l)); no retraining.

For every saved prediction file (job / federation / seed / client):
  pr_auc         scikit-learn average_precision_score (average precision) on test
  roc_auc        ROC-AUC on test
  recall_at_1pct k = ceil(0.01 * n_test) highest-scored test rows (stable sort on -score, so ties keep row
                 order); recall = positives among them / all test positives. SPEDIA: n_test = 76, so k = 1.
  thr, precision, recall, f1   threshold that maximises F1 on the client's OWN validation predictions,
                 applied unchanged to test. NOT computed for R5 (zeroshot_mlp) or R6 (loo): there the target is an
                 unseen organisation that gives no labels, so a threshold chosen on its validation labels would
                 break that rule (second independent audit, item A02). PR-AUC, ROC-AUC and recall@1% need no threshold.
Then per (job, federation, seed): worst-client PR-AUC and best-worst gap; summarised as mean over seeds.
Output: results/review/metrics_all.csv, metrics_by_cell.csv, metrics_worst_client.csv
"""
from __future__ import annotations

import os as _os, sys as _sys  # repository layout: modules live in ../src
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import glob
import math
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from config import RESULTS
from fitd_common import PROTOCOL

PRED = os.path.join(RESULTS, "preds")
_keys = {}


def keys(client, seed):
    sid = "temporal" if PROTOCOL[client] == "temporal" else f"s{seed}"
    k = (client, sid)
    if k not in _keys:
        z = np.load(os.path.join(PRED, "keys", f"{client}_{sid}.npz"), allow_pickle=False)
        _keys[k] = (z["yva"].astype(int), z["yte"].astype(int))
    return _keys[k]


def recall_at(y, s, frac=0.01):
    k = math.ceil(frac * len(y))
    top = np.argsort(-s, kind="stable")[:k]
    return y[top].sum() / max(y.sum(), 1), k


def best_f1_threshold(y, s):
    p, r, t = precision_recall_curve(y, s)
    f = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0)[:-1]
    return t[int(np.argmax(f))]


def metrics(job, fed, seed, client, f):
    z = np.load(f)
    yva, yte = keys(client, seed)
    sv, st = z["val"], z["test"]
    r1, k = recall_at(yte, st)
    if job in ("zeroshot_mlp", "loo"):
        thr = prec = rec = f1 = np.nan
        alerts = -1
    else:
        thr = best_f1_threshold(yva, sv)
        pred = st >= thr
        tp = int((pred & (yte == 1)).sum())
        prec = tp / pred.sum() if pred.sum() else 0.0
        rec = tp / yte.sum()
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        alerts = int(pred.sum())
    return dict(job=job, federation=fed, seed=seed, client=client, n_test=len(yte), pos_test=int(yte.sum()),
                pr_auc=average_precision_score(yte, st), lift=average_precision_score(yte, st) / yte.mean(),
                roc_auc=roc_auc_score(yte, st),
                recall_at_1pct=r1, k_at_1pct=k, thr_val_f1=float(thr), precision=prec, recall=rec,
                f1=f1, alerts=alerts)


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(PRED, "*", "*", "s*_*.npz"))):
        parts = f.split(os.sep)
        job, fed, name = parts[-3], parts[-2], parts[-1][:-4]
        if job == "keys":
            continue
        s, client = name.split("_", 1)
        rows.append(metrics(job, fed, int(s[1:]), client, f))
    d = pd.DataFrame(rows)
    out = os.path.join(RESULTS, "review")
    d.to_csv(os.path.join(out, "metrics_all.csv"), index=False)
    cell = d.groupby(["job", "federation", "client"]).agg(
        n_seeds=("seed", "nunique"), pr_auc=("pr_auc", "mean"), roc_auc=("roc_auc", "mean"),
        lift=("lift", "mean"), recall_at_1pct=("recall_at_1pct", "mean"), k_at_1pct=("k_at_1pct", "first"),
        precision=("precision", "mean"), recall=("recall", "mean"), f1=("f1", "mean")).reset_index()
    cell.to_csv(os.path.join(out, "metrics_by_cell.csv"), index=False)
    fedjobs = d[~d.job.isin(["mlp_local", "zeroshot_mlp", "loo", "sanity"])]
    # PR-AUC is not comparable across clients with very different base rates (LANL 0.0004 vs SPEDIA 0.39),
    # so the worst client is also reported on lift (PR-AUC / base rate).
    w = fedjobs.groupby(["job", "federation", "seed"]).agg(pmin=("pr_auc", "min"), pmax=("pr_auc", "max"),
                                                          lmin=("lift", "min"), lmax=("lift", "max")).reset_index()
    w["gap"] = w.pmax - w.pmin
    wc = w.groupby(["job", "federation"]).agg(n_seeds=("seed", "nunique"), worst_client_pr_auc=("pmin", "mean"),
                                              best_worst_gap=("gap", "mean"), worst_client_lift=("lmin", "mean"),
                                              best_lift=("lmax", "mean")).reset_index()
    wc.to_csv(os.path.join(out, "metrics_worst_client.csv"), index=False)
    print(cell.head(20).to_string(index=False))
    print(wc.to_string(index=False))


if __name__ == "__main__":
    main()
