"""E1 (R0 local-only) and E2 (R1 pooled) per protocol v1.0 FROZEN (13 Oct 2026).

Per client: train on its own training data, score its own test period/users.
Pooled: train on all four clients' training data, score each client's own test set.
Models: MLP 128-64 (torch, Adam 1e-3, batch 256, class-weighted BCE, <=100 epochs, early stop on
validation PR-AUC, patience 10); RF (IJACSA config, trained on train+val); XGBoost (fixed params).
Scalers (signed log1p + standardise) are fitted on each client's own training rows only.
Usage: python run_e1_e2.py <part>   part in {mlp_local, mlp_pooled, xgb_local, rf_local, rf_pooled, xgb_pooled}
"""
from __future__ import annotations

import os as _os, sys as _sys  # repository layout: modules live in ../src
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from fitd_common import CLIENTS, DAY, LABEL, USER, feature_cols, load, signed_log1p, three_way

torch.set_num_threads(int(__import__("os").environ.get("FITD_THREADS", "2")))
SEEDS = (0, 1, 2, 3, 4)
from config import RESULTS as OUT
os.makedirs(OUT, exist_ok=True)


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def prep(seed):
    """Per-client splits and per-client scaled matrices."""
    P = {}
    for n in CLIENTS:
        d = load(n)
        cols = feature_cols(d)
        tv, tr, va, te = three_way(d, n, seed)
        sc = StandardScaler().fit(signed_log1p(tr[cols].to_numpy(np.float64)))
        f = lambda x: sc.transform(signed_log1p(x[cols].to_numpy(np.float64))).astype(np.float32)
        P[n] = dict(cols=cols,
                    Xtr=f(tr), ytr=tr[LABEL].to_numpy(), Xva=f(va), yva=va[LABEL].to_numpy(),
                    Xte=f(te), yte=te[LABEL].to_numpy(),
                    Xtv_raw=tv[cols].to_numpy(np.float32), ytv=tv[LABEL].to_numpy(),
                    Xte_raw=te[cols].to_numpy(np.float32),
                    # row keys (used only to save predictions)
                    kva_user=va[USER].astype(str).to_numpy(), kva_day=va[DAY].astype(str).to_numpy(),
                    kte_user=te[USER].astype(str).to_numpy(), kte_day=te[DAY].astype(str).to_numpy())
    return P


class MLP(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d, 128), torch.nn.ReLU(), torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def predict(model, X, bs=8192):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(torch.sigmoid(model(torch.from_numpy(X[i:i + bs]))).numpy())
    return np.concatenate(out) if out else np.array([])


TRAIN_HIST = []   # per-epoch validation score of the latest train_mlp call (diagnostic only)


def train_mlp(Xtr, ytr, wtr, val_sets, seed, max_epochs=100, patience=10, optimizer="adam", lr=1e-3):
    """val_sets: list of (Xva, yva); selection = mean validation PR-AUC across them.
    optimizer="sgd" (E-1 only): plain SGD (no momentum) with learning rate lr; the default is the frozen Adam 1e-3."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLP(Xtr.shape[1])
    opt = (torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0) if optimizer == "sgd"
           else torch.optim.Adam(model.parameters(), lr=lr))
    X = torch.from_numpy(Xtr)
    y = torch.from_numpy(ytr.astype(np.float32))
    w = torch.from_numpy(wtr.astype(np.float32))
    best, best_state, best_ep, bad = -1.0, None, 0, 0
    TRAIN_HIST.clear()
    g = torch.Generator().manual_seed(seed)
    for ep in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(len(X), generator=g)
        for i in range(0, len(X), 256):
            idx = perm[i:i + 256]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(X[idx]), y[idx], weight=w[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        score = float(np.mean([average_precision_score(yv, predict(model, Xv)) for Xv, yv in val_sets]))
        TRAIN_HIST.append(score)
        if score > best:
            best, best_ep, bad = score, ep, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best, best_ep


def class_weights(y):
    pos = max(int(y.sum()), 1)
    neg = len(y) - pos
    return np.where(y == 1, neg / pos, 1.0)


def record(rows, part, regime, model, seed, n, y, p, extra=None):
    r = dict(part=part, regime=regime, model=model, seed=seed, client=n,
             n_test=int(len(y)), pos_test=int(y.sum()), base_rate=float(y.mean()),
             pr_auc=float(average_precision_score(y, p)))
    r["lift"] = r["pr_auc"] / r["base_rate"]
    if extra:
        r.update(extra)
    rows.append(r)
    log(f"{part} {model} seed{seed} {n}: PR-AUC={r['pr_auc']:.4f} lift={r['lift']:.1f} {extra or ''}")


def main(part):
    rows = []
    out_csv = f"{OUT}/e1e2_{part}.csv"
    for seed in SEEDS:
        P = prep(seed)
        if part == "mlp_local":
            for n in CLIENTS:
                c = P[n]
                m, v, ep = train_mlp(c["Xtr"], c["ytr"], class_weights(c["ytr"]), [(c["Xva"], c["yva"])], seed)
                record(rows, part, "R0_local", "MLP", seed, n, c["yte"], predict(m, c["Xte"]), dict(val_pr_auc=v, best_epoch=ep))
        elif part == "mlp_pooled":
            Xtr = np.concatenate([P[n]["Xtr"] for n in CLIENTS])
            ytr = np.concatenate([P[n]["ytr"] for n in CLIENTS])
            wtr = np.concatenate([class_weights(P[n]["ytr"]) for n in CLIENTS])
            m, v, ep = train_mlp(Xtr, ytr, wtr, [(P[n]["Xva"], P[n]["yva"]) for n in CLIENTS], seed)
            for n in CLIENTS:
                record(rows, part, "R1_pooled", "MLP", seed, n, P[n]["yte"], predict(m, P[n]["Xte"]), dict(val_pr_auc=v, best_epoch=ep))
        elif part == "xgb_local":
            import xgboost as xgb
            for n in CLIENTS:
                c = P[n]
                spw = (len(c["ytv"]) - c["ytv"].sum()) / max(c["ytv"].sum(), 1)
                m = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1, subsample=0.8,
                                      colsample_bytree=0.8, scale_pos_weight=spw, n_jobs=2,
                                      random_state=seed, tree_method="hist", eval_metric="aucpr")
                m.fit(c["Xtv_raw"], c["ytv"])
                record(rows, part, "R0_local", "XGB", seed, n, c["yte"], m.predict_proba(c["Xte_raw"])[:, 1])
        elif part == "rf_local":
            for n in CLIENTS:
                c = P[n]
                m = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", n_jobs=2, random_state=seed)
                m.fit(c["Xtv_raw"], c["ytv"])
                record(rows, part, "R0_local", "RF", seed, n, c["yte"], m.predict_proba(c["Xte_raw"])[:, 1])
        elif part == "rf_pooled":
            X = np.concatenate([P[n]["Xtv_raw"] for n in CLIENTS])
            y = np.concatenate([P[n]["ytv"] for n in CLIENTS])
            m = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", n_jobs=2, random_state=seed)
            m.fit(X, y)
            for n in CLIENTS:
                record(rows, part, "R1_pooled", "RF", seed, n, P[n]["yte"], m.predict_proba(P[n]["Xte_raw"])[:, 1])
        elif part == "xgb_pooled":
            # D-3 (optional): pooled XGBoost with the same settings as local XGBoost; scale_pos_weight from the pool
            import xgboost as xgb
            X = np.concatenate([P[n]["Xtv_raw"] for n in CLIENTS])
            y = np.concatenate([P[n]["ytv"] for n in CLIENTS])
            spw = (len(y) - y.sum()) / max(y.sum(), 1)
            m = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1, subsample=0.8,
                                  colsample_bytree=0.8, scale_pos_weight=spw, n_jobs=2,
                                  random_state=seed, tree_method="hist", eval_metric="aucpr")
            m.fit(X, y)
            for n in CLIENTS:
                record(rows, part, "R1_pooled", "XGB", seed, n, P[n]["yte"], m.predict_proba(P[n]["Xte_raw"])[:, 1])
        pd.DataFrame(rows).to_csv(out_csv, index=False)
    log(f"done {part}")


if __name__ == "__main__":
    main(sys.argv[1])
