"""F-2 (review plan of 11 Nov 2026, M3a): single-feature baseline for C1-C5. Deterministic, no training.
For each of the 42 features and its negative, the raw value is used as the score (ranking only; tied values are
grouped by sklearn's average_precision_score). Feature and sign chosen on VALIDATION PR-AUC (ties -> first in column
order, positive sign first); its TEST PR-AUC reported. Splits as in the runs: CERT/SPEDIA temporal, LANL
user-disjoint (20 seed splits), TWOS day split. SPEDIA also: cnt_cmd_exec, and the share of positive / negative test
days with cnt_cmd_exec > 0.
Outputs: results/review2/f2_single_feature.csv (one row per member and split), f2_single_feature_summary.csv"""
import os as _os, sys as _sys  # repository layout: modules live in ../../src and ../
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_os.path.join(_HERE, "..", "..", "src"), _os.path.join(_HERE, "..")]
from config import ROOT as _ROOT
_os.chdir(_ROOT)  # relative paths below (results/, data/) are relative to FITD_ROOT
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap

from fitd_common import LABEL, feature_cols, load, three_way
from run_r2 import twos_frame, twos_split

R = "results/review2"
rows = []


def best(va, te, cols):
    yv, yt = va[LABEL].to_numpy(), te[LABEL].to_numpy()
    cand = []
    for c in cols:
        for sgn in (1, -1):
            v = ap(yv, sgn * va[c].to_numpy(np.float64)) if yv.sum() else np.nan
            cand.append((v, c, sgn))
    vbest = max(v for v, _, _ in cand if not np.isnan(v))
    v, c, sgn = next(x for x in cand if x[0] == vbest)
    return dict(feature=c, sign=sgn, val_pr_auc=v, test_pr_auc=ap(yt, sgn * te[c].to_numpy(np.float64)),
                test_base_rate=yt.mean(), test_pos_days=int(yt.sum()), test_pos_users=int(te.groupby("user")[LABEL].max().sum()),
                val_pos_days=int(yv.sum()))


for name in ("cert42", "cert52", "spedia", "lanl"):
    d = load(name)
    cols = feature_cols(d)
    for s in (range(20) if name == "lanl" else [0]):
        _, tr, va, te = three_way(d, name, s)
        r = dict(member=name, split_seed=s, **best(va, te, cols))
        if name == "spedia":
            yt = te[LABEL].to_numpy()
            x = te["cnt_cmd_exec"].to_numpy()
            r.update(cmd_test_pr_auc=ap(yt, x), cmd_val_pr_auc=ap(va[LABEL].to_numpy(), va["cnt_cmd_exec"].to_numpy()),
                     cmd_share_pos_gt0=float((x[yt == 1] > 0).mean()), cmd_share_neg_gt0=float((x[yt == 0] > 0).mean()))
        rows.append(r)
d = twos_frame()
cols = feature_cols(d)
tr, va, te = twos_split(d, "day", 0)
rows.append(dict(member="twos", split_seed=0, **best(va, te, cols)))
D = pd.DataFrame(rows)
D.to_csv(f"{R}/f2_single_feature.csv", index=False)

# comparison: local MLP (20 seeds) and XGBoost (stored R0 runs)
mlp = pd.concat(pd.read_csv(f"results/review/mlp_local_s{s}.csv") for s in range(20))
mlp = mlp[mlp.federation == mlp.client].groupby("client").pr_auc.mean().to_dict()
e11 = pd.read_csv(f"{R}/e11_twos_cells.csv").set_index("regime")
mlp["twos"] = float(e11.loc["R0_local", "pr_auc"])
xg = pd.read_csv("results/e1e2_xgb_local.csv")
xgb = xg.groupby("client").agg(pr=("pr_auc", "mean"), n=("seed", "nunique"))
xgb_d = xgb.pr.to_dict()
xgb_d["twos"] = float(e11.loc["R0_XGB", "pr_auc"])
xgb_n = xgb.n.to_dict()
xgb_n["twos"] = int(e11.loc["R0_XGB", "n_seeds"])
S = []
for m, g in D.groupby("member", sort=False):
    r = dict(member=m, n_splits=len(g), chosen_feature=g.feature.mode().iloc[0] + ("" if g.sign.mode().iloc[0] > 0 else " (neg)"),
             n_distinct_choices=g[["feature", "sign"]].drop_duplicates().shape[0],
             val_pr_auc=g.val_pr_auc.mean(), test_pr_auc=g.test_pr_auc.mean(), test_min=g.test_pr_auc.min(),
             test_max=g.test_pr_auc.max(), base_rate=g.test_base_rate.mean(), local_mlp=mlp[m], xgb=xgb_d[m], xgb_seeds=xgb_n[m])
    if m == "spedia":
        r.update({k: g[k].iloc[0] for k in ("cmd_val_pr_auc", "cmd_test_pr_auc", "cmd_share_pos_gt0", "cmd_share_neg_gt0")})
    S.append(r)
S = pd.DataFrame(S)
S.to_csv(f"{R}/f2_single_feature_summary.csv", index=False)
pd.set_option("display.width", 250)
print(S.round(4).to_string(index=False))
print(D[D.member == "lanl"][["split_seed", "feature", "sign", "val_pr_auc", "test_pr_auc"]].round(4).to_string(index=False))
