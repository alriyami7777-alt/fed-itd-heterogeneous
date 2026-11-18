"""F-1 scope 1 (review plan of 11 Nov 2026, M4): two-level paired bootstrap for the 156 cases of the appendix table of every scored case
(Table A.14; stats_review_s0-19.csv).
Test scores: saved predictions in results/preds/<job>/<federation>/s<seed>_<member>.npz (C2+C4: c2c4_<job>; FedNova
C2+C4 inside fednova_*/C2+C4; pooled: mlp_pooled/pooled). Local: mlp_local. Labels and users: results/preds/keys.
Step 1 reproduction check: mean PR-AUC over seeds recomputed from the saved scores vs the stored table (tol 0.001).
Step 2: twolevel.run_member per member; Holm (a) 102 planned (6 rules x 17 planned cells), (b) all 156, (c) FedNova
(Adam) family (38). Output: results/review2/stats_twolevel_s0-19.csv, f1_repro_check.csv"""
import os as _os, sys as _sys  # repository layout: modules live in ../../src and ../
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_os.path.join(_HERE, "..", "..", "src"), _os.path.join(_HERE, "..")]
from config import ROOT as _ROOT
_os.chdir(_ROOT)  # relative paths below (results/, data/) are relative to FITD_ROOT
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap

import twolevel as T

PRED = "results/preds"
ST = pd.read_csv("results/review/stats_review_s0-19.csv")
CODE = {"CERT r4.2": "cert42", "CERT r5.2": "cert52", "SPEDIA": "spedia", "LANL": "lanl"}
SEEDS = range(20)


def keys(n, s):
    k = np.load(f"{PRED}/keys/{n}_{'s%d' % s if n == 'lanl' else 'temporal'}.npz", allow_pickle=True)
    return k["yte"].astype(int), k["test_user"].astype(str)


def pred_path(job, fed, n, s):
    if job == "mlp_pooled":
        return f"{PRED}/mlp_pooled/pooled/s{s}_{n}.npz"
    if fed == "C2+C4" and not job.startswith("fednova"):
        return f"{PRED}/c2c4_{job}/C2+C4/s{s}_{n}.npz"
    return f"{PRED}/{job}/{fed}/s{s}_{n}.npz"


ld = lambda p: np.load(p)["test"]
local = {n: {s: ld(f"{PRED}/mlp_local/{n}/s{s}_{n}.npz") for s in SEEDS} for n in CODE.values()}
yu = {n: {s: keys(n, s) for s in SEEDS} for n in CODE.values()}

# Step 1: reproduction check
rep = []
cells = {n: {} for n in CODE.values()}
for i, r in ST.iterrows():
    n = CODE[r.client]
    fed = {s: ld(pred_path(r.job, r.federation, n, s)) for s in SEEDS}
    cells[n][i] = fed
    fpr = np.mean([ap(yu[n][s][0], fed[s]) for s in SEEDS])
    lpr = np.mean([ap(yu[n][s][0], local[n][s]) for s in SEEDS])
    rep.append(dict(row=i, regime=r.regime, federation=r.federation, client=r.client, fed_stored=r.fed_pr_auc,
                    fed_recomputed=fpr, local_stored=r.local_pr_auc, local_recomputed=lpr,
                    max_abs_diff=max(abs(fpr - r.fed_pr_auc), abs(lpr - r.local_pr_auc))))
R = pd.DataFrame(rep)
R.to_csv("results/review2/f1_repro_check.csv", index=False)
bad = R[R.max_abs_diff > 0.001]
print(f"reproduction: {len(R)} cells, max |diff| = {R.max_abs_diff.max():.2e}, cells over 0.001: {len(bad)}", flush=True)
if len(bad) > 3:
    raise SystemExit("more than a few cells do not reproduce - stopping before Step 2")

# Step 2: two-level bootstrap
idx = T.seed_index()
res = {}
for code, n in enumerate(["cert42", "cert52", "spedia", "lanl"]):
    fixed = n != "lanl"
    y = yu[n][0][0] if fixed else {s: yu[n][s][0] for s in SEEDS}
    u = yu[n][0][1] if fixed else {s: yu[n][s][1] for s in SEEDS}
    res.update(T.run_member(code, fixed, y, u, local[n], cells[n], idx))
    print(f"{n}: {len(cells[n])} cells done", flush=True)
rows = []
for i, r in ST.iterrows():
    d = res[i]
    rows.append(dict(group=r.federation, member=r.client, rule=r.regime, job=r.job, planned=bool(r.protocol_cell),
                     mean_diff=r.mean_diff, seed_lo=r.boot_lo, seed_hi=r.boot_hi, old_verdict=r.verdict_boot,
                     old_verdict_holm=r.verdict_holm, old_verdict_holm_all=r.verdict_holm_all,
                     tl_mean=float(np.nanmean(d)), **T.summarise(d)))
O = pd.DataFrame(rows)
fa = O.planned & ~O.job.isin(["fednova_size", "fednova_equal", "mlp_pooled"])
fc = O.job.isin(["fednova_size", "fednova_equal"])
assert fa.sum() == 102 and fc.sum() == 38, (fa.sum(), fc.sum())
for col, mask in (("holm_a", fa), ("holm_b", np.ones(len(O), bool)), ("holm_c", fc)):
    O[col] = np.nan
    O.loc[mask, col] = T.holm(O.loc[mask, "tl_p"].to_numpy())
    O[f"verdict_{col}"] = np.where(O[col].isna(), "", np.where(O[col] < 0.05, np.where(O.mean_diff > 0, "gain", "loss"),
                                                                 "no clear difference"))
O["family_a_102"], O["family_c_fednova"] = fa, fc
O.to_csv("results/review2/stats_twolevel_s0-19.csv", index=False)
pd.set_option("display.width", 250)
print(O.groupby(["rule", "tl_verdict"]).size().unstack(fill_value=0))
print("dropped draws (max over cells):", O.tl_dropped.max())
