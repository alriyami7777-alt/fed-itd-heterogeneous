"""F-4 (review plan of 11 Nov 2026, M3c) analysis: two-level paired bootstrap (twolevel.py) for the label-meaning test.
For each variant (R, L1, L2, L3), rule (FedAvg size, FedAvg equal, private first layer) and member (m1 = A, m2-m4):
effect = federated PR-AUC - that member's local MLP PR-AUC under the SAME variant (A's labels differ per variant;
the other members' labels are as in R). Test sets differ per seed (E-2 partition), so users are resampled inside each
drawn seed's own test set (the LANL path of run_member). Member codes 40+k; the same user draws are used for every
variant, so per-draw effects are paired across variants (used for the descriptive A contrasts L1-L3, L2-L3, L1-R).
Holm within F-4 main family (4 variants x 3 rules x 4 members = 48, seeds 0-19). L2 sensitivity: seeds with A
positives in both training and validation (logged 11 Nov before running), its own Holm family.
Also: reproduction check of every mean PR-AUC vs the run CSVs, A's local PR-AUC per variant, positives per split.
Outputs (results/review2): f4_twolevel.csv, f4_twolevel_L2_sens.csv, f4_contrasts_A.csv, f4_local_A.csv,
f4_positives_summary.csv, f4_repro_check.csv"""
import os as _os, sys as _sys  # repository layout: modules live in ../../src and ../
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_os.path.join(_HERE, "..", "..", "src"), _os.path.join(_HERE, "..")]
from config import ROOT as _ROOT
_os.chdir(_ROOT)  # relative paths below (results/, data/) are relative to FITD_ROOT
import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap

import twolevel as T

R2 = "results/review2"
PRED = "results/preds"
KEYS = f"{PRED}/keys/f4"
VARS = ["R", "L1", "L2", "L3"]
RULES = ["fedavg_size", "fedavg_equal", "fedper_in"]
NAME = {"fedavg_size": "FedAvg (size)", "fedavg_equal": "FedAvg (equal)", "fedper_in": "Private first layer"}
MEMS = ["m1", "m2", "m3", "m4"]
SEEDS = list(range(20))


def labels(v, s):
    f = f"{KEYS}/y_{v}_s{s}.npz"
    if not os.path.exists(f):
        import f4_label as F
        M, _ = F.members(v, s)
        np.savez(f, **{m: M[m]["yte"].astype(np.int8) for m in M})
    z = np.load(f)
    return {m: z[m].astype(int) for m in MEMS}


ld = lambda v, rule, s, m: np.load(f"{PRED}/r2_f4_{v}/{v}_{rule}__{rule}/s{s}_{m}.npz")["test"]
ldl = lambda v, s, m: np.load(f"{PRED}/r2_f4_{v}/{v}_local__local/s{s}_{m}.npz")["test"]
users = {s: np.load(f"{KEYS}/s{s}.npz", allow_pickle=True) for s in SEEDS}
Y = {(v, s): labels(v, s) for v in VARS for s in SEEDS}
CSV = pd.concat([pd.read_csv(f) for f in glob.glob(f"{R2}/f4_*_s*.csv")])

# reproduction check
rep = []
for v in VARS:
    for m in MEMS:
        for rule in ["local"] + RULES:
            g = lambda s: ldl(v, s, m) if rule == "local" else ld(v, rule, s, m)
            rc = np.mean([ap(Y[(v, s)][m], g(s)) for s in SEEDS])
            st = CSV[(CSV.variant == v) & (CSV.client == m) & (CSV.federation == f"{v}_{rule}")].pr_auc.mean()
            rep.append(dict(variant=v, member=m, rule=rule, stored=st, recomputed=rc, abs_diff=abs(st - rc)))
RP = pd.DataFrame(rep)
RP.to_csv(f"{R2}/f4_repro_check.csv", index=False)
print(f"reproduction: {len(RP)} cells, max |diff| = {RP.abs_diff.max():.2e}", flush=True)


SKIP = False


def family(seeds):
    idx = T.seed_index(seeds)
    rows, vecs = [], {}
    for v in VARS:
        for k, m in enumerate(MEMS):
            y = {s: Y[(v, s)][m] for s in seeds}
            u = {s: users[s][f"{m}_test_user"].astype(str) for s in seeds}
            base = {"local": {s: ldl(v, s, m) for s in seeds}}
            cells = {rule: ({s: ld(v, rule, s, m) for s in seeds}, "local") for rule in RULES}
            res = T.run_member(40 + k, False, y, u, base, cells, idx, skip_empty=SKIP)
            for rule in RULES:
                d = res[rule]
                vecs[(v, rule, m)] = d
                fed = np.mean([ap(y[s], cells[rule][0][s]) for s in seeds])
                loc = np.mean([ap(y[s], base["local"][s]) for s in seeds])
                rows.append(dict(variant=v, rule=NAME[rule], member="A (m1)" if m == "m1" else m, n_seeds=len(seeds),
                                 fed=fed, local=loc, mean_diff=fed - loc, tl_mean=float(np.nanmean(d)), **T.summarise(d)))
            print(f"{v} {m} done", flush=True)
    O = pd.DataFrame(rows)
    O["tl_holm"] = T.holm(O.tl_p.to_numpy())
    O["tl_verdict_holm"] = np.where(O.tl_holm < 0.05, np.where(O.mean_diff > 0, "gain", "loss"), "no clear difference")
    return O, vecs


O, V = family(SEEDS)
O.to_csv(f"{R2}/f4_twolevel.csv", index=False)

# descriptive paired contrasts for A (same user draws across variants); not part of the Holm family
con = []
for rule in RULES:
    for a, b in (("L1", "L3"), ("L2", "L3"), ("L1", "L2"), ("L1", "R"), ("L2", "R"), ("L3", "R")):
        d = V[(a, rule, "m1")] - V[(b, rule, "m1")]
        con.append(dict(rule=NAME[rule], contrast=f"A effect {a} - A effect {b}", mean=float(np.nanmean(d)),
                        **T.summarise(d)))
pd.DataFrame(con).to_csv(f"{R2}/f4_contrasts_A.csv", index=False)

# L2 sensitivity: seeds with A positives in both train and val
fz = pd.read_csv(f"{R2}/f4_feasibility.csv")
ok = fz[(fz.variant == "L2") & (fz.pos_tr > 0) & (fz.pos_va > 0)].seed.tolist()
print("L2 sensitivity seeds:", ok, flush=True)
VARS_ALL = VARS
VARS = ["L2"]
S2, _ = family(ok)
VARS = VARS_ALL
S2.to_csv(f"{R2}/f4_twolevel_L2_sens.csv", index=False)

# A's local PR-AUC per variant, and positives per split
la = []
for v in VARS:
    pr = [ap(Y[(v, s)]["m1"], ldl(v, s, "m1")) for s in SEEDS]
    br = [Y[(v, s)]["m1"].mean() for s in SEEDS]
    la.append(dict(variant=v, local_pr_auc_mean=np.mean(pr), local_pr_auc_min=np.min(pr), local_pr_auc_max=np.max(pr),
                   base_rate_mean=np.mean(br), lift_mean=np.mean(np.array(pr) / np.array(br))))
pd.DataFrame(la).to_csv(f"{R2}/f4_local_A.csv", index=False)
fz.groupby("variant")[["pos_tr", "pos_va", "pos_te", "posusers_tr", "posusers_va", "posusers_te"]] \
    .agg(["min", "mean", "max"]).round(1).to_csv(f"{R2}/f4_positives_summary.csv")

# sensitivity (logged 11 Nov 22:55, after the primary run): skip empty seed slots instead of dropping whole draws
import sys
if "--skip" in sys.argv:
    SKIP = True
    Os, Vs = family(SEEDS)
    Os.to_csv(f"{R2}/f4_twolevel_skipempty.csv", index=False)
    cs = []
    for rule in RULES:
        for a, b in (("L1", "L3"), ("L2", "L3"), ("L1", "L2"), ("L1", "R"), ("L2", "R"), ("L3", "R")):
            d = Vs[(a, rule, "m1")] - Vs[(b, rule, "m1")]
            cs.append(dict(rule=NAME[rule], contrast=f"A effect {a} - A effect {b}", mean=float(np.nanmean(d)), **T.summarise(d)))
    pd.DataFrame(cs).to_csv(f"{R2}/f4_contrasts_A_skipempty.csv", index=False)
    VARS = ["L2"]
    S2s, _ = family(ok)
    VARS = VARS_ALL
    S2s.to_csv(f"{R2}/f4_twolevel_L2_sens_skipempty.csv", index=False)
pd.set_option("display.width", 250)
print(O[["variant", "rule", "member", "fed", "local", "mean_diff", "tl_lo", "tl_hi", "tl_verdict", "tl_verdict_holm"]]
      .round(4).to_string())
print(pd.DataFrame(con).round(4).to_string())
print(S2[["rule", "member", "mean_diff", "tl_lo", "tl_hi", "tl_verdict", "tl_verdict_holm"]].round(4).to_string())
print(pd.DataFrame(la).round(4).to_string())
