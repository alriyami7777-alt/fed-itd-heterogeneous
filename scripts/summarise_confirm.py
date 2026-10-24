"""Summaries of the confirmatory runs C-1 and C-2 (change log 2026-10-14 C-1, C-2), seeds 5-19.
Intervals: paired bootstrap over seeds via review_stats.boot_ci (10,000 fixed-seed resamples for n = 15);
* = 95% CI excludes 0. Output: results/review/CONFIRM_summary.md
"""
import os as _os, sys as _sys  # repository layout: modules live in ../src
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import glob
import os

import numpy as np
import pandas as pd

from config import RESULTS
from review_stats import boot_ci

NAMES = {"cert42": "CERT r4.2", "cert52": "CERT r5.2", "spedia": "SPEDIA", "lanl": "LANL"}
REV = os.path.join(RESULTS, "review")


def ci(v):
    v = np.asarray(v, float)
    lo, hi, _ = boot_ci(v)
    return f"{v.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]{'*' if lo > 0 or hi < 0 else ''}"


def rd(pat):
    fs = sorted(glob.glob(os.path.join(REV, pat)))
    return pd.concat([pd.read_csv(f) for f in fs]) if fs else None


L = ["# Confirmatory runs C-1 and C-2 (designs fixed before running; fresh seeds 5-19)", ""]
c1 = rd("confirm_c1_s[0-9]*.csv")
if c1 is not None:
    seeds = sorted(c1.seed.unique())
    ctrl = c1[c1.regime == "spedia_alone_S100"].set_index("seed").pr_auc
    r0 = rd("mlp_local_s[0-9]*.csv")
    r0 = r0[r0.client == "spedia"].set_index("seed").pr_auc
    L += [f"## C-1 Extra local steps for SPEDIA only ({len(seeds)} seeds: {seeds[0]}-{seeds[-1]})", "",
          f"SPEDIA alone, same schedule (S = 100 steps/round, own-validation selection): PR-AUC {ctrl.mean():.3f}; "
          f"SPEDIA alone, standard local MLP (R0), same seeds: {r0.loc[ctrl.index].mean():.3f}.", "",
          "**Primary outcome:** SPEDIA in federation minus SPEDIA alone on the same schedule (paired by seed).", "",
          "| Weights | Federation | SPEDIA federated | Δ vs same-schedule control [95% CI] | Δ vs standard local R0 |",
          "|---|---|---|---|---|"]
    sp = c1[(c1.client == "spedia") & (c1.regime != "spedia_alone_S100")]
    for (w, fed), g in sp.groupby(["weighting", "federation"], sort=False):
        g = g.set_index("seed").pr_auc
        L.append(f"| {w} | {fed.rsplit('_', 1)[0]} | {g.mean():.3f} | {ci(g - ctrl.loc[g.index])} | "
                 f"{ci(g - r0.loc[g.index])} |")
    # secondary: other clients vs the main runs (same seeds, same federation, frozen FedAvg)
    L += ["", "**Secondary:** other clients in the same federations, compared with the main FedAvg runs on the same "
          "seeds (note: here each client is selected on its own validation set; in the main runs on the mean).", "",
          "| Weights | Federation | Client | C-1 PR-AUC | Main FedAvg PR-AUC | Δ [95% CI] |", "|---|---|---|---|---|---|"]
    oth = c1[(c1.client != "spedia") & (c1.regime != "spedia_alone_S100")]
    for (w, fed, cl), g in oth.groupby(["weighting", "federation", "client"], sort=False):
        base = rd(f"fedavg_{w}_s[0-9]*.csv")
        f0 = fed.rsplit("_", 1)[0]
        b = base[(base.federation == f0) & (base.client == cl)].set_index("seed").pr_auc
        g = g.set_index("seed").pr_auc
        k = [s for s in g.index if s in b.index]
        if len(k) >= 2:
            L.append(f"| {w} | {f0} | {NAMES[cl]} | {g[k].mean():.3f} | {b[k].mean():.3f} | {ci(g[k] - b[k])} |")
    L.append("")

c2 = rd("confirm_c2_s[0-9]*.csv")
if c2 is not None:
    seeds = sorted(c2.seed.unique())
    d = c2[c2.client == "cert42"]
    L += [f"## C-2 Low-data CERT r4.2 ({len(seeds)} seeds: {seeds[0]}-{seeds[-1]})", "",
          "Scaler fitted on the subsample only; validation and test sets unchanged (full validation set). "
          "Baselines: (a) CERT r4.2 local on the subsample; (b) zero-shot CERT r5.2 -> CERT r4.2. "
          "**Primary outcome:** federated minus the better baseline (better = higher mean PR-AUC at that fraction).", "",
          "| Share | Train rows (pos) | (a) Local | (b) Zero-shot r5.2 | FedAvg size | Δ size − better [95% CI] | "
          "FedAvg equal | Δ equal − better [95% CI] | Δ equal − local [95% CI] |", "|---|---|---|---|---|---|---|---|---|"]
    for f, g in d.groupby("frac"):
        a = g[g.regime == "local_subsample"].set_index("seed").pr_auc
        z = g[g.regime == "zeroshot_r52"].set_index("seed").pr_auc
        s = g[g.regime == "fedavg_size"].set_index("seed").pr_auc
        e = g[g.regime == "fedavg_equal"].set_index("seed").pr_auc
        best = a if a.mean() >= z.mean() else z
        nm = "(a)" if best is a else "(b)"
        k = sorted(set(a.index) & set(z.index) & set(s.index) & set(e.index))
        if len(k) < 2:
            continue
        L.append(f"| {f:.0%} | {g.n_train.mean():,.0f} ({g.pos_train.mean():.0f}) | {a[k].mean():.3f} | {z[k].mean():.3f} | "
                 f"{s[k].mean():.3f} | {ci(s[k] - best[k])} vs {nm} | {e[k].mean():.3f} | {ci(e[k] - best[k])} vs {nm} | "
                 f"{ci(e[k] - a[k])} |")
    L.append("")
open(os.path.join(REV, "CONFIRM_summary.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
