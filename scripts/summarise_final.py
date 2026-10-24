"""Compact tables for the results-and-discussion note: regime comparison in the all-four federation,
and verdict counts per regime over all 17 client-federation cells. Paired seed bootstrap (exact, 3,125 resamples for 5 seeds)."""
import os as _os, sys as _sys  # repository layout: modules live in ../src
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import os

import numpy as np
import pandas as pd

from config import RESULTS as R
NAMES = {"cert42": "CERT r4.2", "cert52": "CERT r5.2", "spedia": "SPEDIA", "lanl": "LANL"}
ORDER = ["cert42", "cert52", "spedia", "lanl"]
JOBS = [("fedavg_size", "FedAvg (size)"), ("fedavg_equal", "FedAvg (equal)"), ("fedprox", "FedProx"),
        ("fedper", "FedPer (local output)"), ("fedper_in", "Local input layer (C2-b)"),
        ("finetune", "FedAvg + fine-tune")]


def boot(v, *_, **__):
    """Paired bootstrap 95% CI of the mean: exact enumeration when n = 5 (3,125 resamples), otherwise 10,000
    fixed-seed resamples (review_stats.boot_ci). Replaces the earlier 5,000 random draws."""
    from review_stats import boot_ci
    v = np.asarray(v, float)
    lo, hi, _ = boot_ci(v)
    return v.mean(), lo, hi


r0 = pd.read_csv(f"{R}/e1e2_mlp_local.csv").set_index(["client", "seed"]).pr_auc
r1 = pd.read_csv(f"{R}/e1e2_mlp_pooled.csv").set_index(["client", "seed"]).pr_auc
L = ["### All-four federation: PR-AUC per client (mean of 5 seeds; Δ vs local-only MLP with 95% CI)", "",
     "| Regime | " + " | ".join(NAMES[c] for c in ORDER) + " |", "|---" * 5 + "|",
     "| Local-only MLP (R0) | " + " | ".join(f"{r0.loc[c].mean():.3f}" for c in ORDER) + " |"]


def cell(s, c):
    k = [x for x in s.index if (c, x) in r0.index]
    mu, lo, hi = boot([s[x] - r0.loc[(c, x)] for x in k])
    star = "*" if (lo > 0 or hi < 0) else ""
    return f"{s.mean():.3f} ({mu:+.3f}{star})"


L.append("| Pooled MLP (R1, no privacy) | " + " | ".join(cell(r1.loc[c], c) for c in ORDER) + " |")
counts = []
for job, lab in JOBS:
    f = f"{R}/fl_{job}.csv"
    if not os.path.exists(f):
        L.append(f"| {lab} | pending | pending | pending | pending |")
        continue
    d = pd.read_csv(f)
    if d.seed.nunique() < 5:
        L.append(f"| {lab} | running | | | |")
        continue
    a = d[d.federation == "C1+C2+C3+C4"]
    L.append(f"| {lab} | " + " | ".join(cell(a[a.client == c].set_index("seed").pr_auc, c) for c in ORDER) + " |")
    g = n = z = 0
    sp = []
    for (fed, c), x in d.groupby(["federation", "client"]):
        x = x.set_index("seed").pr_auc
        mu, lo, hi = boot([x[s] - r0.loc[(c, s)] for s in x.index])
        g += lo > 0
        n += hi < 0
        z += not (lo > 0 or hi < 0)
        if c == "spedia":
            sp.append(mu)
    counts.append(f"| {lab} | {g} | {z} | {n} | {np.mean(sp):+.3f} |")
L += ["", "Cells show PR-AUC (Δ vs local-only; * = 95% CI excludes 0).", "",
      "### Verdicts over all 17 client-in-federation cells", "",
      "| Regime | Gain | No clear difference | Negative transfer | Mean Δ for SPEDIA (5 federations) |",
      "|---|---|---|---|---|"] + counts
open(f"{R}/FINAL_tables.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
