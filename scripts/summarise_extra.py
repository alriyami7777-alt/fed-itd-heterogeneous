"""Summaries for the post-freeze additions (change log h): low-data CERT r4.2 and the step-budget control."""
import os as _os, sys as _sys  # repository layout: modules live in ../src
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import os

import numpy as np
import pandas as pd

from config import RESULTS as R
NAMES = {"cert42": "CERT r4.2", "cert52": "CERT r5.2", "spedia": "SPEDIA", "lanl": "LANL"}


def boot(v, *_, **__):
    """Paired bootstrap 95% CI of the mean: exact enumeration when n = 5 (3,125 resamples), otherwise 10,000
    fixed-seed resamples (review_stats.boot_ci). Replaces the earlier 5,000 random draws."""
    from review_stats import boot_ci
    v = np.asarray(v, float)
    lo, hi, _ = boot_ci(v)
    return v.mean(), lo, hi


L = []
p = f"{R}/fl_lowdata.csv"
if os.path.exists(p):
    d = pd.read_csv(p)
    d = d[d.client == "cert42"]
    L += ["## E8 Low-data CERT r4.2 federated with full CERT r5.2", "",
          "Δ = federated − CERT r4.2 local-only MLP trained on the same subset (paired by seed).", "",
          "| r4.2 training share | Train rows (pos) | Local-only on subset | FedAvg (size) | Δ [95% CI] | FedAvg (equal) | Δ [95% CI] | Local input layer | Δ [95% CI] |",
          "|---|---|---|---|---|---|---|---|---|"]
    for f, g in d.groupby("frac"):
        loc = g[g.regime == "local"].set_index("seed").pr_auc
        row = f"| {f:.0%} | {g.n_train.mean():,.0f} ({g.pos_train.mean():.0f}) | {loc.mean():.3f} |"
        for reg in ("fedavg_size", "fedavg_equal", "local_input"):
            x = g[g.regime == reg].set_index("seed").pr_auc
            k = [s for s in x.index if s in loc.index]
            if len(k) < 2:
                row += " running | |"
                continue
            mu, lo, hi = boot([x[s] - loc[s] for s in k])
            row += f" {x.mean():.3f} | {mu:+.3f} [{lo:+.3f}, {hi:+.3f}]{'*' if lo > 0 or hi < 0 else ''} |"
        L.append(row)
    L += ["", "Full-data reference: CERT r4.2 local-only 0.530; with r5.2 under FedAvg (equal) 0.574.", ""]

p = f"{R}/fl_eqsteps_local.csv"
if os.path.exists(p):
    c = pd.read_csv(p).set_index(["client", "seed"]).pr_auc
    r0 = pd.read_csv(f"{R}/e1e2_mlp_local.csv").set_index(["client", "seed"]).pr_auc
    L += ["## E9 control: equal-steps federation vs each client alone on the same 100-steps-per-round schedule", "",
          "| Federation | Client | Federated (equal steps) | Alone, same schedule | Δ vs same-schedule local [95% CI] | Alone, standard epochs (R0) |",
          "|---|---|---|---|---|---|"]
    for job in ("eqsteps", "eqsteps_in"):
        q = f"{R}/fl_{job}.csv"
        if not os.path.exists(q):
            continue
        e = pd.read_csv(q)
        L.append(f"| **{job}** | | | | | |")
        for (fed, cl), g in e.groupby(["federation", "client"], sort=False):
            g = g.set_index("seed").pr_auc
            k = [s for s in g.index if (cl, s) in c.index]
            if len(k) < 2:
                continue
            mu, lo, hi = boot([g[s] - c.loc[(cl, s)] for s in k])
            L.append(f"| {fed} | {NAMES[cl]} | {g.mean():.3f} | {c.loc[cl].mean():.3f} | {mu:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                     f"{'*' if lo > 0 or hi < 0 else ''} | {r0.loc[cl].mean():.3f} |")
p = f"{R}/fl_fedper_in_ft.csv"
if os.path.exists(p):
    e = pd.read_csv(p)
    r0 = pd.read_csv(f"{R}/e1e2_mlp_local.csv").set_index(["client", "seed"]).pr_auc
    L += ["## Private first layer + local fine-tuning of all layers (<=10 epochs, patience 3)", "",
          "| Federation | Client | Federated PR-AUC | Local-only (R0) | Δ [95% CI] |", "|---|---|---|---|---|"]
    nl = 0
    for (fed, cl), g in e.groupby(["federation", "client"], sort=False):
        g = g.set_index("seed").pr_auc
        mu, lo, hi = boot([g[s] - r0.loc[(cl, s)] for s in g.index])
        nl += hi < 0
        L.append(f"| {fed} | {NAMES[cl]} | {g.mean():.3f} | {r0.loc[cl].mean():.3f} | {mu:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                 f"{'*' if lo > 0 or hi < 0 else ''} |")
    L += ["", f"Clear losses: {nl} of {e.groupby(['federation', 'client']).ngroups}.", ""]

HEAD = ["# Exploratory runs (change log 2026-10-14 (h)); outside the frozen protocol; 5 seeds (0-4)", "",
        "Intervals: exact paired bootstrap over 5 seeds (3,125 resamples); * = 95% CI excludes 0.", "",
        "Design notes (second independent audit, items A06, A07): equal-steps runs draw each 256-row minibatch WITH replacement",
        "(SPEDIA, 155 rows: ~100 bootstrap batches per round; CERT r4.2 / r5.2: 13% / 6% of an epoch per round) and",
        "select the round on the mean validation PR-AUC across clients, while the local control selects on its own",
        "validation set. Low-data runs fit the scaler on 100% of CERT r4.2 training rows and select on the full",
        "validation set (33,851 rows, 179 positives); zero-shot CERT r5.2 -> CERT r4.2 with 0% of CERT r4.2 labels",
        "already scores 0.356. Confirmatory runs with corrected designs: change log entries C-1 / C-2.", ""]
L = HEAD + L
open(f"{R}/EXTRA_summary.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
