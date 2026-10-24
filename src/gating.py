"""E11 (post hoc, no new training): similarity-gated joining.
Rule: organisations federate only with partners whose Jaccard overlap of observed action types is >= tau;
an organisation with no such partner stays local. With the observed overlaps (CERT pair 0.78; every other
pair <= 0.43), any tau in (0.43, 0.78] forms one federation {CERT r4.2, CERT r5.2} and leaves SPEDIA and
LANL local. Compared with 'all four join' and 'all stay local', for each federated method.
Per client: delta vs local-only MLP, paired by seed, exact paired bootstrap (3,125 resamples for 5 seeds)."""
import os

import numpy as np
import pandas as pd

from config import RESULTS as R
NAMES = {"cert42": "CERT r4.2", "cert52": "CERT r5.2", "spedia": "SPEDIA", "lanl": "LANL"}
ORDER = list(NAMES)
JOBS = [("fedavg_size", "FedAvg (size)"), ("fedavg_equal", "FedAvg (equal)"), ("fedprox", "FedProx"),
        ("fedper", "FedPer"), ("fedper_in", "Local input layer"), ("finetune", "FedAvg + fine-tune")]
r0 = pd.read_csv(f"{R}/e1e2_mlp_local.csv").set_index(["client", "seed"]).pr_auc


def boot(v, *_, **__):
    """Paired bootstrap 95% CI of the mean: exact enumeration when n = 5 (3,125 resamples), otherwise 10,000
    fixed-seed resamples (review_stats.boot_ci). Replaces the earlier 5,000 random draws."""
    from review_stats import boot_ci
    v = np.asarray(v, float)
    lo, hi, _ = boot_ci(v)
    return v.mean(), lo, hi


def deltas(d, fed, c):
    x = d[(d.federation == fed) & (d.client == c)].set_index("seed").pr_auc
    return np.array([x[s] - r0.loc[(c, s)] for s in x.index])


L = ["| Method | Policy | " + " | ".join(f"Δ {NAMES[c]}" for c in ORDER) + " | Clients losing | Mean Δ (4 clients) |",
     "|---|---|" + "---|" * 4 + "---|---|"]
for job, lab in JOBS:
    p = f"{R}/fl_{job}.csv"
    if not os.path.exists(p):
        continue
    d = pd.read_csv(p)
    for pol in ("All four join", "Gated (overlap ≥ τ)"):
        cells, lose, means = [], 0, []
        for c in ORDER:
            fed = "C1+C2+C3+C4" if pol == "All four join" else ("C1+C2" if c in ("cert42", "cert52") else None)
            if fed is None:
                cells.append("0 (local)")
                means.append(0.0)
                continue
            mu, lo, hi = boot(deltas(d, fed, c))
            lose += hi < 0
            means.append(mu)
            cells.append(f"{mu:+.3f}{'*' if lo > 0 or hi < 0 else ''}")
        L.append(f"| {lab} | {pol} | " + " | ".join(cells) + f" | {lose} | {np.mean(means):+.3f} |")
L.append("| — | All stay local | 0 | 0 | 0 | 0 | 0 | 0.000 |")
txt = "\n".join(L)
open(f"{R}/gating_summary.md", "w").write("## E11 Similarity-gated joining (post hoc)\n\n" + txt + "\n\n* = 95% CI excludes 0.\n")
print(txt)
