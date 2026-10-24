"""Summarise E1/E2 results: per-client mean PR-AUC (seed bootstrap 95% CI) and pooled-minus-local
paired deltas (exact paired bootstrap over seeds: 3,125 resamples for 5 seeds)."""
import os as _os, sys as _sys  # repository layout: modules live in ../src
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import glob

import numpy as np
import pandas as pd

from config import RESULTS as R


def boot_mean(v, *_, **__):
    """Paired bootstrap 95% CI of the mean: exact enumeration when n = 5 (3,125 resamples), otherwise 10,000
    fixed-seed resamples (review_stats.boot_ci). Replaces the earlier 5,000 random draws."""
    from review_stats import boot_ci
    v = np.asarray(v, float)
    lo, hi, _ = boot_ci(v)
    return v.mean(), lo, hi


fs = sorted(glob.glob(f"{R}/e1e2_*.csv"))
d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()
lines = ["# E1–E2 results (local-only vs pooled), protocol v1.0 FROZEN", "",
         f"Parts finished: {', '.join(sorted(d.part.unique())) if len(d) else 'none'}", ""]
if len(d):
    order = ["cert42", "cert52", "spedia", "lanl"]
    names = {"cert42": "CERT r4.2", "cert52": "CERT r5.2", "spedia": "SPEDIA", "lanl": "LANL (user-disjoint)"}
    lines += ["## PR-AUC per client (mean over seeds, 95% bootstrap CI; lift = PR-AUC ÷ base rate)", "",
              "| Client | Base rate | " + " | ".join(f"{m} {r.split('_')[1]}" for (r, m) in
                                                    sorted(d.groupby(['regime', 'model']).groups)) + " |",
              "|---|---|" + "---|" * d.groupby(['regime', 'model']).ngroups]
    combos = sorted(d.groupby(['regime', 'model']).groups)
    for c in order:
        sub = d[d.client == c]
        if not len(sub):
            continue
        cells = []
        for (r, m) in combos:
            v = sub[(sub.regime == r) & (sub.model == m)].pr_auc
            if len(v) == 0:
                cells.append("—")
                continue
            mu, lo, hi = boot_mean(v)
            cells.append(f"{mu:.3f} [{lo:.3f}, {hi:.3f}] (n={len(v)})")
        lines.append(f"| {names[c]} | {sub.base_rate.mean():.4f} | " + " | ".join(cells) + " |")
    lines += ["", "## Pooled minus local-only (paired by seed)", "",
              "| Client | Model | Δ PR-AUC | 95% CI | Verdict |", "|---|---|---|---|---|"]
    for m in sorted(d.model.unique()):
        a = d[(d.model == m) & (d.regime == "R0_local")]
        b = d[(d.model == m) & (d.regime == "R1_pooled")]
        if not len(a) or not len(b):
            continue
        for c in order:
            x = a[a.client == c].set_index("seed").pr_auc
            y = b[b.client == c].set_index("seed").pr_auc
            k = x.index.intersection(y.index)
            if len(k) < 2:
                continue
            mu, lo, hi = boot_mean((y[k] - x[k]).to_numpy())
            verdict = "pooled better" if lo > 0 else ("pooled worse" if hi < 0 else "no clear difference")
            lines.append(f"| {names[c]} | {m} | {mu:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {verdict} |")
open(f"{R}/E1E2_summary.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
