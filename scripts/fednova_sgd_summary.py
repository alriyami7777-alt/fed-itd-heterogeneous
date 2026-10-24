"""FedNova with plain local SGD (lr 0.01) vs local-only MLP and vs FedNova with Adam, seeds 0-4 (change log D-1 check).
Paired by seed; exact bootstrap (3,125 resamples) via review_stats.boot_ci; * = 95% CI excludes 0. Descriptive.
Output: results/review/fednova_sgd_s0-4.csv and results/review/FEDNOVA_SGD_summary.md
"""
import os as _os, sys as _sys  # repository layout: modules live in ../src
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import glob
import os

import numpy as np
import pandas as pd

from config import RESULTS
from review_stats import boot_ci, verdict

REV = os.path.join(RESULTS, "review")
NAMES = {"cert42": "CERT r4.2", "cert52": "CERT r5.2", "spedia": "SPEDIA", "lanl": "LANL"}
SEEDS = range(5)


def rd(job):
    fs = sorted(glob.glob(f"{REV}/{job}_s[0-9]*.csv")) + sorted(glob.glob(f"{REV}/c2c4_{job}_s[0-9]*.csv"))
    d = pd.concat([pd.read_csv(f) for f in fs])
    return d[d.seed.isin(SEEDS)]


r0 = rd("mlp_local").set_index(["client", "seed"]).pr_auc
rows = []
for w in ("size", "equal"):
    sgd, adam = rd(f"fednova_sgd_{w}"), rd(f"fednova_{w}")
    for (fed, c), g in sgd.groupby(["federation", "client"], sort=False):
        g = g.set_index("seed").pr_auc
        a = adam[(adam.federation == fed) & (adam.client == c)].set_index("seed").pr_auc
        k = sorted(s for s in g.index if s in a.index and (c, s) in r0.index)
        if len(k) < 2:
            continue
        d0 = np.array([g[s] - r0.loc[(c, s)] for s in k])
        da = np.array([g[s] - a[s] for s in k])
        lo0, hi0, kind = boot_ci(d0)
        loa, hia, _ = boot_ci(da)
        rows.append(dict(weighting=w, federation=fed, client=NAMES[c], n_seeds=len(k),
                         local=np.mean([r0.loc[(c, s)] for s in k]), fednova_adam=a[k].mean(), fednova_sgd=g[k].mean(),
                         diff_vs_local=d0.mean(), lo_local=lo0, hi_local=hi0, verdict_vs_local=verdict(lo0, hi0),
                         diff_vs_adam=da.mean(), lo_adam=loa, hi_adam=hia, verdict_vs_adam=verdict(loa, hia),
                         boot_kind=kind))
S = pd.DataFrame(rows)
S["protocol_cell"] = S.federation != "C2+C4"
S.to_csv(f"{REV}/fednova_sgd_s0-4.csv", index=False)

L = ["# FedNova with plain local SGD (lr 0.01), seeds 0-4", ""]
for w in ("size", "equal"):
    x = S[S.weighting == w]
    for scope, y in (("17 protocol cells", x[x.protocol_cell]), ("19 incl. C2+C4", x)):
        vl, va = y.verdict_vs_local.value_counts(), y.verdict_vs_adam.value_counts()
        L.append(f"- {w}, {scope} (n seeds = {sorted(set(y.n_seeds))}): vs local {vl.get('loss', 0)} losses / "
                 f"{vl.get('gain', 0)} gains; vs FedNova-Adam {va.get('loss', 0)} lower / {va.get('gain', 0)} higher")
L.append("")
L.append("| Weights | Federation | Client | Local | FedNova Adam | FedNova SGD | SGD − local [95% CI] | SGD − Adam [95% CI] |")
L.append("|---|---|---|---|---|---|---|---|")
f = lambda m, lo, hi: f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]{'*' if lo > 0 or hi < 0 else ''}"
for r in S.itertuples():
    L.append(f"| {r.weighting} | {r.federation} | {r.client} | {r.local:.3f} | {r.fednova_adam:.3f} | {r.fednova_sgd:.3f} | "
             f"{f(r.diff_vs_local, r.lo_local, r.hi_local)} | {f(r.diff_vs_adam, r.lo_adam, r.hi_adam)} |")
open(f"{REV}/FEDNOVA_SGD_summary.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
