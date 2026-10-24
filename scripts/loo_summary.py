"""R6 leave-one-out summary over seeds 0-19 (reconstructed 24 Oct 2026 as a script; the 18 Oct files were written by
an inline helper that was not saved). Writes *_regenerated.csv next to the 18 Oct files; --check compares with them.
Check result (24 Oct): every mean and label reproduces exactly; the bootstrap CIs differ by <= 0.002 because the lost
helper drew its resamples differently. No verdict changes (same CI signs).

Per unseen organisation c (the member left out of the federation, scored zero-shot):
  loo_pr_auc        mean over seeds of the R6 leave-one-out FedAvg (size) model's test PR-AUC on c, with a 95% bootstrap CI
  best_single_source the single source whose R5 zero-shot model has the highest mean test PR-AUC on c (hindsight)
  loo_minus_best    paired (by seed) difference R6 - that source, with its 95% bootstrap CI
  mean_over_sources mean over the three sources of their mean R5 test PR-AUC on c
  own_local_pr_auc  c's own local MLP (R0), mean test PR-AUC
CIs: review_stats.boot_ci (10,000 resamples, seed 20261014)."""
import os as _os, sys as _sys  # repository layout: modules live in ../src
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import glob
import os
import sys

import numpy as np
import pandas as pd

from config import RESULTS
from review_stats import boot_ci

REV = os.path.join(RESULTS, "review")
NAMES = {"cert42": "CERT r4.2", "cert52": "CERT r5.2", "spedia": "SPEDIA", "lanl": "LANL"}
SEEDS = range(20)


def rd(job):
    return pd.concat([pd.read_csv(f"{REV}/{job}_s{s}.csv") for s in SEEDS])


loo, z, loc = rd("loo"), rd("zeroshot_mlp"), rd("mlp_local")
ALL = loo.sort_values(["client", "seed"]).reset_index(drop=True)
rows = []
for c in NAMES:
    l = loo[loo.client == c].set_index("seed").pr_auc.sort_index()
    zc = z[z.client == c]
    means = zc.groupby("source").pr_auc.mean()
    best = means.idxmax()
    b = zc[zc.source == best].set_index("seed").pr_auc.sort_index()
    k = sorted(set(l.index) & set(b.index))
    llo, lhi, kind = boot_ci(l[k])
    dlo, dhi, _ = boot_ci(l[k] - b[k])
    rows.append(dict(unseen_client=NAMES[c], trained_on=loo[loo.client == c].federation.iloc[0], n_seeds=len(k),
                     loo_pr_auc=l[k].mean(), loo_ci_lo=llo, loo_ci_hi=lhi,
                     base_rate=float(loo[loo.client == c].base_rate.iloc[0]), best_single_source=NAMES[best],
                     best_single_pr_auc=b[k].mean(), loo_minus_best=(l[k] - b[k]).mean(), diff_ci_lo=dlo, diff_ci_hi=dhi,
                     mean_over_sources=means.mean(), own_local_pr_auc=loc[(loc.client == c) & (loc.federation == c)].pr_auc.mean(),
                     ci_method=kind))
S = pd.DataFrame(rows)
fa, fs = f"{REV}/loo_all_seeds_s0-19.csv", f"{REV}/loo_summary_s0-19.csv"
OUT_A, OUT_S = fa.replace(".csv", "_regenerated.csv"), fs.replace(".csv", "_regenerated.csv")
if "--check" in sys.argv:
    old = pd.read_csv(fs)
    num = [c for c in S.columns if S[c].dtype.kind == "f"]
    print("summary: max |diff| over numeric columns =", float(np.nanmax(np.abs(S[num].to_numpy() - old[num].to_numpy()))))
    print("summary: text columns equal =", all((S[c].astype(str) == old[c].astype(str)).all()
                                               for c in S.columns if c not in num))
    oa = pd.read_csv(fa)
    print("all seeds: rows", len(oa), "vs", len(ALL), "; max |pr_auc diff| =",
          float(np.abs(np.sort(oa.pr_auc.to_numpy()) - np.sort(ALL.pr_auc.to_numpy())).max()))
else:
    ALL.to_csv(OUT_A, index=False)
    S.to_csv(OUT_S, index=False)
    print(S.to_string(index=False))
