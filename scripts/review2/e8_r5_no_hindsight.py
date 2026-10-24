"""E-8 (doc 31): R5 without hindsight, and a real ensemble. No training: uses the saved zero-shot (R5) and
leave-one-out (R6) predictions of the 20-seed review runs (results/preds/zeroshot_mlp, results/preds/loo).

For each target t and seed:
  oracle R5      best single source by mean TEST PR-AUC over seeds (hindsight; as in the paper)
  average source mean over the three sources of their test PR-AUC
  (a) selected   the source s whose model has the best mean validation PR-AUC on the validation sets of the OTHER
                 candidate sources (the sources other than s and t); no target labels used. Chosen per seed.
  (b) ensemble   each source model's target test scores turned into within-target ranks (average ranks for ties,
                 divided by n), averaged over the three sources, then scored (PR-AUC).
  R6             leave-one-out FedAvg (size) model scored on t.
Paired differences vs R6 and vs oracle R5: 10,000-resample bootstrap (seed 20261014), paired t; Holm within E-8.
Output: results/review2/e8_per_seed.csv, results/review2/e8_summary.csv
"""
import os as _os, sys as _sys  # repository layout: modules live in ../../src and ../
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_os.path.join(_HERE, "..", "..", "src"), _os.path.join(_HERE, "..")]
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

from config import RESULTS
from review_stats import boot_ci, holm, t_ci, verdict

PRED = os.path.join(RESULTS, "preds")
ALL = ("cert42", "cert52", "spedia", "lanl")
NAMES = {"cert42": "CERT r4.2", "cert52": "CERT r5.2", "spedia": "SPEDIA", "lanl": "LANL"}


def keyfile(n, s):
    return os.path.join(PRED, "keys", f"{n}_{'s%d' % s if n == 'lanl' else 'temporal'}.npz")


rows = []
for s in range(20):
    for t in ALL:
        k = np.load(keyfile(t, s), allow_pickle=True)
        yte = k["yte"].astype(int)
        srcs = [x for x in ALL if x != t]
        test = {x: np.load(os.path.join(PRED, "zeroshot_mlp", x, f"s{s}_{t}.npz"))["test"] for x in srcs}
        pr = {x: average_precision_score(yte, test[x]) for x in srcs}
        sel_val = {}
        for x in srcs:
            others = [o for o in srcs if o != x]
            v = []
            for o in others:
                ko = np.load(keyfile(o, s), allow_pickle=True)
                v.append(average_precision_score(ko["yva"].astype(int),
                                                 np.load(os.path.join(PRED, "zeroshot_mlp", x, f"s{s}_{o}.npz"))["val"]))
            sel_val[x] = float(np.mean(v))
        chosen = max(srcs, key=lambda x: sel_val[x])
        ens = np.mean([rankdata(test[x]) / len(yte) for x in srcs], axis=0)
        loo_dir = "-".join(x for x in ALL if x != t)
        loo = np.load(os.path.join(PRED, "loo", loo_dir, f"s{s}_{t}.npz"))["test"]
        rows.append(dict(seed=s, target=t, **{f"r5_{x}": pr[x] for x in srcs}, avg_source=float(np.mean(list(pr.values()))),
                         selected_source=chosen, selected=pr[chosen], ensemble=float(average_precision_score(yte, ens)),
                         r6=float(average_precision_score(yte, loo)),
                         **{f"selval_{x}": sel_val[x] for x in srcs}))
D = pd.DataFrame(rows)
for t in ALL:
    m = D[D.target == t]
    best = max([x for x in ALL if x != t], key=lambda x: m[f"r5_{x}"].mean())
    D.loc[D.target == t, "oracle_r5"] = m[f"r5_{best}"].values
    D.loc[D.target == t, "oracle_source"] = best
os.makedirs(os.path.join(RESULTS, "review2"), exist_ok=True)
D.to_csv(os.path.join(RESULTS, "review2", "e8_per_seed.csv"), index=False)
out = []
for t in ALL:
    m = D[D.target == t]
    for a in ("selected", "ensemble"):
        for b in ("r6", "oracle_r5", "avg_source"):
            d = (m[a] - m[b]).to_numpy()
            lo, hi, _ = boot_ci(d)
            tlo, thi, p = t_ci(d)
            out.append(dict(target=NAMES[t], method=a, vs=b, method_mean=m[a].mean(), vs_mean=m[b].mean(),
                            diff=d.mean(), boot_lo=lo, boot_hi=hi, verdict=verdict(lo, hi), t_p=p, n=len(d),
                            oracle_source=NAMES[m.oracle_source.iloc[0]],
                            selected_counts=m.selected_source.map(NAMES).value_counts().to_dict() if a == "selected" else ""))
O = pd.DataFrame(out)
O["t_p_holm"] = holm(O.t_p.to_numpy())
O["verdict_holm"] = np.where(O.t_p_holm < 0.05, np.where(O["diff"] > 0, "gain", "loss"), "no clear difference")
O.to_csv(os.path.join(RESULTS, "review2", "e8_summary.csv"), index=False)
print(O.round(3).to_string(index=False))
