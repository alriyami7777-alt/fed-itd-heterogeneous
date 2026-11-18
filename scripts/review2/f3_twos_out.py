"""F-3 (review plan of 11 Nov 2026): the missing transfer cells TWOS -> C1..C4. For each seed, TWOS's local MLP (R0) is retrained exactly as
in E-11 (e11_prep(seed, 'day'), train_mlp with the same seed and settings) and scored on each other member's test set
(that member's own training scaler, the R5 rule). Reproduction check: the retrained TWOS model's TWOS test scores
are compared with the stored E-11 R0 predictions. Thread count is set as in the original run (seeds 0-4: 1 thread,
5-19: 2 threads) via FITD_THREADS before import.
Usage: FITD_THREADS=1 python3 f3_twos_out.py 0-4 ; FITD_THREADS=2 python3 f3_twos_out.py 5-19
Output: results/preds/f3_transfer/twos/s<seed>_<target>.npz, results/review2/f3_twos_repro_s<a>-<b>.csv"""
import os as _os, sys as _sys  # repository layout: modules live in ../../src and ../
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_os.path.join(_HERE, "..", "..", "src"), _os.path.join(_HERE, "..")]
from config import ROOT as _ROOT
_os.chdir(_ROOT)  # relative paths below (results/, data/) are relative to FITD_ROOT
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap

from run_e1_e2 import class_weights, predict, train_mlp
from run_r2 import e11_prep

lo, hi = map(int, sys.argv[1].split("-"))
OUT = "results/preds/f3_transfer/twos"
os.makedirs(OUT, exist_ok=True)
rows = []
for s in range(lo, hi + 1):
    P, _ = e11_prep(s, "day")
    c = P["twos"]
    m, v, ep = train_mlp(c["Xtr"], c["ytr"], class_weights(c["ytr"]), [(c["Xva"], c["yva"])], s)
    pt = predict(m, c["Xte"])
    old = np.load(f"results/preds/r2_e11_main/twos__R0_local/s{s}_twos.npz")["test"]
    r = dict(seed=s, threads=os.environ.get("FITD_THREADS", "2"), pr_new=ap(c["yte"], pt), pr_stored=ap(c["yte"], old),
             max_abs_score_diff=float(np.abs(pt - old).max()))
    for t in ("cert42", "cert52", "spedia", "lanl"):
        ps = predict(m, P[t]["Xte"])
        np.savez(f"{OUT}/s{s}_{t}.npz", test=np.asarray(ps, np.float32))
        r[f"pr_{t}"] = ap(P[t]["yte"], ps)
    rows.append(r)
    print(r, flush=True)
pd.DataFrame(rows).to_csv(f"results/review2/f3_twos_repro_s{lo}-{hi}.csv", index=False)
