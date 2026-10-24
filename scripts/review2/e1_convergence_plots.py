"""E-1 convergence plots: per-round validation PR-AUC per member, four-member group (C1+C2+C3+C4), seeds 0-4,
for the four SGD rules (one pass per round) and the fixed-step variant (S = 100). Dashed line: the member's SGD local
baseline validation PR-AUC (best epoch, mean over seeds 0-4). Output: results/review2/plots/e1_convergence_*.png"""
import os as _os, sys as _sys  # repository layout: modules live in ../../src and ../
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_os.path.join(_HERE, "..", "..", "src"), _os.path.join(_HERE, "..")]
from config import ROOT as _ROOT
_os.chdir(_ROOT)  # relative paths below (results/, data/) are relative to FITD_ROOT
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

H = "results/review2/hist"
NAMES = {"cert42": "CERT r4.2", "cert52": "CERT r5.2", "spedia": "SPEDIA", "lanl": "LANL"}
RULES = ["fedavg_size", "fedavg_equal", "fednova_size", "fednova_equal"]
lr = json.load(open("results/review2/e1_lr.json"))
loc = pd.concat(pd.read_csv(f) for f in glob.glob("results/review2/e1_local_s[0-4].csv"))
locval = loc.groupby("client").val.mean()
for variant, pat in (("one_pass", "e1_{r}__C1+C2+C3+C4__s{s}.json"), ("fixed_S100", "e1_fixed_{r}__C1+C2+C3+C4__s{s}.json")):
    fig, axes = plt.subplots(len(RULES), 4, figsize=(14, 11), sharex=True)
    for i, r in enumerate(RULES):
        for s in range(5):
            p = os.path.join(H, pat.format(r=r, s=s))
            if not os.path.exists(p):
                continue
            h = json.load(open(p))
            for j, n in enumerate(NAMES):
                y = [x.get(n, np.nan) for x in h]
                axes[i, j].plot(range(1, len(y) + 1), y, lw=1, alpha=0.8, label=f"seed {s}")
        for j, n in enumerate(NAMES):
            ax = axes[i, j]
            ax.axhline(locval[n], ls="--", c="k", lw=1)
            if i == 0:
                ax.set_title(NAMES[n])
            if j == 0:
                ax.set_ylabel(f"{r}\nlr {lr[r]}\nval PR-AUC")
            if i == len(RULES) - 1:
                ax.set_xlabel("round")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"E-1 plain SGD, four-member group, {variant.replace('_', ' ')}: validation PR-AUC per round "
                 "(dashed = SGD local baseline, best-epoch validation, mean of seeds 0-4)")
    fig.tight_layout()
    fig.savefig(f"results/review2/plots/e1_convergence_{variant}.png", dpi=110)
    print("saved", variant)
