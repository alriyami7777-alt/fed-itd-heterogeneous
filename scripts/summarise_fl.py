"""Federated results vs local-only (R0 MLP), paired by seed; exact paired bootstrap (3,125 resamples for 5 seeds) over seeds.
Negative transfer = federated PR-AUC below R0 with the 95% CI of the difference excluding 0."""
import os as _os, sys as _sys  # repository layout: modules live in ../src
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import glob

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


r0 = pd.read_csv(f"{R}/e1e2_mlp_local.csv").set_index(["client", "seed"]).pr_auc
fs = [f for f in sorted(glob.glob(f"{R}/fl_*.csv"))]
L = ["# Federated results vs local-only (MLP), protocol v1.0 FROZEN", ""]
if not fs:
    L.append("No federated job finished yet.")
for f in fs:
    d = pd.read_csv(f)
    if not len(d):
        continue
    job = d.job.iloc[0]
    if job == "loo":
        L += [f"## {job}: leave-one-client-out (train on three, test on the unseen fourth)", "",
              "| Unseen client | Seeds | PR-AUC LOO-FL | Local-only (has own data) | Lift |", "|---|---|---|---|---|"]
        for c, g in d.groupby("client"):
            mu, lo, hi = boot(g.pr_auc)
            base = r0.loc[c]
            L.append(f"| {NAMES[c]} | {len(g)} | {mu:.3f} [{lo:.3f}, {hi:.3f}] | {base.mean():.3f} | {g.lift.mean():.1f} |")
        zp = f"{R}/fl_zeroshot_mlp.csv"
        if __import__("os").path.exists(zp):
            Z = pd.read_csv(zp)
            L += ["", "**Unseen organisation: federated (other three) vs single-source zero-shot (same MLP, same test sets).** "
                  "'Best single source' is chosen with hindsight, which favours single-source transfer.", "",
                  "| Unseen client | Zero-shot by source | Best single source | Mean over sources | Federated (LOO) | LOO − best single [95% CI] |",
                  "|---|---|---|---|---|---|"]
            for c in ["cert42", "cert52", "spedia", "lanl"]:
                z = Z[Z.client == c]
                if not len(z):
                    continue
                means = z.groupby("source").pr_auc.mean()
                best = means.idxmax()
                l = d[d.client == c].set_index("seed").pr_auc
                b = z[z.source == best].set_index("seed").pr_auc
                dd = (l - b).dropna().to_numpy()
                mu, lo, hi = boot(dd)
                L.append(f"| {NAMES[c]} | " + ", ".join(f"{NAMES[k]} {v:.3f}" for k, v in means.items()) +
                         f" | {NAMES[best]} {means[best]:.3f} | {means.mean():.3f} | {l.mean():.3f} | {mu:+.3f} [{lo:+.3f}, {hi:+.3f}] |")
        L.append("")
        continue
    L += [f"## {job} ({d.regime.iloc[0]})", "",
          "| Federation | Client | Seeds | Federated PR-AUC | Local-only | Δ (fed − local) [95% CI] | Verdict |",
          "|---|---|---|---|---|---|---|"]
    for (fed, c), g in d.groupby(["federation", "client"], sort=False):
        g = g.set_index("seed").pr_auc
        k = [s for s in g.index if (c, s) in r0.index]
        delta = np.array([g[s] - r0.loc[(c, s)] for s in k])
        mu, lo, hi = boot(delta) if len(delta) > 1 else (delta.mean() if len(delta) else np.nan, np.nan, np.nan)
        verdict = "gain" if lo > 0 else ("**negative transfer**" if hi < 0 else "no clear difference")
        L.append(f"| {fed} | {NAMES[c]} | {len(g)} | {g.mean():.3f} | {np.mean([r0.loc[(c, s)] for s in k]):.3f} | "
                 f"{mu:+.3f} [{lo:+.3f}, {hi:+.3f}] | {verdict} |")
    L.append("")
open(f"{R}/FL_summary.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
