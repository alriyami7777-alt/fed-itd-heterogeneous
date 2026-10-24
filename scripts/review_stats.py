"""Statistics for regime − R0 comparisons (change log 2026-10-14 (o)). Replaces the 5,000 random-draw bootstrap.

Per regime, federation and client (paired by seed):
  mean difference, seed SD, n seeds
  paired bootstrap 95% CI: exact enumeration of all n^n resamples when n = 5 (3,125), otherwise 10,000 resamples
    with a fixed seed (20261014)
  paired t 95% CI (sensitivity)
  verdict under each CI: gain (CI > 0), loss (CI < 0), no clear difference
  Holm-adjusted count of losses over all comparisons, from two-sided paired t p-values (sensitivity)
Verdicts are descriptive labels, not formal hypothesis tests.
"PR-AUC" is scikit-learn average_precision_score (average precision).

Usage: python review_stats.py [--seeds 0-19] [--source review|legacy]
  legacy = the original 5-seed CSVs (results/e1e2_mlp_local.csv, results/fl_*.csv)
"""
from __future__ import annotations

import os as _os, sys as _sys  # repository layout: modules live in ../src
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import argparse
import glob
import itertools
import os

import numpy as np
import pandas as pd
from scipy import stats

from config import RESULTS

NAMES = {"cert42": "CERT r4.2", "cert52": "CERT r5.2", "spedia": "SPEDIA", "lanl": "LANL"}
REGIMES = [("fedavg_size", "FedAvg (size)"), ("fedavg_equal", "FedAvg (equal)"), ("fedprox", "FedProx (μ=0.001)"),
           ("fedper", "FedPer (local output layer)"), ("fedper_in", "Private first layer"),
           ("finetune", "FedAvg + fine-tune"), ("mlp_pooled", "Pooled MLP (no privacy)"),
           ("fednova_size", "FedNova (size)"), ("fednova_equal", "FedNova (equal)"),
           ("fednova_sgd_size", "FedNova (size, local SGD)"), ("fednova_sgd_equal", "FedNova (equal, local SGD)")]
PROTOCOL_JOBS = ("fedavg_size", "fedavg_equal", "fedprox", "fedper", "fedper_in", "finetune")  # Holm family: 6 x 17 = 102
BOOT_SEED, BOOT_N = 20261014, 10_000


def load(source, seeds):
    if source == "legacy":
        r0 = pd.read_csv(f"{RESULTS}/e1e2_mlp_local.csv")
        jobs = {j: pd.read_csv(f"{RESULTS}/fl_{j}.csv") for j, _ in REGIMES if os.path.exists(f"{RESULTS}/fl_{j}.csv")}
        if os.path.exists(f"{RESULTS}/e1e2_mlp_pooled.csv"):
            jobs["mlp_pooled"] = pd.read_csv(f"{RESULTS}/e1e2_mlp_pooled.csv").assign(federation="C1+C2+C3+C4")
    else:
        def rd(job):
            # also picks up the D-2 "c2c4_<job>" files (CERT r5.2 + LANL federation, change log D-2)
            fs = sorted(glob.glob(f"{RESULTS}/review/{job}_s[0-9]*.csv")) + \
                sorted(glob.glob(f"{RESULTS}/review/c2c4_{job}_s[0-9]*.csv"))
            return pd.concat([pd.read_csv(f) for f in fs]) if fs else None
        r0 = rd("mlp_local")
        jobs = {j: rd(j) for j, _ in REGIMES}
        jobs = {j: d for j, d in jobs.items() if d is not None}
        if "mlp_pooled" in jobs:
            jobs["mlp_pooled"] = jobs["mlp_pooled"].assign(federation="C1+C2+C3+C4")
    r0 = r0[r0.seed.isin(seeds)]
    jobs = {j: d[d.seed.isin(seeds)] for j, d in jobs.items()}
    return r0.set_index(["client", "seed"]).pr_auc, jobs


def boot_ci(d):
    d = np.asarray(d, float)
    n = len(d)
    if n == 5:
        idx = np.array(list(itertools.product(range(n), repeat=n)))
        means = d[idx].mean(1)
        kind = "exact (3,125)"
    else:
        idx = np.random.default_rng(BOOT_SEED).integers(0, n, (BOOT_N, n))
        means = d[idx].mean(1)
        kind = f"{BOOT_N:,} resamples, seed {BOOT_SEED}"
    return np.percentile(means, 2.5), np.percentile(means, 97.5), kind


def t_ci(d):
    d = np.asarray(d, float)
    n = len(d)
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
    if se == 0:
        return m, m, 1.0 if m == 0 else 0.0
    h = stats.t.ppf(0.975, n - 1) * se
    p = 2 * stats.t.sf(abs(m / se), n - 1)
    return m - h, m + h, p


def verdict(lo, hi):
    return "gain" if lo > 0 else ("loss" if hi < 0 else "no clear difference")


def compare(r0, jobs):
    rows = []
    for job, lab in REGIMES:
        if job not in jobs:
            continue
        d = jobs[job]
        for (fed, c), g in d.groupby(["federation", "client"], sort=False):
            g = g.set_index("seed").pr_auc
            k = sorted(s for s in g.index if (c, s) in r0.index)
            if len(k) < 2:
                continue
            diff = np.array([g[s] - r0.loc[(c, s)] for s in k])
            blo, bhi, kind = boot_ci(diff)
            tlo, thi, p = t_ci(diff)
            rows.append(dict(regime=lab, job=job, federation=fed, client=NAMES[c], n_seeds=len(k),
                             fed_pr_auc=g[k].mean(), local_pr_auc=np.mean([r0.loc[(c, s)] for s in k]),
                             mean_diff=diff.mean(), sd_diff=diff.std(ddof=1), boot_lo=blo, boot_hi=bhi, boot_kind=kind,
                             verdict_boot=verdict(blo, bhi), t_lo=tlo, t_hi=thi, t_p=p, verdict_t=verdict(tlo, thi)))
    out = pd.DataFrame(rows)
    # Holm (sensitivity). Primary family: the federated comparisons only (6 methods x 17 = 102), as in the paper.
    # Also reported over all comparisons including the pooled MLP (106) - columns *_all.
    out["protocol_cell"] = out.federation != "C2+C4"
    fedmask = (out.job.isin(PROTOCOL_JOBS) & out.protocol_cell).to_numpy()
    out["t_p_holm"], out["t_p_holm_all"] = np.nan, holm(out.t_p.to_numpy())   # *_all: every comparison here
    out.loc[fedmask, "t_p_holm"] = holm(out.t_p.to_numpy()[fedmask])
    for col, v in (("t_p_holm", "verdict_holm"), ("t_p_holm_all", "verdict_holm_all")):
        out[v] = np.where(out[col] < 0.05, np.where(out.mean_diff > 0, "gain", "loss"), "no clear difference")
        out.loc[out[col].isna(), v] = "not in family"
    return out


def holm(p):
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(np.argsort(p)):
        run = max(run, min(1.0, (m - rank) * p[i]))
        adj[i] = run
    return adj


def counts(out):
    rows = []
    for (job, lab), g0 in out.groupby(["job", "regime"], sort=False):
      for scope, g in (("17 protocol cells", g0[g0.protocol_cell]), ("19 incl. C2+C4", g0)):
        if scope.startswith("19") and g0.protocol_cell.all():
            continue
        r = dict(regime=lab, scope=scope, comparisons=len(g))
        for v in ("boot", "t", "holm", "holm_all"):
            vc = g[f"verdict_{v}"].value_counts()
            if v == "holm" and (job not in PROTOCOL_JOBS or scope.startswith("19")):
                r[f"loss_{v}"] = r[f"gain_{v}"] = "–"          # not in the 102-comparison Holm family
            else:
                r[f"loss_{v}"], r[f"gain_{v}"] = int(vc.get("loss", 0)), int(vc.get("gain", 0))
        rows.append(r)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-19")
    ap.add_argument("--source", default="review")
    a = ap.parse_args()
    lo, hi = map(int, a.seeds.split("-"))
    r0, jobs = load(a.source, list(range(lo, hi + 1)))
    out = compare(r0, jobs)
    tag = f"{a.source}_s{a.seeds}"
    os.makedirs(f"{RESULTS}/review", exist_ok=True)
    out.to_csv(f"{RESULTS}/review/stats_{tag}.csv", index=False)
    cnt = counts(out)
    cnt.to_csv(f"{RESULTS}/review/stats_counts_{tag}.csv", index=False)
    nf = int((out.job.isin(PROTOCOL_JOBS) & out.protocol_cell).sum())
    print(f"Comparisons: {nf} in the Holm family ('holm': 6 protocol methods x 17 cells); "
          f"{len(out)} in total ('holm_all': incl. pooled, FedNova, C2+C4)")
    print(cnt.to_string(index=False))
