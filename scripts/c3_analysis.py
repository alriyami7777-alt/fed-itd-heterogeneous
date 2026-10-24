"""C3 analysis: pre-training similarity vs gain from federation (protocol §8; change log 2026-10-14 (p)).

Inputs: results/c3_similarity.csv (from similarity.py: training partitions only, before any federated training)
        two-client FedAvg (size) results, paired with the local-only MLP by seed.
Output: results/c3_points.csv (one row per scored client in each two-client federation: 10 directed points)
        results/c3_spearman.csv (Spearman rho and an exact permutation p-value per measure)

The exact permutation p-value enumerates every permutation of the gains (10! = 3,628,800 for all points,
5! = 120 for the CERT-scored subset) and counts |rho_perm| >= |rho_obs| (two-sided). It treats the points as
exchangeable and so ignores that the two directions of a pair share one similarity value and one federation.

Usage: python c3_analysis.py [--source legacy|review] [--seeds 0-4]
"""
from __future__ import annotations

import os as _os, sys as _sys  # repository layout: modules live in ../src
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import argparse
import glob
import itertools

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from config import RESULTS

PAIRS = {"C1+C2": ("cert42", "cert52"), "C1+C3": ("cert42", "spedia"), "C2+C3": ("cert52", "spedia"),
         "C3+C4": ("spedia", "lanl"), "C1+C4": ("cert42", "lanl")}
CODE = {"cert42": "C1", "cert52": "C2", "spedia": "C3", "lanl": "C4"}


def load(source, seeds):
    if source == "legacy":
        r0 = pd.read_csv(f"{RESULTS}/e1e2_mlp_local.csv")
        fa = pd.read_csv(f"{RESULTS}/fl_fedavg_size.csv")
    else:
        rd = lambda j: pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{RESULTS}/review/{j}_s[0-9]*.csv"))])
        r0, fa = rd("mlp_local"), rd("fedavg_size")
    r0 = r0[r0.seed.isin(seeds)].set_index(["client", "seed"]).pr_auc
    return r0, fa[fa.seed.isin(seeds)]


def exact_perm_p(x, y):
    rx, ry = rankdata(x), rankdata(y)
    obs = abs(np.corrcoef(rx, ry)[0, 1])
    n = len(x)
    rxc = rx - rx.mean()
    den = np.sqrt((rxc ** 2).sum() * ((ry - ry.mean()) ** 2).sum())
    hits = total = 0
    perms = itertools.permutations(ry - ry.mean())
    while True:
        chunk = np.array(list(itertools.islice(perms, 200_000)))
        if not len(chunk):
            break
        r = np.abs(chunk @ rxc / den)
        hits += int((r >= obs - 1e-12).sum())
        total += len(chunk)
    return hits / total, total


def main(source, seeds, tag=("", "")):
    sim = pd.read_csv(f"{RESULTS}/c3_similarity.csv")
    r0, fa = load(source, seeds)
    rows = []
    for fed, (a, b) in PAIRS.items():
        s = sim[((sim.a == a) & (sim.b == b)) | ((sim.a == b) & (sim.b == a))].iloc[0]
        for c, partner in ((a, b), (b, a)):
            g = fa[(fa.federation == fed) & (fa.client == c)].set_index("seed").pr_auc
            d = np.array([g[k] - r0.loc[(c, k)] for k in g.index])
            rows.append(dict(client=c, partner=partner, js=s.js_distance, jac=s.jaccard_coverage,
                             prev=s.prevalence_ratio, delta=d.mean(), n_seeds=len(d)))
    P = pd.DataFrame(rows)
    P.to_csv(f"{RESULTS}/{tag[0]}c3_points{tag[1]}.csv", index=False)
    out = []
    for subset, Q in (("all 10 points", P), ("CERT-scored (5 points)", P[P.client.isin(["cert42", "cert52"])])):
        for col, lab in (("jac", "Jaccard overlap of observed actions"), ("js", "Jensen-Shannon distance"),
                         ("prev", "prevalence ratio")):
            rho = spearmanr(Q[col], Q.delta).statistic
            p, n_perm = exact_perm_p(Q[col].to_numpy(), Q.delta.to_numpy())
            out.append(dict(subset=subset, measure=lab, n=len(Q), spearman_rho=rho, exact_perm_p=p,
                            permutations=n_perm))
    S = pd.DataFrame(out)
    S.to_csv(f"{RESULTS}/{tag[0]}c3_spearman{tag[1]}.csv", index=False)
    print(P.round(4).to_string(index=False))
    print(S.round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="legacy")
    ap.add_argument("--seeds", default="0-4")
    a = ap.parse_args()
    lo, hi = map(int, a.seeds.split("-"))
    # the legacy 5-seed run keeps the original file names (results/c3_*.csv); any other run writes tagged
    # files under results/review/ so it never overwrites them
    legacy = a.source == "legacy" and a.seeds == "0-4"
    main(a.source, list(range(lo, hi + 1)), tag=("", "") if legacy else ("review/", f"_{a.source}_s{a.seeds}"))
