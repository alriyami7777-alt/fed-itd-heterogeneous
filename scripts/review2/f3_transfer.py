"""F-3 (review plan of 11 Nov 2026, M3b): 5 x 5 cross-dataset transfer matrix and the RQ3 extension.
Cell A->B = member A's local MLP (R0) scored on B's test set with B's own training scaler (R5 rule); diagonal = R0.
Seeds 0-19. Sources: results/preds/zeroshot_mlp (C1-C4 -> C1-C4), r2_e11_main/<A>->C5__R5_zeroshot (C1-C4 -> TWOS),
f3_transfer/twos (TWOS -> C1-C4, retrained, bit-identical check in f3_twos_repro_*.csv), mlp_local and
r2_e11_main/twos__R0_local (diagonal). Labels from results/preds/keys (TWOS: e11_prep, day split).
T(A->B) = PR-AUC(A->B) / PR-AUC(B->B) per seed; mean and a 10,000-resample seed bootstrap CI (seed 20261111).
RQ3: Spearman between a member's gain in its pair (FedAvg size, 20 seeds, c3_analysis.load) and T(partner -> member),
and the mean of T in both directions; exact permutation p (c3_analysis.exact_perm_p); 5 pairs, + C2+C4, and with
each dataset left out (points whose member or partner is that dataset dropped). The three existing measures are
computed on the same points for comparison.
Outputs: results/review2/f3_transfer_matrix.csv, f3_rq3_points.csv, f3_rq3_spearman.csv"""
import os as _os, sys as _sys  # repository layout: modules live in ../../src and ../
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_os.path.join(_HERE, "..", "..", "src"), _os.path.join(_HERE, "..")]
from config import ROOT as _ROOT
_os.chdir(_ROOT)  # relative paths below (results/, data/) are relative to FITD_ROOT
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score as ap

from c3_analysis import PAIRS, exact_perm_p, load as c3load
from run_r2 import e11_prep

PRED = "results/preds"
M = ["cert42", "cert52", "spedia", "lanl", "twos"]
NM = {"cert42": "CERT r4.2", "cert52": "CERT r5.2", "spedia": "SPEDIA", "lanl": "LANL", "twos": "TWOS"}
SEEDS = range(20)
P0, _ = e11_prep(0, "day")
y_twos = P0["twos"]["yte"].astype(int)


def ytest(b, s):
    if b == "twos":
        return y_twos
    k = np.load(f"{PRED}/keys/{b}_{'s%d' % s if b == 'lanl' else 'temporal'}.npz", allow_pickle=True)
    return k["yte"].astype(int)


def scores(a, b, s):
    if a == b:
        f = f"{PRED}/r2_e11_main/twos__R0_local/s{s}_twos.npz" if b == "twos" else f"{PRED}/mlp_local/{b}/s{s}_{b}.npz"
    elif b == "twos":
        f = f"{PRED}/r2_e11_main/{a}->C5__R5_zeroshot/s{s}_twos.npz"
    elif a == "twos":
        f = f"{PRED}/f3_transfer/twos/s{s}_{b}.npz"
    else:
        f = f"{PRED}/zeroshot_mlp/{a}/s{s}_{b}.npz"
    return np.load(f)["test"]


pr = {(a, b): np.array([ap(ytest(b, s), scores(a, b, s)) for s in SEEDS]) for a in M for b in M}
base = {b: np.mean([ytest(b, s).mean() for s in SEEDS]) for b in M}
rng = np.random.default_rng(20261111)
idx = rng.integers(0, 20, (10_000, 20))
rows = []
for a in M:
    for b in M:
        t = pr[(a, b)] / pr[(b, b)]
        bm = t[idx].mean(1)
        rows.append(dict(source=NM[a], target=NM[b], pr_auc=pr[(a, b)].mean(), target_base_rate=base[b],
                         lift=pr[(a, b)].mean() / base[b], T=t.mean(), T_lo=np.percentile(bm, 2.5),
                         T_hi=np.percentile(bm, 97.5), n_seeds=len(t)))
X = pd.DataFrame(rows)
X.to_csv("results/review2/f3_transfer_matrix.csv", index=False)
Tm = {(a, b): (pr[(a, b)] / pr[(b, b)]).mean() for a in M for b in M}

# RQ3
sim = pd.read_csv("results/c3_similarity.csv")
r0, fa = c3load("review", list(SEEDS))
pairs = dict(PAIRS, **{"C2+C4": ("cert52", "lanl")})
pts = []
for fed, (a, b) in pairs.items():
    s = sim[((sim.a == a) & (sim.b == b)) | ((sim.a == b) & (sim.b == a))].iloc[0]
    for c, p in ((a, b), (b, a)):
        g = fa[(fa.federation == fed) & (fa.client == c)].set_index("seed").pr_auc
        pts.append(dict(federation=fed, member=c, partner=p, gain=np.mean([g[k] - r0.loc[(c, k)] for k in g.index]),
                        T_partner_to_member=Tm[(p, c)], T_mean_both=(Tm[(p, c)] + Tm[(c, p)]) / 2,
                        js=s.js_distance, jac=s.jaccard_coverage, prev=s.prevalence_ratio))
Pt = pd.DataFrame(pts)
Pt.to_csv("results/review2/f3_rq3_points.csv", index=False)
MEAS = [("T_partner_to_member", "transfer T(partner -> member)"), ("T_mean_both", "transfer, mean of both directions"),
        ("jac", "Jaccard overlap of observed actions"), ("js", "Jensen-Shannon distance"), ("prev", "prevalence ratio")]
subsets = [("5 pairs (10 points)", Pt[Pt.federation != "C2+C4"]), ("6 pairs (12 points)", Pt)]
for d in ("cert42", "cert52", "spedia", "lanl"):
    Q = Pt[(Pt.federation != "C2+C4") & (Pt.member != d) & (Pt.partner != d)]
    subsets.append((f"5 pairs, {NM[d]} left out ({len(Q)} points)", Q))
out = []
for name, Q in subsets:
    for col, lab in MEAS:
        rho = spearmanr(Q[col], Q.gain).statistic
        p, nperm = exact_perm_p(Q[col].to_numpy(), Q.gain.to_numpy())
        out.append(dict(subset=name, measure=lab, n=len(Q), spearman_rho=rho, perm_p=p, permutations=nperm))
S = pd.DataFrame(out)
S.to_csv("results/review2/f3_rq3_spearman.csv", index=False)
pd.set_option("display.width", 250)
print(X.pivot(index="source", columns="target", values="pr_auc").round(4).to_string())
print(X.pivot(index="source", columns="target", values="T").round(3).to_string())
print(Pt.round(4).to_string(index=False))
print(S.round(4).to_string(index=False))
