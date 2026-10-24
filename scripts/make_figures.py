"""Paper figures (static, light, print). Colours from the validated reference palette:
categorical slots 1-3 (blue, orange, aqua; validated all-pairs), diverging blue<->red with a grey
midpoint, text in neutral inks. Thin marks, recessive grid, no dual axes.
Outputs PDF (for Overleaf) and PNG (preview) into results/figures/."""
import os as _os, sys as _sys  # repository layout: modules live in ../src
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from config import RESULTS as R
F = f"{R}/figures"
os.makedirs(F, exist_ok=True)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
NAMES = {"cert42": "CERT r4.2", "cert52": "CERT r5.2", "spedia": "SPEDIA", "lanl": "LANL"}
ORDER = ["cert42", "cert52", "spedia", "lanl"]
FED_ORDER = ["C1+C2", "C1+C3", "C2+C3", "C1+C4", "C3+C4", "C1+C2+C3", "C1+C2+C3+C4"]
FED_LABEL = {"C1+C2": "r4.2 + r5.2", "C1+C3": "r4.2 + SPEDIA", "C2+C3": "r5.2 + SPEDIA",
             "C1+C4": "r4.2 + LANL", "C3+C4": "SPEDIA + LANL", "C1+C2+C3": "r4.2 + r5.2 + SPEDIA",
             "C1+C2+C3+C4": "All four"}
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "font.family": "DejaVu Sans", "pdf.fonttype": 42})
DIV = LinearSegmentedColormap.from_list("div", ["#d03b3b", "#f0efec", "#2a78d6"])


def boot(v, *_, **__):
    """Paired bootstrap 95% CI of the mean: exact enumeration when n = 5 (3,125 resamples), otherwise 10,000
    fixed-seed resamples (review_stats.boot_ci). Replaces the earlier 5,000 random draws."""
    from review_stats import boot_ci
    v = np.asarray(v, float)
    lo, hi, _ = boot_ci(v)
    return v.mean(), lo, hi


def save(fig, name):
    fig.savefig(f"{F}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{F}/{name}.png", dpi=200, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)


r0 = pd.read_csv(f"{R}/e1e2_mlp_local.csv").set_index(["client", "seed"]).pr_auc


def deltas(df):
    out = []
    for (fed, c), g in df.groupby(["federation", "client"]):
        g = g.set_index("seed").pr_auc
        d = np.array([g[s] - r0.loc[(c, s)] for s in g.index if (c, s) in r0.index])
        mu, lo, hi = boot(d)
        out.append(dict(federation=fed, client=c, delta=mu, lo=lo, hi=hi, fed_pr=g.mean()))
    return pd.DataFrame(out)


# ---------- Fig 2: gain/loss heat map per regime ----------
def complete(job):
    lg = f"{R}/log_fl_{job}.txt"
    return os.path.exists(lg) and f"done {job}" in open(lg).read()


def fig_heatmap(job, title, name):
    p = f"{R}/fl_{job}.csv"
    if not os.path.exists(p) or not complete(job):
        return
    D = deltas(pd.read_csv(p))
    feds = [f for f in FED_ORDER if f in set(D.federation)]
    M = np.full((len(feds), 4), np.nan)
    sig = np.zeros_like(M, bool)
    for _, r in D.iterrows():
        i, j = feds.index(r.federation), ORDER.index(r.client)
        M[i, j] = r.delta
        sig[i, j] = (r.lo > 0) or (r.hi < 0)
    fig, ax = plt.subplots(figsize=(3.6, 0.36 * len(feds) + 0.9))
    lim = 0.5  # shared scale across all regimes so heat maps are comparable
    im = ax.imshow(M, cmap=DIV, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    for i in range(len(feds)):
        for j in range(4):
            if np.isnan(M[i, j]):
                ax.text(j, i, "–", ha="center", va="center", color="#9a9893", fontsize=7)
                continue
            ax.text(j, i, (f"{M[i, j]:+.3f}" if abs(M[i, j]) < 0.01 else f"{M[i, j]:+.2f}") + ("*" if sig[i, j] else ""), ha="center", va="center",
                    fontsize=7, color=INK, fontweight="bold" if sig[i, j] else "normal")
    ax.set_xticks(range(4), [NAMES[c] for c in ORDER])
    ax.set_yticks(range(len(feds)), [FED_LABEL[f] for f in feds])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, 4), minor=True)
    ax.set_yticks(np.arange(-.5, len(feds)), minor=True)
    ax.grid(which="minor", color=SURF, linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.xaxis.tick_top()
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("Δ PR-AUC vs local-only", color=INK)
    cb.outline.set_visible(False)
    ax.set_title(title, fontsize=8, color=INK, pad=22, loc="left")
    fig.text(0.0, -0.02, "* 95% CI of the paired difference excludes 0 (5 seeds, exact bootstrap, 3,125 resamples). "
             "Red = loss (negative transfer), blue = gain.", fontsize=6.5, color=INK2)
    save(fig, name)


fig_heatmap("fedavg_size", "FedAvg (size-weighted): change in PR-AUC per client", "fig2_heatmap_fedavg_size")
fig_heatmap("fedavg_equal", "FedAvg (equal weights): change in PR-AUC per client", "fig2b_heatmap_fedavg_equal")
fig_heatmap("fedper", "FedPer (local output layer): change in PR-AUC per client", "fig2c_heatmap_fedper")
fig_heatmap("fedprox", "FedProx: change in PR-AUC per client", "fig2d_heatmap_fedprox")
fig_heatmap("fedper_in", "Local input layer (C2-b): change in PR-AUC per client", "fig2e_heatmap_fedper_in")
fig_heatmap("finetune", "FedAvg + local fine-tuning: change in PR-AUC per client", "fig2f_heatmap_finetune")

# ---------- Fig 3: regimes in the four-client federation, one panel per client ----------
REG = [("Local only", None), ("Pooled", "e1e2_mlp_pooled"), ("FedAvg-size", "fl_fedavg_size"),
       ("FedAvg-equal", "fl_fedavg_equal"), ("FedProx", "fl_fedprox"), ("FedPer", "fl_fedper"),
       ("Local input", "fl_fedper_in"), ("FedAvg+FT", "fl_finetune")]
fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.3))
for ax, c in zip(axes, ORDER):
    xs, labels = [], []
    for k, (lab, f) in enumerate(REG):
        if f is None:
            v = r0.loc[c].to_numpy()
        else:
            p = f"{R}/{f}.csv"
            if not os.path.exists(p) or (f.startswith("fl_") and not complete(f[3:])):
                continue
            d = pd.read_csv(p)
            if "federation" in d:
                d = d[d.federation == "C1+C2+C3+C4"]
            v = d[d.client == c].pr_auc.to_numpy()
            if len(v) < 2:
                continue
        mu, lo, hi = boot(v)
        col = INK2 if f is None else S1
        ax.errorbar(len(xs), mu, yerr=[[mu - lo], [hi - mu]], fmt="o", ms=4.5, color=col, ecolor=col,
                    elinewidth=1.2, capsize=0, markeredgecolor=SURF, markeredgewidth=0.8, zorder=3)
        xs.append(len(xs))
        labels.append(lab)
    loc = r0.loc[c].mean()
    ax.axhline(loc, color=INK2, linewidth=0.8, zorder=1)
    ax.set_xticks(xs, labels, rotation=90, fontsize=6.3)
    ax.set_title(NAMES[c], fontsize=8, color=INK, loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
axes[0].set_ylabel("Test PR-AUC (mean, 95% CI)")
fig.text(0.0, -0.08, "Federated regimes trained on all four clients; MLP; 5 seeds. Grey line and grey point = the "
         "client trained alone (R0). Each panel has its own y-scale.", fontsize=6.5, color=INK2)
fig.tight_layout()
save(fig, "fig3_regimes_all_four")

# ---------- Fig 4: C3 similarity vs gain (two-client federations, FedAvg size) ----------
sim = pd.read_csv(f"{R}/c3_similarity.csv")
pair_of = {"C1+C2": ("cert42", "cert52"), "C1+C3": ("cert42", "spedia"), "C2+C3": ("cert52", "spedia"),
           "C3+C4": ("spedia", "lanl"), "C1+C4": ("cert42", "lanl")}
if os.path.exists(f"{R}/fl_fedavg_size.csv"):
    D = deltas(pd.read_csv(f"{R}/fl_fedavg_size.csv"))
    pts = []
    for _, r in D.iterrows():
        if r.federation not in pair_of:
            continue
        a, b = pair_of[r.federation]
        s = sim[((sim.a == a) & (sim.b == b)) | ((sim.a == b) & (sim.b == a))].iloc[0]
        partner = b if r.client == a else a
        pts.append(dict(client=r.client, partner=partner, js=s.js_distance, jac=s.jaccard_coverage,
                        prev=s.prevalence_ratio, delta=r.delta, lo=r.lo, hi=r.hi))
    P = pd.DataFrame(pts)
    P.to_csv(f"{R}/c3_points.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.4), sharey=True)
    for ax, (col, lab, logx) in zip(axes, [("js", "Action-mix distance (JS)", False),
                                           ("jac", "Log-coverage overlap (Jaccard)", False),
                                           ("prev", "Prevalence ratio (log scale)", True)]):
        ax.axhline(0, color=INK2, linewidth=0.8)
        STY = {"cert42": (S1, "o"), "cert52": (S1, "s"), "spedia": (S2, "^"), "lanl": (S3, "D")}
        for cl in ORDER:
            Q = P[P.client == cl]
            if not len(Q):
                continue
            colr, mk = STY[cl]
            ax.errorbar(Q[col], Q.delta, yerr=[Q.delta - Q.lo, Q.hi - Q.delta], fmt=mk, ms=5, color=colr,
                        ecolor=colr, elinewidth=1.0, capsize=0, markeredgecolor=SURF, markeredgewidth=0.8,
                        label=f"{NAMES[cl]} (scored client)")
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(lab)
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        rho = pd.Series(P[col]).corr(P.delta, method="spearman")
        ax.set_title(f"Spearman ρ = {rho:+.2f} (n = {len(P)})", fontsize=7, color=INK, loc="left")
    axes[0].set_ylabel("Δ PR-AUC vs local-only")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 1.08))
    fig.text(0.0, -0.06, "Each point: one client in a two-client FedAvg (size-weighted) federation, "
             "mean and 95% CI over 5 seeds. Similarity computed from training data before any training.",
             fontsize=6.5, color=INK2)
    fig.tight_layout()
    save(fig, "fig4_similarity_vs_gain")

# ---------- Fig 5: C4 unseen organisation ----------
if os.path.exists(f"{R}/fl_loo.csv"):
    L = pd.read_csv(f"{R}/fl_loo.csv")
    Z = pd.read_csv(f"{R}/fl_zeroshot_mlp.csv") if os.path.exists(f"{R}/fl_zeroshot_mlp.csv") else None
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.2))
    for ax, c in zip(axes, ORDER):
        series = []
        if Z is not None and len(Z[Z.client == c]):
            z = Z[Z.client == c].groupby("source").pr_auc.mean()
            best = z.idxmax()
            series.append(("Best single source (chosen with hindsight)", Z[(Z.client == c) & (Z.source == best)].pr_auc, S2, NAMES[best]))
        series.append(("Federated on the other three (LOO)", L[L.client == c].pr_auc, S1, None))
        series.append(("Own data (local-only, reference)", r0.loc[c], S3, None))
        top = 0
        for k, (lab, v, col, note) in enumerate(series):
            mu, lo, hi = boot(v)
            ax.bar(k, mu, width=0.6, color=col, edgecolor=SURF, linewidth=1.0, label=lab)
            ax.errorbar(k, mu, yerr=[[mu - lo], [hi - mu]], fmt="none", ecolor=INK2, elinewidth=0.9, capsize=0)
            top = max(top, hi)
            if note:
                ax.annotate(f"from\n{note}", (k, hi), xytext=(0, 2), textcoords="offset points",
                            fontsize=5.5, color=INK2, ha="center", va="bottom")
        br = L[L.client == c].base_rate.mean()
        ax.axhline(br, color=INK, linewidth=0.8, linestyle=(0, (3, 2)), label="Base rate (random ranking)")
        ax.set_ylim(0, top * 1.22)
        ax.set_xticks([])
        ax.set_title(NAMES[c], fontsize=8, color=INK, loc="left")
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Test PR-AUC (mean, 95% CI)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, fontsize=6.3, frameon=False, bbox_to_anchor=(0.5, 1.1))
    fig.text(0.0, -0.06, "Unseen-organisation test: no data or labels from the target are used in training. "
             "Same MLP and same test partitions throughout. Each panel has its own y-scale.", fontsize=6.5, color=INK2)
    fig.tight_layout()
    save(fig, "fig5_unseen_organisation")
print(sorted(os.listdir(F)))
