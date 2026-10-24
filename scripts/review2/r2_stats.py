"""Statistics for the review-2 items (doc 31 §0.3): effect = new − its own matched baseline, paired by seed;
10,000-resample paired bootstrap (seed 20261014; exact 3,125 enumeration when n = 5, as review_stats.boot_ci);
paired t; Holm within the item's family. Verdicts are descriptive.
Usage: python r2_stats.py [item ...]   (items: e1 e12 e4 e2 e3 e5 e6 e11; default: all that have results)
Outputs: results/review2/stats_<item>.csv (+ printed summary)."""
from __future__ import annotations

import os as _os, sys as _sys  # repository layout: modules live in ../../src and ../
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_os.path.join(_HERE, "..", "..", "src"), _os.path.join(_HERE, "..")]
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

from config import RESULTS
from review_stats import boot_ci, holm, t_ci, verdict

R2 = os.path.join(RESULTS, "review2")
REV = os.path.join(RESULTS, "review")
NAMES = {"cert42": "CERT r4.2", "cert52": "CERT r5.2", "spedia": "SPEDIA", "lanl": "LANL", "twos": "TWOS"}


def rd(job, d=R2):
    fs = sorted(glob.glob(os.path.join(d, f"{job}_s[0-9]*.csv")))
    return pd.concat([pd.read_csv(f) for f in fs]) if fs else None


def rd_rev(job):
    fs = sorted(glob.glob(os.path.join(REV, f"{job}_s[0-9]*.csv"))) + \
        sorted(glob.glob(os.path.join(REV, f"c2c4_{job}_s[0-9]*.csv")))
    return pd.concat([pd.read_csv(f) for f in fs]) if fs else None


def paired(new: pd.Series, base: pd.Series, **lab):
    """new, base: indexed by seed."""
    k = sorted(set(new.dropna().index) & set(base.dropna().index))
    if len(k) < 2:
        return None
    d = np.array([new[s] - base[s] for s in k], float)
    lo, hi, kind = boot_ci(d)
    tlo, thi, p = t_ci(d)
    return dict(**lab, n_seeds=len(k), new=float(new[k].mean()), base=float(base[k].mean()), diff=float(d.mean()),
                sd=float(d.std(ddof=1)), boot_lo=lo, boot_hi=hi, verdict=verdict(lo, hi), t_lo=tlo, t_hi=thi, t_p=p)


def finish(rows, families, name):
    """families: column whose value names the Holm family of each row."""
    S = pd.DataFrame([r for r in rows if r])
    if not len(S):
        return S
    S["t_p_holm"] = np.nan
    for f, g in S.groupby(families):
        S.loc[g.index, "t_p_holm"] = holm(g.t_p.to_numpy())
    S["verdict_holm"] = np.where(S.t_p_holm < 0.05, np.where(S["diff"] > 0, "gain", "loss"), "no clear difference")
    S.to_csv(os.path.join(R2, f"stats_{name}.csv"), index=False)
    print(f"\n=== {name}: {len(S)} comparisons; families: {S[families].value_counts().to_dict()}")
    cnt = S.groupby([families] + [c for c in ("comparison",) if c in S]).agg(
        n=("diff", "size"), loss=("verdict", lambda v: (v == "loss").sum()), gain=("verdict", lambda v: (v == "gain").sum()),
        loss_holm=("verdict_holm", lambda v: (v == "loss").sum()), gain_holm=("verdict_holm", lambda v: (v == "gain").sum()))
    print(cnt.to_string())
    return S


def ser(df, **flt):
    x = df
    for k, v in flt.items():
        x = x[x[k] == v]
    return x.set_index("seed").pr_auc


# ---------------------------------------------------------------- E-1
def e1():
    base = rd("e1_local")
    rows = []
    rules = ("fedavg_size", "fedavg_equal", "fednova_size", "fednova_equal")
    D = {r: rd(f"e1_{r}") for r in rules}
    for r, d in D.items():
        if d is None:
            continue
        for (fed, c), g in d.groupby(["federation", "client"], sort=False):
            rows.append(paired(g.set_index("seed").pr_auc, ser(base, client=c), family="E-1 vs SGD local",
                               comparison=f"{r} − local(SGD)", federation=fed, client=NAMES[c],
                               protocol_cell=fed != "C2+C4", lr=float(g.lr.iloc[0]),
                               tau=int(g.tau.iloc[0]), tau_eff=float(g.tau_eff_calc.iloc[0])))
    for w in ("size", "equal"):
        a, b = D.get(f"fednova_{w}"), D.get(f"fedavg_{w}")
        if a is None or b is None:
            continue
        for (fed, c), g in a.groupby(["federation", "client"], sort=False):
            rows.append(paired(g.set_index("seed").pr_auc, ser(b, federation=fed, client=c),
                               family="E-1 FedNova − FedAvg (same solver)", comparison=f"fednova_{w} − fedavg_{w}",
                               federation=fed, client=NAMES[c], protocol_cell=fed != "C2+C4"))
    fx = rd("e1_fixed")
    if fx is not None:
        for (reg, c), g in fx.groupby(["regime", "client"], sort=False):
            r = reg.replace("sgd_", "").replace("_S100", "")
            rows.append(paired(g.set_index("seed").pr_auc, ser(base, client=c), family="E-1 fixed-step vs SGD local",
                               comparison=f"{r} S=100 − local(SGD)", federation="C1+C2+C3+C4", client=NAMES[c]))
            if D.get(r) is not None:
                rows.append(paired(g.set_index("seed").pr_auc, ser(D[r], federation="C1+C2+C3+C4", client=c),
                                   family="E-1 fixed-step − fixed-epoch", comparison=f"{r} S=100 − {r} one pass",
                                   federation="C1+C2+C3+C4", client=NAMES[c]))
    return finish(rows, "family", "e1")


# ---------------------------------------------------------------- E-12
def e12():
    d = rd("e12")
    if d is None:
        return None
    ctrl = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(REV, "confirm_c1_s*.csv"))])
    ctrl_sp = ser(ctrl, regime="spedia_alone_S100")
    r0 = rd_rev("mlp_local")
    rows = []
    for (fed, c), g in d.groupby(["federation", "client"], sort=False):
        s = g.set_index("seed").pr_auc
        fed0, w = fed.rsplit("_", 1)
        rows.append(paired(s, ser(r0, client=c), family="E-12 vs usual local", comparison=f"{w} − local",
                           federation=fed0, client=NAMES[c]))
        if c == "spedia":
            rows.append(paired(s, ctrl_sp, family="E-12 SPEDIA vs same-steps alone", comparison=f"{w} − SPEDIA alone S=100",
                               federation=fed0, client=NAMES[c]))
        own = ctrl[(ctrl.federation == fed) & (ctrl.client == c)]
        if len(own):
            rows.append(paired(s, own.set_index("seed").pr_auc, family="E-12 main rule − C-1 own rule",
                               comparison=f"{w}: main − own checkpoint", federation=fed0, client=NAMES[c]))
    return finish(rows, "family", "e12")


# ---------------------------------------------------------------- E-4
def e4():
    d = rd("e4")
    if d is None:
        return None
    rows, win = [], []
    for (frac, vs), g in d.groupby(["frac", "val_set"]):
        opt = {r: g[g.regime == r].set_index("seed").pr_auc for r in g.regime.unique()}
        base = ["a_local", "b_r52_as_is", "c_r52_finetune"]
        best = max(base, key=lambda r: opt[r].mean())
        perseed_max = pd.concat([opt[r] for r in base], axis=1).max(axis=1)
        for w in ("size", "equal"):
            dd = opt[f"d_fedavg_{w}"]
            lab = dict(frac=frac, val_set=vs, n_val=int(g.n_val.iloc[0]), pos_val_mean=float(g.pos_val.mean()),
                       pos_train_mean=float(g.pos_train.mean()))
            rows.append(paired(dd, opt[best], family=f"E-4 {vs}", comparison=f"d_{w} − best of a–c ({best})", **lab))
            rows.append(paired(dd, perseed_max, family=f"E-4 {vs} (per-seed max, strict)",
                               comparison=f"d_{w} − per-seed max(a–c)", **lab))
        for r in base:
            rows.append(paired(opt[r], opt["a_local"], family=f"E-4 {vs} baselines vs local",
                               comparison=f"{r} − a_local", frac=frac, val_set=vs)) if r != "a_local" else None
        allo = pd.concat({r: opt[r] for r in opt}, axis=1)
        for s, row in allo.iterrows():
            win.append(dict(frac=frac, val_set=vs, seed=s, winner=row.idxmax(), **row.to_dict()))
    W = pd.DataFrame(win)
    W.to_csv(os.path.join(R2, "e4_per_seed_winner.csv"), index=False)
    print(W.groupby(["val_set", "frac"]).winner.value_counts().unstack(fill_value=0).to_string())
    return finish(rows, "family", "e4")


# ---------------------------------------------------------------- E-2
def e2():
    rows = []
    for f in sorted(glob.glob(os.path.join(R2, "e2_*_s[0-9]*.csv"))):
        pass
    jobs = sorted({os.path.basename(f).rsplit("_s", 1)[0] for f in glob.glob(os.path.join(R2, "e2_*_s[0-9]*.csv"))})
    for job in jobs:
        d = rd(job)
        v = job[3:]
        loc = d[d.regime == "local"]
        for (reg, c), g in d[~d.regime.isin(["local", "local_same_steps"])].groupby(["regime", "client"], sort=False):
            rows.append(paired(g.set_index("seed").pr_auc, ser(loc, client=c), family="E-2", variant=v,
                               comparison=f"{reg} − local", member=c, A_rate_all=float(loc[loc.client == "m1"].rate_all.mean()),
                               A_users=float(loc[loc.client == "m1"].n_users.mean())))
            ss = d[d.regime == "local_same_steps"]
            if c == "m1" and len(ss):
                rows.append(paired(g.set_index("seed").pr_auc, ss.set_index("seed").pr_auc, family="E-2 A vs same-steps local",
                                   variant=v, comparison=f"{reg} − local(same steps)", member=c))
    return finish(rows, "family", "e2")


# ---------------------------------------------------------------- E-3, E-5, E-6 (vs the protocol local MLP)
def vs_local(jobs, fam, name, base=None):
    base = base if base is not None else rd_rev("mlp_local")
    rows = []
    for job, family in jobs:
        d = rd(job)
        if d is None:
            continue
        for (reg, fed, c), g in d.groupby(["regime", "federation", "client"], sort=False):
            rows.append(paired(g.set_index("seed").pr_auc, ser(base, client=c), family=family, comparison=f"{reg} − local",
                               federation=fed, client=NAMES[c], protocol_cell=fed != "C2+C4"))
    return finish(rows, "family", name)


def e3():
    return vs_local([("e3_fedperh", "E-3"), ("e3_fedper2", "E-3"), ("e3_ditto", "E-3"),
                     ("e3_dittofed", "E-3 sensitivity (per-group lambda)")], "family", "e3")


def e5():
    jobs = [(f"e5_{r}", "E-5 six planned rules") for r in ("fedavg_size", "fedavg_equal", "fedprox", "fedper", "fedper_in", "finetune")]
    jobs += [(f"e5_{r}", "E-5 FedNova (exploratory)") for r in ("fednova_size", "fednova_equal")]
    return vs_local(jobs, "family", "e5")


def e6():
    rows = []
    for v in ("ii_desc", "iii_sev7_or_desc"):
        d = rd(f"e6_{v}")
        if d is None:
            continue
        loc = d[d.regime == "R0_local"]
        r0 = rd_rev("mlp_local")
        for (reg, fed, c), g in d[d.regime != "R0_local"].groupby(["regime", "federation", "client"], sort=False):
            b = ser(loc, client="spedia") if c == "spedia" else ser(r0, client=c)
            rows.append(paired(g.set_index("seed").pr_auc, b, family="E-6", variant=v, comparison=f"{reg} − local",
                               federation=fed, client=NAMES[c]))
    return finish(rows, "family", "e6")


def e11():
    rows = []
    for job in ("e11_main", "e11_user"):
        d = rd(job)
        if d is None:
            continue
        loc_t = ser(d[d.regime == "R0_local"], client="twos")
        r0 = rd_rev("mlp_local")
        for (reg, fed, c), g in d[d.regime != "R0_local"].groupby(["regime", "federation", "client"], sort=False):
            b = loc_t if c == "twos" else ser(r0, client=c)
            fam = "E-11 federated" if reg.startswith(("fedavg", "fedper")) else "E-11 references (R1/R5/R6)"
            if job == "e11_user":
                fam += " user-split sensitivity"
            rows.append(paired(g.set_index("seed").pr_auc, b, family=fam, comparison=f"{reg} − local", federation=fed,
                               client=NAMES[c], split=job[4:]))
    return finish(rows, "family", "e11")


if __name__ == "__main__":
    items = sys.argv[1:] or ["e1", "e12", "e4", "e2", "e3", "e5", "e6", "e11"]
    for it in items:
        try:
            globals()[it]()
        except Exception as ex:      # noqa: BLE001 - an item without results yet
            print(f"{it}: not available ({ex})")
