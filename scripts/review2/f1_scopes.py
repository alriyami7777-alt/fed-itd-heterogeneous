"""F-1 scopes 2-5 (review plan of 11 Nov 2026) and the E-5 coalitions (5-seed exploratory extra of scope 1): the same two-level paired
bootstrap (twolevel.py) applied to the rows of stats_e5 / stats_e1 / stats_e4 / stats_e3 / stats_e11.
Each row's compared and baseline test scores come from the saved predictions; seeds are the row's own seeds
(E-1, E-3, E-11 main: 0-19; E-4: 5-19; E-5, E-3 per-group check, E-11 user split: 0-4), resampled with seed_index().
Not computed: E-4 "per-seed max (strict)" rows (their baseline is a max over options per seed, not one paired run).
Holm within each original family. Output: results/review2/stats_twolevel_review2.csv
Usage: python3 f1_scopes.py [e5 e1 e4 e3 e11]"""
import os as _os, sys as _sys  # repository layout: modules live in ../../src and ../
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_os.path.join(_HERE, "..", "..", "src"), _os.path.join(_HERE, "..")]
from config import ROOT as _ROOT
_os.chdir(_ROOT)  # relative paths below (results/, data/) are relative to FITD_ROOT
import sys

import numpy as np
import pandas as pd

import twolevel as T
from run_r2 import e11_prep

PRED = "results/preds"
R2 = "results/review2"
CODE = {"CERT r4.2": "cert42", "CERT r5.2": "cert52", "SPEDIA": "spedia", "LANL": "lanl", "TWOS": "twos"}
MC = {"cert42": 0, "cert52": 1, "spedia": 2, "lanl": 3, "twos": 4, "twos_user": 5}
ld = lambda p: np.load(p)["test"]


def keys(n, s):
    k = np.load(f"{PRED}/keys/{n}_{'s%d' % s if n == 'lanl' else 'temporal'}.npz", allow_pickle=True)
    return k["yte"].astype(int), k["test_user"].astype(str)


TW = {}


def twos_keys(split, s):
    if (split, s) not in TW:
        P, _ = e11_prep(s if split == "user" else 0, split)
        TW[(split, s)] = (P["twos"]["yte"].astype(int), P["twos"]["kte_user"].astype(str))
    return TW[(split, s)]


def rows_for(item):
    """Yield (row_dict, member, member_key, seeds, fed_path(s), base_key, base_path(s))."""
    S = pd.read_csv(f"{R2}/stats_{item}.csv")
    loc_main = lambda n: (("mlp_local", n), lambda s, n=n: f"{PRED}/mlp_local/{n}/s{s}_{n}.npz")
    for _, r in S.iterrows():
        n = CODE[r.client] if "client" in r and r.client in CODE else "cert42"
        if item == "e5":
            rule = r.comparison.split(" − ")[0]
            yield r, n, n, range(5), (lambda s, n=n, f=r.federation, ru=rule: f"{PRED}/r2_e5_{ru}/{f}__{ru}/s{s}_{n}.npz"), *loc_main(n)
        elif item == "e1":
            comp, base = r.comparison.split(" − ")
            loc = (("e1_local", n), lambda s, n=n: f"{PRED}/r2_e1_local/{n}/s{s}_{n}.npz")
            if r.family == "E-1 vs SGD local":
                ru = comp
                yield r, n, n, range(20), (lambda s, n=n, f=r.federation, ru=ru: f"{PRED}/r2_e1_{ru}/{f}/s{s}_{n}.npz"), *loc
            elif r.family.startswith("E-1 FedNova"):
                b = base
                yield r, n, n, range(20), (lambda s, n=n, f=r.federation, ru=comp: f"{PRED}/r2_e1_{ru}/{f}/s{s}_{n}.npz"), \
                    ("e1", b, r.federation, n), (lambda s, n=n, f=r.federation, b=b: f"{PRED}/r2_e1_{b}/{f}/s{s}_{n}.npz")
            else:
                ru = comp.split(" S=100")[0]
                fx = lambda s, n=n, ru=ru: f"{PRED}/r2_e1_fixed/C1+C2+C3+C4_{ru}__sgd_{ru}_S100/s{s}_{n}.npz"
                if r.family == "E-1 fixed-step vs SGD local":
                    yield r, n, n, range(20), fx, *loc
                else:
                    yield r, n, n, range(20), fx, ("e1", ru, "C1+C2+C3+C4", n), \
                        (lambda s, n=n, ru=ru: f"{PRED}/r2_e1_{ru}/C1+C2+C3+C4/s{s}_{n}.npz")
        elif item == "e4":
            if "strict" in r.family:
                continue
            fr, vs = r.frac, r.val_set
            comp, base = r.comparison.split(" − ")
            dirs = {"a_local": "a__a_local", "b_r52_as_is": "b__b_r52_as_is", "c_r52_finetune": "c__c_r52_finetune",
                    "d_size": "d_size__d_fedavg_size", "d_equal": "d_equal__d_fedavg_equal"}
            b = base.split("(")[1].rstrip(")") if "best of" in base else base
            p = lambda key, s, fr=fr, vs=vs: f"{PRED}/r2_e4/r42_{fr}_{vs}_{dirs[key]}/s{s}_cert42.npz"
            yield r, "cert42", "cert42", range(5, 20), (lambda s, c=comp: p(c, s)), ("e4", fr, vs, b), (lambda s, b=b: p(b, s))
        elif item == "e3":
            rule = r.comparison.split(" − ")[0]
            seeds = range(5) if "sensitivity" in r.family else range(20)
            job = "e3_dittofed" if "sensitivity" in r.family else f"e3_{rule}"
            yield r, n, n, seeds, (lambda s, n=n, f=r.federation, j=job, ru=rule: f"{PRED}/r2_{j}/{f}__{ru}/s{s}_{n}.npz"), *loc_main(n)
        elif item == "e11":
            reg = r.comparison.split(" − ")[0]
            user = "user-split" in r.family
            job = "e11_user" if user else "e11_main"
            seeds = range(5) if user else range(20)
            fp = lambda s, n=n, f=r.federation, j=job, reg=reg: f"{PRED}/r2_{j}/{f}__{reg}/s{s}_{n if n != 'twos' else 'twos'}.npz"
            if n == "twos":
                mk = "twos_user" if user else "twos"
                yield r, n, mk, seeds, fp, (job, "twos_R0"), (lambda s, j=job: f"{PRED}/r2_{j}/twos__R0_local/s{s}_twos.npz")
            else:
                yield r, n, n, seeds, fp, *loc_main(n)


def run(item):
    groups = {}
    for r, n, mk, seeds, fp, bk, bp in rows_for(item):
        groups.setdefault((mk, tuple(seeds)), []).append((r, n, fp, bk, bp))
    out = []
    for (mk, seeds), lst in groups.items():
        n = lst[0][1]
        idx = T.seed_index(seeds)
        fixed = n != "lanl" and mk != "twos_user"
        if n == "twos":
            kk = {s: twos_keys("user" if mk == "twos_user" else "day", s) for s in seeds}
        else:
            kk = {s: keys(n, s) for s in seeds}
        y = kk[seeds[0]][0] if fixed else {s: kk[s][0] for s in seeds}
        u = kk[seeds[0]][1] if fixed else {s: kk[s][1] for s in seeds}
        bases, cells = {}, {}
        for i, (r, _, fp, bk, bp) in enumerate(lst):
            bases.setdefault(bk, {s: ld(bp(s)) for s in seeds})
            cells[i] = ({s: ld(fp(s)) for s in seeds}, bk)
        res = T.run_member(MC[mk], fixed, y, u, bases, cells, idx)
        for i, (r, *_rest) in enumerate(lst):
            d = res[i]
            out.append(dict(item=item, family=r.family, comparison=r.comparison, group=r.get("federation", ""),
                            member=r.get("client", "CERT r4.2"), frac=r.get("frac", np.nan), val_set=r.get("val_set", ""),
                            n_seeds=int(r.n_seeds), new=r.new, base=r.base, mean_diff=r["diff"], seed_lo=r.boot_lo,
                            seed_hi=r.boot_hi, old_verdict=r.verdict, old_verdict_holm=r.verdict_holm,
                            tl_mean=float(np.nanmean(d)), **T.summarise(d)))
        print(f"{item} {mk} seeds {seeds[0]}-{seeds[-1]}: {len(lst)} cells", flush=True)
    O = pd.DataFrame(out)
    O["tl_holm"] = np.nan
    for fam, g in O.groupby("family"):
        O.loc[g.index, "tl_holm"] = T.holm(g.tl_p.to_numpy())
    O["tl_verdict_holm"] = np.where(O.tl_holm < 0.05, np.where(O.mean_diff > 0, "gain", "loss"), "no clear difference")
    return O


if __name__ == "__main__":
    items = sys.argv[1:] or ["e5", "e1", "e4", "e3", "e11"]
    for it in items:
        O = run(it)
        O.to_csv(f"{R2}/stats_twolevel_{it}.csv", index=False)
        print(O.groupby(["family", "tl_verdict"]).size().unstack(fill_value=0).to_string(), flush=True)
