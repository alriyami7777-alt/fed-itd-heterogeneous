"""Review-2 runs (doc 31, CONSOLIDATED experiment plan v10, 20 Oct 2026; protocol change log E-1 ... E-12).
Designs are fixed here before running (and logged in the change log first). Defaults as the frozen protocol:
MLP 42-128-64-1, Adam 1e-3, batch 256, class-weighted BCE, T = 100, patience 10, best round by mean validation
PR-AUC over members, splits and scalers as in fitd_common.py. Every training call reseeds itself.

Usage: python run_r2.py <job> [seeds]
Outputs: results/review2/<job>_s<seed>.csv (one row per scored member), predictions under
results/preds/r2_<job>/<federation>/s<seed>_<member>.npz, per-round histories under results/review2/hist/.

Jobs
  E-1  e1_sel (seed 0): SGD lr chosen separately for each method from {0.003, 0.01, 0.03} on the mean validation
       PR-AUC of the four-member group (seed 0); each member's local SGD baseline chooses its own lr on its own
       validation PR-AUC (seed 0). Written to results/review2/e1_lr.json.
       e1_local; e1_fedavg_size|equal, e1_fednova_size|equal (7 protocol groups + C2+C4);
       e1_fixed: four-member group, every member makes S = 100 steps per round (without replacement within a
       pass), the 4 rules, each at its selected lr.
       e1_ref: FedNova reference check (synthetic logistic regression; see e1_ref()).
  E-12 e12 (seeds 5-19): C-1 schedule (SPEDIA 100 steps/round) with the MAIN checkpoint rule.
  E-4  e4 (seeds 5-19): equal total label budget (f of training AND of validation rows), options a-d, plus the
       same with the full validation set.
  E-2  e2_<variant>, variant in R S1 S2 C1 (C2, S1C2 after the event rebuild; P held - see change log).
  E-3  e3_fedperh, e3_fedper2, e3_ditto_sel (seed 0), e3_ditto.
  E-5  e5_<rule>: 3 triples x 8 rules.
"""
from __future__ import annotations

import os as _os, sys as _sys  # repository layout: modules live in ../../src and ../
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_os.path.join(_HERE, "..", "..", "src"), _os.path.join(_HERE, "..")]
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

import fl
import run_e1_e2
from config import RESULTS
from fitd_common import LABEL, USER, feature_cols, load, signed_log1p, three_way
from fl import finetune, run_fl
from run_confirm import FRACS, SPEDIA_FEDS, lowdata_client
from run_e1_e2 import class_weights, predict, prep, train_mlp
from run_fl import FEDS

R2 = os.path.join(RESULTS, "review2")
PRED = os.path.join(RESULTS, "preds")
HIST = os.path.join(R2, "hist")
os.makedirs(HIST, exist_ok=True)
FEDS8 = dict(FEDS, **{"C2+C4": ("cert52", "lanl")})
TRIPLES = {"C1+C2+C4": ("cert42", "cert52", "lanl"), "C1+C3+C4": ("cert42", "spedia", "lanl"),
           "C2+C3+C4": ("cert52", "spedia", "lanl")}
ALL4 = FEDS["C1+C2+C3+C4"]
LR_GRID = (0.003, 0.01, 0.03)
E1_RULES = ("fedavg_size", "fedavg_equal", "fednova_size", "fednova_equal")
_ABS = ['cnt_file_read', 'cnt_file_write', 'cnt_file_delete', 'cnt_usb_connect', 'cnt_usb_disconnect', 'cnt_http',
        'cnt_email_send', 'cnt_email_recv', 'cnt_cmd_exec', 'cnt_ssh', 'cnt_ftp', 'cnt_process_start', 'cnt_unknown']
LANL_ABSENT = _ABS + ["dev_" + c for c in _ABS]          # the 26 columns that are all zero in LANL


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def save_pred(job, fed, seed, n, val, test):
    d = os.path.join(PRED, f"r2_{job}", fed)
    os.makedirs(d, exist_ok=True)
    np.savez(os.path.join(d, f"s{seed}_{n}.npz"), val=np.asarray(val, np.float32), test=np.asarray(test, np.float32))


def rec(rows, job, regime, fed, seed, n, P, model, extra, target=None):
    t = target or n
    pv, pt = predict(model, P[t]["Xva"]), predict(model, P[t]["Xte"])
    y = P[t]["yte"]
    pr = float(average_precision_score(y, pt)) if y.sum() else float("nan")
    # predictions are stored per (federation, regime) so that several rules in one job never overwrite each other
    # (fixed 21 Oct; earlier E-1/E-12/E-4/E-2/E-3/E-5 runs used one rule per job or rule-specific federation names)
    save_pred(job, f"{fed}__{regime}", seed, t, pv, pt)
    rows.append(dict(job=job, regime=regime, federation=fed, seed=seed, client=t, n_test=len(y), pos_test=int(y.sum()),
                     base_rate=float(y.mean()), pr_auc=pr, lift=pr / y.mean() if y.sum() else float("nan"), **extra))
    log(f"{job} {regime} {fed} seed{seed} {t}: PR-AUC={pr:.4f}")


def fl_cost(wall):
    c = dict(best_round=fl.LAST["best_round"], stop_round=fl.LAST["stop_round"],
             params_shared=fl.LAST["params_shared"], params_local=fl.LAST["params_local"],
             bytes_per_round=fl.LAST["bytes_per_round"], wall_s=wall)
    if "tau_eff" in fl.LAST:
        c["tau_eff"] = fl.LAST["tau_eff"]
    c["diverged_round"] = fl.LAST.get("diverged_round", 0)
    return c


def save_hist(job, fed, seed, h):
    json.dump(h, open(os.path.join(HIST, f"{job}__{fed}__s{seed}.json"), "w"))


def taus(C, w, steps=None):
    t = {n: int((steps or {}).get(n) or np.ceil(len(C[n]["ytr"]) / 256)) for n in C}
    s = np.array([len(C[n]["ytr"]) for n in C], float)
    p = s / s.sum() if w == "size" else np.ones(len(C)) / len(C)
    return t, float((p * np.array(list(t.values()))).sum())


def lr_of(key):
    return json.load(open(os.path.join(R2, "e1_lr.json")))[key]


def run_rule(rule, C, seed, lr=None, client_steps=None, hist=False):
    """Returns (member models, regime label, best val, cost, per-member extras)."""
    T = time.perf_counter
    t0 = T()
    ext = {n: {} for n in C}
    kw = dict(client_steps=client_steps, history=hist)
    if rule in ("fedavg_size", "fedavg_equal"):
        ms, br, bv = run_fl(C, seed, rule.split("_")[1], **kw)
    elif rule in ("fednova_size", "fednova_equal"):
        ms, br, bv = run_fl(C, seed, rule.split("_")[1], agg="fednova", **kw)
    elif rule == "fedprox":
        mu = json.load(open(os.path.join(RESULTS, "fedprox_mu.json")))["mu"]
        ms, br, bv = run_fl(C, seed, "size", mu=mu, **kw)
        rule = f"fedprox_mu{mu}"
    elif rule in ("fedper", "fedper_in", "fedper2"):
        ms, br, bv = run_fl(C, seed, "size", mode=rule, **kw)
    elif rule == "fedperh":
        ms, br, bv = run_fl(C, seed, "size", mode="fedper", arch="head83", **kw)
    elif rule == "finetune":
        g, br, bv = run_fl(C, seed, "size", **kw)
        cost = fl_cost(T() - t0)
        ms = {}
        for n in C:
            ms[n] = finetune(g[n], C[n], seed)
            ext[n] = dict(ft_epochs_run=fl.LAST["ft_epochs_run"], ft_best_epoch=fl.LAST["ft_best_epoch"])
        return ms, rule, bv, cost, ext
    elif rule.startswith("sgd_"):                         # E-1: sgd_fedavg_size etc.
        _, agg, w = rule.split("_")
        ms, br, bv = run_fl(C, seed, w, agg=agg, optimizer="sgd", lr=lr, **kw)
        t, te = taus(C, w, client_steps)
        for n in C:
            ext[n] = dict(tau=t[n], tau_eff_calc=te, lr=lr)
    else:
        raise ValueError(rule)
    return ms, rule, bv, fl_cost(T() - t0), ext


def fed_job(job, rule, feds, seed, lr=None, client_steps=None, hist=False):
    P = prep(seed)
    rows = []
    for fed, members in feds.items():
        C = {n: P[n] for n in members}
        ms, regime, bv, cost, ext = run_rule(rule, C, seed, lr, client_steps, hist)
        if hist:
            save_hist(job, fed, seed, fl.LAST["history"])
        for n in members:
            rec(rows, job, regime, fed, seed, n, P, ms[n], dict(val=bv, **cost, **ext[n]))
    return rows


# ---------------------------------------------------------------- E-1
def e1_sel(job, seed):
    P = prep(0)
    rows = []
    C = {n: P[n] for n in ALL4}
    for lr in LR_GRID:
        for rule in E1_RULES:
            ms, regime, bv, cost, ext = run_rule("sgd_" + rule, C, 0, lr, hist=True)
            save_hist(f"{job}_{rule}_lr{lr}", "C1+C2+C3+C4", 0, fl.LAST["history"])
            for n in C:
                rec(rows, job, f"{regime}_lr{lr}", "C1+C2+C3+C4", 0, n, P, ms[n], dict(val=bv, method=rule, **cost, **ext[n]))
        for n in ALL4:
            c = P[n]
            m, v, ep = train_mlp(c["Xtr"], c["ytr"], class_weights(c["ytr"]), [(c["Xva"], c["yva"])], 0,
                                 optimizer="sgd", lr=lr)
            save_hist(f"{job}_local_lr{lr}", n, 0, {n: list(run_e1_e2.TRAIN_HIST)})
            rec(rows, job, f"local_sgd_lr{lr}", n, 0, n, P, m, dict(val=v, method=f"local_{n}", lr=lr, best_round=ep))
    d = pd.DataFrame(rows)
    sel = {}
    for meth, g in d.groupby("method"):
        v = g.groupby("lr").val.first()
        sel[meth] = float(v.idxmax())
        sel[f"{meth}__val_by_lr"] = {str(k): float(x) for k, x in v.items()}
    sel["rule"] = ("seed 0; FL methods: mean validation PR-AUC over the 4 members of C1+C2+C3+C4; local baselines: the "
                   "member's own validation PR-AUC; ties -> smaller lr")
    json.dump(sel, open(os.path.join(R2, "e1_lr.json"), "w"), indent=1)
    log(f"E-1 selected: { {k: v for k, v in sel.items() if '__' not in k and k != 'rule'} }")
    return rows


LR_EXT = {"fedavg_equal": 0.001, "fednova_equal": 0.001, "fednova_size": 0.1, "local_cert52": 0.001, "local_spedia": 0.1}


def e1_selx(job, seed):
    """Grid extension (writing team, 20 Oct 19:34): every method/baseline whose selected lr was at the edge of
    {0.003, 0.01, 0.03} gets one more value beyond that edge (0.001 or 0.1), same rule (seed 0; FL: mean validation
    PR-AUC over the four members of C1+C2+C3+C4; local: the member's own validation PR-AUC). The final lr is the best
    over the extended grid (ties -> smaller lr). A new edge is reported, not extended again.
    e1_lr.json is updated in place; the first selection is kept in e1_lr_v1.json."""
    v1 = os.path.join(R2, "e1_lr_v1.json")
    if not os.path.exists(v1):
        json.dump(json.load(open(os.path.join(R2, "e1_lr.json"))), open(v1, "w"), indent=1)
    sel = json.load(open(v1))
    P = prep(0)
    rows = []
    C = {n: P[n] for n in ALL4}
    for meth, lr in LR_EXT.items():
        if meth.startswith("local_"):
            n = meth[6:]
            c = P[n]
            m, v, ep = train_mlp(c["Xtr"], c["ytr"], class_weights(c["ytr"]), [(c["Xva"], c["yva"])], 0, optimizer="sgd", lr=lr)
            save_hist(f"e1_sel_local_lr{lr}", n, 0, {n: list(run_e1_e2.TRAIN_HIST)})
            rec(rows, job, f"local_sgd_lr{lr}", n, 0, n, P, m, dict(val=v, method=meth, lr=lr, best_round=ep))
        else:
            ms, regime, v, cost, ext = run_rule("sgd_" + meth, C, 0, lr, hist=True)
            save_hist(f"e1_sel_{meth}_lr{lr}", "C1+C2+C3+C4", 0, fl.LAST["history"])
            if fl.LAST.get("never_finite"):
                v = float("-inf")             # diverged from round 1: cannot be selected
                rows.append(dict(job=job, regime=f"{regime}_lr{lr}", method=meth, lr=lr, val=float("nan"),
                                 note="diverged from round 1 (no finite validation score)"))
                log(f"e1_selx {meth} lr{lr}: diverged from round 1")
            else:
                for n in C:
                    rec(rows, job, f"{regime}_lr{lr}", "C1+C2+C3+C4", 0, n, P, ms[n], dict(val=v, method=meth, **cost, **ext[n]))
        grid = {float(k): x for k, x in sel[f"{meth}__val_by_lr"].items()}
        grid[lr] = float(v)
        best = max(sorted(grid), key=lambda k: grid[k])
        sel[f"{meth}__val_by_lr"] = {str(k): grid[k] for k in sorted(grid)}
        sel[meth] = best
        sel[f"{meth}__new_edge"] = best in (min(grid), max(grid)) and best == lr
    sel["rule"] += "; extended one step beyond an edge (0.001 / 0.1) on 20 Oct, no further extension"
    json.dump(sel, open(os.path.join(R2, "e1_lr.json"), "w"), indent=1)
    log(f"E-1 extended selection: { {k: sel[k] for k in LR_EXT} }")
    return rows


def e1_local(job, seed):
    P = prep(seed)
    rows = []
    for n in P:
        lr = lr_of(f"local_{n}")
        c = P[n]
        t0 = time.perf_counter()
        m, v, ep = train_mlp(c["Xtr"], c["ytr"], class_weights(c["ytr"]), [(c["Xva"], c["yva"])], seed,
                             optimizer="sgd", lr=lr)
        save_hist(job, n, seed, {n: list(run_e1_e2.TRAIN_HIST)})
        rec(rows, job, "R0_local_sgd", n, seed, n, P, m, dict(val=v, lr=lr, best_round=ep, wall_s=time.perf_counter() - t0))
    return rows


def e1_fixed(job, seed):
    P = prep(seed)
    rows = []
    C = {n: P[n] for n in ALL4}
    steps = {n: 100 for n in ALL4}
    for rule in E1_RULES:
        ms, regime, bv, cost, ext = run_rule("sgd_" + rule, C, seed, lr_of(rule), steps, hist=True)
        save_hist(f"{job}_{rule}", "C1+C2+C3+C4", seed, fl.LAST["history"])
        for n in C:
            rec(rows, job, f"{regime}_S100", f"C1+C2+C3+C4_{rule}", seed, n, P, ms[n], dict(val=bv, **cost, **ext[n]))
    return rows


def e1_ref(job, seed):
    """FedNova reference check. 3 clients, logistic regression (d = 5), unequal sizes 300 / 900 / 2,000 rows
    -> tau = 2 / 4 / 8 steps of 256 rows, plain SGD lr 0.05, size weights, 5 rounds.
    Our run_fl(agg="fednova") versus an independent NumPy implementation of Wang et al. (2020), Alg. 1 / eq. (7):
      x_{t+1} = x_t - tau_eff * sum_i p_i * eta * d_i,  d_i = G_i a_i / ||a_i||_1,  a_i = 1 (vanilla SGD),
    with the minibatch gradients of the class-weighted mean logistic loss computed by hand. Both use the same
    minibatch order (the run_fl shuffle generators), so only the update rule is compared."""
    import torch
    rng = np.random.default_rng(7)
    d, sizes, rounds, eta = 5, (300, 900, 2000), 5, 0.05
    beta = rng.normal(size=d)
    C = {}
    for i, n in enumerate(sizes):
        X = rng.normal(loc=i * 0.5, size=(n, d)).astype(np.float32)
        y = (rng.random(n) < 1 / (1 + np.exp(-(X @ beta - 1)))).astype(int)
        yv = np.r_[1, np.zeros(49, int)]
        C[f"c{i}"] = dict(Xtr=X, ytr=y, Xva=X[:50], yva=yv, Xte=X[:50], yte=yv)
    run_fl(C, 0, "size", agg="fednova", optimizer="sgd", lr=eta, arch="logreg", T=rounds, patience=10 ** 6, trace=True)
    ours = [np.r_[s["net.0.weight"].numpy().ravel(), s["net.0.bias"].numpy()].astype(np.float64)
            for s in fl.LAST["global_trace"]]
    names = list(C)
    p = np.array(sizes, float) / sum(sizes)
    tau = np.array([int(np.ceil(n / 256)) for n in sizes])
    tau_eff = float((p * tau).sum())
    gens = {n: torch.Generator().manual_seed(0 * 100 + i) for i, n in enumerate(names)}
    torch.manual_seed(0)
    np.random.seed(0)
    m0 = fl.LogReg(d)                                       # same initialisation as run_fl with seed 0
    x = np.r_[m0.net[0].weight.detach().numpy().ravel(), m0.net[0].bias.detach().numpy()].astype(np.float64)
    rows = []
    for r in range(rounds):
        upd = np.zeros_like(x)
        for i, n in enumerate(names):
            X, y = C[n]["Xtr"].astype(np.float64), C[n]["ytr"].astype(np.float64)
            wc = class_weights(C[n]["ytr"]).astype(np.float64)
            xi, gsum = x.copy(), np.zeros_like(x)
            perm = torch.randperm(len(X), generator=gens[n]).numpy()
            for k in range(0, len(X), 256):
                b = perm[k:k + 256]
                s = 1 / (1 + np.exp(-(X[b] @ xi[:d] + xi[d])))
                gz = wc[b] * (s - y[b]) / len(b)
                g = np.r_[X[b].T @ gz, gz.sum()]
                gsum += g
                xi = xi - eta * g
            upd += p[i] * eta * gsum / tau[i]               # eta * G_i a_i / ||a_i||_1
        x = x - tau_eff * upd
        rows.append(dict(job=job, round=r + 1, max_abs_diff=float(np.abs(ours[r] - x).max()),
                         max_abs_param=float(np.abs(x).max()), tau="2/4/8", tau_eff=tau_eff, lr=eta))
        log(f"e1_ref round {r + 1}: max |ours - independent| = {rows[-1]['max_abs_diff']:.3e}")
    return rows


# ---------------------------------------------------------------- E-12
def e12(job, seed):
    P = prep(seed)
    rows = []
    for w in ("size", "equal"):
        for fed, members in SPEDIA_FEDS.items():
            t0 = time.perf_counter()
            ms, br, bv = run_fl({n: P[n] for n in members}, seed, w, client_steps={"spedia": 100})
            cost = fl_cost(time.perf_counter() - t0)
            for n in members:
                rec(rows, job, f"spedia_S100_{w}_mainrule", f"{fed}_{w}", seed, n, P, ms[n], dict(weighting=w, val=bv, **cost))
    return rows


# ---------------------------------------------------------------- E-4
def cut_val(c, frac, seed):
    """Label-stratified cut of the validation set to share f (new draw per seed; at least one row per class)."""
    rng = np.random.default_rng(30_000 + seed)
    keep = []
    for lab in (0, 1):
        idx = np.flatnonzero(c["yva"] == lab)
        keep.append(rng.choice(idx, max(1, int(round(frac * len(idx)))), replace=False))
    keep = np.sort(np.concatenate(keep))
    return dict(c, Xva=c["Xva"][keep], yva=c["yva"][keep])


def e4(job, seed):
    P = prep(seed)
    rows = []
    c52 = P["cert52"]
    m52, v52, ep52 = train_mlp(c52["Xtr"], c52["ytr"], class_weights(c52["ytr"]), [(c52["Xva"], c52["yva"])], seed)
    for frac in FRACS:
        cf = lowdata_client(seed, frac)                    # f of training rows, scaler on the subsample
        for vs, c in (("budget", cut_val(cf, frac, seed)), ("fullval", cf)):
            ex = dict(frac=frac, val_set=vs, n_train=len(c["ytr"]), pos_train=int(c["ytr"].sum()),
                      n_val=len(c["yva"]), pos_val=int(c["yva"].sum()))
            tag = f"r42_{frac}_{vs}"
            one = {"cert42": c}
            m, v, ep = train_mlp(c["Xtr"], c["ytr"], class_weights(c["ytr"]), [(c["Xva"], c["yva"])], seed)
            rec(rows, job, "a_local", f"{tag}_a", seed, "cert42", one, m, dict(ex, best_round=ep))
            rec(rows, job, "b_r52_as_is", f"{tag}_b", seed, "cert42", one, m52, ex)
            m = finetune(m52, c, seed)
            rec(rows, job, "c_r52_finetune", f"{tag}_c", seed, "cert42", one, m,
                dict(ex, ft_epochs_run=fl.LAST["ft_epochs_run"], ft_best_epoch=fl.LAST["ft_best_epoch"]))
            Q = {"cert42": c, "cert52": c52}
            for w in ("size", "equal"):
                ms, br, bv = run_fl(Q, seed, w)
                rec(rows, job, f"d_fedavg_{w}", f"{tag}_d_{w}", seed, "cert42", Q, ms["cert42"], dict(ex, val=bv, best_round=br))
    return rows


# ---------------------------------------------------------------- E-2
def e2_partition(seed):
    """CERT r5.2 users -> four user-disjoint members, stratified by positive-day pattern over train/val/test
    (a refinement of 'has a positive day'); same chronological split as the protocol. Draw fixed by seed."""
    d = load("cert52")
    cols = feature_cols(d)
    _, tr, va, te = three_way(d, "cert52", 0)
    pat = pd.DataFrame({s: x.groupby(USER)[LABEL].max() for s, x in (("tr", tr), ("va", va), ("te", te))})
    pat = pat.reindex(sorted(d[USER].unique())).fillna(0).astype(int)
    key = pat.tr.astype(str) + pat.va.astype(str) + pat.te.astype(str)
    rng = np.random.default_rng(20_000 + seed)
    mem = {f"m{i}": [] for i in range(1, 5)}
    off = 0
    for k in sorted(key.unique()):
        us = rng.permutation(key.index[key == k].to_numpy())
        for j, u in enumerate(us):
            mem[f"m{(j + off) % 4 + 1}"].append(u)
        off += len(us)
    return d, cols, (tr, va, te), key, mem, rng


def e2_members(variant, seed):
    d, cols, (tr, va, te), key, mem, rng = e2_partition(seed)
    info = {}
    if variant in ("S1", "S2", "S1C2"):
        # member A (m1) keeps 1% of its users: one positive user with training positives (preferring one that
        # also has validation positives), one positive user with test positives, and normal users to make
        # round(0.01 * |A|) users in total (at least 2 positive users are needed, so at least 2 users)
        A = mem["m1"]
        k = max(2, int(round(0.01 * len(A))))
        r = np.random.default_rng(40_000 + seed)
        trva = [u for u in A if key[u] == "110"] or [u for u in A if key[u][0] == "1"]
        u1 = trva[r.integers(len(trva))]
        tep = [u for u in A if key[u][2] == "1" and u != u1]
        tep = [u for u in tep if key[u][1] == "1"] if key[u1][1] == "0" and any(key[u][1] == "1" for u in tep) else tep
        u2 = tep[r.integers(len(tep))]
        normal = [u for u in A if key[u] == "000"]
        mem["m1"] = [u1, u2] + list(r.choice(normal, k - 2, replace=False))
    if variant == "S1b":
        # (proposed, pending approval) size only: A keeps 10% of its users; its positive users (round(10%) of A's
        # positive users, at least 2) are drawn so that A has training, validation and test positives; the rest are
        # normal users. Fixed by seed (rng 41,000 + seed).
        r = np.random.default_rng(41_000 + seed)
        A = mem["m1"]
        n_keep = int(round(0.10 * len(A)))
        posA = [u for u in A if key[u] != "000"]
        n_pos = max(2, int(round(0.10 * len(posA))))
        for _ in range(10_000):
            pk = list(r.choice(posA, n_pos, replace=False))
            if all(any(key[u][i] == "1" for u in pk) for i in range(3)):
                break
        else:
            raise RuntimeError("no draw with positives in all three splits")
        normal = [u for u in A if key[u] == "000"]
        mem["m1"] = pk + list(r.choice(normal, n_keep - n_pos, replace=False))
    if variant == "P":
        # (proposed, pending approval) prevalence only: A swaps normal users for positive users of m2-m4, one for
        # one (the normal user goes to the member the positive user came from), taking positive users in a fixed
        # random order (round robin over m2-m4) until A's training positive-day rate reaches 2x its R rate.
        r = np.random.default_rng(42_000 + seed)
        trc = tr.groupby(USER)[LABEL].agg(["sum", "size"])
        rate_tr = lambda us: trc.reindex(us).fillna(0)["sum"].sum() / trc.reindex(us).fillna(0)["size"].sum()
        target = 2 * rate_tr(mem["m1"])
        normA = list(r.permutation([u for u in mem["m1"] if key[u] == "000"]))
        queues = {m: list(r.permutation([u for u in mem[m] if key[u] != "000"])) for m in ("m2", "m3", "m4")}
        moved, i = 0, 0
        while rate_tr(mem["m1"]) < target and any(queues.values()):
            m = ("m2", "m3", "m4")[i % 3]
            i += 1
            if not queues[m]:
                continue
            u, g = queues[m].pop(0), normA.pop(0)
            mem[m] = [x for x in mem[m] if x != u] + [g]
            mem["m1"] = [x for x in mem["m1"] if x != g] + [u]
            moved += 1
        info = dict(swapped_users=moved, target_rate_tr=float(target))
    M = {}
    for m, us in mem.items():
        s = set(us)
        parts = [x[x[USER].isin(s)] for x in (tr, va, te)]
        X = [p[cols].to_numpy(np.float64).copy() for p in parts]
        if m == "m1" and variant in ("C2", "S1C2"):
            # coverage by rebuild: A's 42 features recomputed (IJACSA user_day_features, parity-checked) from its
            # events with every action LANL does not record dropped; same user-days and labels as the cache
            # (1,004 of 693,649 user-days have no kept event and are zero-filled)
            rb = pd.read_parquet(os.path.join(os.path.dirname(os.path.dirname(R2)), "data", "derived",
                                              "cert52_user_day_cov_logon_rebuild.parquet"))
            assert (rb[USER].values == d[USER].values).all() and (rb[LABEL].values == d[LABEL].values).all()
            X = [rb.loc[p.index, cols].to_numpy(np.float64).copy() for p in parts]
        if m == "m1" and variant == "C1":
            idx = [cols.index(c) for c in LANL_ABSENT]
            for x in X:
                x[:, idx] = 0.0
        sc = StandardScaler().fit(signed_log1p(X[0]))
        f = lambda x: sc.transform(signed_log1p(x)).astype(np.float32)
        ys = [p[LABEL].to_numpy() for p in parts]
        M[m] = dict(Xtr=f(X[0]), ytr=ys[0], Xva=f(X[1]), yva=ys[1], Xte=f(X[2]), yte=ys[2],
                    desc=dict(variant=variant, n_users=len(us), n_pos_users=int((key[list(us)] != "000").sum()),
                              n_train=len(ys[0]), rate_tr=float(ys[0].mean()), rate_all=float(np.concatenate(ys).mean()),
                              pos_tr=int(ys[0].sum()), pos_va=int(ys[1].sum()), pos_te=int(ys[2].sum()),
                              posusers_tr=int(parts[0].groupby(USER)[LABEL].max().sum()),
                              posusers_va=int(parts[1].groupby(USER)[LABEL].max().sum()),
                              posusers_te=int(parts[2].groupby(USER)[LABEL].max().sum()), **info))
    steps = None
    if variant in ("S1", "S1C2"):
        steps = {"m1": int(round(np.mean([np.ceil(len(M[m]["ytr"]) / 256) for m in ("m2", "m3", "m4")])))}
    return M, steps, info


def e2(job, seed):
    variant = job.split("_", 1)[1]
    M, steps, info = e2_members(variant, seed)
    rows = []
    for m in M:
        c = M[m]
        m_, v, ep = train_mlp(c["Xtr"], c["ytr"], class_weights(c["ytr"]), [(c["Xva"], c["yva"])], seed)
        rec(rows, job, "local", f"{variant}_local", seed, m, M, m_, dict(val=v, best_round=ep, **c["desc"]))
    if steps:   # same-schedule local baseline for member A (as the C-1 control)
        ms, br, bv = run_fl({"m1": M["m1"]}, seed, "size", client_steps=steps)
        rec(rows, job, "local_same_steps", f"{variant}_local_same_steps", seed, "m1", M, ms["m1"],
            dict(val=bv, steps_m1=steps["m1"], **fl_cost(0), **M["m1"]["desc"]))
    for rule in ("fedavg_size", "fedavg_equal", "fedper_in"):
        ms, regime, bv, cost, ext = run_rule(rule, M, seed, client_steps=steps)
        for m in M:
            rec(rows, job, regime, f"{variant}_{rule}", seed, m, M, ms[m],
                dict(val=bv, steps_m1=(steps or {}).get("m1", 0), **cost, **M[m]["desc"]))
    return rows


# ---------------------------------------------------------------- E-3
DITTO_GRID = (0.01, 0.1, 1.0)


def e3_ditto_sel(job, seed):
    """Main rule (amendment A3): lambda chosen ONCE, four-member group, seed 0, by mean validation PR-AUC of the
    personal models (ties -> smaller lambda), then used for every group - the same tuning budget as FedProx's mu.
    job e3_dittofed_sel (labelled sensitivity): lambda chosen per group on seed 0."""
    P = prep(0)
    per_fed = job == "e3_dittofed_sel"
    feds = FEDS8 if per_fed else {"C1+C2+C3+C4": ALL4}
    rows, sel = [], {}
    for fed, members in feds.items():
        C = {n: P[n] for n in members}
        vals = {}
        for lam in DITTO_GRID:
            t0 = time.perf_counter()
            ms, br, bv = run_fl(C, 0, "size", ditto_lam=lam)
            vals[lam] = bv
            for n in members:
                rec(rows, job, f"ditto_lam{lam}", fed, 0, n, P, ms[n], dict(val=bv, lam=lam, **fl_cost(time.perf_counter() - t0)))
        sel[fed] = max(DITTO_GRID, key=lambda k: (vals[k], -k))
        sel[f"{fed}__val"] = {str(k): float(v) for k, v in vals.items()}
    if not per_fed:
        sel = dict(sel, **{f: sel["C1+C2+C3+C4"] for f in FEDS8})
    sel["rule"] = ("seed 0, per group (SENSITIVITY)" if per_fed else "seed 0, four-member group, used for every group") + \
        "; highest mean validation PR-AUC of the personal models; ties -> smaller lambda"
    json.dump(sel, open(os.path.join(R2, "e3_dittofed_lambda.json" if per_fed else "e3_ditto_lambda.json"), "w"), indent=1)
    return rows


def e3_ditto(job, seed):
    lam = json.load(open(os.path.join(R2, "e3_dittofed_lambda.json" if job == "e3_dittofed" else "e3_ditto_lambda.json")))
    P = prep(seed)
    rows = []
    for fed, members in FEDS8.items():
        C = {n: P[n] for n in members}
        t0 = time.perf_counter()
        ms, br, bv = run_fl(C, seed, "size", ditto_lam=lam[fed])
        cost = fl_cost(time.perf_counter() - t0)
        for n in members:
            rec(rows, job, "ditto" if job == "e3_ditto" else "ditto_perfed_sensitivity", fed, seed, n, P, ms[n],
                dict(val=bv, lam=lam[fed], **cost))
    return rows


# ---------------------------------------------------------------- E-6
E6_FEDS = {"C1+C3": ("cert42", "spedia"), "C3+C4": ("spedia", "lanl"), "C1+C2+C3+C4": ALL4}


def e6_prep(seed, variant):
    """prep(seed) with SPEDIA's user-day labels replaced by the variant's labels (features and splits unchanged:
    SPEDIA's split is chronological, so it does not depend on labels)."""
    P = prep(seed)
    d = pd.read_parquet(os.path.join(os.path.dirname(os.path.dirname(R2)), "data", "derived",
                                     f"spedia_user_day_{variant}.parquet"))
    d[LABEL] = d[LABEL].astype(int)
    _, tr, va, te = three_way(d, "spedia", seed)
    for k, part in (("tr", tr), ("va", va), ("te", te)):
        assert (part[USER].astype(str).values == P["spedia"]["kva_user" if k == "va" else "kte_user"]).all() if k != "tr" else True
        P["spedia"]["y" + k] = part[LABEL].to_numpy()
    cnt = {f"{k}_{s}": v for k, part in (("tr", tr), ("va", va), ("te", te))
           for s, v in (("posdays", int(part[LABEL].sum())), ("posusers", int(part.groupby(USER)[LABEL].max().sum())),
                        ("rows", len(part)))}
    return P, cnt


def e6(job, seed):
    variant = job.split("_", 1)[1]
    P, cnt = e6_prep(seed, variant)
    rows = []
    c = P["spedia"]
    m, v, ep = train_mlp(c["Xtr"], c["ytr"], class_weights(c["ytr"]), [(c["Xva"], c["yva"])], seed)
    rec(rows, job, "R0_local", "spedia", seed, "spedia", P, m, dict(val=v, best_round=ep, variant=variant, **cnt))
    for fed, members in E6_FEDS.items():
        C = {n: P[n] for n in members}
        for rule in ("fedavg_size", "fedavg_equal", "fedper_in"):
            ms, regime, bv, cost, ext = run_rule(rule, C, seed)
            for n in members:
                rec(rows, job, regime, fed, seed, n, P, ms[n], dict(val=bv, variant=variant, **cost, **cnt))
    return rows


# ---------------------------------------------------------------- E-11 TWOS
TWOS_FILE = "twos_user_day_dev_v1_logon.parquet"
TWOS_SHA = "fedadd4a707a92d0b8977bc134115121a4e2ddd03a727c067c55d1bd31c5bd31"
E11_PAIRS = {"C1+C5": ("cert42", "twos"), "C2+C5": ("cert52", "twos"), "C3+C5": ("spedia", "twos"),
             "C4+C5": ("lanl", "twos")}
E11_ALL5 = {"C1+C2+C3+C4+C5": ("cert42", "cert52", "spedia", "lanl", "twos")}
E11_RULES = ("fedavg_size", "fedavg_equal", "fedper_in")


def twos_frame():
    import hashlib
    p = os.path.join(os.path.dirname(os.path.dirname(R2)), "data", "twos", TWOS_FILE)
    assert hashlib.sha256(open(p, "rb").read()).hexdigest() == TWOS_SHA
    d = pd.read_parquet(p)
    d[LABEL] = d[LABEL].astype(int)
    return d


def twos_split(d, split, seed):
    """'day' (pre-specified, doc 31): train Mon 20-Tue 21, validation Wed 22, test Thu 23-Fri 24 March 2017.
    'user' (sensitivity): user-disjoint as LANL (70/30 of users, then 15% of training users for validation,
    stratified by having a positive day, redrawn per seed)."""
    from fitd_common import DAY, split_user_day_frame
    if split == "day":
        day = pd.to_datetime(d[DAY])
        tr = d[day <= "2017-03-21"]
        va = d[day == "2017-03-22"]
        te = d[day >= "2017-03-23"]
        return tr, va, te
    tv, te = split_user_day_frame(d, "user_disjoint", 0.7, seed=seed)
    tr, va = split_user_day_frame(tv, "user_disjoint", 0.85, seed=seed + 1000)
    return tr, va, te


def e11_prep(seed, split="day"):
    P = prep(seed)
    d = twos_frame()
    cols = feature_cols(d)
    assert cols == P["cert42"]["cols"], "TWOS feature order differs"
    tr, va, te = twos_split(d, split, seed)
    sc = StandardScaler().fit(signed_log1p(tr[cols].to_numpy(np.float64)))
    f = lambda x: sc.transform(signed_log1p(x[cols].to_numpy(np.float64))).astype(np.float32)
    P["twos"] = dict(cols=cols, Xtr=f(tr), ytr=tr[LABEL].to_numpy(), Xva=f(va), yva=va[LABEL].to_numpy(),
                     Xte=f(te), yte=te[LABEL].to_numpy(), Xtv_raw=pd.concat([tr, va])[cols].to_numpy(np.float32),
                     ytv=pd.concat([tr, va])[LABEL].to_numpy(), Xte_raw=te[cols].to_numpy(np.float32),
                     kte_user=te[USER].astype(str).to_numpy())
    cnt = {f"twos_{k}_{s}": v for k, part in (("tr", tr), ("va", va), ("te", te))
           for s, v in (("rows", len(part)), ("posdays", int(part[LABEL].sum())),
                        ("posusers", int(part.groupby(USER)[LABEL].max().sum())), ("users", part[USER].nunique()))}
    return P, cnt


def e11(job, seed):
    """e11_main (day split): TWOS local MLP; pairs and all five under the 3 rules; pooled MLP (all five);
    R6 (TWOS outside: FedAvg size over the other four, scored on TWOS with its own training scaler); R5 (each single
    source's local MLP scored on TWOS). e11_user: user-disjoint sensitivity (local + all five, 3 rules).
    e11_trees (seeds 0-4): RF and XGBoost R0 on TWOS (train + validation, protocol settings)."""
    split = "user" if job == "e11_user" else "day"
    P, cnt = e11_prep(seed, split)
    rows = []
    c = P["twos"]
    ex = dict(split=split, **cnt)
    if job == "e11_trees":
        import xgboost as xgb
        from sklearn.ensemble import RandomForestClassifier
        m = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", n_jobs=2, random_state=seed)
        m.fit(c["Xtv_raw"], c["ytv"])
        pt = m.predict_proba(c["Xte_raw"])[:, 1]
        save_pred(job, "RF", seed, "twos", np.zeros(0), pt)
        rows.append(dict(job=job, regime="R0_RF", federation="twos", seed=seed, client="twos",
                         pr_auc=float(average_precision_score(c["yte"], pt)), base_rate=float(c["yte"].mean()), **ex))
        spw = (len(c["ytv"]) - c["ytv"].sum()) / max(c["ytv"].sum(), 1)
        m = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                              scale_pos_weight=spw, n_jobs=2, random_state=seed, tree_method="hist", eval_metric="aucpr")
        m.fit(c["Xtv_raw"], c["ytv"])
        pt = m.predict_proba(c["Xte_raw"])[:, 1]
        save_pred(job, "XGB", seed, "twos", np.zeros(0), pt)
        rows.append(dict(job=job, regime="R0_XGB", federation="twos", seed=seed, client="twos",
                         pr_auc=float(average_precision_score(c["yte"], pt)), base_rate=float(c["yte"].mean()), **ex))
        for r in rows:
            r["lift"] = r["pr_auc"] / r["base_rate"]
            log(f"{job} {r['regime']} seed{seed}: PR-AUC={r['pr_auc']:.4f}")
        return rows
    m, v, ep = train_mlp(c["Xtr"], c["ytr"], class_weights(c["ytr"]), [(c["Xva"], c["yva"])], seed)
    rec(rows, job, "R0_local", "twos", seed, "twos", P, m, dict(val=v, best_round=ep, **ex))
    feds = dict(E11_ALL5) if split == "user" else dict(E11_PAIRS, **E11_ALL5)
    for fed, members in feds.items():
        C = {n: P[n] for n in members}
        for rule in E11_RULES:
            ms, regime, bv, cost, ext = run_rule(rule, C, seed)
            for n in members:
                rec(rows, job, regime, fed, seed, n, P, ms[n], dict(val=bv, **cost, **ex))
    if split == "day":
        five = ("cert42", "cert52", "spedia", "lanl", "twos")
        Xtr = np.concatenate([P[n]["Xtr"] for n in five])
        ytr = np.concatenate([P[n]["ytr"] for n in five])
        wtr = np.concatenate([class_weights(P[n]["ytr"]) for n in five])
        m, v, ep = train_mlp(Xtr, ytr, wtr, [(P[n]["Xva"], P[n]["yva"]) for n in five], seed)
        for n in five:
            rec(rows, job, "R1_pooled", "pooled5", seed, n, P, m, dict(val=v, best_round=ep, **ex))
        ms, br, bv = run_fl({n: P[n] for n in ALL4}, seed, "size")
        rec(rows, job, "R6_loo", "C1+C2+C3+C4->C5", seed, "cert42", P, ms["cert42"], dict(val=bv, **ex), target="twos")
        for src in ALL4:
            s = P[src]
            m, v, ep = train_mlp(s["Xtr"], s["ytr"], class_weights(s["ytr"]), [(s["Xva"], s["yva"])], seed)
            rec(rows, job, "R5_zeroshot", f"{src}->C5", seed, src, P, m, dict(val=v, source=src, **ex), target="twos")
    return rows


# ---------------------------------------------------------------- E-7 trees, training rows only
RF_TREES = (100, 300, 500)
XGB_GRID = [(t, dep) for dep in (4, 6) for t in (100, 200, 400)]


def e7(job, seed):
    """RF and XGBoost fitted on TRAINING rows only (raw features, as the protocol's trees); the number of trees (and
    XGBoost depth from {4, 6}) chosen on validation PR-AUC: R0 per client (own validation), R1 pooled (mean
    validation PR-AUC over the four clients). Grids fixed here: RF trees {100, 300, 500}; XGBoost trees
    {100, 200, 400} x depth {4, 6}; other settings as the protocol (RF balanced_subsample; XGBoost lr 0.1, subsample
    0.8, colsample 0.8, scale_pos_weight = neg/pos of the training rows, hist)."""
    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier
    d = {n: load(n) for n in ALL4}
    raw = {}
    for n in ALL4:
        cols = feature_cols(d[n])
        _, tr, va, te = three_way(d[n], n, seed)
        raw[n] = {k: (x[cols].to_numpy(np.float32), x[LABEL].to_numpy()) for k, x in (("tr", tr), ("va", va), ("te", te))}
    def fit(kind, X, y, h):
        if kind == "RF":
            m = RandomForestClassifier(n_estimators=h[0], class_weight="balanced_subsample", n_jobs=2, random_state=seed)
        else:
            spw = (len(y) - y.sum()) / max(y.sum(), 1)
            m = xgb.XGBClassifier(n_estimators=h[0], max_depth=h[1], learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                                  scale_pos_weight=spw, n_jobs=2, random_state=seed, tree_method="hist", eval_metric="aucpr")
        return m.fit(X, y)
    ap = lambda m, n, k: float(average_precision_score(raw[n][k][1], m.predict_proba(raw[n][k][0])[:, 1]))
    rows = []
    for kind, grid in (("RF", [(t, None) for t in RF_TREES]), ("XGB", XGB_GRID)):
        for n in ALL4:                                       # R0
            best = None
            for h in grid:
                m = fit(kind, *raw[n]["tr"], h)
                v = ap(m, n, "va")
                if best is None or v > best[0]:
                    best = (v, h, m)
            v, h, m = best
            rows.append(dict(job=job, regime=f"R0_{kind}_trainonly", federation=n, seed=seed, client=n, val=v,
                             n_trees=h[0], depth=h[1], pr_auc=ap(m, n, "te"), base_rate=float(raw[n]["te"][1].mean())))
        X = np.concatenate([raw[n]["tr"][0] for n in ALL4])          # R1
        y = np.concatenate([raw[n]["tr"][1] for n in ALL4])
        best = None
        for h in grid:
            m = fit(kind, X, y, h)
            v = float(np.mean([ap(m, n, "va") for n in ALL4]))
            if best is None or v > best[0]:
                best = (v, h, m)
        v, h, m = best
        for n in ALL4:
            rows.append(dict(job=job, regime=f"R1_{kind}_trainonly", federation="pooled", seed=seed, client=n, val=v,
                             n_trees=h[0], depth=h[1], pr_auc=ap(m, n, "te"), base_rate=float(raw[n]["te"][1].mean())))
    for r in rows:
        r["lift"] = r["pr_auc"] / r["base_rate"]
        log(f"{job} {r['regime']} seed{seed} {r['client']}: trees={r['n_trees']} depth={r['depth']} PR-AUC={r['pr_auc']:.4f}")
    return rows


# ---------------------------------------------------------------- E-9 more cut points (CERT only)
def e9_prep(seed, q):
    """prep(seed) with CERT r4.2 and r5.2 cut at the date where the cumulative share of user-days reaches q (the
    protocol rule with q instead of 0.7); validation = latest 15% of training, same rule. SPEDIA and LANL unchanged."""
    from fitd_common import split_user_day_frame
    P = prep(seed)
    cnt = {}
    for n in ("cert42", "cert52"):
        d = load(n)
        cols = feature_cols(d)
        tv, te = split_user_day_frame(d, "temporal", q)
        tr, va = split_user_day_frame(tv, "temporal", 0.85)
        sc = StandardScaler().fit(signed_log1p(tr[cols].to_numpy(np.float64)))
        f = lambda x: sc.transform(signed_log1p(x[cols].to_numpy(np.float64))).astype(np.float32)
        P[n].update(Xtr=f(tr), ytr=tr[LABEL].to_numpy(), Xva=f(va), yva=va[LABEL].to_numpy(), Xte=f(te), yte=te[LABEL].to_numpy())
        for k, part in (("tr", tr), ("va", va), ("te", te)):
            cnt[f"{n}_{k}_rows"] = len(part)
            cnt[f"{n}_{k}_posdays"] = int(part[LABEL].sum())
            cnt[f"{n}_{k}_posusers"] = int(part.groupby(USER)[LABEL].max().sum())
        cnt[f"{n}_cut_day"] = str(tv["day"].max())
    return P, cnt


def e9(job, seed):
    q = float(job.split("_q")[1])
    P, cnt = e9_prep(seed, q)
    rows = []
    for n in ("cert42", "cert52"):
        c = P[n]
        m, v, ep = train_mlp(c["Xtr"], c["ytr"], class_weights(c["ytr"]), [(c["Xva"], c["yva"])], seed)
        rec(rows, job, "R0_local", n, seed, n, P, m, dict(val=v, best_round=ep, q=q, **cnt))
    for fed in ("C1+C2", "C1+C2+C3+C4"):
        C = {n: P[n] for n in FEDS[fed]}
        for rule in ("fedavg_size", "fedavg_equal"):
            ms, regime, bv, cost, ext = run_rule(rule, C, seed)
            for n in C:
                rec(rows, job, regime, fed, seed, n, P, ms[n], dict(val=bv, q=q, **cost, **cnt))
    return rows


# ---------------------------------------------------------------- E-10 LANL sampling
E10_FEDS = {"C1+C4": ("cert42", "lanl"), "C3+C4": ("spedia", "lanl"), "C1+C2+C3+C4": ALL4}


def e10_prep(seed, build):
    """prep(seed) with LANL replaced by a rebuilt sample (lanl_rebuild_E10.py: f05_s7, f02_s8, f02_s9; f02_s7 is the
    parity rebuild of the protocol cache). Same user-disjoint split rule and seed, scaler on LANL training rows."""
    P = prep(seed)
    d = pd.read_parquet(os.path.join(os.path.dirname(os.path.dirname(R2)), "data", "derived", f"lanl_user_day_{build}_fixed.parquet"))
    d[LABEL] = d[LABEL].astype(int)
    cols = feature_cols(d)
    assert cols == P["lanl"]["cols"]
    _, tr, va, te = three_way(d, "lanl", seed)
    sc = StandardScaler().fit(signed_log1p(tr[cols].to_numpy(np.float64)))
    f = lambda x: sc.transform(signed_log1p(x[cols].to_numpy(np.float64))).astype(np.float32)
    P["lanl"].update(Xtr=f(tr), ytr=tr[LABEL].to_numpy(), Xva=f(va), yva=va[LABEL].to_numpy(), Xte=f(te), yte=te[LABEL].to_numpy())
    cnt = {f"lanl_{k}_{x}": v for k, part in (("tr", tr), ("va", va), ("te", te))
           for x, v in (("rows", len(part)), ("posdays", int(part[LABEL].sum())),
                        ("posusers", int(part.groupby(USER)[LABEL].max().sum())))}
    return P, dict(build=build, **cnt)


def e10(job, seed):
    build = job.split("_", 1)[1]
    P, cnt = e10_prep(seed, build)
    rows = []
    c = P["lanl"]
    m, v, ep = train_mlp(c["Xtr"], c["ytr"], class_weights(c["ytr"]), [(c["Xva"], c["yva"])], seed)
    rec(rows, job, "R0_local", "lanl", seed, "lanl", P, m, dict(val=v, best_round=ep, **cnt))
    for fed, members in E10_FEDS.items():
        C = {n: P[n] for n in members}
        for rule in ("fedavg_size", "fedavg_equal"):
            ms, regime, bv, cost, ext = run_rule(rule, C, seed)
            for n in members:
                rec(rows, job, regime, fed, seed, n, P, ms[n], dict(val=bv, **cost, **cnt))
    return rows


# ---------------------------------------------------------------- dispatch
def run_seed(job, seed):
    if job == "e1_sel":
        return e1_sel(job, seed)
    if job == "e1_local":
        return e1_local(job, seed)
    if job == "e1_selx":
        return e1_selx(job, seed)
    if job == "e1_fixed":
        return e1_fixed(job, seed)
    if job == "e1_ref":
        return e1_ref(job, seed)
    if job in tuple("e1_" + r for r in E1_RULES):
        r = job[3:]
        return fed_job(job, "sgd_" + r, FEDS8, seed, lr=lr_of(r), hist=True)
    if job == "e12":
        return e12(job, seed)
    if job == "e4":
        return e4(job, seed)
    if job.startswith("e2_"):
        return e2(job, seed)
    if job in ("e3_fedperh", "e3_fedper2"):
        return fed_job(job, job[3:], FEDS8, seed)
    if job in ("e3_ditto_sel", "e3_dittofed_sel"):
        return e3_ditto_sel(job, seed)
    if job in ("e3_ditto", "e3_dittofed"):
        return e3_ditto(job, seed)
    if job.startswith("e5_"):
        return fed_job(job, job[3:], TRIPLES, seed)
    if job.startswith("e6_"):
        return e6(job, seed)
    if job.startswith("e10_"):
        return e10(job, seed)
    if job == "e7":
        return e7(job, seed)
    if job.startswith("e9_q"):
        return e9(job, seed)
    if job in ("e11_main", "e11_user", "e11_trees"):
        return e11(job, seed)
    raise ValueError(job)


def main(job, seeds):
    for seed in seeds:
        out = os.path.join(R2, f"{job}_s{seed}.csv")
        if os.path.exists(out):
            log(f"skip {job} seed{seed} (done)")
            continue
        rows = run_seed(job, seed)
        pd.DataFrame(rows).to_csv(out + ".tmp", index=False)
        os.replace(out + ".tmp", out)
    log(f"done {job}")


if __name__ == "__main__":
    a, b = ((sys.argv[2] if len(sys.argv) > 2 else "0").split("-") + [None])[:2]
    main(sys.argv[1], list(range(int(a), int(b or a) + 1)))
