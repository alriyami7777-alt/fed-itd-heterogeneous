"""Federated regimes for protocol v1.0 FROZEN (change-log entries 2026-10-13 b–e).

Plain-PyTorch simulation of cross-silo FL (Flower could not be installed in the compute workspace;
the algorithms are the standard ones):
  * FedAvg (McMahan et al., 2017): clients start each round from the global weights, run E=1 local
    epoch (Adam lr 1e-3, batch 256, class-weighted BCE with the client's own prevalence), the server
    averages weights, weighted by client training size ("size") or equally ("equal").
  * FedProx (Li et al., 2020): FedAvg + (mu/2)*||w - w_global||^2 in each client's local loss.
  * FedPer (Arivazhagan et al., 2019): the output layer stays local; the rest is averaged.
  * FedPer-in (C2 variant b): the first (input) layer stays local; the rest is averaged.
  * FedAvg + local fine-tuning: FedAvg, then each client fine-tunes the selected global model for up
    to 10 local epochs with early stopping on its own validation PR-AUC.
Each client keeps its own Adam state across rounds. The reported model is the round with the best
mean validation PR-AUC over clients (each client scored with its own personalised layers, if any).
Rounds are capped at T=100 and stop early after 10 rounds without improvement.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from sklearn.metrics import average_precision_score

import os as _os, sys as _sys  # repository layout: MLP helpers live in ../scripts/run_e1_e2.py
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "scripts"))
from run_e1_e2 import MLP, class_weights, predict

torch.set_num_threads(int(__import__("os").environ.get("FITD_THREADS", "2")))


def _local_epoch(model, opt, X, y, w, gen, prox=None, mu=0.0, steps=None, cycle_steps=None):
    """One local epoch (default).
    steps=S (exploratory runs, change log h): S minibatches of 256 rows each drawn WITH replacement.
    cycle_steps=S (confirmatory run C-1): S minibatches drawn WITHOUT replacement, cycling through freshly
    shuffled epochs (a minibatch never crosses an epoch boundary)."""
    model.train()
    if cycle_steps:
        batches = []
        while len(batches) < cycle_steps:
            perm = torch.randperm(len(X), generator=gen)
            batches += [perm[i:i + 256] for i in range(0, len(X), 256)]
        batches = batches[:cycle_steps]
    elif steps:
        batches = [torch.randint(len(X), (min(256, len(X)),), generator=gen) for _ in range(steps)]
    else:
        perm = torch.randperm(len(X), generator=gen)
        batches = [perm[i:i + 256] for i in range(0, len(X), 256)]
    for idx in batches:
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(X[idx]), y[idx], weight=w[idx])
        if prox is not None and mu > 0:
            loss = loss + (mu / 2) * sum(((p - q) ** 2).sum() for p, q in zip(model.parameters(), prox))
        opt.zero_grad()
        loss.backward()
        opt.step()


def _local_keys(model, mode):
    keys = list(model.state_dict().keys())
    if mode == "fedper":      # output layer local
        return {k for k in keys if k.startswith("net.6.")}
    if mode == "fedper_in":   # input layer local
        return {k for k in keys if k.startswith("net.0.")}
    if mode == "fedper2":     # E-3: last two layers local (8,321 private weights)
        return {k for k in keys if k.startswith("net.3.") or k.startswith("net.6.")}
    return set()


class MLPHead(torch.nn.Module):
    """E-3 FedPer-H: shared trunk 42-128-64 (same layer indices as MLP) + head Linear(64,83)-ReLU-Linear(83,1).
    Head = 64*83+83 + 83+1 = 5,479 weights (private under mode='fedper')."""
    def __init__(self, d):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d, 128), torch.nn.ReLU(), torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Dropout(0.2),
            torch.nn.Sequential(torch.nn.Linear(64, 83), torch.nn.ReLU(), torch.nn.Linear(83, 1)))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class LogReg(torch.nn.Module):
    """E-1 reference check only: logistic regression."""
    def __init__(self, d):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(d, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


ARCH = {"mlp": MLP, "head83": MLPHead, "logreg": LogReg}

def _val_ap(y, p):
    """Validation PR-AUC; NaN when the model has diverged (non-finite scores), so a divergent round can never be
    selected and counts as a round without improvement (review-2 change log, E-1 divergence handling)."""
    p = np.asarray(p)
    if not np.isfinite(p).all():
        return float("nan")
    return average_precision_score(y, p)


LAST_DIV = {}
LAST = {}  # cost/diagnostic info from the most recent run_fl / finetune call (does not affect training)


def run_fl(clients: dict, seed: int, weighting="size", mu=0.0, mode="fedavg", T=100, patience=10, steps=None,
           match_local=False, client_steps=None, select="mean", stop_on=None, agg="fedavg", optimizer="adam", lr=None,
           ditto_lam=None, arch="mlp", history=False, trace=False):
    """clients: name -> dict(Xtr, ytr, Xva, yva). Returns (per-client models, best_round, best_val).
    match_local=True (single-client sanity check only): the shuffle generator is seeded exactly as in
    run_e1_e2.train_mlp, so FedAvg with one client and E=1 must reproduce the local-only MLP.
    Confirmatory run C-1 options (defaults keep the frozen behaviour):
      client_steps={name: S}: that client makes S without-replacement minibatch steps per round; the others one epoch.
      select="own": each client reports the round with its own best validation PR-AUC (personalised selection);
        training stops when client `stop_on` has not improved its own validation PR-AUC for `patience` rounds.
        best_round / best_val returned refer to `stop_on`.
    FedNova (D-1; Wang et al., 2020), agg="fednova": client i runs tau_i local steps (one epoch: ceil(n_i/256)),
      sends d_i = (w - w_i) / tau_i; the server sets w <- w - tau_eff * sum_i p_i d_i, tau_eff = sum_i p_i tau_i,
      with p_i the aggregation weights (size or equal). With one client this returns w_i exactly.
    optimizer="sgd" (D-1 sensitivity only): plain SGD, lr 0.01, no momentum, instead of Adam lr 1e-3.
    lr (E-1): overrides the learning rate of the chosen optimizer (default: Adam 1e-3, SGD 0.01).
    ditto_lam (E-3; Ditto, Li et al. ICML 2021): the global model is trained exactly as FedAvg; in addition each
      client keeps a personal model v_i (own Adam, lr 1e-3) and, every round, trains it for one local epoch on
      loss_i(v_i) + (lam/2)||v_i - w_global||^2, with w_global the global model at the start of the round.
      Selection and reporting use the personal models (best round by mean validation PR-AUC over clients).
      Review2 (E-3): the personal pass is taken AFTER the global (FedAvg) step of the round, anchored on the new
      global model.
    arch: "mlp" (default), "head83" (E-3 FedPer-H), "logreg" (E-1 reference check).
    history=True (E-1): LAST["history"] = per-round validation PR-AUC of every member."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    names = list(clients)
    d = next(iter(clients.values()))["Xtr"].shape[1]
    glob = ARCH[arch](d)
    local_keys = _local_keys(glob, mode)
    models = {n: copy.deepcopy(glob) for n in names}
    if optimizer == "sgd":
        opts = {n: torch.optim.SGD(models[n].parameters(), lr=lr or 0.01, momentum=0.0) for n in names}
    else:
        opts = {n: torch.optim.Adam(models[n].parameters(), lr=lr or 1e-3) for n in names}
    if ditto_lam is not None:
        pers = {n: copy.deepcopy(glob) for n in names}
        popts = {n: torch.optim.Adam(pers[n].parameters(), lr=1e-3) for n in names}
        pgens = {n: torch.Generator().manual_seed(seed * 100 + 50 + i) for i, n in enumerate(names)}
    data = {}
    for n in names:
        c = clients[n]
        data[n] = (torch.from_numpy(c["Xtr"]), torch.from_numpy(c["ytr"].astype(np.float32)),
                   torch.from_numpy(class_weights(c["ytr"]).astype(np.float32)))
    sizes = np.array([len(clients[n]["ytr"]) for n in names], float)
    agg_w = sizes / sizes.sum() if weighting == "size" else np.ones(len(names)) / len(names)
    gens = {n: torch.Generator().manual_seed(seed if match_local else seed * 100 + i) for i, n in enumerate(names)}
    n_shared = sum(v.numel() for k, v in glob.state_dict().items() if k not in local_keys)
    n_local = sum(v.numel() for k, v in glob.state_dict().items() if k in local_keys)
    cs = client_steps or {}
    taus = np.array([cs.get(n) or int(np.ceil(len(clients[n]["ytr"]) / 256)) for n in names], float)
    tau_eff = float((agg_w * taus).sum())
    best, best_round, best_states, bad = -1.0, 0, None, 0
    hist, gtrace = [], []
    LAST_DIV.clear()
    r = 0
    for r in range(1, T + 1):
        gstate = glob.state_dict()
        prox = [p.detach().clone() for p in glob.parameters()] if mu > 0 else None
        for n in names:
            st = models[n].state_dict()
            for k in gstate:
                if k not in local_keys:
                    st[k] = gstate[k].clone()
            models[n].load_state_dict(st)
            X, y, w = data[n]
            _local_epoch(models[n], opts[n], X, y, w, gens[n], prox, mu, steps,
                         (client_steps or {}).get(n))
        new = {}
        for k in gstate:
            if k in local_keys:
                new[k] = gstate[k]
            elif agg == "fednova":
                g = gstate[k]
                new[k] = g - tau_eff * sum(float(a) * (g - models[n].state_dict()[k]) / float(t)
                                           for a, t, n in zip(agg_w, taus, names))
            else:
                new[k] = sum(float(a) * models[n].state_dict()[k] for a, n in zip(agg_w, names))
        glob.load_state_dict(new)
        if trace:
            gtrace.append({k: v.detach().clone() for k, v in new.items()})
        if ditto_lam is not None:          # personal pass after the global step, anchored on the new global model
            anchor = [p.detach().clone() for p in glob.parameters()]
            for n in names:
                X, y, w = data[n]
                _local_epoch(pers[n], popts[n], X, y, w, pgens[n], anchor, ditto_lam)
        # evaluate: shared layers from the new global model, local layers from each client
        evals = {}
        for n in (names if ditto_lam is None else []):
            st = models[n].state_dict()
            for k in new:
                if k not in local_keys:
                    st[k] = new[k].clone()
            m = copy.deepcopy(models[n])
            m.load_state_dict(st)
            evals[n] = m
        if ditto_lam is not None:
            evals = pers
        if select == "own":
            if best_states is None:
                best_states, own_best, own_round = {}, {n: -1.0 for n in names}, {n: 0 for n in names}
            for n in names:
                v = _val_ap(clients[n]["yva"], predict(evals[n], clients[n]["Xva"]))
                if v > own_best[n]:
                    own_best[n], own_round[n] = v, r
                    best_states[n] = copy.deepcopy(evals[n].state_dict())
            if own_round[stop_on] == r:
                best, best_round, bad = own_best[stop_on], r, 0
            else:
                bad += 1
                if bad >= patience:
                    break
            continue
        vals = {n: _val_ap(clients[n]["yva"], predict(evals[n], clients[n]["Xva"])) for n in names}
        if not np.isfinite(list(vals.values())).all() and "diverged_round" not in LAST_DIV:
            LAST_DIV["diverged_round"] = r
        if history:
            hist.append({n: float(v) for n, v in vals.items()})
        score = float(np.mean(list(vals.values())))
        if score > best:
            best, best_round, bad = score, r, 0
            best_states = {n: copy.deepcopy(evals[n].state_dict()) for n in names}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_states is None:            # no round ever had finite validation scores (diverged from round 1)
        LAST_DIV["never_finite"] = True
        best_states = {n: copy.deepcopy(evals[n].state_dict()) for n in names}
        best = float("nan")
    out = {}
    for n in names:
        m = ARCH[arch](d)
        m.load_state_dict(best_states[n])
        out[n] = m
    LAST.clear()
    LAST.update(best_round=best_round, stop_round=r, params_shared=n_shared, params_local=n_local,
                bytes_per_round=n_shared * 4 * 2 * len(names), n_clients=len(names))
    LAST["diverged_round"] = LAST_DIV.get("diverged_round", 0)
    LAST["never_finite"] = LAST_DIV.get("never_finite", False)
    if history:
        LAST["history"] = hist
    if trace:
        LAST["global_trace"] = gtrace
    if agg == "fednova":
        LAST["tau"] = {n: int(t) for n, t in zip(names, taus)}
        LAST["tau_eff"] = tau_eff
    if select == "own":
        LAST["own_round"] = dict(own_round)
        LAST["own_val"] = dict(own_best)
    return out, best_round, best


def finetune(model, c, seed, epochs=10, patience=3):
    """FedAvg + local fine-tuning: start from the selected global model, train locally."""
    torch.manual_seed(seed)
    m = copy.deepcopy(model)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    X = torch.from_numpy(c["Xtr"])
    y = torch.from_numpy(c["ytr"].astype(np.float32))
    w = torch.from_numpy(class_weights(c["ytr"]).astype(np.float32))
    g = torch.Generator().manual_seed(seed)
    best = average_precision_score(c["yva"], predict(m, c["Xva"]))
    best_state, bad, best_ep, used = copy.deepcopy(m.state_dict()), 0, 0, 0
    for ep in range(1, epochs + 1):
        used = ep
        _local_epoch(m, opt, X, y, w, g)
        s = average_precision_score(c["yva"], predict(m, c["Xva"]))
        if s > best:
            best, best_state, bad, best_ep = s, copy.deepcopy(m.state_dict()), 0, ep
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(best_state)
    LAST["ft_epochs_run"] = used
    LAST["ft_best_epoch"] = best_ep
    return m
