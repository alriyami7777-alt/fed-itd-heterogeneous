"""Two-level paired bootstrap engine (review plan of 11 Nov 2026, F-1; change log 2026-11-11 10:55).

In each draw b: the seeds are resampled with replacement; each member's TEST USERS are resampled with replacement
(a user's days kept together); the same user draw is used for the federated run and for the member's local run, so the
difference stays paired. Members whose test set does not depend on the seed (CERT, SPEDIA, TWOS) get one user draw per
draw b, shared by all drawn seeds. LANL (user split changes with the seed) gets an independent user draw inside each
drawn seed's own test set, one per (b, slot). Statistic: mean over the drawn seeds of PR-AUC(fed) - PR-AUC(local).
Draws whose resampled test set has no positive day (in any slot) are dropped and counted.

PR-AUC is sklearn's average precision with sample weights = the user's multiplicity in the draw. It is computed
exactly from precomputed per-user counts at each distinct score threshold that contains a positive row:
  cumW[b,k] = sum_u m[b,u] * C[u,k],  cumP[b,k] = sum_u m[b,u] * Cp[u,k]
  AP[b] = sum_k (cumP[b,k] - cumP[b,k-1]) * cumP[b,k] / cumW[b,k] / cumP[b,K]
(checked against sklearn with sample_weight in self_test()).

Seed draws: seed_index() (default_rng(SEED)). User draws: per member from default_rng([SEED, member code]) (LANL:
default_rng([SEED, code, seed])), so every cell of a member uses the same draws and results do not depend on the
order in which cells are processed."""
import numpy as np
from sklearn.metrics import average_precision_score

B = 10_000
SEED = 20261111


def prep_counts(score, y, uidx, U):
    """Per-user cumulative row and positive counts at the end of each distinct score value containing a positive."""
    o = np.argsort(-score, kind="mergesort")
    s, yy, uu = score[o], y[o], uidx[o]
    last = np.r_[np.flatnonzero(s[1:] != s[:-1]), len(s) - 1]          # end index of each distinct-score group
    gpos = np.add.reduceat(yy, np.r_[0, last[:-1] + 1]) > 0
    ends = last[gpos]
    K = len(ends)
    C = np.zeros((U, K), np.float32)
    Cp = np.zeros((U, K), np.float32)
    prev = 0
    cw = np.zeros(U)
    cp = np.zeros(U)
    for k, e in enumerate(ends):
        seg = slice(prev, e + 1)
        cw += np.bincount(uu[seg], minlength=U)
        cp += np.bincount(uu[seg], weights=yy[seg], minlength=U)
        C[:, k], Cp[:, k] = cw, cp
        prev = e + 1
    return C, Cp


def ap_weighted(Mw, C, Cp):
    """AP for each row of the user-multiplicity matrix Mw (n_draws x U)."""
    if C.shape[1] == 0:
        return np.full(Mw.shape[0], np.nan)
    cw = Mw @ C
    cp = Mw @ Cp
    tot = cp[:, -1]
    dp = np.diff(np.c_[np.zeros(len(cp)), cp], axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.where(cw > 0, cp / cw, 0.0)
        ap = (dp * prec).sum(1) / tot
    ap[tot <= 0] = np.nan
    return ap


def user_index(users):
    uniq, inv = np.unique(users, return_inverse=True)
    return inv, len(uniq)


def draw_users(rng, U, n):
    """n draws of U users with replacement -> multiplicity matrix (n x U), float32."""
    picks = rng.integers(0, U, (n, U))
    flat = (np.arange(n)[:, None] * U + picks).ravel()
    return np.bincount(flat, minlength=n * U).astype(np.float32).reshape(n, U)


def seed_index(seeds=tuple(range(20)), rng_seed=SEED):
    """B x len(seeds) matrix of resampled seed VALUES (first use of the RNG). For 20 seeds 0-19 this is the same
    matrix as in the scope-1 run."""
    seeds = np.asarray(sorted(seeds))
    return seeds[np.random.default_rng(rng_seed).integers(0, len(seeds), (B, len(seeds)))]


def run_member(code, fixed, y, users, bases, cells, idx, skip_empty=False):
    """Two-level statistic for every cell of one member.
    fixed: True -> y, users are arrays (same test set for every seed); False (LANL) -> dicts seed -> array.
    bases: dict base_key -> dict seed -> baseline test scores (e.g. the local model);
    cells: dict cell_id -> (dict seed -> compared test scores, base_key).
    User draws: fixed members, one draw per b from default_rng([SEED, code]); LANL, for seed s one draw per slot (b, j)
    with idx[b, j] == s, from default_rng([SEED, code, s]). Returns dict cell_id -> B-vector (NaN = dropped draw)."""
    nB, nS = idx.shape
    seeds = sorted(set(idx.ravel().tolist()))
    pos = {s: i for i, s in enumerate(seeds)}
    if fixed:
        uidx, U = user_index(users)
        M = draw_users(np.random.default_rng([SEED, code]), U, nB)
        yy = y.astype(np.float64)
        apd = lambda sc: ap_weighted(M, *prep_counts(sc.astype(np.float64), yy, uidx, U))
        Db = {k: {s: apd(b[s]) for s in seeds} for k, b in bases.items()}
        rowpos = np.vectorize(pos.get)(idx)
        out = {}
        for cid, (fed, bk) in cells.items():
            D = np.stack([apd(fed[s]) - Db[bk][s] for s in seeds])
            out[cid] = D[rowpos, np.arange(nB)[:, None]].mean(1)
        return out
    acc = {cid: np.full((nB, nS), np.nan) for cid in cells}
    for s in seeds:
        occ = np.argwhere(idx == s)
        uidx, U = user_index(users[s])
        M = draw_users(np.random.default_rng([SEED, code, s]), U, len(occ))
        yy = y[s].astype(np.float64)
        apd = lambda sc: ap_weighted(M, *prep_counts(sc.astype(np.float64), yy, uidx, U))
        db = {k: apd(b[s]) for k, b in bases.items()}
        for cid, (fed, bk) in cells.items():
            acc[cid][occ[:, 0], occ[:, 1]] = apd(fed[s]) - db[bk]
    # skip_empty (F-4 sensitivity, added 11 Nov 22:55): a seed slot whose resampled test set has no positive user is
    # left out of that draw's mean instead of dropping the whole draw (default False = the rule used everywhere else)
    if skip_empty:
        return {cid: np.nanmean(a, 1) for cid, a in acc.items()}
    return {cid: a.mean(1) for cid, a in acc.items()}


def summarise(d):
    ok = d[~np.isnan(d)]
    lo, hi = np.percentile(ok, [2.5, 97.5])
    p = min(1.0, 2 * min((ok <= 0).mean(), (ok >= 0).mean()))
    return dict(tl_lo=lo, tl_hi=hi, tl_p=p, tl_dropped=int(np.isnan(d).sum()),
                tl_verdict="gain" if lo > 0 else ("loss" if hi < 0 else "no clear difference"))


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def self_test():
    rng = np.random.default_rng(1)
    n, U = 3000, 40
    y = (rng.random(n) < 0.05).astype(float)
    sc = np.round(rng.random(n) + y * 0.3, 2)        # ties on purpose
    u = rng.integers(0, U, n)
    M = draw_users(rng, U, 5)
    C, Cp = prep_counts(sc, y, u, U)
    a = ap_weighted(M, C, Cp)
    b = [average_precision_score(y, sc, sample_weight=M[i][u]) for i in range(5)]
    assert np.allclose(a, b, atol=1e-9), (a, b)
    return float(np.abs(a - np.array(b)).max())


if __name__ == "__main__":
    print("self-test max |AP - sklearn| =", self_test())
