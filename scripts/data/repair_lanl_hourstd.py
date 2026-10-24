"""E-10 repair: lanl_rebuild_E10.py computed hour_std from int32 sums on Windows (dt.hour is int32 there), so
hs**2 overflowed (wrapped mod 2^32) on high-volume user-days. hs is recovered exactly as round(hour_mean * n)
(hour_mean = hs / n is unaffected), the wrapped term is reproduced, the true sum of squares is recovered from the
stored (wrong) variance, and hour_std and all deviation columns are recomputed with the IJACSA code. Rows whose
wrong variance was clipped at 0 cannot be recovered and are counted. Parity on f02_s7 vs the protocol cache decides.
Run:  python repair_lanl_hourstd.py --ijacsa-repo <clone>   (inputs in $FITD_ROOT/data/derived/)"""
import argparse, sys, os, json
import numpy as np, pandas as pd
ap = argparse.ArgumentParser(description="E-10 repair of hour_std in the LANL rebuilds.")
ap.add_argument("--ijacsa-repo", required=True, help="local clone of the cross-dataset study repository (provides src.data)")
A = ap.parse_args()
sys.path.insert(0, A.ijacsa_repo)
from src.data.features import _add_causal_deviations, BASE_FEATURE_COLUMNS
os.chdir(os.environ.get("FITD_ROOT", "."))  # paths below are relative to FITD_ROOT (data/, data/derived/)
rep = {}
cache = pd.read_parquet("data/lanl_user_day_dev.parquet")
for b in ("f02_s7", "f05_s7", "f02_s8", "f02_s9"):
    d = pd.read_parquet(f"data/derived/lanl_user_day_{b}.parquet")
    d = d.sort_values(["user", "day"], kind="mergesort").reset_index(drop=True)
    n = d.n_events.to_numpy(np.int64)
    hs = np.rint(d.hour_mean.to_numpy() * n).astype(np.int64)
    sq = hs * hs
    wrapped = ((sq + 2**31) % 2**32) - 2**31
    sw = d.hour_std.to_numpy(float)
    bad = (sq != wrapped) & (n > 1)
    clipped = int((bad & (sw == 0)).sum())
    h2 = sw ** 2 * (n - 1) + wrapped / n
    var = (h2 - sq / n) / np.maximum(n - 1, 1)
    new = np.where(bad, np.sqrt(np.clip(var, 0, None)), sw)
    base = d[["user", "day", "label"] + BASE_FEATURE_COLUMNS].copy()
    base["hour_std"] = new
    out = _add_causal_deviations(base)[list(cache.columns)]
    out["day"] = out["day"].astype(cache["day"].dtype)
    out.to_parquet(f"data/derived/lanl_user_day_{b}_fixed.parquet", index=False)
    r = dict(rows_overflowed=int(bad.sum()), clipped_unrecoverable=clipped)
    if b == "f02_s7":
        m = out.merge(cache, on=["user", "day"], suffixes=("", "_c"), how="outer", indicator=True)
        both = m[m._merge == "both"]
        fc = [c for c in cache.columns if c not in ("user", "day")]
        dif = {c: float(np.abs(both[c].astype(float) - both[c + "_c"].astype(float)).max()) for c in fc}
        r.update(parity_one_side=int((m._merge != "both").sum()), parity_max_abs_diff=max(dif.values()),
                 parity_worst_col=max(dif, key=dif.get))
    rep[b] = r
    print(b, r, flush=True)
json.dump(rep, open("data/derived/lanl_repair_report.json", "w"), indent=1)
