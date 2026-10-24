"""E-2 C2: CERT r5.2 user-day table rebuilt from logon/logoff events only (the actions LANL also records; CERT has no
'auth'). Rewritten on 24 Oct 2026 from the recipe in the change log (2026-10-20, E-2 C2) because the 20 Oct copy of
this script was not kept; the output is checked against the 20 Oct file (see --check).
Check result (24 Oct 2026): the rebuilt table equals the 20 Oct file exactly (pandas DataFrame.equals: same 693,649
user-days, labels, all 42 feature values and dtypes; 1,004 zero-filled user-days). The parquet bytes differ (file
metadata), so the SHA-256 of a new write is not f5a762b9...a17a; compare contents, not hashes.

Recipe:
  events  : CERT r5.2 logon.csv, mapped exactly as the IJACSA CERT loader does (timestamp format, activity -> action)
  features: IJACSA user_day_features(deviation=True) building blocks: base features per (user, day) from the kept
            events; reindexed onto the full-cache user-days (labels and user-days kept from the cache; user-days with
            no kept event are zero-filled BEFORE the causal deviations are computed); then _add_causal_deviations.
Usage: IJACSA_REPO=<clone of crossdataset-insider-threat> python3 rebuild_c52.py <path to r5.2 logon.csv> [--check]
Output: data/derived/cert52_user_day_cov_logon_rebuild.parquet (with --check: written to *_check.parquet and compared)."""
import os as _os, sys as _sys  # repository layout: config lives in ../../src
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "src"))
from config import ROOT as _ROOT
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.environ["IJACSA_REPO"])  # local clone of the cross-dataset study repository
from src.data import features as F  # noqa: E402
from src.data.loaders import _ACT, _CERT_DATE  # noqa: E402

LOGON = os.path.abspath(sys.argv[1])
os.chdir(_ROOT)  # relative paths below (data/) are relative to FITD_ROOT
CHECK = "--check" in sys.argv
OUT = "data/derived/cert52_user_day_cov_logon_rebuild.parquet"
cache = pd.read_parquet("data/cert52_user_day_dev.parquet")

raw = pd.read_csv(LOGON, dtype=str, keep_default_na=False)
ts = pd.to_datetime(raw["date"], format=_CERT_DATE, errors="coerce")
ev = pd.DataFrame({F.TIMESTAMP: ts, F.USER: raw["user"].astype(str), F.DST_HOST: raw["pc"].astype(str),
                   F.ACTION: raw["activity"].map(_ACT).fillna("unknown"), F.LABEL: 0}).dropna(subset=[F.TIMESTAMP])
print("logon events:", len(ev), "actions:", ev[F.ACTION].value_counts().to_dict())

# base features from the kept events (same code path as user_day_features, without deviations)
base = F.user_day_features(ev, deviation=False).drop(columns=[F.LABEL])
del raw, ev, ts
base["day"] = base["day"].astype(cache["day"].dtype)
key = cache[[F.USER, "day", F.LABEL]].copy()
out = key.merge(base, on=[F.USER, "day"], how="left")
n_missing = int(out["n_events"].isna().sum())
extra = len(base) - (len(out) - n_missing)
print(f"cache user-days {len(cache):,}; with no kept event (zero-filled) {n_missing:,}; rebuilt days not in cache {extra:,}")
for c in F.BASE_FEATURE_COLUMNS:
    out[c] = out[c].fillna(0)
out = F._add_causal_deviations(out)
out = cache[[F.USER, "day"]].merge(out.drop(columns=[F.LABEL]), on=[F.USER, "day"], how="left").assign(**{F.LABEL: cache[F.LABEL].values})
out = out[list(cache.columns)]
for c in (F.USER, "day", F.LABEL):          # feature columns stay float64 (zero-fill made them float), as in 20 Oct
    out[c] = out[c].astype(cache[c].dtype)
assert (out[F.USER].values == cache[F.USER].values).all() and (out[F.LABEL].values == cache[F.LABEL].values).all()

if CHECK:
    old = pd.read_parquet(OUT)
    num = [c for c in cache.columns if c not in (F.USER, "day")]
    same_keys = (old[F.USER].values == out[F.USER].values).all() and (old["day"].values == out["day"].values).all()
    dmax = float(np.nanmax(np.abs(old[num].to_numpy(float) - out[num].to_numpy(float))))
    print("check vs 20 Oct file: same user-days/labels:", same_keys, "| max |diff| over features:", dmax)
    out.to_parquet(OUT.replace(".parquet", "_check.parquet"), index=False)
else:
    out.to_parquet(OUT, index=False)
    print("written", OUT)
