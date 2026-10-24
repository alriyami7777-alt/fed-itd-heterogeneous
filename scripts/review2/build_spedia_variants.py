"""E-6: SPEDIA label variants. Features unchanged; only the user-day label is rebuilt.
Event table: IJACSA load_spedia(real_only=True) on logs_SPEDIA.csv v1.0 (20,619 real-exercise events, no sampling),
features: IJACSA user_day_features(deviation=True). Parity: the default rule must reproduce the cached SPEDIA
user-day table exactly (features and labels). Variants (event-level, then user-day label = max over its events):
  default: description Highly|Midly Suspicious OR Wazuh level >= 8   (protocol)
  i_sev8:  level >= 8 only
  ii_desc: description Highly|Midly Suspicious only
  iii_sev7_or_desc: level >= 7 OR description
Output: data/derived/spedia_user_day_<variant>.parquet (same rows/features as the cache; label replaced).
Run:  IJACSA_REPO=<clone of crossdataset-insider-threat> python build_spedia_variants.py <path to logs_SPEDIA.csv>"""
import os as _os, sys as _sys  # repository layout: modules live in ../../src and ../
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_os.path.join(_HERE, "..", "..", "src"), _os.path.join(_HERE, "..")]
from config import ROOT as _ROOT
import os
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.environ["IJACSA_REPO"])  # local clone of the cross-dataset study repository
from src.data.loaders import load_spedia
from src.data.features import user_day_features

CSV = os.path.abspath(sys.argv[1])
_os.chdir(_ROOT)  # relative paths below (data/) are relative to FITD_ROOT
raw = pd.read_csv(CSV, low_memory=False)
raw = raw[raw["Decoder_name"] != "cert"].reset_index(drop=True)
desc = raw["Description"].fillna("").astype(str).str.contains("Highly Suspicious|Midly Suspicious", case=False)
lev = pd.to_numeric(raw["Level"], errors="coerce").fillna(0)
ev = load_spedia(CSV, real_only=True)          # IJACSA loader (default rule); drops rows with no timestamp
assert len(ev) == len(raw) == 20619, (len(ev), len(raw))
cache = pd.read_parquet("data/spedia_user_day_dev.parquet")
rules = {"default": desc | (lev >= 8), "i_sev8": lev >= 8, "ii_desc": desc, "iii_sev7_or_desc": desc | (lev >= 7)}
assert (rules["default"].astype(int).values == ev["label"].values).all()
os.makedirs("data/derived", exist_ok=True)
rows = []
for name, lab in rules.items():
    e = ev.copy()
    e["label"] = lab.astype(int).values
    f = user_day_features(e, deviation=True)
    f = f.sort_values(["user", "day"]).reset_index(drop=True)
    c = cache.sort_values(["user", "day"]).reset_index(drop=True)
    fc = [x for x in c.columns if x not in ("user", "day", "label")]
    assert (f["user"].values == c["user"].values).all() and (f["day"].values == c["day"].values).all()
    dmax = float(np.abs(f[fc].to_numpy(float) - c[fc].to_numpy(float)).max())
    assert dmax == 0.0, dmax
    out = cache.copy()
    key = f.set_index(["user", "day"])["label"]
    out["label"] = key.reindex(pd.MultiIndex.from_frame(cache[["user", "day"]])).values.astype(int)
    if name == "default":
        assert (out["label"].values == cache["label"].values).all(), "default rule does not reproduce the cache"
    out.to_parquet(f"data/derived/spedia_user_day_{name}.parquet", index=False)
    rows.append(dict(variant=name, events_pos=int(lab.sum()), userdays_pos=int(out.label.sum()),
                     users_pos=int(out.groupby("user").label.max().sum()), feature_max_abs_diff=dmax))
print(pd.DataFrame(rows).to_string(index=False))
