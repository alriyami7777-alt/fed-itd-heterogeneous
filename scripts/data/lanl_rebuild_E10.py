"""Run:  python lanl_rebuild_E10.py --ijacsa-repo <clone> --lanl-dir <folder with auth.txt.gz>   (needs pandas, numpy; pyarrow optional)
Same as lanl_rebuild_E10_original.py, with a pickle fallback. It was needed only on a machine where a security policy
blocked pyarrow; use the original wherever pyarrow works. Reads <lanl-dir>/auth.txt.gz once (1-2 h); writes 4 files
+ lanl_rebuild_report.json to --out.

E-10 LANL sampling rebuild (post-review plan; change log E-10 in docs/protocol.md). One pass over the raw auth.txt.gz, four label-agnostic sampling streams,
mirroring IJACSA load_lanl (strict uniform keep rule) + user_day_features(deviation=True):
  f02_s7 : 2%, RNG seed 7  -> must reproduce the strict cache (parity check)
  f05_s7 : 5%, RNG seed 7  (same uniform draws as f02_s7, threshold 0.05: a superset of the 2% sample)
  f02_s8 : 2%, RNG seed 8
  f02_s9 : 2%, RNG seed 9
Machine accounts dropped exactly as the loader (endswith $ or ^C\\d+$); the RNG draw happens per 1,000,000-row raw
chunk after that filter, as in the loader. Per (user, day) aggregates are accumulated per chunk (counts per action,
n, off-hours, hour sum and sum of squares, distinct dst hosts) so memory stays small."""
import os, sys, json, time
import numpy as np, pandas as pd
import argparse
ap = argparse.ArgumentParser(description="E-10 LANL sampling rebuild (four label-agnostic sampling streams).")
ap.add_argument("--ijacsa-repo", required=True, help="local clone of the cross-dataset study repository (provides src.data)")
ap.add_argument("--lanl-dir", required=True, help="folder holding auth.txt.gz and redteam.txt.gz")
ap.add_argument("--out", default=os.path.join(os.environ.get("FITD_ROOT", "."), "data", "derived"),
                help="output folder (default: $FITD_ROOT/data/derived)")
ap.add_argument("--cache", default=None, help="strict LANL user-day cache used for the parity check "
                "(default: <ijacsa-repo>/results/strict_camera_ready_2026_09_15/cache/lanl_user_day_dev.parquet)")
ap.add_argument("--cache-pkl", default=None, help="pickle copy of the cache, read only if parquet cannot be read (default: <out>/lanl_cache_for_check.pkl)")
A = ap.parse_args()
REPO = A.ijacsa_repo
sys.path.insert(0, REPO)
from src.data.features import _add_causal_deviations, BASE_FEATURE_COLUMNS
from src.data.schema import ACTIONS
LANL = A.lanl_dir
OUT = A.out
os.makedirs(OUT, exist_ok=True)
CACHE = A.cache or os.path.join(REPO, "results", "strict_camera_ready_2026_09_15", "cache", "lanl_user_day_dev.parquet")
EPOCH = pd.Timestamp("2015-01-01")
AUTH = ["time", "src_user", "dst_user", "src_comp", "dst_comp", "auth_type", "logon_type", "auth_orientation", "success"]
ORIENT = {"LogOn": "logon", "LogOff": "logoff"}
BUILDS = {"f02_s7": (7, 0.02), "f05_s7": (7, 0.05), "f02_s8": (8, 0.02), "f02_s9": (9, 0.02)}
STREAMS = {s: np.random.default_rng(s) for s in (7, 8, 9)}
rt = pd.read_csv(os.path.join(LANL, "redteam.txt.gz"), header=None, names=["time", "user", "src_comp", "dst_comp"], dtype=str, keep_default_na=False)
rt_user = rt["user"].astype(str).str.split("@", n=1).str[0]
rt_day = (EPOCH + pd.to_timedelta(pd.to_numeric(rt["time"]), unit="s")).dt.strftime("%Y-%m-%d")
mal = set(zip(rt_user, rt_day))
agg = {b: [] for b in BUILDS}; hosts = {b: [] for b in BUILDS}
def add(b, sub):
    ts = EPOCH + pd.to_timedelta(pd.to_numeric(sub["time"], errors="coerce"), unit="s")
    d = pd.DataFrame({"user": sub["user"].to_numpy(), "day": ts.dt.floor("D").to_numpy(), "hour": ts.dt.hour.to_numpy(),
                      "action": sub["auth_orientation"].map(ORIENT).fillna("auth").to_numpy(), "dst": sub["dst_comp"].to_numpy()})
    d = d[d["day"].notna()]
    d["off"] = ((d.hour < 7) | (d.hour > 19)).astype(int); d["h2"] = d.hour.astype(float) ** 2
    for a in ("logon", "logoff", "auth"):
        d["c_" + a] = (d.action == a).astype(int)
    g = d.groupby(["user", "day"]).agg(n=("hour", "size"), off=("off", "sum"), hs=("hour", "sum"), h2=("h2", "sum"),
                                       c_logon=("c_logon", "sum"), c_logoff=("c_logoff", "sum"), c_auth=("c_auth", "sum"))
    agg[b].append(g); hosts[b].append(d[["user", "day", "dst"]].drop_duplicates())
def compact(b):
    agg[b] = [pd.concat(agg[b]).groupby(level=[0, 1]).sum()]
    hosts[b] = [pd.concat(hosts[b]).drop_duplicates()]
t0 = time.time(); n_raw = n_user = 0; kept = {b: 0 for b in BUILDS}
for i, chunk in enumerate(pd.read_csv(os.path.join(LANL, "auth.txt.gz"), header=None, names=AUTH, dtype=str, keep_default_na=False, chunksize=1_000_000)):
    n_raw += len(chunk)
    user = chunk["src_user"].astype(str).str.split("@", n=1).str[0]
    mach = user.str.endswith("$") | user.str.match(r"^C\d+$", na=False)
    chunk = chunk.loc[~mach]; user = user.loc[~mach]
    if len(chunk) == 0:
        continue
    n_user += len(chunk)
    chunk = chunk.assign(user=user.to_numpy())
    u = {s: r.random(len(chunk)) for s, r in STREAMS.items()}
    for b, (s, f) in BUILDS.items():
        k = u[s] < f
        kept[b] += int(k.sum())
        if k.any():
            add(b, chunk.loc[k])
    if i % 25 == 24:
        for b in BUILDS: compact(b)
        print(f"chunk {i+1} raw={n_raw:,} kept={kept} {time.time()-t0:.0f}s", flush=True)
report = dict(raw_rows=n_raw, user_rows=n_user, kept_events=kept, wall_s=round(time.time() - t0))
try:
    cache = pd.read_parquet(CACHE)
except Exception as e:  # pyarrow unavailable: read the same cache saved as a pickle (--cache-pkl)
    print("parquet read failed (", e, "); using lanl_cache_for_check.pkl", flush=True)
    cache = pd.read_pickle(A.cache_pkl or os.path.join(OUT, "lanl_cache_for_check.pkl"))
for b in BUILDS:
    compact(b)
    g = agg[b][0].reset_index()
    nh = hosts[b][0].groupby(["user", "day"]).size().rename("n_distinct_hosts").reset_index()
    g = g.merge(nh, on=["user", "day"])
    out = pd.DataFrame({"user": g.user, "day": g.day})
    for a in ACTIONS:
        out["cnt_" + a] = g["c_" + a] if a in ("logon", "logoff", "auth") else 0
    out["n_events"] = g.n; out["n_distinct_hosts"] = g.n_distinct_hosts
    out["frac_offhours"] = g.off / g.n; out["hour_mean"] = g.hs / g.n
    var = (g.h2 - g.hs ** 2 / g.n) / (g.n - 1)
    out["hour_std"] = np.sqrt(var.clip(lower=0)).where(g.n > 1, 0.0)
    out["label"] = [int((x, dd.strftime("%Y-%m-%d")) in mal) for x, dd in zip(out.user, out.day)]
    out = _add_causal_deviations(out)
    out = out[list(cache.columns)]
    p = os.path.join(OUT, f"lanl_user_day_{b}.parquet")
    try:
        out.to_parquet(p, index=False)
    except Exception as e:  # pyarrow blocked: write a pickle instead (convert to parquet elsewhere)
        print("parquet write failed (", e, "); writing pickle", flush=True)
        out.to_pickle(p.replace(".parquet", ".pkl"), protocol=4)
    pos = out[out.label == 1]
    ev_pos = g.set_index(["user", "day"]).n.reindex(pd.MultiIndex.from_frame(pos[["user", "day"]])).to_numpy()
    rep = dict(rows=len(out), users=int(out.user.nunique()), pos_userdays=int(out.label.sum()),
               pos_users=int(pos.user.nunique()), events_per_pos_day_mean=float(ev_pos.mean()),
               events_per_pos_day_median=float(np.median(ev_pos)), malicious_userdays_total=len(mal))
    if b == "f02_s7":
        m = out.merge(cache, on=["user", "day"], how="outer", suffixes=("", "_c"), indicator=True)
        both = m[m._merge == "both"]
        fc = [c for c in cache.columns if c not in ("user", "day")]
        rep["parity_rows_only_one_side"] = int((m._merge != "both").sum())
        rep["parity_max_abs_diff"] = float(max(np.nanmax(np.abs(both[c].to_numpy(float) - both[c + "_c"].to_numpy(float))) for c in fc))
    report[b] = rep
    print(b, rep, flush=True)
json.dump(report, open(os.path.join(OUT, "lanl_rebuild_report.json"), "w"), indent=1, default=str)
print("DONE", flush=True)
