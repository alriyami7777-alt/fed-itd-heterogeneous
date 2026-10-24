"""Seeds 0-4 of the review runs must reproduce the stored 5-seed scores (same code path, same seeds).
Also checks the single-client sanity runs (protocol §6): fedavg1_matched_streams must equal R0 exactly, in
PR-AUC and in the saved prediction arrays. Note: fedavg1_fl_seeding equals the matched mode at seed 0
(generator seed = seed*100 + 0 = 0), so only seeds 1-4 test a different shuffle stream.
A job is reported identical only when all stored rows (all 5 seeds) are matched.
Writes results/review/reproduction_check.csv and results/review/sanity_check.csv."""
import os as _os, sys as _sys  # repository layout: modules live in ../src
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import glob
import os

import numpy as np
import pandas as pd

from config import RESULTS

LEGACY = {"mlp_local": "e1e2_mlp_local.csv", "mlp_pooled": "e1e2_mlp_pooled.csv", "fedavg_size": "fl_fedavg_size.csv",
          "fedavg_equal": "fl_fedavg_equal.csv", "fedper": "fl_fedper.csv", "fedper_in": "fl_fedper_in.csv",
          "fedprox": "fl_fedprox.csv", "finetune": "fl_finetune.csv", "loo": "fl_loo.csv",
          "zeroshot_mlp": "fl_zeroshot_mlp.csv"}
rows = []
for job, f in LEGACY.items():
    new = [pd.read_csv(x) for x in sorted(glob.glob(f"{RESULTS}/review/{job}_s[0-4].csv"))]
    if not new:
        continue
    n = pd.concat(new)
    o = pd.read_csv(os.path.join(RESULTS, f))
    key = ["seed", "client"] + (["federation"] if "federation" in o.columns and job != "mlp_pooled" else [])
    if job == "zeroshot_mlp":
        key = ["seed", "client", "source"]
    if job == "loo":
        key = ["seed", "client"]
    m = o.merge(n, on=key, suffixes=("_old", "_new"))
    d = (m.pr_auc_old - m.pr_auc_new).abs()
    complete = len(m) == len(o)
    rows.append(dict(job=job, stored_rows=len(o), matched_rows=len(m), complete=complete,
                     max_abs_diff=d.max(), identical=bool(complete and (d == 0).all())))
r = pd.DataFrame(rows)
r.to_csv(f"{RESULTS}/review/reproduction_check.csv", index=False)
print(r.to_string(index=False))

fs = sorted(glob.glob(f"{RESULTS}/review/sanity_s[0-4].csv"))
if fs:
    s = pd.concat([pd.read_csv(x) for x in fs])
    p = s.pivot_table(index=["seed", "client"], columns="regime", values="pr_auc")
    out = []
    for (seed, c), x in p.iterrows():
        z0 = np.load(f"{RESULTS}/preds/sanity/local/s{seed}_{c}.npz")
        z1 = np.load(f"{RESULTS}/preds/sanity/fedavg1_matched_streams/s{seed}_{c}.npz")
        out.append(dict(seed=seed, client=c,
                        matched_abs_diff=abs(x.fedavg1_matched_streams - x.R0_local_recomputed),
                        matched_arrays_identical=bool(np.array_equal(z0["val"], z1["val"]) and np.array_equal(z0["test"], z1["test"])),
                        fl_seeding_abs_diff=abs(x.fedavg1_fl_seeding - x.R0_local_recomputed),
                        fl_seeding_distinct_stream=seed != 0))
    q = pd.DataFrame(out)
    q.to_csv(f"{RESULTS}/review/sanity_check.csv", index=False)
    print(f"sanity: seeds {sorted(q.seed.unique())}; matched max |diff| = {q.matched_abs_diff.max()}; "
          f"all arrays identical = {q.matched_arrays_identical.all()}; "
          f"FL-seeding max |diff| (seeds 1-4) = {q[q.seed > 0].fl_seeding_abs_diff.max():.4f}")
