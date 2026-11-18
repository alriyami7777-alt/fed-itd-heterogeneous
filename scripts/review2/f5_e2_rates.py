"""F-5 (review plan of 11 Nov 2026, M7): positive user-day rates in every E-2 / S2 variant, from the E-2 data preparation itself
(run_r2.e2_members, same seeds 0-4 as the runs). No training. Output: results/review2/f5_e2_rates.csv (per seed,
member, split) and f5_e2_rates_summary.csv (mean over seeds; min-max of the rate)."""
import os as _os, sys as _sys  # repository layout: modules live in ../../src and ../
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_os.path.join(_HERE, "..", "..", "src"), _os.path.join(_HERE, "..")]
from config import ROOT as _ROOT
_os.chdir(_ROOT)  # relative paths below (results/, data/) are relative to FITD_ROOT
import pandas as pd

from run_r2 import e2_members

rows = []
for variant in ("R", "S1", "S1b", "S2", "P", "C1", "C2", "S1C2"):
    for s in range(5):
        M, _, _ = e2_members(variant, s)
        for m, c in M.items():
            for sp, y in (("train", c["ytr"]), ("val", c["yva"]), ("test", c["yte"])):
                rows.append(dict(variant=variant, seed=s, member="A" if m == "m1" else m, split=sp, rows=len(y),
                                 pos_days=int(y.sum()), rate=float(y.mean()),
                                 pos_users=c["desc"]["posusers_" + {"train": "tr", "val": "va", "test": "te"}[sp]]))
D = pd.DataFrame(rows)
D.to_csv("results/review2/f5_e2_rates.csv", index=False)
D["role"] = D.member.where(D.member == "A", "others (m2-m4)")
g = D.groupby(["variant", "role", "split"], sort=False)
S = g.agg(rows=("rows", "mean"), pos_days=("pos_days", "mean"), pos_users=("pos_users", "mean"), rate=("rate", "mean"),
          rate_min=("rate", "min"), rate_max=("rate", "max")).reset_index()
S.to_csv("results/review2/f5_e2_rates_summary.csv", index=False)
pd.set_option("display.width", 250)
print(S.round(5).to_string(index=False))
