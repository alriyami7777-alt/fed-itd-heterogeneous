"""F-1 summary counts for the paper (review plan of 11 Nov 2026), computed from stats_twolevel_s0-19.csv (scope 1) and
stats_twolevel_<item>.csv (scopes 2-5). 'clear' = two-level 95% interval excludes 0 (new single criterion).
Output: results/review2/f1_summary_counts.csv and printed tables."""
import os as _os, sys as _sys  # repository layout: modules live in ../../src and ../
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_os.path.join(_HERE, "..", "..", "src"), _os.path.join(_HERE, "..")]
from config import ROOT as _ROOT
_os.chdir(_ROOT)  # relative paths below (results/, data/) are relative to FITD_ROOT
import os

import numpy as np
import pandas as pd

R2 = "results/review2"
O = pd.read_csv(f"{R2}/stats_twolevel_s0-19.csv")
SIX = ["FedAvg (size)", "FedAvg (equal)", "FedProx (μ=0.001)", "FedPer (local output layer)", "Private first layer",
       "FedAvg + fine-tune"]
out = []


def add(section, item, value, old=None):
    out.append(dict(section=section, item=item, new=value, old=old))


P = O[O.planned]
# Fig. 6: per rule over the 17 planned cases
for rule, g in P.groupby("rule", sort=False):
    if rule.startswith("Pooled"):
        continue
    for v in ("gain", "no clear difference", "loss"):
        add("Fig. 6 (17 planned cases per rule, two-level CI)", f"{rule}: {v}", int((g.tl_verdict == v).sum()),
            int((g.old_verdict == v).sum()))
# abstract
g = P[P.rule == "FedAvg (size)"]
add("Abstract", "FedAvg (size): clear losses of 17", int((g.tl_verdict == "loss").sum()), 12)
add("Abstract", "FedAvg (size): losses after Holm (a)", int((g.verdict_holm_a == "loss").sum()), 8)
# Sec. 4.2.1 over the 102
A = O[O.family_a_102]
cg = A[A.tl_verdict == "gain"]
add("Sec. 4.2.1 (102 planned)", "clear gains (two-level CI)", len(cg), 13)
add("Sec. 4.2.1 (102 planned)", "  of which LANL", int((cg.member == "LANL").sum()), 8)
add("Sec. 4.2.1 (102 planned)", "  of which CERT r4.2 with CERT r5.2", int(((cg.member == "CERT r4.2") & cg.group.str.contains("C2")).sum()), 5)
add("Sec. 4.2.1 (102 planned)", "clear losses (two-level CI)", int((A.tl_verdict == "loss").sum()))
add("Sec. 4.2.1 (102 planned)", "losses after Holm (a)", int((A.verdict_holm_a == "loss").sum()), 37)
add("Sec. 4.2.1 (102 planned)", "gains after Holm (a)", int((A.verdict_holm_a == "gain").sum()), 3)
add("All 156 cases", "clear gains / losses (two-level CI)", f"{(O.tl_verdict == 'gain').sum()} / {(O.tl_verdict == 'loss').sum()}")
add("All 156 cases", "gains / losses after Holm (b)", f"{(O.verdict_holm_b == 'gain').sum()} / {(O.verdict_holm_b == 'loss').sum()}")
F = O[O.family_c_fednova]
add("FedNova (Adam) family, 38", "clear gains / losses (two-level CI)", f"{(F.tl_verdict == 'gain').sum()} / {(F.tl_verdict == 'loss').sum()}")
add("FedNova (Adam) family, 38", "gains / losses after Holm (c)", f"{(F.verdict_holm_c == 'gain').sum()} / {(F.verdict_holm_c == 'loss').sum()}")
# Sec. 4.2.3: CERT r4.2 and CERT r5.2 in C1+C2 under the six planned rules
c12 = O[(O.group == "C1+C2") & O.rule.isin(SIX)]
for m, v in (("CERT r4.2", "gain"), ("CERT r5.2", "loss")):
    g = c12[c12.member == m]
    add("Sec. 4.2.3 (C1+C2, six planned rules)", f"{m}: clear {v} (two-level CI) of 6", int((g.tl_verdict == v).sum()),
        int((g.old_verdict == v).sum()))
    add("Sec. 4.2.3 (C1+C2, six planned rules)", f"{m}: {v} after Holm (a) of 6", int((g.verdict_holm_a == v).sum()))

# scopes 2-5
for it in ("e1", "e4", "e3", "e11", "e5"):
    f = f"{R2}/stats_twolevel_{it}.csv"
    if not os.path.exists(f):
        continue
    S = pd.read_csv(f)
    for (fam, comp), g in S.groupby(["family", "comparison"], sort=False):
        add(f"{it}: {fam}", comp, f"{(g.tl_verdict == 'gain').sum()} gain / {(g.tl_verdict == 'loss').sum()} loss "
            f"of {len(g)} (Holm: {(g.tl_verdict_holm == 'gain').sum()} / {(g.tl_verdict_holm == 'loss').sum()})",
            f"{(g.old_verdict == 'gain').sum()} / {(g.old_verdict == 'loss').sum()} "
            f"(Holm {(g.old_verdict_holm == 'gain').sum()} / {(g.old_verdict_holm == 'loss').sum()})")
S = pd.DataFrame(out)
S.to_csv(f"{R2}/f1_summary_counts.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 300)
pd.set_option("display.max_colwidth", 90)
print(S.to_string(index=False))
# flips
fl = pd.crosstab(O.old_verdict, O.tl_verdict)
print(fl)
