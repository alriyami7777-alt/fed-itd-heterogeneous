"""C3 similarity measures (protocol Section 8), from each client's training partition only (seed 0)."""
import itertools, json
import numpy as np, pandas as pd
from scipy.spatial.distance import jensenshannon
import os
from config import RESULTS
from fitd_common import CLIENTS, load, three_way, feature_cols

cnt = None
S = {}
for n in CLIENTS:
    d = load(n); tv, tr, va, te = three_way(d, n, 0)
    cc = [c for c in feature_cols(d) if c.startswith("cnt_")]
    tot = tr[cc].clip(lower=0).sum()
    S[n] = dict(prev=float(tr.label.mean()), mix=(tot / tot.sum()).to_numpy(), observed=set(t for t in cc if tot[t] > 0), cols=cc)
rows = []
for a, b in itertools.combinations(CLIENTS, 2):
    pa, pb = S[a]["prev"], S[b]["prev"]
    rows.append(dict(pair=f"{a}-{b}", a=a, b=b,
                     prevalence_ratio=max(pa, pb) / min(pa, pb),
                     js_distance=float(jensenshannon(S[a]["mix"], S[b]["mix"], base=2)),
                     jaccard_coverage=len(S[a]["observed"] & S[b]["observed"]) / len(S[a]["observed"] | S[b]["observed"])))
t = pd.DataFrame(rows); t.to_csv(os.path.join(RESULTS, "c3_similarity.csv"), index=False)
print("count features:", len(S['cert42']['cols']))
for n in CLIENTS: print(n, "prev", round(S[n]["prev"], 5), "observed actions", sorted(x.replace('cnt_','') for x in S[n]["observed"]))
print(t.round(3).to_string(index=False))
