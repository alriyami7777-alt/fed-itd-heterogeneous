"""After e1_selx: for every E-1 method whose learning rate changed, move its finished seed files (and their
predictions/histories stay as they are, overwritten on rerun) to results/review2/archive_e1_lrv1/ so the queue reruns
them at the final rate. e1_local is rerun if any local rate changed; e1_fixed if any FL rate changed."""
import os as _os, sys as _sys  # repository layout: modules live in ../../src and ../
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_os.path.join(_HERE, "..", "..", "src"), _os.path.join(_HERE, "..")]
from config import ROOT as _ROOT
_os.chdir(_ROOT)  # relative paths below (results/, data/) are relative to FITD_ROOT
import glob, json, os, shutil
R2 = "results/review2"
v1, v2 = json.load(open(f"{R2}/e1_lr_v1.json")), json.load(open(f"{R2}/e1_lr.json"))
meths = ["fedavg_size", "fedavg_equal", "fednova_size", "fednova_equal"]
changed = [m for m in meths if v1[m] != v2[m]]
local_changed = [k for k in v2 if k.startswith("local_") and "__" not in k and v1[k] != v2[k]]
jobs = [f"e1_{m}" for m in changed] + (["e1_local"] if local_changed else []) + (["e1_fixed"] if changed else [])
os.makedirs(f"{R2}/archive_e1_lrv1", exist_ok=True)
for j in jobs:
    for f in glob.glob(f"{R2}/{j}_s[0-9]*.csv"):
        shutil.move(f, f"{R2}/archive_e1_lrv1/{os.path.basename(f)}")
print("changed FL:", {m: (v1[m], v2[m]) for m in changed}, "changed local:", {k: (v1[k], v2[k]) for k in local_changed},
      "rerun:", jobs)
