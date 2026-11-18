"""F-4 input: scenario of each insider day in CERT r5.2, from the dataset's answer key.

The CERT answer key (answers.tar.bz2 of the CERT Insider Threat Test Dataset) holds one folder per release and
scenario, r5.2-<scenario>/, with one file per insider, r5.2-<scenario>-<USER>.csv. Each line of such a file is one
malicious event: <type>,<id>,<MM/DD/YYYY HH:MM:SS>,<user>,... This script lists every distinct (scenario, insider,
date), taking the insider from the file name, and writes them, sorted, as a headerless CSV: scenario,user,MM/DD/YYYY. f4_label.py reads that file from
data/answers_r52/f4_r52_answer_days.csv to give each positive user-day of CERT r5.2 its scenario.

The output is derived from the dataset, so it is not kept in the repository. Rewritten on 18 Nov 2026 (the first
version was an unsaved one-off); checked to reproduce the file used for the F-4 runs byte for byte (1,309 rows:
scenario 1: 85, scenario 2: 863, scenario 3: 20, scenario 4: 341).

Usage: python scripts/data/build_f4_answer_days.py <answers folder> [output csv]"""
import csv
import glob
import os
import sys

src = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join("data", "answers_r52", "f4_r52_answer_days.csv")
rows = set()
for scen in (1, 2, 3, 4):
    files = sorted(glob.glob(os.path.join(src, f"r5.2-{scen}", f"r5.2-{scen}-*.csv")))
    assert files, f"no files for scenario {scen} under {src}"
    for f in files:
        user = os.path.basename(f)[len(f"r5.2-{scen}-"):-4]
        with open(f, newline="", encoding="utf-8", errors="replace") as fh:
            for r in csv.reader(fh):
                if len(r) >= 4:  # the insider is the one named in the file name (2 events carry another user id)
                    rows.add((scen, user, r[2].split(" ")[0]))
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
with open(out, "w", newline="") as fh:
    fh.write("".join(f"{s},{u},{d}\n" for s, u, d in sorted(rows)))
print(len(rows), "rows ->", out)
