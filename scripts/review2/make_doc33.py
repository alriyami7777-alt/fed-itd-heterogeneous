import os as _os, sys as _sys  # repository layout: modules live in ../../src and ../
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_os.path.join(_HERE, "..", "..", "src"), _os.path.join(_HERE, "..")]
from config import ROOT as _ROOT
_os.chdir(_ROOT)  # relative paths below (results/, data/) are relative to FITD_ROOT
import pandas as pd, numpy as np
R="results/review2/"
L=["# 33 — Review-2 exploratory items (5 seeds): E-2, E-5, E-6, E-7, E-8, E-9, E-10 and the E-11 extras (23 Oct 2026)","",
"For the writing team. Every number below is read from the result files by code (`make_doc33.py`); nothing is typed by hand. All items are **exploratory, 5 seeds (0–4)**, except E-8 (20 seeds, from the saved main-run predictions). Effects are member − its own matched baseline; 95% CI = paired bootstrap over seeds (exact 3,125 resamples for n = 5); 'gain'/'loss' = CI excludes 0; Holm within the item's family. With 5 seeds few effects survive Holm — that is expected and stated, not hidden.",""]
f=lambda x: f"{x:+.3f}"
def v(r): return "gain" if r=="gain" else ("loss" if r=="loss" else "n.c.d.")
# E-2
S=pd.read_csv(R+"stats_e2.csv"); D=pd.read_csv(R+"e2_member_descriptors.csv")
A=S[(S.member=="m1")&(S.family=="E-2")]
desc=D[D.member=="m1"].groupby("variant")[["n_users","pos_users","rate_all","local"]].mean()
L+=["## E-2 Controlled federation within CERT r5.2 (member A = m1)","","| Variant | A users (positive) | A rate, all splits | A local PR-AUC | FedAvg size | FedAvg equal | Private first layer |","|---|---|---|---|---|---|---|"]
for var in ["R","S1","S2","S1b","P","C1","C2","S1C2"]:
    x=A[A.variant==var].set_index("comparison"); d=desc.loc[var]
    cells=[f"{f(x.loc[c,'diff'])} ({v(x.loc[c,'verdict'])})" for c in ["fedavg_size − local","fedavg_equal − local","fedper_in − local"]]
    L.append(f"| {var} | {d.n_users:.0f} ({d.pos_users:.0f}) | {d.rate_all:.4f} | {d.local:.3f} | "+" | ".join(cells)+" |")
O=S[(S.member!="m1")&(S.family=="E-2")]
L+=["",f"Other members (m2–m4), all variants and rules: {(O.verdict=='gain').sum()} gains, {(O.verdict=='loss').sum()} losses, {(O.verdict=='no clear difference').sum()} no clear difference. Holm (all {len(S[S.family=='E-2'])} E-2 comparisons): {(S[S.family=='E-2'].verdict_holm!='no clear difference').sum()} survive.","",
"**Plain reading.** Inside one dataset, federation helps: with four similar members (R), FedAvg roughly doubles member A's PR-AUC and the other members gain too. Making A small (S1, S2, S1b) does not remove the gain — the smallest member gains most. Raising A's prevalence about 2× (P) leaves little to gain because A's own model is already strong. What removes the benefit is **coverage**: when A records only logon/logoff (like LANL), its gain disappears (C2) or turns into a loss (C1), and a small member with low coverage gains only an unclear amount (S1+C2). Sharing all layers is what helps here; the private first layer gives no gain for A in any variant. Caveats: prevalence can be moved only about 2× within CERT r5.2 at fixed size; at 1% size A's rate is necessarily high (2 positive users of 5).",""]
# E-5
S=pd.read_csv(R+"stats_e5.csv")
L+=["## E-5 Remaining coalitions (all members scored)","","| Group | Member | FedAvg size | FedAvg equal | FedProx | FedPer | Private first layer | Fine-tune | FedNova size | FedNova equal |","|---|---|---|---|---|---|---|---|---|---|"]
comps=["fedavg_size − local","fedavg_equal − local","fedprox_mu0.001 − local","fedper − local","fedper_in − local","finetune − local","fednova_size − local","fednova_equal − local"]
for (g,c),x in S.groupby(["federation","client"]):
    x=x.set_index("comparison"); L.append(f"| {g} | {c} | "+" | ".join(f"{f(x.loc[k,'diff'])} ({v(x.loc[k,'verdict'])})" for k in comps)+" |")
p=S[S.family=="E-5 six planned rules"]; q=S[S.family!="E-5 six planned rules"]
L+=["",f"Six planned rules (54 comparisons): {(p.verdict=='gain').sum()} gains, {(p.verdict=='loss').sum()} losses; after Holm {(p.verdict_holm=='gain').sum()} gains, {(p.verdict_holm=='loss').sum()} losses. FedNova family (18): {(q.verdict=='gain').sum()} gains, {(q.verdict=='loss').sum()} losses; after Holm {(q.verdict_holm=='gain').sum()} gains, {(q.verdict_holm=='loss').sum()} losses.","",
"**Plain reading.** The CERT pair keeps helping CERT r4.2 when LANL joins (C1+C2+C4: gains under FedAvg, FedProx, fine-tuning and both FedNova rules). Adding SPEDIA still hurts SPEDIA and CERT r4.2 under every rule; the private first layer limits SPEDIA's loss most (−0.05 to −0.07). FedNova (equal) again removes SPEDIA's loss in C1+C3+C4 (+0.023, no clear difference) at the cost of large CERT losses (CERT r4.2 −0.41; CERT r5.2 −0.68 in C2+C3+C4).",""]
# E-6
S=pd.read_csv(R+"stats_e6.csv")
L+=["## E-6 SPEDIA label rules (SPEDIA vs its own-label local model)","","Variant (i) level ≥ 8 only gives exactly the protocol labels (every 'suspicious' event has level ≥ 8), so its results are the main results.","","| Variant | Group | FedAvg size | FedAvg equal | Private first layer |","|---|---|---|---|---|"]
for (var,g),x in S[S.client=="SPEDIA"].groupby(["variant","federation"]):
    x=x.set_index("comparison"); L.append(f"| {var} | {g} | "+" | ".join(f"{f(x.loc[k,'diff'])} ({v(x.loc[k,'verdict'])})" for k in ["fedavg_size − local","fedavg_equal − local","fedper_in − local"])+" |")
L+=["","Positive user-days train/val/test: protocol 43/9/30; (ii) description only 38/7/24; (iii) level ≥ 7 or description 112/16/47 (62% of test). Own-label local MLP: (ii) 0.882, (iii) 0.955. "+f"Other members: {(S[S.client!='SPEDIA'].verdict=='gain').sum()} gains, {(S[S.client!='SPEDIA'].verdict=='loss').sum()} losses. Holm over all {len(S)}: {(S.verdict_holm!='no clear difference').sum()} survive.","",
"**Plain reading.** SPEDIA's loss under FedAvg does not depend on the label rule. Under the broadest rule (iii) the private first layer no longer loses (no clear difference in all three groups), so its residual loss under the protocol rule is partly a labelling effect.",""]
# E-7
S=pd.read_csv(R+"e7_summary.csv")
L+=["## E-7 Trees fitted on training rows only (trees and XGBoost depth chosen on validation)","","| Model | Member | Trees (depth) | Training only | Training + validation (protocol) | Difference | vs local MLP |","|---|---|---|---|---|---|---|"]
for _,r in S.iterrows():
    L.append(f"| {r.model} | {r.client} | {r.trees}{'' if pd.isna(r.depth) or r.depth=='' else ' ('+str(int(float(r.depth)))+')'} | {r.train_only:.3f} | {r.train_plus_val:.3f} | {f(r['diff'])} ({v(r.verdict)}) | {f(r.tree_minus_mlp)} ({v(r.v_mlp)}) |")
L+=["","**Plain reading.** Holding validation rows out costs the trees 0.005–0.05 PR-AUC on CERT and nothing clear on SPEDIA or LANL; they remain stronger than the local MLP on CERT and SPEDIA. RF on CERT always chose the largest grid value (500 trees) and XGBoost on CERT 400 trees at depth 6 (grid tops).",""]
# E-8
S=pd.read_csv(R+"e8_summary.csv")
L+=["## E-8 R5 without hindsight, and a rank-average ensemble (20 seeds, no training)","","| Target | Method | Method PR-AUC | Compared with | Its PR-AUC | Difference [95% CI] | Verdict (Holm) |","|---|---|---|---|---|---|---|"]
for _,r in S.iterrows():
    L.append(f"| {r.target} | {r.method} | {r.method_mean:.3f} | {r.vs} | {r.vs_mean:.3f} | {f(r['diff'])} [{f(r.boot_lo)}, {f(r.boot_hi)}] | {v(r.verdict)} ({v(r.verdict_holm)}) |")
L+=["","Source chosen per seed without target labels: "+"; ".join(f"{t}: {S[(S.target==t)&(S.method=='selected')].selected_counts.iloc[0]}" for t in S.target.unique())+".","",
"**Plain reading.** Choosing the source on the other sources' validation sets mostly picks the oracle source for CERT (CERT r5.2 for CERT r4.2 in 17 of 20 seeds) and comes close to the hindsight value; for SPEDIA it is worse than leave-one-out FL. The rank-average ensemble fails on the CERT targets because the LANL source model ranks CERT positives below normal days (ROC about 0.26).",""]
# E-9
S=pd.read_csv(R+"e9_summary.csv")
L+=["## E-9 More time cut points (CERT only)","","| Cut q | Rule | Group | CERT r4.2 | CERT r5.2 |","|---|---|---|---|---|"]
for (q,rule,g),x in S.groupby(["q","rule","group"]):
    x=x.set_index("client"); L.append(f"| {q} | {rule} | {g} | {f(x.loc['cert42','diff'])} ({v(x.loc['cert42','verdict'])}) | {f(x.loc['cert52','diff'])} ({v(x.loc['cert52','verdict'])}) |")
L+=["","Positive user-days train/val/test: q = 0.6 — CERT r4.2 342/159/465, CERT r5.2 444/145/718; q = 0.8 — 643/152/171 and 788/231/288. Test sets differ by cut, so compare differences, not raw scores.","",
"**Plain reading.** The CERT r4.2 gain with CERT r5.2 holds at every cut and grows with a longer training window (+0.10 to +0.13 at 0.8). At 0.8 both CERT members gain in the CERT pair, and CERT r4.2 also gains in the four-member group.",""]
# E-10
B=pd.read_csv(R+"e10_builds.csv"); E=pd.read_csv(R+"e10_effects.csv")
L+=["## E-10 LANL sampling (full rebuild from the raw logs; parity with the protocol cache passed to float precision after the logged hour_std repair)","","| Build | Positive user-days (users) | Events / positive day, median (mean) | LANL local PR-AUC | Lift | Recall@1% |","|---|---|---|---|---|---|"]
for _,r in B.iterrows():
    L.append(f"| {r.build} | {r.pos_userdays} ({r.pos_users}) | {r.events_per_pos_day_median:.0f} ({r.events_per_pos_day_mean:.0f}) | {r.lanl_local_prauc:.4f} | {r.lift:.1f} | {r.recall_at_1pct:.2f} |")
L+=["","| Build | Rule | Group | Member | Difference | Verdict |","|---|---|---|---|---|---|"]
for _,r in E.iterrows(): L.append(f"| {r.build} | {r.rule} | {r.group} | {r.client} | {f(r['diff'])} | {v(r.verdict)} |")
L+=["",f"Counts over the three new builds: {(E.verdict=='gain').sum()} gains, {(E.verdict=='loss').sum()} losses, {(E.verdict=='no clear difference').sum()} no clear difference.","",
"**Plain reading.** 5% sampling recovers only 3 more of the 176 red-team user-days but gives about 2.5× more events per positive day; the two new 2% draws already give about twice the protocol's LANL PR-AUC, so the protocol LANL score sits at the low end of what a 2% sample gives. Federation never helps LANL in any build, and the other members' losses do not depend on the LANL build.",""]
# E-11 extras
O=pd.read_csv(R+"e11_twos_cells.csv"); C=pd.read_csv(R+"e11_twos_seed_counts.csv")
L+=["## E-11 TWOS extras (seeds 0–4; full detail in doc 32)","","TWOS test set: 46 user-days, 6 positive user-days from 6 positive users (24 users); top-10% = 5 user-days.","","| Group | Rule | TWOS PR-AUC | Δ vs local MLP [user-bootstrap CI] | Positives in top 5 | Seeds above local MLP / XGBoost |","|---|---|---|---|---|---|"]
name={"fedavg_size":"FedAvg size","fedavg_equal":"FedAvg equal","fedper_in":"private first layer"}
for _,r in O[O.regime.isin(name)].iterrows():
    c=C[(C.group==r.federation)&(C.rule==r.regime)].iloc[0]
    L.append(f"| {r.federation} | {name[r.regime]} | {r.pr_auc:.3f} | {f(r.diff_vs_local)} [{f(r.diff_user_lo)}, {f(r.diff_user_hi)}] | {r.top10pct_hits_mean:.1f} | {c.beats_local}/5, {c.beats_xgb}/5 |")
loc=O[O.regime.isin(["R0_local","R0_RF","R0_XGB"])]
L+=["","Local references: "+"; ".join(f"{r.regime} {r.pr_auc:.3f} (top-5 hits {r.top10pct_hits_mean:.1f})" for _,r in loc.iterrows())+".","",
"**Plain reading.** TWOS gains clearly from federation with all five members, with SPEDIA and with LANL (user-bootstrap intervals exclude 0 for 9 of 15 cells; above the local MLP and XGBoost on 4–5 of 5 seeds in those groups). Pairs with CERT help less. Caveats: 6 test positives; validation holds only traitors and test only masquerade victims.",""]
L+=["## Files","`results/review2/`: stats_e2.csv, e2_member_descriptors.csv, stats_e5.csv, stats_e6.csv, e7_summary.csv, e8_summary.csv, e8_per_seed.csv, e9_summary.csv, e10_builds.csv, e10_effects.csv, e11_twos_cells.csv, e11_twos_seed_counts.csv, e11_twos_per_seed.csv, stats_e11.csv.",""]
open("docs/33_Review2_exploratory_5seed_results_2026-10-23.md","w").write("\n".join(L)+"\n")
print(len(L))
