# Paper-to-code mapping

This file links the names used in the paper to the names used in the code and result files. Figure, table and section numbers are those of the submitted version of the paper. All paths are relative to the repository root.

Members: C1 = CERT r4.2, C2 = CERT r5.2, C3 = SPEDIA, C4 = LANL, C5 = TWOS.

## 1. Supplementary experiments S1 to S7 (paper Table 7)

| Paper | Change-log name | What it tests | Run script and jobs | Statistics script | Result files |
|---|---|---|---|---|---|
| S1 Same training method | E-1 | FedAvg and FedNova with plain gradient descent, 20 runs | `scripts/review2/run_r2.py`: `e1_sel`, `e1_selx`, `e1_local`, `e1_fedavg_size`, `e1_fedavg_equal`, `e1_fednova_size`, `e1_fednova_equal`, `e1_fixed`, `e1_ref` | `scripts/review2/r2_stats.py`; two-level intervals: `scripts/review2/f1_scopes.py` | `results/review2/stats_e1.csv`, `results/review2/stats_twolevel_e1.csv` |
| S2 Controlled split | E-2 | One property of member A changed within CERT r5.2, 5 runs | `scripts/review2/run_r2.py`: `e2_R`, `e2_S1`, `e2_S1b`, `e2_S2`, `e2_P`, `e2_C1`, `e2_C2`, `e2_S1C2`; data for `e2_C2` and `e2_S1C2`: `scripts/data/rebuild_c52.py`; positive rates: `scripts/review2/f5_e2_rates.py` (F-5) | `scripts/review2/r2_stats.py` | `results/review2/stats_e2.csv`, `results/review2/f5_e2_rates_summary.csv` |
| S3 Few labels | E-4 | CERT r4.2 with 1–25% of its labels, 15 runs (seeds 5–19) | `scripts/review2/run_r2.py`: `e4` | `scripts/review2/r2_stats.py`, `scripts/review2/summarise_e4.py`; two-level intervals: `scripts/review2/f1_scopes.py` | `results/review2/stats_e4.csv`, `results/review2/e4_per_seed_winner.csv`, `results/review2/stats_twolevel_e4.csv` |
| S4 Matched personalisation | E-3 | Private output head, last two layers private, Ditto; 20 runs | `scripts/review2/run_r2.py`: `e3_fedperh`, `e3_fedper2`, `e3_ditto_sel`, `e3_ditto` (sensitivity: `e3_dittofed_sel`, `e3_dittofed`) | `scripts/review2/r2_stats.py`, `scripts/review2/summarise_e3.py`, `scripts/review2/summarise_e3_ditto.py`; two-level intervals: `scripts/review2/f1_scopes.py` | `results/review2/stats_e3.csv`, `results/review2/e3_interim_vs_local.csv`, `results/review2/e3_interim_head_vs_input.csv`, `results/review2/e3_ditto_vs_input.csv`, `results/review2/stats_twolevel_e3.csv` |
| S5 SPEDIA's limitations | C-1, then E-12 | SPEDIA makes 100 steps per round (C-1); repeated with one round selected for all members (E-12); 15 runs (seeds 5–19) | C-1: `scripts/run_confirm.py` job `c1`; E-12: `scripts/review2/run_r2.py` job `e12` | C-1: `scripts/summarise_confirm.py`; E-12: `scripts/review2/r2_stats.py` | E-12: `results/review2/stats_e12.csv`. C-1 has no summary CSV in the repository: `summarise_confirm.py` prints its table from the per-seed run files |
| S6 Robustness | E-5, E-6, E-7, E-9, E-10 | Remaining three-member groups (E-5); two other SPEDIA label rules (E-6); trees tuned on training rows only (E-7); CERT cut points at 60% and 80% (E-9); LANL rebuilt three other ways (E-10); 5 runs each | `scripts/review2/run_r2.py`: `e5_<rule>`, `e6_ii_desc`, `e6_iii_sev7_or_desc`, `e7`, `e9_q0.6`, `e9_q0.8`, `e10_f05_s7`, `e10_f02_s8`, `e10_f02_s9`; data: `scripts/review2/build_spedia_variants.py`, `scripts/data/lanl_rebuild_E10_original.py` | `scripts/review2/r2_stats.py`, `scripts/review2/summarise_exploratory.py`; two-level intervals for E-5: `scripts/review2/f1_scopes.py` | `results/review2/stats_e5.csv`, `results/review2/stats_twolevel_e5.csv`, `results/review2/stats_e6.csv`, `results/review2/e7_summary.csv`, `results/review2/e9_summary.csv`, `results/review2/e10_builds.csv`, `results/review2/e10_effects.csv` |
| S7 Label meaning | F-4 | Only what counts as a positive day in member A changes (R, L1, L2, L3); 20 runs | Input: `scripts/data/build_f4_answer_days.py`; runs: `scripts/review2/f4_label.py` (`check`, then `R`, `L1`, `L2`, `L3` with seeds `0-19`, one thread) | `scripts/review2/f4_stats.py` (add `--skip` for the sensitivity check), using `scripts/review2/twolevel.py` | `results/review2/f4_feasibility.csv`, `results/review2/f4_twolevel.csv`, `results/review2/f4_twolevel_L2_sens.csv`, `results/review2/f4_contrasts_A.csv`, `results/review2/f4_local_A.csv`, `results/review2/f4_positives_summary.csv`, `results/review2/f4_repro_check.csv`; sensitivity: `results/review2/f4_twolevel_skipempty.csv`, `results/review2/f4_twolevel_L2_sens_skipempty.csv`, `results/review2/f4_contrasts_A_skipempty.csv` |

Other tests that have no S number in the paper:

| Paper | Change-log name | Run script and jobs | Statistics script | Result files |
|---|---|---|---|---|
| TWOS as a fifth member (Section 4.10) | E-11 | `scripts/data/build_twos_userday.py`; `scripts/review2/run_r2.py`: `e11_main`, `e11_user`, `e11_trees`; `scripts/review2/e11_extra.py` | `scripts/review2/r2_stats.py`, `scripts/review2/summarise_e11.py`; two-level intervals: `scripts/review2/f1_scopes.py` | `results/review2/stats_e11.csv`, `results/review2/stats_twolevel_e11.csv` (group level only) |
| RQ4 checks: source chosen without hindsight, and an ensemble of the source models | E-8 | No training: `scripts/review2/e8_r5_no_hindsight.py` | the same script | `results/review2/e8_summary.csv` |
| Earlier low-label test (Table A.18) | C-2 | `scripts/run_confirm.py` job `c2` | `scripts/summarise_confirm.py` | No summary CSV in the repository (printed from the per-seed run files) |
| Intervals over runs and test users (Section 3.11 and every "clear" result) | F-1 | No training: reads the saved per-row predictions | `scripts/review2/twolevel.py`, `scripts/review2/f1_scope1.py`, `scripts/review2/f1_scopes.py`, `scripts/review2/f1_summary.py` | `results/review2/stats_twolevel_s0-19.csv`, `results/review2/stats_twolevel_e1.csv`, `results/review2/stats_twolevel_e3.csv`, `results/review2/stats_twolevel_e4.csv`, `results/review2/stats_twolevel_e5.csv`, `results/review2/stats_twolevel_e11.csv`, `results/review2/f1_summary_counts.csv` |
| Single-feature check (Section 4.1) | F-2 | No training | `scripts/review2/f2_single_feature.py` | `results/review2/f2_single_feature.csv`, `results/review2/f2_single_feature_summary.csv` |
| Cross-member transfer (Table A.15) and its use in RQ3 | F-3 | `scripts/review2/f3_twos_out.py` (the TWOS → C1–C4 cells) | `scripts/review2/f3_transfer.py` | `results/review2/f3_transfer_matrix.csv`, `results/review2/f3_rq3_spearman.csv` |

C-1 and C-2 have no summary file here, but both can be repeated: `scripts/run_confirm.py` (jobs `c1` and `c2`) runs them and `scripts/summarise_confirm.py` prints their tables.

## 2. Set-up and rule codes R0 to R7 (paper Fig. 4 and Table 5)

| Code | Paper name | Job name in the code | Name in the statistics files (`regime` or `rule` column) | Run script |
|---|---|---|---|---|
| R0 | Local training (MLP) | `mlp_local` | the baseline of every row (column `local_pr_auc`) | `scripts/run_review.py` |
| R1 | Pooled data, no privacy (MLP) | `mlp_pooled` | `Pooled MLP (no privacy)` | `scripts/run_review.py` |
| R2 | FedAvg, size weights | `fedavg_size` | `FedAvg (size)` | `scripts/run_review.py` |
| R2 | FedAvg, equal weights | `fedavg_equal` | `FedAvg (equal)` | `scripts/run_review.py` |
| R3 | FedProx, μ = 0.001 | `fedprox` (μ chosen by `fedprox_tune` in `scripts/run_fl.py`) | `FedProx (μ=0.001)` | `scripts/run_review.py` |
| R3b | FedNova, size or equal weights (with Adam) | `fednova_size`, `fednova_equal` | `FedNova (size)`, `FedNova (equal)` | `scripts/run_review.py` |
| R4a | FedPer (own output layer) | `fedper` | `FedPer (local output layer)` | `scripts/run_review.py` |
| R4b | FedAvg, then fine-tuning | `finetune` | `FedAvg + fine-tune` | `scripts/run_review.py` |
| R5 | Model from one other member, no labels from the target | `zeroshot_mlp` | columns `best_single_source`, `best_single_pr_auc` of the leave-one-out summary | `scripts/run_review.py` |
| R6 | Federated on the other members, target left out | `loo` | column `loo_pr_auc` of the leave-one-out summary | `scripts/run_review.py`, `scripts/loo_summary.py` |
| R7 | Private first layer | `fedper_in` | `Private first layer` | `scripts/run_review.py` |

The federated rules themselves are in `src/fl.py`. The sixth pair (C2+C4) uses the same jobs with the prefix `c2c4_`. In S1 the jobs are `e1_fedavg_size`, `e1_fedavg_equal`, `e1_fednova_size` and `e1_fednova_equal`; in S4 they are `e3_fedperh` (private output head), `e3_fedper2` (last two layers private) and `e3_ditto`.

## 3. Figures and tables of the paper

"Plot script: not in the repository" (Figs. 7, 8, 13 and 15) means the chart was drawn from the values in the result file named, with a short plotting script that is not part of this repository. In `scripts/figures/`, `verdicts_bar.py` and `controlled_bar.py` write their PDF to the folder given by `FITD_FIGURES` (default `figures/`); `twos_bar.py` and `sgd_bar.py` write to the current folder.

| Paper item | What it shows | Result file(s) | Script that makes the numbers | Plot script |
|---|---|---|---|---|
| Fig. 6 | Gain, no clear change and loss for each rule over the 17 planned cases | `results/review2/f1_summary_counts.csv` (from `results/review2/stats_twolevel_s0-19.csv`) | `scripts/review2/f1_scope1.py`, `scripts/review2/f1_summary.py` | `scripts/figures/verdicts_bar.py` (reads `results/review2/stats_twolevel_s0-19.csv`) |
| Fig. 7 | Change in SPEDIA's PR-AUC by rule | `results/review/stats_review_s0-19.csv` (SPEDIA rows) | `scripts/review_stats.py` | not in the repository |
| Fig. 8 | Position of the private layer (S4) | `results/review2/stats_e3.csv` (SPEDIA rows); FedPer and the private first layer from `results/review/stats_review_s0-19.csv` | `scripts/review2/r2_stats.py`, `scripts/review_stats.py` | not in the repository |
| Fig. 9 | Controlled test of member differences (S2), FedAvg with equal weights | `results/review2/stats_e2.csv` (member `m1` = A, comparison "fedavg_equal − local") | `scripts/review2/r2_stats.py` | `scripts/figures/controlled_bar.py` (values typed from Table A.16) |
| Fig. 10 | SPEDIA by aggregation weight and local steps (S5) | One step per round: `results/review/stats_review_s0-19.csv`. 100 steps per round: C-1 (no summary CSV in the repository); its repeat with the main round rule is in `results/review2/stats_e12.csv` | `scripts/review_stats.py`, `scripts/summarise_confirm.py`, `scripts/review2/r2_stats.py` | drawn in LaTeX |
| Fig. 11 | How FedNova scales member updates | A diagram; no result file | — | drawn in LaTeX |
| Fig. 12 | SPEDIA under plain gradient descent (S1) | `results/review2/stats_e1.csv` (SPEDIA rows, family "E-1 vs SGD local") | `scripts/review2/r2_stats.py` | `scripts/figures/sgd_bar.py` |
| Fig. 13 | CERT r4.2 with a reduced share of labels (S3) | `results/review2/stats_e4.csv` (families "E-4 budget" and "E-4 budget baselines vs local") | `scripts/review2/r2_stats.py` | not in the repository |
| Fig. 14 | Change in TWOS's PR-AUC, with two-level intervals | `results/review2/stats_twolevel_e11.csv` (TWOS rows, family "E-11 federated") | `scripts/review2/f1_scopes.py` | `scripts/figures/twos_bar.py` |
| Fig. 15 | Trade-off between members in the four-member federation | `results/review/stats_review_s0-19.csv` (group C1+C2+C3+C4) | `scripts/review_stats.py` | not in the repository |
| Table 8 | Local (R0) and pooled (R1) training | MLP, 20 runs: `results/review/stats_review_s0-19.csv` (rule "Pooled MLP (no privacy)": local and pooled means). Trees, 5 runs: `results/e1e2_rf_local.csv`, `results/e1e2_rf_pooled.csv`, `results/e1e2_xgb_local.csv`, `results/e1e2_xgb_pooled.csv`. Test rates: `results/splits_record.csv` | `scripts/run_review.py`, `scripts/run_e1_e2.py`, `scripts/review_stats.py` | — |
| Table 9 | PR-AUC in the four-member federation | Means: `results/review/stats_review_s0-19.csv` (group C1+C2+C3+C4). Asterisks: `results/review2/stats_twolevel_s0-19.csv` (column `tl_verdict`) | `scripts/review_stats.py`, `scripts/review2/f1_scope1.py` | — |
| Table 10 | Effect of label meaning on member A (S7) | `results/review2/f4_twolevel.csv`, `results/review2/f4_twolevel_L2_sens.csv` (the "L2, 15 runs" row), `results/review2/f4_local_A.csv`, `results/review2/f4_positives_summary.csv`; the L1 − L3 comparison in the text: `results/review2/f4_contrasts_A.csv` | `scripts/review2/f4_label.py`, `scripts/review2/f4_stats.py` | — |
| Table 11 | A member left out of federated training (RQ4) | `results/review/loo_summary_s0-19_regenerated.csv` | `scripts/loo_summary.py` | — |
| Table 12 | PR-AUC of TWOS under federated training | `results/review2/stats_twolevel_e11.csv` (TWOS rows, family "E-11 federated": `new`, `mean_diff`, `tl_lo`, `tl_hi`) | `scripts/review2/f1_scopes.py` | — |
| Table 13 | Comparison with earlier studies | From the literature; no result file | — | — |
| Table A.14 | Change against local training for every scored case | Means: `results/review/stats_review_s0-19.csv`. Asterisks and daggers: `results/review2/stats_twolevel_s0-19.csv` (asterisk: `tl_verdict`; dagger: `verdict_holm_b`, Holm over all 156 cases). This file also holds the interval, p-value and Holm-adjusted p-values for each cell | `scripts/review_stats.py`, `scripts/review2/f1_scope1.py` | — |
| Table A.15 | Cross-member transfer of locally trained models | `results/review2/f3_transfer_matrix.csv` | `scripts/review2/f3_twos_out.py`, `scripts/review2/f3_transfer.py` | — |
| Table A.16 | Controlled test of member differences (S2), all three rules | Changes: `results/review2/stats_e2.csv`. Member A's positive rates: `results/review2/f5_e2_rates_summary.csv` | `scripts/review2/r2_stats.py`, `scripts/review2/f5_e2_rates.py` | — |
| Table A.17 | Outcomes under plain gradient descent (S1) | Counts: `results/review2/stats_twolevel_e1.csv` (family "E-1 vs SGD local", column `tl_verdict_holm`). SPEDIA's PR-AUC: `results/review2/stats_e1.csv` (column `new`) | `scripts/review2/f1_scopes.py`, `scripts/review2/r2_stats.py` | — |
| Table A.18 | Earlier low-label test for CERT r4.2 (C-2) | No summary CSV in the repository | `scripts/run_confirm.py` (job `c2`), `scripts/summarise_confirm.py` | — |

Other numbers in the text:

| Where in the paper | Result file |
|---|---|
| Section 3: sizes of the splits (Table 4) | `results/splits_record.csv` |
| Section 3: similarity between members (Table 6) | `results/c3_similarity.csv` |
| Section 4.1: single-feature check | `results/review2/f2_single_feature_summary.csv` |
| Section 4.2: counts of clear gains and losses, before and after Holm correction | `results/review2/f1_summary_counts.csv` |
| Section 4.2: secondary measures (ROC-AUC, recall in the top 1%) | `results/review/metrics_by_cell.csv` |
| Section 4.4 (RQ3): rank correlations | `results/review2/f3_rq3_spearman.csv`, `results/review/c3_spearman_review_s0-19.csv` |
| Section 4.7 (RQ4): source chosen without hindsight, ensemble | `results/review2/e8_summary.csv` |

## 4. Test plan and change log

The frozen test plan and its dated change log are in `docs/protocol.md` (Section 10). The rows dated 11 November 2026 cover F-1 to F-5.
