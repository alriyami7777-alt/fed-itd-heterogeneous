# Who Gains from Federated Insider-Threat Detection? Collaboration across Synthetic and Real Security Logs

**Status:** under review at *Applied Soft Computing* (Elsevier).

Code, test plan and summary results for the paper. No data is included.

## What it does

The study treats public insider-threat datasets as separate organisations: CERT r4.2 and CERT r5.2 (synthetic), SPEDIA v1.0 (attacks by people, simulated normal activity) and LANL (real enterprise authentication logs with red-team activity). Each dataset is turned into the same user-day table with 42 features.

One small neural network (an MLP) is trained in three ways:

- **alone**, on each organisation's own data;
- **pooled**, on all organisations' data together (no privacy, as a reference);
- **in federations**, with FedAvg (size or equal weights), FedProx, FedPer, a private first layer, FedAvg with local fine-tuning, and FedNova.

For each member of each federation, the paper reports the gain or loss in PR-AUC against training alone, over 20 runs (seeds 0–19). Leave-one-out and single-source transfer to an unseen organisation are also run.

## Data (not included)

Get each dataset from its provider and accept its terms.

| Dataset | Source | Licence |
|---|---|---|
| CERT r4.2 / r5.2 | https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247 | See provider |
| SPEDIA v1.0 | https://doi.org/10.5281/zenodo.15495572 (collection tools: https://github.com/UPM-RSTI/SPEDIA-Dataset) | CC BY 4.0 |
| LANL | https://doi.org/10.17021/1179829 (if the DOI page does not load: https://csr.lanl.gov/data/cyber1/) | CC0 |
| TWOS (post-review test E-11 only) | From its authors, under a release agreement (Harilal et al., 2017) | Not redistributable |

TWOS may not be redistributed, so request it from its authors yourself. This repository holds no TWOS file of any kind: only the code that builds the table and group-level results.

## Building the user-day tables

The four user-day tables come from the pipeline of our earlier cross-dataset study: [crossdataset-insider-threat](https://github.com/alriyami7777-alt/crossdataset-insider-threat). They are built by `scripts/strict_camera_ready_2026_09_15.py` in that repository (stages `build-c2` and `build-rest`) and copied unchanged to `$FITD_ROOT/data/` as `{cert42,cert52,spedia,lanl}_user_day_dev.parquet`.

Check your copies against these SHA-256 hashes:

| Table | SHA-256 |
|---|---|
| cert42 | `99bbcd739f6f3fd718b37406415cf9d2b8dc4d1b772b62950b0ee7f39582c011` |
| cert52 | `8850a1eef4fc4b64b6509ddd00182fce4f88695593f0123d6cd6fddb1b151156` |
| spedia | `5d1727eb7461e1c5de117bee613d315d1999711d61a9402111161334c6e523d3` |
| lanl | `e78d635a029aa8b7380edbb1e65b56ca9249ba7d7148f29ac16065b5d5c93d9b` |

Each table holds 42 features over 16 actions, plus `user`, `day` and `label`.

**LANL rebuild (E-10).** `scripts/data/` rebuilds LANL from the raw logs with four sampling streams. This tests whether the results depend on the 2% sample.

```bash
python scripts/data/lanl_rebuild_E10_original.py --ijacsa-repo <clone of crossdataset-insider-threat> --lanl-dir <folder with auth.txt.gz and redteam.txt.gz>
python scripts/data/repair_lanl_hourstd.py --ijacsa-repo <clone>
```

- `lanl_rebuild_E10_original.py` is the rebuild script.
- `lanl_rebuild_E10.py` does the same work but can fall back to pickle files. It was needed only on a machine where a security policy blocked `pyarrow`. Use the original wherever `pyarrow` works.
- `repair_lanl_hourstd.py` fixes an integer overflow in `hour_std` that appears only on Windows.
- The rebuild reads about one billion raw rows and takes 1–2 hours.

**Other tables for the post-review tests.** Put the outputs in `$FITD_ROOT/data/`:

```bash
# E-11: TWOS user-day table (logon events only) -> data/twos/twos_user_day_dev_v1_logon.parquet
python scripts/data/build_twos_userday.py --ijacsa-repo <clone> --twos-dir <TWOS release folder> --traitors <five traitor users>
# E-6: SPEDIA label variants -> data/derived/spedia_user_day_<variant>.parquet
IJACSA_REPO=<clone> python scripts/review2/build_spedia_variants.py <path to logs_SPEDIA.csv>
# E-2 (C2, S1C2): CERT r5.2 table from logon/logoff events only -> data/derived/cert52_user_day_cov_logon_rebuild.parquet
IJACSA_REPO=<clone> python scripts/data/rebuild_c52.py <path to CERT r5.2 logon.csv>
```

For TWOS, the builder takes the five traitor user names from the TWOS release documentation, so they are not repeated here. `run_r2.py` checks the TWOS table against the SHA-256 `fedadd4a707a92d0b8977bc134115121a4e2ddd03a727c067c55d1bd31c5bd31`. A different `pyarrow` version can change this hash without changing the table.

## Setup

Python 3.13.16 on Linux x86_64. The exact package versions are in `requirements.txt`: numpy 2.5.3, pandas 3.0.5, pyarrow 26.0.0, scikit-learn 1.9.1, scipy 1.18.1, torch 2.14.1, xgboost 3.4.1, matplotlib 3.11.2, openpyxl 3.1.5.

```bash
git clone https://github.com/alriyami7777-alt/fed-itd-heterogeneous.git
cd fed-itd-heterogeneous
pip install -r requirements.txt
export FITD_ROOT=/path/to/workdir   # must contain data/; results/ is written there
```

If `FITD_ROOT` is not set, the repository folder is used. No other path is hard-coded. `FITD_THREADS` sets the number of torch threads (default 2).

## Reproduce

Run every command from the repository root.

**Main study, 5 seeds (as first run):**

```bash
python scripts/run_e1_e2.py mlp_local; python scripts/run_e1_e2.py mlp_pooled
python scripts/run_e1_e2.py rf_local;  python scripts/run_e1_e2.py rf_pooled
python scripts/run_e1_e2.py xgb_local; python scripts/run_e1_e2.py xgb_pooled
python src/similarity.py
for j in fedavg_size loo zeroshot_mlp fedavg_equal fedper fedper_in fedprox_tune fedprox finetune; do python scripts/run_fl.py $j; done
python scripts/summarise.py; python scripts/summarise_fl.py; python scripts/make_figures.py
```

**Main study, 20 seeds (the results in the paper):**

```bash
python scripts/run_review.py mlp_local 0-19
python scripts/run_review.py sanity 0-4
for j in fedavg_size fedavg_equal fedper_in finetune; do python scripts/run_review.py $j 0-19; done
python scripts/run_confirm.py c1 5-19
python scripts/run_confirm.py c2 5-19
python scripts/run_review.py fednova_sanity 0-0
for j in fednova_size fednova_equal; do python scripts/run_review.py $j 0-19; done
for j in fedavg_size fedavg_equal fedper_in finetune fedprox fedper; do python scripts/run_review.py c2c4_$j 0-19; done
for j in fednova_sgd_size fednova_sgd_equal; do python scripts/run_review.py $j 0-4; done
for j in fedprox fedper mlp_pooled loo zeroshot_mlp; do python scripts/run_review.py $j 0-19; done
```

`run_review.py` saves per-row predictions under `results/preds/`. These are needed by `review_metrics.py` and `spedia_cluster_boot.py`, and are not committed. Each job skips seeds that are already done, so you can restart a job after an interruption.

**Statistics:**

```bash
python scripts/check_reproduction.py          # seeds 0-4 reproduce the 5-seed scores; sanity runs equal training alone
python scripts/review_stats.py --seeds 0-19   # paired bootstrap, paired t-test, Holm correction
python scripts/review_metrics.py              # ROC-AUC, recall at 1%, thresholded P/R/F1, worst client
python scripts/spedia_cluster_boot.py         # user-level bootstrap for SPEDIA
python scripts/c3_analysis.py --source review --seeds 0-19
python scripts/summarise_confirm.py
python scripts/loo_summary.py                 # leave-one-out summary -> results/review/*_regenerated.csv
python scripts/fednova_sgd_summary.py         # FedNova with local SGD, seeds 0-4 -> results/review/fednova_sgd_s0-4.csv
```

**Two rewritten scripts.** The original scripts behind two inputs were not kept, so they were rewritten on 24 October 2026 and checked against the files they replace. The checks are stated in each script's docstring.

- `scripts/data/rebuild_c52.py` rebuilds the CERT r5.2 logon/logoff-only table for E-2. Its output equals the original table exactly: the same 693,649 user-days, labels, all 42 feature values and data types. Only the parquet file bytes differ, so the SHA-256 differs too. Compare contents, not hashes.
- `scripts/loo_summary.py` rebuilds the leave-one-out summary. It reproduces every mean and label of `results/review/loo_summary_s0-19.csv` exactly. Its bootstrap intervals differ by at most 0.002, because the original helper drew its resamples differently. No interval changes sign. Its output is published as `results/review/loo_summary_s0-19_regenerated.csv` and `loo_all_seeds_s0-19_regenerated.csv`, so the published numbers come from published code. The original summary file is kept unchanged. Run it with `--check` to compare the two.

**Post-review tests (E-1 to E-12).** Each test is logged in the change log of `docs/protocol.md` before it was run. Most are exploratory with 5 seeds (0–4). E-1, E-3 and E-11 use 20 seeds. E-12 and E-4 use seeds 5–19.

```bash
# E-1: plain-SGD FedAvg vs FedNova (learning rate chosen on seed 0, validation only)
python scripts/review2/run_r2.py e1_sel 0
for j in e1_local e1_fedavg_size e1_fedavg_equal e1_fednova_size e1_fednova_equal e1_fixed; do python scripts/review2/run_r2.py $j 0-19; done
python scripts/review2/run_r2.py e1_ref 0            # FedNova reference check on synthetic data
# the learning-rate grid was later extended at its edges (change log, 20 Oct): run e1_selx 0, then
# scripts/review2/e1_apply_ext.py, then the E-1 jobs again (finished seeds are skipped)
# E-11: TWOS as a fifth member
for j in e11_main e11_user e11_trees; do python scripts/review2/run_r2.py $j 0-4; done
python scripts/review2/run_r2.py e11_main 0-19
python scripts/review2/e11_extra.py 0-4
# E-3: matched personalisation (FedPer-H, FedPer-2, Ditto)
python scripts/review2/run_r2.py e3_ditto_sel 0
for j in e3_fedperh e3_fedper2 e3_ditto; do python scripts/review2/run_r2.py $j 0-19; done
# E-6 label variants, E-5 controlled triples, E-7 trees on training rows, E-9 time cuts, E-10 LANL rebuilds
for j in e6_ii_desc e6_iii_sev7_or_desc; do python scripts/review2/run_r2.py $j 0-4; done
for j in e5_fedavg_size e5_fedavg_equal e5_fedprox e5_fedper e5_fedper_in e5_finetune e5_fednova_size e5_fednova_equal; do python scripts/review2/run_r2.py $j 0-4; done
python scripts/review2/run_r2.py e7 0-4
for j in e9_q0.6 e9_q0.8 e10_f05_s7 e10_f02_s8 e10_f02_s9; do python scripts/review2/run_r2.py $j 0-4; done
# E-2: controlled federation within CERT r5.2
for j in e2_R e2_S1 e2_S1b e2_S2 e2_P e2_C1 e2_C2 e2_S1C2; do python scripts/review2/run_r2.py $j 0-4; done
# E-12 and E-4
python scripts/review2/run_r2.py e12 5-19
python scripts/review2/run_r2.py e4 5-19
# statistics, E-8 (no training) and plots
python scripts/review2/r2_stats.py
python scripts/review2/e8_r5_no_hindsight.py
python scripts/review2/e1_convergence_plots.py
python scripts/review2/make_doc33.py
python scripts/review2/make_doc37.py             # E-4 summary tables
```

`e2_C2` and `e2_S1C2` also need `data/derived/cert52_user_day_cov_logon_rebuild.parquet`, built by `scripts/data/rebuild_c52.py` (see above).

**Exploratory, outside the protocol:** `scripts/run_extra.py`, `src/gating.py` and `scripts/summarise_extra.py`.

**Run times.** These are measured on 2 CPU cores with 2 torch threads, per seed:

| Job | Time per seed |
|---|---|
| Local MLP (`mlp_local`) | about 4 min |
| Zero-shot transfer (`zeroshot_mlp`) | about 13 min |
| Leave-one-out (`loo`) | about 15 min |
| Confirmatory C-1 (`run_confirm.py c1`) | about 40 min |

Every output CSV records its own wall-clock time in a `wall_s` column. **Run one job at a time.** Two training jobs sharing 2 cores slowed each other badly.

## Results

Summary tables only. Per-row predictions and per-seed raw files are not included.

| File | What it holds |
|---|---|
| `results/splits_record.csv` | Split cut-off dates, row counts and positives per organisation (per seed for LANL) |
| `results/e1e2_{mlp,rf,xgb}_{local,pooled}.csv` | Per-seed PR-AUC and lift for local and pooled training: MLP, random forest and XGBoost, 5 seeds |
| `results/c3_similarity.csv` | Similarity between organisations before training: prevalence ratio, Jensen–Shannon distance, Jaccard overlap |
| `results/c3_points.csv` | One row per scored member of each two-member federation: similarity and gain |
| `results/c3_spearman.csv` | Spearman correlation between similarity and gain, with exact permutation p-values (5 seeds) |
| `results/fedprox_mu.json` | FedProx μ chosen on validation |
| `results/review/stats_review_s0-19.csv` | Each federated method vs training alone, per member: mean difference, bootstrap and t intervals, Holm-adjusted p-values (20 seeds) |
| `results/review/stats_counts_review_s0-19.csv` | Number of significant gains and losses per method under each test |
| `results/review/loo_summary_s0-19.csv` | Leave-one-out: PR-AUC on the unseen organisation vs the best single source (original file) |
| `results/review/loo_summary_s0-19_regenerated.csv` | The same summary, regenerated by `scripts/loo_summary.py`: identical means and labels; intervals differ by at most 0.002 |
| `results/review/loo_all_seeds_s0-19_regenerated.csv` | Leave-one-out PR-AUC per seed and unseen organisation (input to the summary above) |
| `results/review/fednova_sgd_s0-4.csv` | FedNova with plain local SGD vs training alone and vs FedNova with Adam (5 seeds) |
| `results/review/metrics_by_cell.csv` | Secondary metrics per method, federation and member: ROC-AUC, recall at 1%, precision, recall, F1 |
| `results/review/spedia_cluster_boot.csv` | SPEDIA user-level bootstrap intervals for PR-AUC and for the difference from training alone |
| `results/review/c3_spearman_review_s0-19.csv` | Spearman correlation between similarity and gain (20 seeds) |
| `results/review2/stats_e1.csv` | E-1: plain-SGD FedAvg and FedNova vs each member's SGD local baseline; FedNova vs FedAvg with the same solver; the fixed-step variant (20 seeds) |
| `results/review2/stats_e2.csv` | E-2: controlled federation within CERT r5.2, per variant and member |
| `results/review2/stats_e5.csv` | E-5: controlled three-member groups under the six planned methods, plus FedNova (exploratory) |
| `results/review2/stats_e6.csv` | E-6: SPEDIA label variants, federated vs local |
| `results/review2/stats_e11.csv` | E-11: TWOS as a fifth member, at group level only (PR-AUC differences and intervals per method and group) |
| `results/review2/stats_e12.csv` | E-12: extra SPEDIA steps per round with the main checkpoint rule, vs usual local training, vs the C-1 checkpoint rule, and SPEDIA vs the same steps alone (seeds 5–19) |
| `results/review2/e7_summary.csv` | E-7: random forest and XGBoost on training rows only vs training plus validation, and vs the MLP |
| `results/review2/e8_summary.csv` | E-8: transfer to an unseen organisation without hindsight (selected source, ensemble) vs leave-one-out, the best single source in hindsight and the average source |
| `results/review2/e9_summary.csv` | E-9: federated vs local at other time cuts (0.6 and 0.8) |
| `results/review2/e10_builds.csv` | E-10: each LANL rebuild: positive user-days and users, events per positive day, local PR-AUC |
| `results/review2/e10_effects.csv` | E-10: federated vs local effects on each LANL rebuild |
| `results/review2/stats_e4.csv` | E-4: CERT r4.2 with 1–25% of its labels: FedAvg with the full CERT r5.2 (size, equal) vs the best of local training, the CERT r5.2 model as is, and that model fine-tuned; with the same label budget for validation and with the full validation set (seeds 5–19) |
| `results/review2/e4_per_seed_winner.csv` | E-4: test PR-AUC of every option on each seed, and which option won |

The E-1 convergence plots (validation PR-AUC per round, seeds 0–4) are made by `scripts/review2/e1_convergence_plots.py`. E-3 results will be added when that run finishes.

## Test plan

The test plan was frozen on 13 October 2026, before any data was processed. It is in [`docs/protocol.md`](docs/protocol.md). Every later change is dated in its change log (Section 10), with the reason for it.

## Cite

Use **"Cite this repository"** on the GitHub page (from `CITATION.cff`). Authors: Qasim Alriyami, Mohd Murtadha Mohamad and Yusuf Idris Muhammad.

## Related work

- Cross-dataset study (IJACSA 2026): [crossdataset-insider-threat](https://github.com/alriyami7777-alt/crossdataset-insider-threat). Its pipeline builds the user-day tables used here.

## Licence and contact

The code is released under the MIT Licence (see [`LICENSE`](LICENSE)). The datasets keep their own licences.

Corresponding author: Qasim Alriyami, Universiti Teknologi Malaysia.
