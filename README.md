# PathSnap — Map-Matching & Route Reconstruction

> **Phase 1 (Review 1):** data processing, EDA, candidate generation and features, the 7-model comparison and Bayesian optimisation. Phase 2 (Review 2) adds HMM/Viterbi decoding, the FastAPI backend, the evaluator UI and the final report.

UE24CS352A Machine Learning Hackathon, PES University (RR Campus).

PathSnap matches noisy GPS trajectories to a road network and rebuilds the route the vehicle drove:

1. **Candidate generation**: for every GPS fix, collect the road arcs within 50 m (up to 8).
2. **Candidate classification**: a supervised model scores each (GPS fix, arc) pair using 17 engineered geometric, heading, context and topology features.
3. **HMM + Viterbi decoding**: the hidden states are the candidate arcs. Emissions come from the classifier probabilities. Transitions compare network distance with GPS distance (Newson–Krumm style). Shortest-path gap filling then produces a connected route.
4. **Deployment**: a FastAPI backend (Laptop 1) runs the full pipeline and scores it server-side. A browser UI (Laptop 2) lists the 10 evaluation trips, calls the API and shows the predicted route on a map.

## Repository layout

```
data/
  train/            80 training trips (.track .nodes .arcs .route)
  student_test/     10 development-test trips
  eval_test/        put the 10 evaluator trips here on Laptop 1 (never used in development)
src/pathsnap/
  core.py           loading, local AEQD projection, reference-label alignment, candidates, features
  metrics.py        Top-1, macro F1, log loss, route continuity, route P/R/F1
scripts/
  01_eda.py             dataset inventory, data-quality checks, EDA plots
  02_build_dataset.py   reference labels + candidate/feature tables (--shard k --nshards 4 for parallel runs)
  02b_merge_shards.py   merges the 4 dataset shards and validates them
  03_train_compare.py   7 models, 5-fold GroupKFold CV + student-test evaluation (--models subset)
  03b_merge_results.py  merges per-member model results
  04_bayes_opt.py       Optuna TPE Bayesian optimisation of the selected model
  06_maps.py            Folium maps (GPS, network, reference, predicted route)
  07_plots.py           comparison plots
  08_screenshot.py      headless rendering of HTML maps to PNG
artifacts/          final tuned model + all compared models
results/            CSV/JSON experiment logs, figures/, maps/
report/             technical report (PDF)
```

## Installation

Python 3.10+ is required.

```bash
git clone <this-repo-url> && cd PathSnap
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Parallel processing across the team

The dataset build and model training can be split across 4 laptops (`02_build_dataset.py --shard k`, `02b_merge_shards.py`, `03_train_compare.py --models ...`, `03b_merge_results.py`). See **TEAM_WORKFLOW.md**.

## Reproduce the experiments

Run the scripts in order from the repo root:

```bash
cd scripts
python 02_build_dataset.py      # labels + candidate features  (results/data/)
python 01_eda.py                # EDA tables and figures
python 03_train_compare.py      # model comparison             (results/model_comparison.csv)
python 04_bayes_opt.py          # Bayesian optimisation        (results/bayes_opt_*.csv/json, artifacts/final_model.*)
python 06_maps.py --eda-only    # Folium EDA maps              (results/maps/eda_*.html)
python 07_plots.py              # figures                      (results/figures/*.png)
```

All random seeds are fixed at 42. The candidate tables in `results/data/cand_*.parquet` are git-ignored because of their size; running `02_build_dataset.py` regenerates them. Run `02` before `01`.

`08_screenshot.py` is only needed to render PNGs of the maps without internet access.

## Results so far (Phase 1)

| Model | CV Macro F1 | CV Top-1 | Test Top-1 | Test Macro F1 | Test LogLoss |
|---|---|---|---|---|---|
| LogisticRegression | 0.9107 | 0.8906 | 0.8703 | 0.9071 | 0.1399 |
| GaussianNB | 0.8571 | 0.8747 | 0.8636 | 0.8438 | 0.5902 |
| KNN | 0.9233 | 0.8878 | 0.8730 | 0.9196 | 0.1588 |
| RandomForest | 0.9360 | 0.9024 | 0.8948 | 0.9340 | 0.0920 |
| LightGBM | 0.9380 | 0.9051 | 0.9049 | 0.9388 | 0.0874 |
| XGBoost | 0.9368 | 0.9049 | 0.9041 | 0.9383 | 0.0876 |
| MLP | 0.9305 | 0.8995 | 0.8880 | 0.9285 | 0.1019 |

The selected model is LightGBM. After Optuna TPE tuning (30 trials) it reaches test Top-1 0.9065 (untuned: 0.9049).

Report: `report/PathSnap_Review1_Report.pdf`. Rebuild it with `python report/build_report.py --phase1`.
