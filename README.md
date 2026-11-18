# PathSnap — Map-Matching & Route Reconstruction

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
notebooks/          walkthrough notebooks (saved with outputs) that call the code in src/ and scripts/
  01_EDA.ipynb                      inventory, data quality, plots, Folium map
  02_Preprocessing_Features.ipynb   projection, label alignment, candidates, 17 features
  03_Model_Training.ipynb           7 models trained live + CV/test comparison
  04_Bayesian_Optimisation.ipynb    live Optuna TPE study + full 30-trial record
src/pathsnap/
  core.py           loading, local AEQD projection, reference-label alignment, candidates, features
  hmm.py            road graph, Viterbi decoder, route reconstruction (gap filling)
  metrics.py        Top-1, macro F1, log loss, route continuity, route P/R/F1
scripts/
  01_eda.py             dataset inventory, data-quality checks, EDA plots
  02_build_dataset.py   reference labels + candidate/feature tables (--shard k --nshards 4 for parallel runs)
  02b_merge_shards.py   merges the 4 dataset shards and validates them
  03_train_compare.py   7 models, 5-fold GroupKFold CV + student-test evaluation (--models subset)
  03b_merge_results.py  merges per-member model results
  04_bayes_opt.py       Optuna TPE Bayesian optimisation of the selected model
  05_hmm_eval.py        decoder tuning + classifier-only vs classifier+HMM comparison
  06_maps.py            Folium maps (GPS, network, reference, predicted route)
  07_plots.py           comparison plots
  08_screenshot.py      headless rendering of HTML maps to PNG
api/app.py          FastAPI backend (Laptop 1)
ui/index.html       evaluator UI (Laptop 2)
artifacts/          final model, HMM config, all compared models
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

## Notebooks (start here)

The notebooks in `notebooks/` are the main walkthrough. They are saved with outputs, so they open ready to read (on GitHub, or in Jupyter / VS Code). They call the same code as the scripts, so the numbers match exactly: the live training in `03_Model_Training` reproduces the logged Top-1 scores with a difference of 0.0.

| Notebook | Covers |
|---|---|
| `01_EDA.ipynb` | Raw file structure, inventory, missing values, sampling gaps, anomalies, connectivity, trajectory plots, Folium map |
| `02_Preprocessing_Features.ipynb` | Local metric projection (checked against haversine), monotone label alignment, candidate generation, the 17 features, nearest-arc baseline |
| `03_Model_Training.ipynb` | LogisticRegression, GaussianNB, KNN, RandomForest, LightGBM, XGBoost and MLP, each trained live, with the comparison table and plots |
| `04_Bayesian_Optimisation.ipynb` | Short live Optuna TPE study, the full 30-trial record, tuned vs untuned |

To re-run them, build the candidate tables first (they are git-ignored because of their size):

```bash
pip install -r requirements.txt jupyter
cd scripts && python 02_build_dataset.py && cd ../notebooks
jupyter notebook
```

`03_Model_Training` loads the cross-validation numbers from the script log by default. Set `RUN_CV = True` in it to re-run the 5-fold CV live (slow for KNN and RandomForest).

## Parallel processing across the team

The dataset build and model training can be split across 4 laptops (`02_build_dataset.py --shard k`, `02b_merge_shards.py`, `03_train_compare.py --models ...`, `03b_merge_results.py`). See **TEAM_WORKFLOW.md**.

## Reproduce the experiments with scripts

The scripts run the full pipeline without notebooks, which is useful for parallel runs. Run them in order from the repo root:

```bash
cd scripts
python 02_build_dataset.py      # labels + candidate features  (results/data/)
python 01_eda.py                # EDA tables and figures
python 03_train_compare.py      # model comparison             (results/model_comparison.csv)
python 04_bayes_opt.py          # Bayesian optimisation        (results/bayes_opt_*.csv/json, artifacts/final_model.*)
python 05_hmm_eval.py           # HMM tuning + evaluation      (results/hmm_*.csv, artifacts/hmm_config.json)
python 06_maps.py               # Folium maps                  (results/maps/*.html)
python 07_plots.py              # figures                      (results/figures/*.png)
```

All random seeds are fixed at 42. The candidate tables in `results/data/cand_*.parquet` are git-ignored because of their size; running `02_build_dataset.py` regenerates them. Run `02` before `01`.

`08_screenshot.py` is only needed to render PNGs of the maps without internet access.

## Deployment (two laptops)

**Laptop 1: model + API**

```bash
# copy the 10 evaluator trip folders into data/eval_test/  (each folder: <id>.track/.nodes/.arcs/.route)
# Windows:     set PATHSNAP_EVAL_DIR=data\eval_test
# macOS/Linux: export PATHSNAP_EVAL_DIR=data/eval_test
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Find Laptop 1's IP address with `ipconfig` (Windows) or `ip addr` / `ifconfig`. Allow port 8000 through the firewall. Every request is logged in this terminal with the trip id, Top-1 accuracy, route F1 and timing. Show these logs to the evaluator.

**Laptop 2: UI**

```bash
cd ui
python -m http.server 5500
# open http://localhost:5500, set API to http://<laptop1-ip>:8000, click Connect
```

Pick a trip and click **Run map-matching**. The UI then shows the raw GPS, the candidate road network and the predicted route, plus the metrics computed on the server.

### API

| Method | Path       | Body                    | Returns                                                         |
|--------|------------|-------------------------|-----------------------------------------------------------------|
| GET    | `/health`  | none                    | model and eval-dir status                                       |
| GET    | `/trips`   | none                    | list of evaluation trip ids                                     |
| POST   | `/predict` | `{"trip_id": "..."}`    | predicted route polyline, GPS, server-side metrics, timing      |

The prediction pipeline reads only `.track`, `.nodes` and `.arcs`. The `.route` reference is read afterwards, on the server, to compute accuracy. It is never returned to the UI.

## Data-integrity rules followed

- Models are fitted and cross-validated (GroupKFold by trip) only on `train`. `student_test` is used only for development comparison.
- The evaluator partition is never read during development, and nothing is hard-coded.
- Feature scaling is fitted inside each training fold or pipeline.


## Results (student_test, 10 trips)

| Stage | Top-1 (observation) | Route F1 | Route continuity |
|---|---|---|---|
| Nearest-arc baseline | 0.532 | – | – |
| Tuned LightGBM (classifier only) | 0.9065 | 0.899 | 0.768 (raw) |
| **Tuned LightGBM + HMM/Viterbi** | **0.9350** | **0.950** | **0.999** (gap-filled) |

Model comparison (5-fold GroupKFold CV on train, then student_test): see `results/model_comparison.csv`.

| Model | CV Macro F1 | Test Top-1 | Test Macro F1 | Test LogLoss | +HMM Top-1 |
|---|---|---|---|---|---|
| LogisticRegression | 0.9107 | 0.8703 | 0.9071 | 0.1399 | 0.9121 |
| GaussianNB | 0.8571 | 0.8636 | 0.8438 | 0.5902 | 0.8924 |
| KNN | 0.9233 | 0.8730 | 0.9196 | 0.1588 | 0.9164 |
| RandomForest | 0.9360 | 0.8948 | 0.9340 | 0.0920 | 0.9371 |
| LightGBM | 0.9380 | 0.9049 | 0.9388 | 0.0874 | 0.9357 |
| XGBoost | 0.9368 | 0.9041 | 0.9383 | 0.0876 | 0.9327 |
| MLP | 0.9305 | 0.8880 | 0.9285 | 0.1019 | 0.9236 |

Bayesian optimisation used Optuna TPE with 30 trials. The tuned configuration is in `results/bayes_opt_summary.json`. HMM decoder: α=2.0, β=5.0 m, gap break at 60 s.

Report: `report/PathSnap_Technical_Report.pdf`. Rebuild it with `python report/build_report.py`, which needs Google Chrome for the PDF step. Edit `report/team.json` first.
