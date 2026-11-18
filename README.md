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
src/pathsnap/
  core.py           loading, local AEQD projection, reference-label alignment, candidates, features
  hmm.py            road graph, Viterbi decoder, route reconstruction (gap filling)
  metrics.py        Top-1, macro F1, log loss, route continuity, route P/R/F1
scripts/
  01_eda.py             dataset inventory, data-quality checks, EDA plots
  02_build_dataset.py   reference labels + candidate/feature tables
  03_train_compare.py   7 models, 5-fold GroupKFold CV + student-test evaluation
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

## Reproduce the experiments

Run the scripts in order from the repo root:

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

All random seeds are fixed at 42.

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

## Results

See `report/` and `results/` for the full results.
