# Parallel team workflow (4 laptops)

Each step writes **different files** on each laptop, so you can all push to the same repo without merge conflicts. Everyone needs the repo and the data first:

```bash
git clone <repo-url> && cd PathSnap
# copy train/ and student_test/ from the Student folder into data/
pip install -r requirements.txt
```

## Step 1: process the dataset in parallel (each member takes one shard)

The trips are split into 4 balanced shards: about 20 train trips and 2–3 student_test trips each, assigned deterministically by file size.

| Member | Command |
|---|---|
| Vasu Tripathi | `cd scripts && python 02_build_dataset.py --shard 0 --nshards 4` |
| Nidhi S | `cd scripts && python 02_build_dataset.py --shard 1 --nshards 4` |
| Varun ML | `cd scripts && python 02_build_dataset.py --shard 2 --nshards 4` |
| Vinod Kumar L | `cd scripts && python 02_build_dataset.py --shard 3 --nshards 4` |

Each shard writes `results/data/shards/{cand,obs}_<partition>_shard<k>of4.parquet`. Then:

```bash
git add results/data/shards && git commit -m "Dataset shard <k>" && git pull --rebase && git push
```

Once all 4 shards are pushed, **everyone** runs:

```bash
git pull
cd scripts && python 02b_merge_shards.py      # builds results/data/{cand,obs}_*.parquet and checks every trip appears exactly once
```

The merged tables are identical to a single-machine run. We verified this with a byte-for-byte comparison.

## Step 2: EDA, model training and tuning in parallel

| Member | Command(s) | Approx. time |
|---|---|---|
| Vasu Tripathi | `python 03_train_compare.py --models LightGBM,XGBoost` then `python 04_bayes_opt.py` | ~2 min + ~10 min |
| Nidhi S | `python 03_train_compare.py --models RandomForest` and `python 01_eda.py` | ~12 min |
| Varun ML | `python 03_train_compare.py --models KNN,GaussianNB` and `python 06_maps.py --eda-only` | ~5 min |
| Vinod Kumar L | `python 03_train_compare.py --models LogisticRegression,MLP` | ~5 min |

Each member's results go to `results/model_comparison_<models>.csv`, `results/model_cv_folds_<models>.csv`, `artifacts/models/<model>.joblib`, `results/data/preds/student_test_<model>.parquet` and `results/history_<model>.csv`. Commit and push those files.

Then one member (Vinod) pulls and runs:

```bash
python 03b_merge_results.py   # -> results/model_comparison.csv + model_cv_folds.csv (warns if a model is missing)
python 07_plots.py            # all comparison figures
python ../report/build_report.py --phase1
```

All seeds are fixed at 42, so the results don't depend on which laptop trains which model.

## Phase 2 (Review 2) split

| Member | Owns |
|---|---|
| Vasu Tripathi | `05_hmm_eval.py`: HMM/Viterbi tuning and evaluation (needs Vasu's `04` output) |
| Nidhi S | Laptop 1: FastAPI backend `api/app.py`, the eval-trip setup and the API logs |
| Varun ML | Laptop 2: evaluator UI `ui/index.html` and the end-to-end test |
| Vinod Kumar L | `06_maps.py` prediction maps, `07_plots.py`, and the final report (`build_report.py`) |
