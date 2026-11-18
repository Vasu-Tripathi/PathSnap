"""Generates the Phase-1 notebooks (they call the same code in src/ and scripts/, so numbers match the scripts).
Run:  python make_notebooks.py   then execute with  jupyter nbconvert --to notebook --execute --inplace *.ipynb"""
import nbformat as nbf

SETUP = """import os, sys, json, time, warnings
ROOT = os.path.abspath("..")
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(os.path.join(ROOT, "scripts"))  # scripts use paths relative to the repo
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image, IFrame, HTML
warnings.filterwarnings("ignore")
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
DATA, RES, FIG = os.path.join(ROOT, "data"), os.path.join(ROOT, "results"), os.path.join(ROOT, "results", "figures")"""


def nb(cells):
    n = nbf.v4.new_notebook()
    n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    n.cells = [nbf.v4.new_markdown_cell(c[1]) if c[0] == "md" else nbf.v4.new_code_cell(c[1]) for c in cells]
    return n


# ---------------------------------------------------------------- 01 EDA
eda = [
("md", "# 01 · Dataset exploration (EDA)\n**Partitions used:** `train` (80 trips) and `student_test` (10 trips). The evaluator partition is never read.\n\nThis notebook calls `scripts/01_eda.py`, which runs the full analysis, then shows its outputs and adds a few interactive views."),
("code", SETUP),
("md", "## 1. Raw files of one trip\nThe files have no headers. `.track` = lon, lat, t(s) · `.nodes` = lon, lat (row = node id) · `.arcs` = from, to (row = arc id) · `.route` = ordered arc ids."),
("code", """tid = "00000048"; d = f"{DATA}/train/{tid}"
for ext in ["track", "nodes", "arcs", "route"]:
    df = pd.read_csv(f"{d}/{tid}.{ext}", sep=r"\\s+", header=None)
    print(f".{ext:6s} rows={len(df):6d} cols={df.shape[1]}  dtypes={list(df.dtypes.astype(str))}")
    display(df.head(3))"""),
("md", "## 2. Run the EDA script (inventory, data quality, network connectivity, plots)"),
("code", """import subprocess
r = subprocess.run([sys.executable, "01_eda.py"], capture_output=True, text=True); print(r.stdout[-3000:], r.stderr[-2000:])"""),
("md", "## 3. Dataset inventory and summary"),
("code", """inv = pd.read_csv(f"{RES}/eda_inventory.csv", dtype={"trip": str})
display(pd.read_csv(f"{RES}/eda_summary.csv").set_index("partition").T)
display(inv.head(10))"""),
("md", "## 4. Missing values, sampling irregularities, anomalies"),
("code", """facts = json.load(open(f"{RES}/eda_facts.json")); display(pd.Series(facts))
print("Trips with the most >5 s gaps:"); display(inv.sort_values("n_gaps_gt5s", ascending=False)[["partition","trip","n_gps","n_gaps_gt5s","dt_max"]].head(5))
lab = pd.read_csv(f"{RES}/eda_label_alignment.csv", dtype={"trip": str})
print("Trips whose GPS sits far from the reference route (systematic offset):"); display(lab.sort_values("label_dist_median", ascending=False).head(5))"""),
("md", "## 5. Plots"),
("code", """for f in ["eda_distributions.png", "eda_trajectories.png", "eda_network.png", "eda_trip_locations.png"]:
    display(Image(f"{FIG}/{f}", width=1000))"""),
("md", "## 6. Folium map: GPS, road network and reference route (train trip 00000048)"),
("code", """import importlib; maps = importlib.import_module("06_maps")
f = maps.render_eda("train", "00000048"); print(f)
IFrame(os.path.relpath(f, os.path.join(ROOT, "notebooks")), width="100%", height=520)"""),
("md", "**Observations.** Sampling is ~97% at 1 Hz. There are no NaNs or duplicate timestamps, but some long gaps exist (up to 153,064 s). Most roads are two-way, which gives pairs of opposite arcs. Trips 33 and 54 have a systematic ~120–140 m offset and are excluded from training. The data are spread around the world, so we use a per-trip local metric projection."),
]

# ---------------------------------------------------------------- 02 preprocessing
pre = [
("md", "# 02 · Preprocessing, reference labels, candidates and features\nWalk-through on train trip `00000048`, using exactly the functions the pipeline uses (`src/pathsnap/core.py`)."),
("code", SETUP),
("code", """from pathsnap.core import load_trip, align_route, build_candidates, arc_geometry, FEATURES, CAND_RADIUS_M, CAND_MAX_K
tr = load_trip(f"{DATA}/train/00000048")
print("GPS points:", len(tr.track), "| nodes:", len(tr.nodes), "| arcs:", len(tr.arcs), "| route arcs:", len(tr.route))
print("Projected GPS (local Azimuthal Equidistant, metres) – first rows:\\n", tr.xy[:3].round(2))"""),
("md", "## 1. Why a local projection?\nDegrees are not metres, and one degree of longitude shrinks with latitude. We compare the projected distance with the geodesic (haversine) distance for consecutive points."),
("code", """lon, lat = np.radians(tr.track[:,0]), np.radians(tr.track[:,1])
hav = 2*6371008.8*np.arcsin(np.sqrt(np.sin(np.diff(lat)/2)**2 + np.cos(lat[:-1])*np.cos(lat[1:])*np.sin(np.diff(lon)/2)**2))
proj = np.hypot(*np.diff(tr.xy, axis=0).T); deg = np.hypot(*np.diff(tr.track[:,:2], axis=0).T)
print(f"max |projected - haversine| = {np.abs(proj-hav).max():.4f} m over {len(hav)} steps")
print(f"ratio of raw-degree distance to metres varies: {np.min(deg[hav>1]/hav[hav>1]):.2e} .. {np.max(deg[hav>1]/hav[hav>1]):.2e} (not a constant!)")"""),
("md", "## 2. Per-observation reference labels (monotone DP alignment)\n`.route` lists arcs, not one arc per point. Each GPS point is assigned to a route arc so that the route order is preserved and the total point-to-segment distance is minimised."),
("code", """lab, ld, k = align_route(tr)
print("label distance (m): median %.2f, p90 %.2f, max %.2f" % (np.median(ld), np.percentile(ld, 90), ld.max()))
print("route index is non-decreasing:", bool(np.all(np.diff(k) >= 0)))
plt.figure(figsize=(10,3)); plt.plot(k); plt.xlabel("GPS observation"); plt.ylabel("route position"); plt.title("Monotone alignment"); plt.show()"""),
("md", f"## 3. Candidate generation and features\nCandidates are arcs within {50} m (max 8 per point). If there are none, the 3 nearest within 300 m are used."),
("code", """t0 = time.time(); cand = build_candidates(tr, lab); print(f"{len(cand)} candidate rows in {time.time()-t0:.2f}s")
print("candidates per point:", cand.groupby("obs").size().describe().round(2).to_dict())
print("reference arc present in candidates:", round(cand.groupby("obs").y.max().mean(), 4))
display(cand[["obs","arc"] + FEATURES + ["y"]].head(8).round(3))"""),
("md", "## 4. Why heading features matter: nearest-arc baseline vs. features"),
("code", """near = cand.loc[cand.groupby("obs").dist.idxmin()]
print("Nearest-arc Top-1 on this trip:", round(near.y.mean(), 4))
pos = cand[cand.y==1]; neg = cand[cand.y==0]
fig, ax = plt.subplots(1, 3, figsize=(15,3.5))
for a, f in zip(ax, ["dist_minus_min", "ctx_bearing_diff", "ctx_dist_mean"]):
    a.hist(np.clip(neg[f], 0, np.percentile(cand[f], 99)), bins=50, alpha=.6, density=True, label="other candidates")
    a.hist(np.clip(pos[f], 0, np.percentile(cand[f], 99)), bins=50, alpha=.6, density=True, label="reference arc"); a.set_title(f); a.legend()
plt.tight_layout(); plt.show()"""),
("md", "## 5. Full dataset build (all trips)\nRun `02_build_dataset.py` (optionally `--shard k --nshards 4` on each laptop, then `02b_merge_shards.py`). Here we load the merged tables."),
("code", """c = pd.read_parquet(f"{RES}/data/obs_train.parquet"); s = pd.read_parquet(f"{RES}/data/obs_student_test.parquet")
print("train observations:", len(c), "| student_test observations:", len(s))
print("candidate miss rate – train: %.4f, student_test: %.4f" % (1-c.has_cand.mean(), 1-s.has_cand.mean()))"""),
]

# ---------------------------------------------------------------- 03 models
mdl = [
("md", "# 03 · Model training and comparison\nThis is binary classification of (GPS point, candidate arc) pairs.\n- **Training/CV:** 78 training trips (2 anomalous trips excluded), 5-fold **GroupKFold by trip**.\n- **Development test:** the 10 `student_test` trips.\n- **Primary metric:** observation-level Top-1 accuracy.\n\nEach model below is trained live on the training partition and evaluated on student_test, using the same code and seed as `03_train_compare.py`.\nSet `RUN_CV = True` to also re-run the 5-fold CV live; it's slow for KNN and RandomForest. Otherwise the CV numbers are loaded from the script's log `results/model_cv_folds.csv`."),
("code", SETUP),
("code", """import importlib
tc = importlib.import_module("03_train_compare")
from pathsnap.core import FEATURES
from pathsnap.metrics import full_eval, candidate_metrics
RUN_CV = False
c, o = tc.load_train()
ct = pd.read_parquet(f"{RES}/data/cand_student_test.parquet"); ot = pd.read_parquet(f"{RES}/data/obs_student_test.parquet")
arcs_tr, arcs_te = tc.arcs_for("train"), tc.arcs_for("student_test")
X, y = c[FEATURES].values, c["y"].values
print("training rows:", X.shape, "positive rate: %.3f" % y.mean(), "| student_test rows:", len(ct))
MODELS = tc.make_models(); logged_folds = pd.read_csv(f"{RES}/model_cv_folds.csv")
results = {}

def run(name):
    fam, cfg, make = MODELS[name]
    print(f"{name}  [{fam}]  {cfg}")
    cvr = tc.cv_eval(make, c, o, arcs_tr) if RUN_CV else logged_folds[logged_folds.model == name]
    t0 = time.time(); m = make()
    if name == "MLP": m.fit(X, y, ct[FEATURES].values, ct["y"].values)
    else: m.fit(X, y)
    fit = time.time() - t0
    t0 = time.time(); p = m.predict_proba(ct[FEATURES].values)[:, 1]; inf = time.time() - t0
    cc = ct.copy(); cc["p"] = p; met, _ = full_eval(cc, ot, arcs_te)
    row = dict(cv_macro_f1=cvr.macro_f1.mean(), cv_top1=cvr.top1_acc.mean(), test_top1=met["top1_acc"], test_macro_f1=met["macro_f1"],
               test_log_loss=met["log_loss"], test_route_continuity=met["route_continuity"], fit_s=fit, infer_ms_per_1k_obs=1e6*inf/len(ot))
    results[name] = row; display(pd.DataFrame([row]).round(4)); return m"""),
]
for n, why in [("LogisticRegression", "linear, calibrated baseline"), ("GaussianNB", "generative probabilistic model"),
               ("KNN", "instance-based (slow inference)"), ("RandomForest", "bagging ensemble (~2 min fit)"),
               ("LightGBM", "gradient boosting (selected)"), ("XGBoost", "gradient boosting (depth-wise)"), ("MLP", "shallow neural network, epoch-wise history")]:
    mdl.append(("md", f"## {n} — {why}"))
    mdl.append(("code", f'm_{n} = run("{n}")'))
mdl += [
("md", "## MLP training history (real, epoch-wise)"),
("code", """h = pd.DataFrame(m_MLP.history); plt.plot(h.epoch, h.train_loss, label="train"); plt.plot(h.epoch, h.val_loss, label="student_test"); plt.legend(); plt.xlabel("epoch"); plt.ylabel("log loss"); plt.show()"""),
("md", "## Comparison (live run) vs. logged script results"),
("code", """live = pd.DataFrame(results).T; display(live.round(4))
logged = pd.read_csv(f"{RES}/model_comparison.csv").set_index("model")
print("max |live - logged| test Top-1:", float(np.abs(live.test_top1 - logged.loc[live.index, "test_top1"]).max()))"""),
("code", """for f in ["cv_macro_f1_top1.png", "student_test_top1_f1.png", "logloss_inference.png", "generalisation_gap.png", "confusion_matrices.png", "training_histories.png"]:
    display(Image(f"{FIG}/{f}", width=1000))"""),
("md", "**Selection: LightGBM.** It has the best CV Macro F1, CV Top-1 and log loss, and the best student-test Top-1/F1/log loss, with fast training and inference. XGBoost is essentially tied. KNN is too slow at inference. LogReg and GaussianNB underfit."),
]

# ---------------------------------------------------------------- 04 BO
bo = [
("md", "# 04 · Bayesian hyperparameter optimisation (Optuna TPE) — LightGBM\nThe full study (30 trials, 3-fold GroupKFold, objective = CV Top-1) is run by `scripts/04_bayes_opt.py`. Its record is in `results/bayes_opt_trials.csv`.\nBelow, a **short live study** shows the mechanism, then the full record is loaded."),
("code", SETUP),
("code", """import importlib, optuna, lightgbm as lgb
optuna.logging.set_verbosity(optuna.logging.WARNING)
bo = importlib.import_module("04_bayes_opt")   # loads training data, defines objective + search space
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=3))
t0 = time.time(); study.optimize(bo.objective, n_trials=6); print(f"6 live trials in {time.time()-t0:.0f}s")
display(study.trials_dataframe()[["number", "value", "params_learning_rate", "params_num_leaves", "params_n_estimators", "params_min_child_samples"]].round(4))"""),
("md", "## Full search record (30 trials)"),
("code", """t = pd.read_csv(f"{RES}/bayes_opt_trials.csv"); s = json.load(open(f"{RES}/bayes_opt_summary.json"))
print("best trial:", int(t.value.idxmax()), "objective %.4f (baseline trial 0: %.4f)" % (t.value.max(), t.value.iloc[0]))
display(pd.Series(s["tuned"]["params"]))
cmp = pd.DataFrame({k: {m: s[k][m] for m in ["cv_macro_f1", "cv_top1", "test_top1", "test_macro_f1", "test_log_loss", "inference_ms_per_1k_obs"]} for k in ["untuned", "tuned"]})
display(cmp.round(4))
display(Image(f"{FIG}/bayes_opt.png", width=1000)); display(Image(f"{FIG}/feature_importance.png", width=600))"""),
("md", "**Result.** Tuning improved the optimised objective (CV Top-1) by about 0.2 points, and student-test Top-1 by about 0.17 points. Macro F1 was unchanged and log loss slightly worse. The default configuration was already close to what these features allow."),
]

for name, cells in [("01_EDA", eda), ("02_Preprocessing_Features", pre), ("03_Model_Training", mdl), ("04_Bayesian_Optimisation", bo)]:
    nbf.write(nb(cells), f"{name}.ipynb")
print("written")
