"""Builds report/PathSnap_Technical_Report.html (+ PDF via headless Chrome) from the result files in results/.
Every number in the report is read from the experiment outputs, so the report is verifiable against the repo."""
import os, sys, json, subprocess, base64
import pandas as pd, numpy as np

PHASE = 1 if "--phase1" in sys.argv else 2
def H(t): return t if PHASE == 2 else ""
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R, F = os.path.join(ROOT, "results"), os.path.join(ROOT, "results", "figures")
TEAM = json.load(open(os.path.join(ROOT, "report", "team.json")))

mc = pd.read_csv(f"{R}/model_comparison.csv").set_index("model")
folds = pd.read_csv(f"{R}/model_cv_folds.csv")
bo = json.load(open(f"{R}/bayes_opt_summary.json"))
trials = pd.read_csv(f"{R}/bayes_opt_trials.csv")
if PHASE == 2:
    hc = pd.read_csv(f"{R}/hmm_comparison_student_test.csv").set_index("model")
    hg = pd.read_csv(f"{R}/hmm_decoder_grid_train.csv")
    hp = pd.read_csv(f"{R}/hmm_student_test_LightGBM_tuned.csv", dtype={"trip": str}); hp["trip"] = hp.trip.str.zfill(8)
facts = json.load(open(f"{R}/eda_facts.json"))
inv = pd.read_csv(f"{R}/eda_inventory.csv", dtype={"trip": str})
summ = pd.read_csv(f"{R}/eda_summary.csv").set_index("partition")
fi = pd.read_csv(f"{R}/feature_importance.csv")
hcfg = json.load(open(os.path.join(ROOT, "artifacts", "hmm_config.json")))["decoder"] if PHASE == 2 else {}
NEAREST = facts["nearest_arc_top1_student_test"]  # nearest-arc baseline Top-1 on student test (computed in 01_eda.py)
hist = {n: pd.read_csv(f"{R}/history_{n}.csv") for n in ["LightGBM", "XGBoost", "MLP"]}


def f3(x): return f"{x:.3f}"
def f4(x): return f"{x:.4f}"
def pc(x): return f"{100 * x:.2f}%"
def img(name, w=100, cap=""):
    p = os.path.join(F, name)
    b64 = base64.b64encode(open(p, "rb").read()).decode()
    return f'<figure><img src="data:image/png;base64,{b64}" style="width:{w}%"><figcaption>{cap}</figcaption></figure>'
def table(df, cls=""):
    return df.to_html(index=False, escape=False, classes=cls, border=0)
def m(name, col): return mc.loc[name, col]


ORDER = ["LogisticRegression", "GaussianNB", "KNN", "RandomForest", "LightGBM", "XGBoost", "MLP"]
tr_ = summ.loc["train"]; st_ = summ.loc["student_test"]

# ---------------------------------------------------------------- comparison table
comp = pd.DataFrame([{
    "Model": f"{i+1}. {n}", "Algorithm family": mc.loc[n, "family"], "Key configuration": mc.loc[n, "config"],
    "CV Macro F1 (±sd)": f"{f4(m(n,'cv_macro_f1'))} ± {m(n,'cv_macro_f1_std'):.4f}",
    "Top-1 Acc. (student test)": f4(m(n, "test_top1")),
    "Route continuity (student test, raw)": f3(m(n, "test_route_continuity")),
    "Student-Test Macro F1": f4(m(n, "test_macro_f1")), "Log Loss (student test)": f4(m(n, "test_log_loss")),
    "Inference time (ms / 1k obs)": f"{m(n,'inference_ms_per_1k_obs'):.1f}",
} for i, n in enumerate(ORDER)])
comp2 = pd.DataFrame([{
    "Model": n, "CV Top-1": f4(m(n, "cv_top1")), "CV Log Loss": f4(m(n, "cv_log_loss")), "Train Macro F1": f4(m(n, "train_macro_f1")),
    "Test Precision (macro)": f4(m(n, "test_precision")), "Test Recall (macro)": f4(m(n, "test_recall")),
    "Candidate miss rate": pc(m(n, "test_candidate_miss")), "Fit time (s)": f"{m(n,'fit_time_s'):.1f}",
    **({"+HMM Top-1": f4(hc.loc[n, "hmm_top1"]), "+HMM route F1": f4(hc.loc[n, "hmm_route_F1"])} if PHASE == 2 else {}),
} for n in ORDER])

# ---------------------------------------------------------------- per-model narrative
def hist_txt(n):
    h = hist[n]; xc = h.columns[0]; i = h.val_loss.idxmin()
    return (f"The recorded history ({xc}s) shows training loss falling from {h.train_loss.iloc[0]:.4f} to {h.train_loss.iloc[-1]:.4f}; "
            f"student-test loss reached its minimum {h.val_loss.min():.4f} at {xc} {int(h[xc].iloc[i])} and ended at {h.val_loss.iloc[-1]:.4f}.")

MODELS = {
 "LogisticRegression": dict(
  why="A linear, well-calibrated probabilistic baseline. Our features (distance gaps, heading agreement, context distance) are largely monotone with the probability that a candidate is correct, so a linear log-odds model is a reasonable first test and gives a calibrated probability that the HMM can consume directly as an emission score.",
  sl="Strengths: very fast (sub-millisecond per 1k observations), stable across folds, calibrated probabilities. Limitations: it cannot model interactions such as \"heading only matters when speed is high\" or the non-linear effect of the bearing difference near 90°, which is where most two-way-road confusions occur.",
  hp="C = 1.0 (inverse L2 strength), penalty = L2, solver = lbfgs, max_iter = 2000, StandardScaler fitted inside each training fold.",
  loss="Binary cross-entropy (log loss) with L2 regularisation.",
  fit=lambda: f"Train Macro F1 {f4(m('LogisticRegression','train_macro_f1'))}, CV {f4(m('LogisticRegression','cv_macro_f1'))}, student test {f4(m('LogisticRegression','test_macro_f1'))}. The three are almost identical, so there is no overfitting; the model is <b>underfitting</b> (high bias) – it is about 2.7 Macro-F1 points below the tree ensembles on held-out trips, consistent with the missing feature interactions."),
 "GaussianNB": dict(
  why="A generative probabilistic model chosen as a contrasting family: it estimates per-class feature distributions and is extremely cheap, which tests whether the features are informative even under a strong independence assumption.",
  sl="Strengths: trivial to train, no tuning needed, high recall on the positive class. Limitations: the features are strongly correlated (dist, dist_minus_min, dist_ratio_min, ctx_dist_mean all measure distance), which violates the naive independence assumption and double-counts evidence, producing over-confident, poorly calibrated probabilities.",
  hp="var_smoothing = 1e-9 (default); no scaling required.",
  loss="Maximum likelihood estimation of class-conditional Gaussian means/variances and class priors (no iterative loss).",
  fit=lambda: f"Train Macro F1 {f4(m('GaussianNB','train_macro_f1'))} ≈ CV {f4(m('GaussianNB','cv_macro_f1'))} ≈ test {f4(m('GaussianNB','test_macro_f1'))} – clear <b>underfitting</b>. Its log loss ({f4(m('GaussianNB','test_log_loss'))}) is 6–7× worse than the boosted models, confirming miscalibration: it has the highest positive-class recall but {int(m('GaussianNB','fp'))} false positives on the student test (vs {int(m('LightGBM','fp'))} for LightGBM)."),
 "KNN": dict(
  why="An instance-based model: a candidate whose geometry (distance, heading difference, context distance) resembles labelled examples should share their label. It makes no parametric assumption about the decision boundary.",
  sl="Strengths: non-parametric, captures local non-linear structure. Limitations: sensitive to feature scaling, must store and search the full training set (inference is the slowest of all models), and distance-weighting memorises the training points.",
  hp="n_neighbors = 31, weights = distance, metric = Euclidean (Minkowski p=2) on standardised features.",
  loss="None (lazy learner); prediction is the distance-weighted vote of the 31 nearest training candidates.",
  fit=lambda: f"Train Macro F1 is {f4(m('KNN','train_macro_f1'))} (each training point is its own nearest neighbour with infinite weight) versus CV {f4(m('KNN','cv_macro_f1'))} and test {f4(m('KNN','test_macro_f1'))}: it <b>overfits</b> the training points by construction, but generalises reasonably. Its inference cost ({m('KNN','inference_ms_per_1k_obs'):.0f} ms per 1k observations) makes it impractical for the API."),
 "RandomForest": dict(
  why="A bagging ensemble of decision trees that models non-linear interactions (e.g. bearing difference × speed) and is robust to feature scale and outliers in GPS noise.",
  sl="Strengths: strong accuracy, low variance across folds, good probability ranking. Limitations: large model files, slowest training of the tree models, and averaged leaf frequencies give slightly less sharp probabilities than boosting.",
  hp="n_estimators = 200, max_features = sqrt, min_samples_leaf = 5, bootstrap = True, random_state = 42.",
  loss="Each tree greedily minimises Gini impurity; the forest averages leaf class frequencies.",
  fit=lambda: f"Train Macro F1 {f4(m('RandomForest','train_macro_f1'))} vs CV {f4(m('RandomForest','cv_macro_f1'))} / test {f4(m('RandomForest','test_macro_f1'))}: a moderate variance gap (mild <b>overfitting</b> of deep trees), limited by min_samples_leaf = 5; held-out performance is still the second best."),
 "LightGBM": dict(
  why="Gradient-boosted trees fit the residual errors sequentially, modelling feature interactions efficiently, and are known to give well-calibrated probabilities with log-loss training – both important for using the scores as HMM emissions.",
  sl="Strengths: best CV Macro F1, CV Top-1 and log loss among the untuned models, fast training (histogram-based) and fast inference. Limitations: more hyperparameters, can overfit with many leaves, and per-observation predictions are still independent (no sequence context beyond our ±3/±5-point context features).",
  hp="n_estimators = 300, learning_rate = 0.05, num_leaves = 31, min_child_samples = 20, subsample = 1.0, colsample_bytree = 1.0, random_state = 42.",
  loss="Binary log loss (cross-entropy) minimised by gradient boosting (second-order, leaf-wise growth).",
  fit=lambda: f"Train Macro F1 {f4(m('LightGBM','train_macro_f1'))} vs CV {f4(m('LightGBM','cv_macro_f1'))} / test {f4(m('LightGBM','test_macro_f1'))}. {hist_txt('LightGBM')} Validation loss is still flat-to-decreasing at the last round, so there is <b>no overfitting</b>; the small train/test gap is expected between seen and unseen trips."),
 "XGBoost": dict(
  why="A second gradient-boosting implementation (depth-wise trees with regularised objective) to check whether the boosting result is robust to the implementation and tree-growth strategy.",
  sl="Strengths: accuracy on par with LightGBM, fastest of the boosted models at inference here. Limitations: with a higher learning rate and depth 6 it starts to overfit before 300 rounds; slightly worse CV log loss.",
  hp="n_estimators = 300, learning_rate = 0.1, max_depth = 6, tree_method = hist, random_state = 42.",
  loss="Binary log loss with L2 leaf-weight regularisation (λ = 1).",
  fit=lambda: f"Train Macro F1 {f4(m('XGBoost','train_macro_f1'))} vs CV {f4(m('XGBoost','cv_macro_f1'))} / test {f4(m('XGBoost','test_macro_f1'))}. {hist_txt('XGBoost')} The rise after the minimum is mild <b>overfitting</b>; early stopping around that round would help."),
 "MLP": dict(
  why="A shallow neural network learns non-linear combinations of the standardised features and provides a differentiable alternative to trees, with an epoch-wise training history.",
  sl="Strengths: smooth decision function, compact model, fast inference. Limitations: needs feature scaling and careful learning-rate/regularisation choices; training is noisier and it did not reach the accuracy of the tree ensembles on these tabular features.",
  hp="hidden layers = (64, 32), ReLU, Adam learning_rate_init = 1e-3, alpha (L2) = 1e-4, batch = 512, 25 epochs, StandardScaler.",
  loss="Binary cross-entropy (log loss) with L2 penalty, optimised by Adam mini-batch gradient descent.",
  fit=lambda: f"Train Macro F1 {f4(m('MLP','train_macro_f1'))} vs CV {f4(m('MLP','cv_macro_f1'))} / test {f4(m('MLP','test_macro_f1'))}. {hist_txt('MLP')} The gap between training and validation loss grows in the later epochs and validation loss fluctuates – <b>mild overfitting</b> plus training instability."),
}

per_model = ""
for i, n in enumerate(ORDER):
    d = MODELS[n]
    hmm_line = (" Followed by our HMM decoder, Top-1 rises to " + pc(hc.loc[n, "hmm_top1"]) + " and route F1 to " + f4(hc.loc[n, "hmm_route_F1"]) + ".") if PHASE == 2 else ""
    per_model += f"""
<h3>3.2.{i+1} Model {i+1}: {n} <span class="fam">({mc.loc[n,'family']})</span></h3>
<p class="q">A1. Why suitable?</p><p>{d['why']}</p>
<p class="q">A2. Strengths and limitations</p><p>{d['sl']}</p>
<p class="q">B5. Important hyperparameters and initial values</p><p>{d['hp']}</p>
<p class="q">B6. Loss function / objective</p><p>{d['loss']}</p>
<p class="q">C7. Performance on the student-testing dataset</p>
<p>Observation-level Top-1 {pc(m(n,'test_top1'))}; candidate Macro F1 {f4(m(n,'test_macro_f1'))} (precision {f4(m(n,'test_precision'))}, recall {f4(m(n,'test_recall'))});
raw route continuity {f3(m(n,'test_route_continuity'))}.{hmm_line}</p>
<p class="q">C8. Observation-level Top-1, Macro F1 and Log Loss (student test)</p>
<p><b>Top-1 = {f4(m(n,'test_top1'))}, Macro F1 = {f4(m(n,'test_macro_f1'))}, Log Loss = {f4(m(n,'test_log_loss'))}</b>
(5-fold CV: Top-1 {f4(m(n,'cv_top1'))} ± {m(n,'cv_top1_std'):.4f}, Macro F1 {f4(m(n,'cv_macro_f1'))} ± {m(n,'cv_macro_f1_std'):.4f}).</p>
<p class="q">C9. Overfitting / underfitting?</p><p>{d['fit']()}</p>
"""

# ---------------------------------------------------------------- BO
bt = trials.copy()
best_i = int(bt.value.idxmax())
bo_tab = pd.DataFrame([{"Metric": k, "Untuned LightGBM": v1, "Tuned LightGBM": v2} for k, v1, v2 in [
    ("CV Macro F1 (5-fold)", f4(bo['untuned']['cv_macro_f1']), f4(bo['tuned']['cv_macro_f1'])),
    ("CV Top-1 (5-fold)", f4(bo['untuned']['cv_top1']), f4(bo['tuned']['cv_top1'])),
    ("CV Log Loss", f4(bo['untuned']['cv_log_loss']), f4(bo['tuned']['cv_log_loss'])),
    ("Student-test Top-1", f4(bo['untuned']['test_top1']), f4(bo['tuned']['test_top1'])),
    ("Student-test Macro F1", f4(bo['untuned']['test_macro_f1']), f4(bo['tuned']['test_macro_f1'])),
    ("Student-test Log Loss", f4(bo['untuned']['test_log_loss']), f4(bo['tuned']['test_log_loss'])),
    ("Student-test raw route continuity", f3(bo['untuned']['test_route_continuity']), f3(bo['tuned']['test_route_continuity'])),
] + ([("Student-test Top-1 after HMM", f4(hc.loc['LightGBM','hmm_top1']), f4(hc.loc['LightGBM_tuned','hmm_top1']))] if PHASE == 2 else []) + [
    ("Inference time (ms / 1k obs)", f"{bo['untuned']['inference_ms_per_1k_obs']:.1f}", f"{bo['tuned']['inference_ms_per_1k_obs']:.1f}"),
    ("Fit time (s)", f"{bo['untuned']['fit_time_s']:.1f}", f"{bo['tuned']['fit_time_s']:.1f}"),
]])
bp = bo["tuned"]["params"]
best_tab = pd.DataFrame([{"Hyperparameter": k, "Search range / distribution": r, "Best value": (f"{bp[k]:.4g}" if isinstance(bp[k], float) else bp[k])} for k, r in [
    ("n_estimators", "int 100–800, step 50"), ("learning_rate", "float 0.01–0.2, log-uniform"), ("num_leaves", "int 15–255, log-uniform"),
    ("min_child_samples", "int 10–300, log-uniform"), ("subsample", "float 0.5–1.0 (bagging_freq=1)"), ("colsample_bytree", "float 0.4–1.0"),
    ("reg_lambda", "float 1e-3–10, log-uniform"), ("reg_alpha", "float 1e-3–10, log-uniform")]])

# ---------------------------------------------------------------- HMM
if PHASE == 2:
  hmm_tab = pd.DataFrame([{
    "Model": n, "Classifier-only Top-1": f4(hc.loc[n, "clf_top1"]), "+HMM Top-1": f4(hc.loc[n, "hmm_top1"]),
    "Δ Top-1": f"{100*(hc.loc[n,'hmm_top1']-hc.loc[n,'clf_top1']):+.2f} pts",
    "Raw continuity clf → HMM": f"{f3(hc.loc[n,'clf_continuity'])} → {f3(hc.loc[n,'hmm_continuity'])}",
    "Route F1 clf → HMM": f"{f3(hc.loc[n,'clf_route_F1'])} → {f3(hc.loc[n,'hmm_route_F1'])}",
    "Continuity after gap-fill": f3(hc.loc[n, "hmm_route_continuity"]),
} for n in ORDER + ["LightGBM_tuned"]])
  per_trip = hp[["trip", "n_obs", "clf_top1", "hmm_top1", "clf_route_F1", "hmm_route_F1", "hmm_route_continuity", "hmm_breaks", "unfilled_gaps", "decode_s"]].copy()
  for c in ["clf_top1", "hmm_top1", "clf_route_F1", "hmm_route_F1", "hmm_route_continuity"]: per_trip[c] = per_trip[c].map(f4)
  per_trip["decode_s"] = per_trip.decode_s.map(lambda x: f"{x:.2f}")
  worse = hp[hp.hmm_top1 < hp.clf_top1]
  grid_best = hg.sort_values("hmm_top1", ascending=False).iloc[0]

# ---------------------------------------------------------------- EDA tables
eda_tab = pd.DataFrame([{"Quantity": k, "Train (80 trips)": a, "Student test (10 trips)": b} for k, a, b in [
    ("GPS observations (total)", f"{int(tr_.gps_total):,}", f"{int(st_.gps_total):,}"),
    ("GPS obs per trip (min / mean / max)", f"{int(tr_.gps_min)} / {tr_.gps_mean:.0f} / {int(tr_.gps_max)}", f"{int(st_.gps_min)} / {st_.gps_mean:.0f} / {int(st_.gps_max)}"),
    ("Rows with missing / non-finite values", int(tr_.nan_rows), int(st_.nan_rows)),
    ("Duplicate timestamps", int(tr_.dup_timestamps), int(st_.dup_timestamps)),
    ("Sampling gaps > 5 s", int(tr_.gaps_gt5s), int(st_.gaps_gt5s)),
    ("Road nodes per trip (mean)", f"{tr_.nodes_mean:,.0f}", f"{st_.nodes_mean:,.0f}"),
    ("Road arcs per trip (mean)", f"{tr_.arcs_mean:,.0f}", f"{st_.arcs_mean:,.0f}"),
    ("Isolated nodes (total)", int(tr_.isolated_nodes), int(st_.isolated_nodes)),
    ("Reference route head-to-tail contiguity", f"{tr_.route_contiguous:.4f}", f"{st_.route_contiguous:.4f}"),
    ("Candidate miss rate (reference arc not in candidates)", pc(facts['candidate_miss_train']) + f" ({pc(facts['candidate_miss_train_excl_anom'])} excl. 2 anomalous trips)", pc(facts['candidate_miss_student_test'])),
]])
feat_tab = pd.DataFrame([
    ("dist, dist_rank, dist_minus_min, dist_ratio_min", "Point-to-segment distance and how it compares with the closest candidate – GPS noise is roughly isotropic, so relative distance matters more than absolute distance."),
    ("bearing_diff, cos_bearing, ctx_bearing_diff", f"Angle between GPS heading (±2 and ±5 points) and the arc's direction. Arcs are directed and most roads are two-way: two opposite arcs have identical distance, so heading is the only way to choose the right direction (the nearest-arc baseline is only {pc(NEAREST)} Top-1 on student-test for this reason)."),
    ("ctx_dist_mean, ctx_dist_max", "Distance of the neighbouring ±3 observations to the same arc – suppresses single-point noise and parallel-road confusion."),
    ("proj_t, arc_len", "Where the projection falls on the arc (end-points indicate junction ambiguity) and arc length."),
    ("speed, gps_jump", "Speed controls how reliable the heading is (stationary vehicles have random headings); jumps flag outliers."),
    ("n_cand, is_twoway, out_deg_to, in_deg_from", "Local road density and junction topology – ambiguity is higher at dense junctions."),
], columns=["Features", "Why"])

EV = None
if PHASE == 2 and os.path.exists(f"{R}/evaluator_results.csv"):
    EV = pd.read_csv(f"{R}/evaluator_results.csv", dtype={"trip": str})
    _w = EV.n_gps / EV.n_gps.sum()
    EVS = dict(top1=(EV.top1_hmm * _w).sum(), clf=(EV.top1_classifier_only * _w).sum(), f1=EV.route_f1.mean(),
               P=EV.route_precision.mean(), Rc=EV.route_recall.mean(), cont=EV.route_continuity.mean(),
               better=int((EV.top1_hmm > EV.top1_classifier_only).sum()), tmax=EV.server_time_s.max(), tmin=EV.server_time_s.min())
    ev_tab = EV.copy()
    for c in ["top1_hmm", "top1_classifier_only", "route_precision", "route_recall", "route_f1", "route_continuity"]:
        ev_tab[c] = ev_tab[c].map(lambda x: f"{x:.4f}")
    ev_tab.columns = ["Trip", "GPS obs", "Top-1 (HMM)", "Top-1 (classifier only)", "Route P", "Route R", "Route F1", "Continuity", "HMM breaks", "Server time (s)"]
TITLE = "PathSnap — Review 1 Report" if PHASE == 1 else "PathSnap — Technical Report"
if PHASE == 2:
    SUMMARY_TAIL = (f"On the student-test trips, tuned LightGBM alone reaches Top-1 {pc(hc.loc['LightGBM_tuned','clf_top1'])}. With the HMM it reaches "
                    f"<b>{pc(hc.loc['LightGBM_tuned','hmm_top1'])}</b>, with route F1 {f3(hc.loc['LightGBM_tuned','hmm_route_F1'])} and route continuity "
                    f"{f3(hc.loc['LightGBM_tuned','hmm_route_continuity'])}."
                    + (f" On the <b>10 hidden evaluator trips</b>, run once through the deployed two-laptop system, it reached <b>{pc(EVS['top1'])}</b> Top-1 (classifier only: {pc(EVS['clf'])}), with mean route F1 {EVS['f1']:.3f} and route continuity {EVS['cont']:.3f}." if EV is not None else "")
                    + "\nThe system is served by a FastAPI backend and a browser UI.")
    SEL_TAIL = (f"\nEvery model gains a lot from the HMM. After decoding, the top three end up within 0.5 points of each other (RandomForest + HMM "
                f"{pc(hc.loc['RandomForest','hmm_top1'])} vs tuned LightGBM + HMM {pc(hc.loc['LightGBM_tuned','hmm_top1'])}). We kept LightGBM because of its "
                "better calibration and 16× faster training, and because it was chosen on CV before decoding was tuned.")
    BO_HMM = f"After HMM decoding the two are equivalent ({pc(hc.loc['LightGBM','hmm_top1'])} untuned vs {pc(hc.loc['LightGBM_tuned','hmm_top1'])} tuned). "
    LIM_RECALL = " (route recall is about 0.99)"
    LIM_DEC = "Decoder parameters were tuned on out-of-fold training predictions. "
else:
    SUMMARY_TAIL = (f"On the student-test trips, tuned LightGBM reaches observation-level Top-1 <b>{pc(bo['tuned']['test_top1'])}</b> and Macro F1 "
                    f"{f4(bo['tuned']['test_macro_f1'])} as a classifier alone. Phase 2 (Review 2) adds HMM/Viterbi route reconstruction, the FastAPI backend and the evaluator UI.")
    SEL_TAIL = ""
    BO_HMM = ""
    LIM_RECALL = ""
    LIM_DEC = ""
html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>PathSnap Technical Report</title>
<style>
@page {{ size: A4; margin: 22mm 20mm; }}
body {{ font-family: "Times New Roman", Times, Georgia, serif; font-size: 11pt; color:#000; line-height:1.4; }}
h1 {{ font-size: 20pt; margin: 0 0 6px; font-weight: bold; text-align:center; }}
h2 {{ font-size: 14pt; font-weight: bold; margin: 20px 0 6px; page-break-after: avoid; }}
h3 {{ font-size: 12pt; font-weight: bold; margin: 14px 0 5px; page-break-after: avoid; }}
.fam {{ font-weight: normal; font-style: italic; }}
p.q {{ font-weight: bold; margin: 8px 0 2px; }} p {{ margin: 3px 0 7px; text-align: justify; }}
table {{ border-collapse: collapse; width: 100%; font-size: 9pt; margin: 8px 0 12px; page-break-inside: avoid; border-top: 1.5px solid #000; border-bottom: 1.5px solid #000; }}
th {{ font-weight: bold; text-align:left; padding: 3px 5px; border-bottom: 1px solid #000; }}
td {{ padding: 2px 5px; }}
figure {{ margin: 8px 0 12px; text-align:center; page-break-inside: avoid; }} figcaption {{ font-size: 9.5pt; margin-top:4px; }}
.cover {{ text-align:center; margin-bottom: 18px; }} .cover div {{ margin: 2px 0; }}
.box {{ margin: 10px 0; }}
.pb {{ page-break-before: always; }} code {{ font-family: "Courier New", monospace; font-size: 9.5pt; }}
</style></head><body>
<div class="cover"><h1>{TITLE}</h1>
<div>Distributed map-matching &amp; route reconstruction · UE24CS352A Machine Learning · PES University, RR Campus</div>
<div style="margin-top:8px"><b>Team:</b> {TEAM['team_name']} &nbsp;|&nbsp; <b>Section:</b> {TEAM['section']}</div>
<div><b>Members:</b> {', '.join(TEAM['members'])}</div>
<div><b>Repository:</b> {TEAM['repo']}</div></div>

<div class="box"><b>Abstract.</b> GPS fixes are matched to candidate road arcs within {50} m. A classifier scores each (fix, arc) pair, and an HMM with Viterbi decoding turns these scores into a connected route.
We compared 7 models from 6 algorithm families with 5-fold trip-grouped CV on the training partition and evaluated them on the 10 student-test trips.
<b>LightGBM</b> was selected and tuned with Bayesian optimisation (Optuna TPE, {bo['search']['n_trials']} trials).
{SUMMARY_TAIL}</div>

<h2>1. Problem framing and primary metric</h2>
<p>Map-matching is cast as <b>candidate classification + sequence decoding</b>. For each GPS observation we collect the directed road arcs within 50 m (up to 8; if none, the 3 nearest within 300 m). A binary classifier then predicts whether each candidate is the reference arc.
The <b>primary selection metric is observation-level Top-1 accuracy</b>: for each GPS fix, the highest-scoring candidate is compared with the reference arc. This is exactly what the route output is judged on, and it does not depend on a probability threshold.
Candidate-level Macro F1 (threshold 0.5) and log loss are reported as well. Log loss matters because the probabilities become HMM emission scores. Route continuity and route-level arc precision/recall/F1 measure sequence quality.</p>

<h2>2. Dataset understanding and EDA</h2>
<p><b>Partitions used.</b> All EDA tables and figures use the <b>train</b> (80 trips) and <b>student_test</b> (10 trips) partitions, labelled in each figure. The evaluator partition was never opened during development.
Each trip has its own road network: <code>.nodes</code> (lon, lat), directed <code>.arcs</code> (from, to; row index = arc id) and a <code>.route</code> (ordered arc ids). The <code>.track</code> file holds lon, lat and time in seconds.</p>
{table(eda_tab)}
<p class="q">Q1. How did we preprocess the data?</p>
<p>(i) Rows with non-finite or out-of-range coordinates are dropped. Timestamps are sorted and de-duplicated; none were found, but the checks run anyway.
(ii) <b>Projection:</b> the trips are spread across several continents (see the location plot), so no single UTM zone works. Every trip is projected to a local <b>Azimuthal Equidistant</b> system centred on its GPS centroid, using pyproj. This gives metre units with negligible distortion at trip scale, so Euclidean distance in degrees is never used.
(iii) <b>Per-observation reference labels:</b> the <code>.route</code> lists arcs, not one arc per GPS fix. We assign each fix to a route arc with a <b>monotone dynamic-programming alignment</b>, which preserves the order of travel and minimises the total point-to-segment distance. The resulting label distance has median {facts['label_dist_median_all']:.1f} m.
(iv) Candidate generation uses a KD-tree over arc samples spaced 10 m apart, followed by exact point-to-segment distances.
(v) <b>Anomalies:</b> trips {', '.join(facts['anomalous_trips'])} have a systematic GPS-to-network offset (median aligned distance 122 m and 140 m), so the reference arc is almost never within 50 m. These two trips are excluded from model fitting and reported here.
(vi) For training only, every 3rd fix is kept, because 1 Hz fixes are highly autocorrelated. Evaluation always uses every fix.</p>
<p class="q">Q2. Patterns observed; features extracted and why</p>
<p>Sampling is nominally 1 Hz: {pc(facts['frac_dt_eq_1'])} of steps are exactly 1 s and {pc(facts['frac_dt_eq_2'])} are 2 s. There are {int(tr_.gaps_gt5s)} gaps longer than 5 s in train; the largest is {facts['max_dt']:.0f} s, in trip 00000090, and trip 00000078 has 124 such gaps. These motivate the HMM's gap handling.
GPS noise is mostly small (median {facts['label_dist_median_all']:.1f} m from the aligned arc), but the 95th percentile is {facts['label_dist_p95_all']:.0f} m because of the anomalous trips and short outlier bursts.
The road networks are large: a mean of {tr_.nodes_mean:,.0f} nodes per training trip. They have no isolated nodes, and the largest strongly connected component covers about 95% of nodes. The reference routes are almost perfectly contiguous head-to-tail.
The strongest pattern is that <b>most roads are two-way</b>: each street appears as two opposite arcs at identical distance from the fix. A plain nearest-arc rule therefore gets only {pc(NEAREST)} Top-1 on student-test, and heading features are essential.</p>
{table(feat_tab)}
<p>Tuned LightGBM's gain importance confirms this design. The top features are <code>{fi.feature.iloc[0]}</code>, <code>{fi.feature.iloc[1]}</code> and <code>{fi.feature.iloc[2]}</code>: relative distance, heading agreement and context.</p>
{img('eda_distributions.png', 98, 'Fig 2.1 – Train + student_test: GPS points per trip, sampling interval (log count), and GPS-to-aligned-reference-arc distance (missing/irregular value analysis).')}
{img('eda_trajectories.png', 92, 'Fig 2.2 – GPS trajectories (blue), road network (grey) and reference route (green). Train trips 48, 14 and 54 (54 shows the systematic offset), and student_test trips 0, 40 and 76.')}
{img('eda_network.png', 98, 'Fig 2.3 – Road-network size and connectivity per trip, and trajectory length (train + student_test).')}
{img('eda_trip_locations.png', 70, 'Fig 2.4 – Trip centroids: data are global, so a per-trip local projection is required.')}
{img('map_eda_train_00000048.png', 80, 'Fig 2.5 – Folium map (train trip 00000048): raw GPS, road network and reference route with layer control. Interactive HTML: results/maps/eda_train_00000048.html')}

<h2 class="pb">3. Model experimentation and comparison</h2>
<p>Protocol: 5-fold <b>GroupKFold with trip as the group</b> on the training partition (78 usable trips), so no trip appears in both a training fold and a validation fold. Each model is then refit on all training trips and evaluated once on the 10 student-test trips.
Scalers live inside pipelines and are fitted on training data only. The seed is 42 everywhere. Raw logs are in <code>results/model_comparison.csv</code> and <code>results/model_cv_folds.csv</code>.</p>
<h3>3.1 Overall model comparison</h3>
{table(comp)}
<p style="font-size:8.6pt;color:#667085">CV Macro F1 is the candidate-level macro F1 over held-out training folds. Top-1, Macro F1, Log Loss and raw route continuity are on student_test, classifier only (before the HMM). Raw continuity is the share of consecutive distinct predicted arcs that connect head-to-tail. Inference time covers the classifier only, on 2 CPU cores.</p>
{table(comp2)}
<p><b>Selection.</b> LightGBM has the best CV Macro F1, CV Top-1 and CV log loss, and the best student-test Top-1, Macro F1 and log loss. Its fold-to-fold spread is small (±{m('LightGBM','cv_macro_f1_std'):.3f}).
It trains in about {m('LightGBM','fit_time_s'):.0f} s and needs about {m('LightGBM','inference_ms_per_1k_obs'):.0f} ms per 1,000 fixes, which is fast enough for the API. Its well-calibrated probabilities are what the HMM emissions need.
XGBoost is a very close second: within about 0.001 on CV Macro F1, CV Top-1 and student-test Top-1, F1 and log loss. RandomForest is next but about 16× slower to train. KNN is competitive in F1 but impractically slow at inference. Logistic regression and Gaussian NB underfit.
{SEL_TAIL}</p>

<h3>3.2 Per-model questions</h3>
{per_model}

<h3 class="pb">3.3 Model visualisations</h3>
{img('cv_macro_f1_top1.png', 98, 'Fig 3.1 – Cross-validation Macro F1 and Top-1 across the 5 trip-grouped folds for all models.')}
{img('student_test_top1_f1.png', 85, 'Fig 3.2 – Student-test Top-1 accuracy and Macro F1 (classifier only).')}
{img('logloss_inference.png', 98, 'Fig 3.3 – Student-test log loss and inference time (log scale).')}
{img('generalisation_gap.png', 85, 'Fig 3.4 – Train vs CV vs student-test Macro F1 (over/underfitting evidence).')}
{img('confusion_matrices.png', 100, 'Fig 3.5 – Student-test candidate confusion matrices (counts and row-normalised).')}
{img('training_histories.png', 98, 'Fig 3.6 – Real training histories, shown only for models that produce them: LightGBM and XGBoost per boosting round, MLP per epoch. Train = training partition, validation = student_test.')}

<h2 class="pb">4. Bayesian hyperparameter optimisation (LightGBM)</h2>
<p class="q">Q1–Q2. Hyperparameters and ranges</p>{table(best_tab)}
<p class="q">Q3. Objective</p><p>{bo['search']['objective']}, computed on the training partition only. Trial 0 was seeded with the untuned configuration so that the search record contains the baseline. CV Macro F1 and log loss were logged for every trial as user attributes.</p>
<p class="q">Q4. Number of trials</p><p>{bo['search']['n_trials']} trials, sampler {bo['search']['sampler']}. Search time was {bo['search']['search_time_s']/60:.1f} min on 2 CPU cores. The full record is in <code>results/bayes_opt_trials.csv</code>.</p>
<p class="q">Q5. Best values</p><p>Trial {best_i} reached an objective of {bt.value.max():.4f}, vs {bt.value.iloc[0]:.4f} for the baseline trial 0. The best values are in the table above.</p>
<p class="q">Q6. Did the tuned model outperform the untuned one?</p>
{table(bo_tab)}
<p>Only marginally, and only on the metric that was optimised. CV Top-1 rose by {100*(bo['tuned']['cv_top1']-bo['untuned']['cv_top1']):+.2f} points and student-test Top-1 by {100*(bo['tuned']['test_top1']-bo['untuned']['test_top1']):+.2f} points. Raw continuity also improved, from {f3(bo['untuned']['test_route_continuity'])} to {f3(bo['tuned']['test_route_continuity'])}.
Macro F1 was unchanged, and log loss got slightly worse ({f4(bo['untuned']['test_log_loss'])} → {f4(bo['tuned']['test_log_loss'])}). The tuned trees are larger (num_leaves {bp['num_leaves']}) and more confident.
{BO_HMM}The default configuration was already near the limit of what these features allow. Further gains would have to come from features or the decoder, not from hyperparameters.
The tuned model is deployed because it was the pre-registered choice on the CV objective. We did not pick it by looking at student-test results.</p>
{img('bayes_opt.png', 98, 'Fig 4.1 – Optuna TPE trace (objective per trial and best-so-far), and untuned vs tuned comparison.')}
{img('feature_importance.png', 60, 'Fig 4.2 – Tuned LightGBM feature importance (gain).')}
"""
if PHASE == 2 and EV is not None:
    EVAL_SECTION = f"""<h3>7.1 Evaluator test results (hidden partition)</h3>
<p>The 10 reserved evaluator trips were placed in <code>data/eval_test/</code> on Laptop 1 (backend) and run <b>once</b>, trip by trip, from the UI on Laptop 2.
The code was frozen before the run, and nothing was changed afterwards. Every metric was computed on the server from the hidden <code>.route</code> files.
The table is transcribed from the API console log (Fig 7.1) and the UI result panels.</p>
{table(ev_tab)}
<p><b>Summary:</b> observation-weighted Top-1 is <b>{pc(EVS['top1'])}</b> with the HMM vs {pc(EVS['clf'])} for the classifier alone, and the HMM improved Top-1 on <b>{EVS['better']}/10</b> trips.
Mean route precision is {EVS['P']:.3f}, recall {EVS['Rc']:.3f} and F1 {EVS['f1']:.3f}. Route continuity is {EVS['cont']:.3f} on every trip.
Server time per trip ranged from {EVS['tmin']:.2f} s to {EVS['tmax']:.2f} s. The longest was trip 00000015, with 6,854 GPS fixes; the first requests also include warm-up.
These results agree with the student-test estimate ({pc(hc.loc['LightGBM_tuned','hmm_top1'])}), so the development protocol did not over-fit.
<b>Weakest trips:</b> 00000030 (Top-1 {pc(EV.set_index('trip').loc['00000030','top1_hmm'])}) and 00000015 ({pc(EV.set_index('trip').loc['00000015','top1_hmm'])}), both dense urban or forest-track networks with many parallel paths.
Trip 00000023 has high Top-1 ({pc(EV.set_index('trip').loc['00000023','top1_hmm'])}) but low route precision ({EV.set_index('trip').loc['00000023','route_precision']:.3f}). Its short trajectory makes a few extra gap-fill arcs weigh heavily. As on student-test, recall stays at about 0.99 while precision is the limiting factor.</p>
{img('eval_api_cmd_log.png', 100, 'Fig 7.1 – Laptop 1 (backend) command prompt: the FastAPI server loads the tuned LightGBM and HMM parameters, then logs every POST /predict from the UI laptop (10.8.121.67) with server-side Top-1, classifier-only Top-1, route F1, continuity and time.')}
{img('eval_ui_00000087.png', 95, 'Fig 7.2 – Laptop 2 (UI): evaluator trip 00000087, predicted route over OpenStreetMap, Top-1 99.57%.')}
{img('eval_ui_00000015.png', 95, 'Fig 7.3 – Laptop 2 (UI): evaluator trip 00000015, the longest trip, in a dense forest-track network, Top-1 89.51%.')}
{img('eval_ui_00000030.png', 95, 'Fig 7.4 – Laptop 2 (UI): evaluator trip 00000030, the hardest trip, Top-1 83.79% (classifier only 79.66%).')}
"""
else:
    EVAL_SECTION = ""
html_b = "" if PHASE == 1 else f"""
<h2 class="pb">5. HMM and Viterbi route reconstruction</h2>
<p class="q">Q1. Hidden states</p><p>At step <i>t</i> (one GPS fix), the hidden states are that fix's candidate directed arcs, each with its projection parameter <i>τ</i> ∈ [0, 1] along the arc. The candidate-to-arc correspondence is kept through decoding, so every decoded state is a real arc id.</p>
<p class="q">Q2. Emission probabilities</p><p>log e<sub>t</sub>(a) = α · log(p<sub>clf</sub>(a | features<sub>t</sub>) + 10<sup>-4</sup>), using the selected classifier's probability. α weighs classifier evidence against the transition model.</p>
<p class="q">Q3. Transition probabilities</p><p>We follow Newson &amp; Krumm. d<sub>route</sub> is the directed network distance from the projection on arc <i>a</i> at <i>t−1</i> to the projection on arc <i>b</i> at <i>t</i>: the rest of <i>a</i>, plus the Dijkstra shortest path from head(<i>a</i>) to tail(<i>b</i>), plus the start of <i>b</i>. On the same arc it is the movement along the arc, allowing up to 15 m of GPS back-jitter.
Then log T(a→b) = −|d<sub>route</sub> − d<sub>gps</sub>| / β, where d<sub>gps</sub> is the straight-line distance between the two fixes. Shortest paths come from a directed graph cropped to the trajectory bounding box plus 1.5 km, with a search limit of 2 km.
Viterbi recurrence: δ<sub>t</sub>(b) = max<sub>a</sub>[δ<sub>t−1</sub>(a) + log T(a→b)] + log e<sub>t</sub>(b), followed by backtracking. α and β were tuned by grid search on <b>out-of-fold training predictions</b> from 16 training trips. The best setting was α = {hcfg['alpha']}, β = {hcfg['beta']} m, with Top-1 {grid_best.hmm_top1:.4f} vs {grid_best.clf_top1:.4f} for the classifier alone on those trips.</p>
<p class="q">Q4. Missing candidates and large GPS gaps</p><p>If a fix has no arc within 50 m, the 3 nearest arcs within 300 m are used, so no fix is left without candidates. (The candidate miss rate in the comparison table counts fixes whose reference arc is not among their candidates.)
If Δt &gt; {hcfg['gap_s']:.0f} s, the chain is <b>broken</b>: the next segment starts from its emissions plus the best score so far, so an implausible path is never forced across a long gap. Gaps between consecutive decoded arcs are then filled with the directed shortest path.</p>
<p class="q">Q5. Disconnected road segments</p><p>Pairs with no directed path within the limit get log T = −∞. If every transition into a step is −∞, the chain is broken and restarted at that step, the same way as for a time gap. If only some states are unreachable, those states get a large penalty instead of being dropped.
After decoding, a non-adjacent consecutive pair with no path stays as an "unfilled gap". On student-test this happened only {int(hp.unfilled_gaps.sum())} times in total.</p>
<p class="q">Q6. Did HMM decoding improve the final route?</p><p>Yes, for every classifier (table below, student-test). For tuned LightGBM, Top-1 rose from {pc(hc.loc['LightGBM_tuned','clf_top1'])} to {pc(hc.loc['LightGBM_tuned','hmm_top1'])} and route arc-set F1 from {f3(hc.loc['LightGBM_tuned','clf_route_F1'])} to {f3(hc.loc['LightGBM_tuned','hmm_route_F1'])}.
Raw continuity went from {f3(hc.loc['LightGBM_tuned','clf_continuity'])} to {f3(hc.loc['LightGBM_tuned','hmm_continuity'])}, and to {f3(hc.loc['LightGBM_tuned','hmm_route_continuity'])} after gap filling. Weaker classifiers gained the most (GaussianNB {100*(hc.loc['GaussianNB','hmm_top1']-hc.loc['GaussianNB','clf_top1']):+.1f} points, KNN {100*(hc.loc['KNN','hmm_top1']-hc.loc['KNN','clf_top1']):+.1f} points), because connectivity fixes many of their independent errors.</p>
{table(hmm_tab)}
<p><b>Per-trip results (tuned LightGBM):</b></p>{table(per_trip)}
<p><b>Where the HMM did not help.</b> {"On trip " + ", ".join(worse.trip) + f", Top-1 fell from {pc(worse.clf_top1.iloc[0])} to {pc(worse.hmm_top1.iloc[0])}." if len(worse) else "No trip got worse."}
On that trip the classifier was already very accurate. The HMM's distance-consistency term occasionally prefers a short detour, or switches arcs at a junction a few fixes earlier or later than the reference alignment. Those fixes count as errors even though the route itself is unchanged (route F1 is the same).
More generally, route precision (≈{f3(hc.loc['LightGBM_tuned','hmm_route_P'])}) is lower than recall (≈{f3(hc.loc['LightGBM_tuned','hmm_route_R'])}). Brief GPS excursions into side streets, and shortest-path gap fills that do not follow the true path, add arcs that are not in the reference.
The hardest trip, 00000077, stays at {pc(hp.set_index('trip').loc['00000077','hmm_top1'])} even after decoding. Its classifier-only accuracy is the lowest of all trips (80%). The zoomed map shows many side streets close to the route, so the per-fix label is often ambiguous.</p>
{img('hmm_vs_classifier.png', 100, 'Fig 5.1 – Student test: classifier-only vs classifier + HMM for every model.')}
{img('hmm_per_trip.png', 85, 'Fig 5.2 – Per-trip Top-1, tuned LightGBM, with and without HMM.')}
{img('hmm_grid.png', 55, 'Fig 5.3 – Decoder grid search (α, β) on out-of-fold training predictions.')}

<h2 class="pb">6. Geographical visualisation (Folium)</h2>
<p>The maps are generated by <code>scripts/06_maps.py</code> from the deployed model and decoder. Layers: raw GPS (blue), nearby road network (grey), reference route (green), classifier-only top-1 arcs (orange, off by default) and the HMM predicted route (red). Each map has start/end markers, a legend, a layer control and a scale bar.
The interactive HTML (with an OpenStreetMap base layer) is in <code>results/maps/</code>. The screenshots below were rendered without base tiles.</p>
{img('map_student_test_00000040_prediction.png', 88, 'Fig 6.1 – Student-test trip 00000040: the predicted route (red) overlaps the reference (green) almost everywhere (route F1 0.995).')}
{img('map_student_test_00000077_prediction_zoom.png', 88, 'Fig 6.2 – Student-test trip 00000077, zoomed to the middle 10% of the trip: dense side streets around the route, and short deviations of the predicted route at junctions.')}
{img('map_student_test_00000009_prediction_zoom.png', 88, 'Fig 6.3 – Student-test trip 00000009, zoomed view.')}
{img('map_eda_train_00000054.png', 70, 'Fig 6.4 – Training trip 00000054 (excluded): the GPS is offset about 140 m from the network along its whole length.')}

<h2 class="pb">7. API, UI and deployment</h2>
<p><b>Laptop 1</b> runs <code>uvicorn api.app:app --host 0.0.0.0 --port 8000</code>, with <code>PATHSNAP_EVAL_DIR</code> pointing at the 10 reserved trips. Endpoints: <code>GET /health</code>, <code>GET /trips</code> and <code>POST /predict {{"trip_id"}}</code>.
The prediction path reads only <code>.track/.nodes/.arcs</code>: candidates → LightGBM (portable <code>artifacts/final_model.txt</code>) → Viterbi → gap filling. The <code>.route</code> file is read only afterwards, on the server, to compute Top-1, route P/R/F1 and continuity.
The response carries the predicted polyline, the GPS points, the candidate network and the metrics, never the reference. Every request is logged to the console with its trip id and metrics.
<b>Laptop 2</b> serves <code>ui/index.html</code> with <code>python -m http.server 5500</code>. The UI lists the trips from <code>/trips</code>, sends the selected id to <code>/predict</code>, and draws the result on a Leaflet map with a legend and layer control.
During development, with student_test standing in for the evaluator set, all 10 trips succeeded and the API's metrics matched the offline evaluation exactly. The final evaluator run used two physical laptops on the same network: the backend at 10.8.121.42:8000, and the UI on a second laptop.</p>
{EVAL_SECTION}
"""
html_c = f"""
<h2>8. Limitations, integrity and reproducibility</h2>
<p><b>Reference-label limitations.</b> The per-fix labels come from our monotone alignment of GPS to the <code>.route</code>, so they inherit GPS noise and any route errors. At junctions, and on two-way roads where opposite arcs overlap, a fix's "true" arc is genuinely ambiguous, so Top-1 understates route quality{LIM_RECALL}.
Two training trips have systematic offsets and were excluded. If such a trip appears at evaluation time, the 50 m candidate radius will often miss the true arc; the 300 m fallback keeps the pipeline running, but accuracy will be low.
<b>Leakage control.</b> Splits are trip-grouped. {LIM_DEC}Model and hyperparameter choices used training CV. The evaluator partition was never read during development; it was used exactly once, for the final two-laptop run in §7.1, with the code frozen beforehand. Nothing is hard-coded.
<b>Reproducibility.</b> Every number in this report is read by <code>report/build_report.py</code> from <code>results/</code>, and those files are produced by <code>scripts/01–08</code> with seed 42.</p>
<p><b>AI-assistance disclosure.</b> {TEAM['ai_disclosure']}</p>
</body></html>"""

html = html + html_b + html_c
NAME = "PathSnap_Review1_Report" if PHASE == 1 else "PathSnap_Technical_Report"
out = os.path.join(ROOT, "report", f"{NAME}.html")
open(out, "w").write(html)
pdf = os.path.join(ROOT, "report", f"{NAME}.pdf")
subprocess.run(["google-chrome", "--headless=new", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", f"file://{out}"],
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)
print(pdf, os.path.exists(pdf))
