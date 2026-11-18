"""Comparison plots from recorded experiment results (no synthetic values)."""
import os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use("grayscale"); plt.rcParams.update({"font.family": "serif", "axes.prop_cycle": plt.cycler(color=["0.0", "0.55", "0.3", "0.75"])})
import _path

R, F = _path.RES, _path.FIG
mc = pd.read_csv(f"{R}/model_comparison.csv")
folds = pd.read_csv(f"{R}/model_cv_folds.csv")
names = mc.model.tolist(); x = np.arange(len(names))
C = [str(v) for v in np.linspace(0.1, 0.7, len(names))]

# 1. CV macro F1 across models (box over folds + mean)
fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
ax[0].boxplot([folds[folds.model == n].macro_f1 for n in names], tick_labels=names)
ax[0].set(title="5-fold GroupKFold CV — candidate Macro F1", ylabel="Macro F1"); ax[0].tick_params(axis="x", rotation=30)
ax[1].boxplot([folds[folds.model == n].top1_acc for n in names], tick_labels=names)
ax[1].set(title="5-fold GroupKFold CV — observation Top-1 accuracy", ylabel="Top-1"); ax[1].tick_params(axis="x", rotation=30)
plt.tight_layout(); plt.savefig(f"{F}/cv_macro_f1_top1.png", dpi=130); plt.close()

# 2. student test top1 + macro F1
fig, ax = plt.subplots(figsize=(11, 4.5)); w = .38
b1 = ax.bar(x - w / 2, mc.test_top1, w, label="Top-1 accuracy (observation)")
b2 = ax.bar(x + w / 2, mc.test_macro_f1, w, label="Macro F1 (candidate)")
ax.bar_label(b1, fmt="%.3f", fontsize=8); ax.bar_label(b2, fmt="%.3f", fontsize=8)
ax.set_xticks(x, names, rotation=20); ax.set_ylim(0.75, 1.0); ax.legend(loc="lower right"); ax.set_title("Student-test performance (classifier only)")
plt.tight_layout(); plt.savefig(f"{F}/student_test_top1_f1.png", dpi=130); plt.close()

# 3. log loss + inference time
fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
b = ax[0].bar(names, mc.test_log_loss, color=C); ax[0].bar_label(b, fmt="%.3f", fontsize=8)
ax[0].set(title="Student-test log loss (lower is better)"); ax[0].tick_params(axis="x", rotation=30)
b = ax[1].bar(names, mc.inference_ms_per_1k_obs, color=C); ax[1].bar_label(b, fmt="%.1f", fontsize=8)
ax[1].set(title="Inference time (ms per 1000 GPS observations)", yscale="log"); ax[1].tick_params(axis="x", rotation=30)
plt.tight_layout(); plt.savefig(f"{F}/logloss_inference.png", dpi=130); plt.close()

# 4. confusion matrices
fig, axs = plt.subplots(1, len(names), figsize=(3 * len(names), 3))
for ax, (_, r) in zip(axs, mc.iterrows()):
    cm = np.array([[r.tn, r.fp], [r.fn, r.tp]]); cmn = cm / cm.sum(1, keepdims=True)
    ax.imshow(cmn, cmap="Greys", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]}\n{cmn[i,j]:.2f}", ha="center", va="center", fontsize=8, color="white" if cmn[i, j] > .5 else "black")
    ax.set(title=r.model, xticks=[0, 1], yticks=[0, 1], xticklabels=["pred 0", "pred 1"], yticklabels=["true 0", "true 1"])
plt.suptitle("Student-test candidate confusion matrices (threshold 0.5)"); plt.tight_layout(); plt.savefig(f"{F}/confusion_matrices.png", dpi=120); plt.close()

# 5. training histories (only models that produce them)
hs = [n for n in ["LightGBM", "XGBoost", "MLP"] if os.path.exists(f"{R}/history_{n}.csv")]
fig, axs = plt.subplots(1, len(hs), figsize=(5 * len(hs), 4))
for ax, n in zip(np.atleast_1d(axs), hs):
    h = pd.read_csv(f"{R}/history_{n}.csv"); xc = h.columns[0]
    ax.plot(h[xc], h.train_loss, label="train (training partition)"); ax.plot(h[xc], h.val_loss, label="validation (student_test)")
    ax.set(title=f"{n} log-loss history", xlabel=xc, ylabel="log loss"); ax.legend()
plt.tight_layout(); plt.savefig(f"{F}/training_histories.png", dpi=130); plt.close()

# 6. train vs CV vs test macro F1 (over/underfitting)
fig, ax = plt.subplots(figsize=(11, 4.5)); w = .27
for k, (col, lab) in enumerate([("train_macro_f1", "Train (fit data)"), ("cv_macro_f1", "CV (held-out trips)"), ("test_macro_f1", "Student test")]):
    ax.bar(x + (k - 1) * w, mc[col], w, label=lab)
ax.set_xticks(x, names, rotation=20); ax.set_ylim(0.8, 1.0); ax.legend(); ax.set_title("Macro F1: train vs CV vs student-test (generalisation gap)")
plt.tight_layout(); plt.savefig(f"{F}/generalisation_gap.png", dpi=130); plt.close()

# 7. Bayesian optimisation
if os.path.exists(f"{R}/bayes_opt_trials.csv"):
    t = pd.read_csv(f"{R}/bayes_opt_trials.csv")
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    ax[0].plot(t.number, t.value, "o", label="trial"); ax[0].plot(t.number, t.value.cummax(), "-", label="best so far")
    ax[0].set(title="Optuna TPE: CV Top-1 per trial", xlabel="trial", ylabel="3-fold CV Top-1"); ax[0].legend()
    s = json.load(open(f"{R}/bayes_opt_summary.json"))
    keys = ["cv_macro_f1", "test_top1", "test_macro_f1", "test_log_loss"]
    xx = np.arange(len(keys))
    ax[1].bar(xx - .2, [s["untuned"][k] for k in keys], .4, label="untuned"); ax[1].bar(xx + .2, [s["tuned"][k] for k in keys], .4, label="tuned")
    for i, k in enumerate(keys):
        ax[1].text(i - .2, s["untuned"][k], f'{s["untuned"][k]:.4f}', ha="center", va="bottom", fontsize=8)
        ax[1].text(i + .2, s["tuned"][k], f'{s["tuned"][k]:.4f}', ha="center", va="bottom", fontsize=8)
    ax[1].set_xticks(xx, keys); ax[1].set_title("LightGBM untuned vs tuned"); ax[1].legend()
    plt.tight_layout(); plt.savefig(f"{F}/bayes_opt.png", dpi=130); plt.close()
    fi = pd.read_csv(f"{R}/feature_importance.csv")
    fig, ax = plt.subplots(figsize=(8, 5)); ax.barh(fi.feature[::-1], fi.gain[::-1]); ax.set_title("Tuned LightGBM feature importance (gain)")
    plt.tight_layout(); plt.savefig(f"{F}/feature_importance.png", dpi=130); plt.close()

# 8. HMM
if os.path.exists(f"{R}/hmm_comparison_student_test.csv"):
    h = pd.read_csv(f"{R}/hmm_comparison_student_test.csv")
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5)); xx = np.arange(len(h))
    for a_, (c1, c2, ttl) in zip(ax, [("clf_top1", "hmm_top1", "Observation Top-1"), ("clf_continuity", "hmm_continuity", "Route continuity (raw arc sequence)"),
                                      ("clf_route_F1", "hmm_route_F1", "Route arc-set F1")]):
        a_.bar(xx - .2, h[c1], .4, label="classifier only"); a_.bar(xx + .2, h[c2], .4, label="classifier + HMM")
        a_.set_xticks(xx, h.model, rotation=30, fontsize=8); a_.set_title(ttl); a_.legend(fontsize=8); a_.set_ylim(max(0, min(h[c1].min(), h[c2].min()) - .05), 1.0)
    plt.suptitle("Student test: classifier-only vs classifier + HMM/Viterbi"); plt.tight_layout(); plt.savefig(f"{F}/hmm_vs_classifier.png", dpi=130); plt.close()
    g = pd.read_csv(f"{R}/hmm_decoder_grid_train.csv")
    pv = g.pivot(index="alpha", columns="beta", values="hmm_top1")
    fig, ax = plt.subplots(figsize=(6, 4)); im = ax.imshow(pv.values, cmap="Greys")
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            ax.text(j, i, f"{pv.values[i,j]:.4f}", ha="center", va="center", color=("w" if pv.values[i,j] > pv.values.mean() else "k"), fontsize=8)
    ax.set(xticks=range(pv.shape[1]), xticklabels=pv.columns, yticks=range(pv.shape[0]), yticklabels=pv.index, xlabel="beta (m)", ylabel="alpha",
           title="HMM decoder grid (OOF train trips): Top-1"); plt.colorbar(im); plt.tight_layout(); plt.savefig(f"{F}/hmm_grid.png", dpi=130); plt.close()
    per = pd.read_csv(f"{R}/hmm_student_test_LightGBM_tuned.csv")
    fig, ax = plt.subplots(figsize=(11, 4)); xx = np.arange(len(per))
    ax.bar(xx - .2, per.clf_top1, .4, label="classifier only"); ax.bar(xx + .2, per.hmm_top1, .4, label="classifier + HMM")
    ax.set_xticks(xx, per.trip, rotation=30); ax.set_ylim(.7, 1); ax.legend(); ax.set_title("Per-trip Top-1, tuned LightGBM, student test")
    plt.tight_layout(); plt.savefig(f"{F}/hmm_per_trip.png", dpi=130); plt.close()
print("plots done")
