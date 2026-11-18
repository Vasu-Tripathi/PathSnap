"""Bayesian hyperparameter optimisation (Optuna TPE) of the selected model (LightGBM).
Objective: mean observation-level Top-1 accuracy over 3-fold GroupKFold (trips as groups) on the training partition.
Then: refit tuned model, 5-fold CV + student_test evaluation (same protocol as 03), out-of-fold
predictions on ALL training observations (used only to tune the HMM decoder), export final artifacts."""
import os, time, json, warnings
import numpy as np, pandas as pd, joblib, optuna
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
import _path
from pathsnap.core import FEATURES
from pathsnap.metrics import full_eval, candidate_metrics
import importlib; tc = importlib.import_module("03_train_compare")

warnings.filterwarnings("ignore")
SEED, N_TRIALS = 42, int(os.environ.get("N_TRIALS", 30))
D = os.path.join(_path.RES, "data")
c, o = tc.load_train()
ct = pd.read_parquet(f"{D}/cand_student_test.parquet"); ot = pd.read_parquet(f"{D}/obs_student_test.parquet")
arcs_tr, arcs_te = tc.arcs_for("train"), tc.arcs_for("student_test")
utrips = np.array(sorted(o.trip.unique()))


def make(params):
    return lambda: lgb.LGBMClassifier(**params, bagging_freq=1, random_state=SEED, n_jobs=2, verbose=-1)


def objective(trial):
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 800, step=50),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 300, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
    )
    t0 = time.time()
    cvr = tc.cv_eval(make(params), c, o, arcs_tr, n_splits=3)
    trial.set_user_attr("cv_macro_f1", float(cvr.macro_f1.mean()))
    trial.set_user_attr("cv_log_loss", float(cvr.log_loss.mean()))
    trial.set_user_attr("seconds", time.time() - t0)
    return float(cvr.top1_acc.mean())


if __name__ == "__main__":
    t0 = time.time()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=8),
                                study_name="lgbm_tpe")
    # trial 0 = untuned baseline configuration, so the search record contains it
    study.enqueue_trial(dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=20,
                             subsample=1.0, colsample_bytree=1.0, reg_lambda=1e-3, reg_alpha=1e-3))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    search_time = time.time() - t0
    df = study.trials_dataframe(); df.to_csv(os.path.join(_path.RES, "bayes_opt_trials.csv"), index=False)
    best = study.best_params
    print("best", best, study.best_value)

    out = {}
    base = dict(n_estimators=300, learning_rate=0.05, num_leaves=31)
    for tag, params in [("untuned", base), ("tuned", best)]:
        mk = (lambda: lgb.LGBMClassifier(**base, random_state=SEED, n_jobs=2, verbose=-1)) if tag == "untuned" else make(best)
        cvr = tc.cv_eval(mk, c, o, arcs_tr, n_splits=5)
        t1 = time.time(); m = mk().fit(c[FEATURES].values, c["y"].values); fit_t = time.time() - t1
        t1 = time.time(); p = m.predict_proba(ct[FEATURES].values)[:, 1]; inf_t = time.time() - t1
        ct2 = ct.copy(); ct2["p"] = p
        met, _ = full_eval(ct2, ot, arcs_te)
        out[tag] = dict(params=params, cv_macro_f1=cvr.macro_f1.mean(), cv_macro_f1_std=cvr.macro_f1.std(), cv_top1=cvr.top1_acc.mean(),
                        cv_log_loss=cvr.log_loss.mean(), test_top1=met["top1_acc"], test_macro_f1=met["macro_f1"],
                        test_precision=met["macro_precision"], test_recall=met["macro_recall"], test_log_loss=met["log_loss"],
                        test_route_continuity=met["route_continuity"], fit_time_s=fit_t,
                        inference_ms_per_1k_obs=1000 * inf_t / len(ot) * 1000)
        print(tag, out[tag], flush=True)
        if tag == "tuned":
            final = m
            ct2[["trip", "obs", "arc", "y", "p"]].to_parquet(os.path.join(D, "preds", "student_test_LightGBM_tuned.parquet"))
            imp = pd.DataFrame({"feature": FEATURES, "gain": m.booster_.feature_importance("gain")}).sort_values("gain", ascending=False)
            imp.to_csv(os.path.join(_path.RES, "feature_importance.csv"), index=False)
    out["search"] = dict(n_trials=N_TRIALS, sampler="TPE (seed 42, 8 random startup trials)", objective="mean 3-fold GroupKFold observation Top-1 accuracy",
                         best_value=study.best_value, search_time_s=search_time)
    json.dump(out, open(os.path.join(_path.RES, "bayes_opt_summary.json"), "w"), indent=2, default=float)

    # out-of-fold predictions on all (non-anomalous) training observations, for HMM decoder tuning
    call = pd.read_parquet(f"{D}/cand_train.parquet"); call = call[~call.trip.isin(tc.ANOM)].reset_index(drop=True)
    call["p"] = np.nan
    for tri, vai in GroupKFold(5).split(utrips, groups=utrips):
        vt = set(utrips[vai])
        m = make(best)().fit(c.loc[~c.trip.isin(vt), FEATURES].values, c.loc[~c.trip.isin(vt), "y"].values)
        msk = call.trip.isin(vt).values
        call.loc[msk, "p"] = m.predict_proba(call.loc[msk, FEATURES].values)[:, 1]
    call[["trip", "obs", "arc", "y", "p", "proj_t"]].to_parquet(os.path.join(D, "preds", "train_oof_LightGBM_tuned.parquet"))
    joblib.dump(final, os.path.join(_path.ART, "final_model.joblib"))
    print("done")
