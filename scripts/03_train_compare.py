"""Train and compare candidate classifiers (>=5 algorithm families).
CV: 5-fold GroupKFold over the 80 training trips (trip = group). Dev test: 10 student_test trips."""
import os, time, json, warnings
import numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, log_loss
import lightgbm as lgb, xgboost as xgb
import _path
from pathsnap.core import FEATURES, load_trip, list_trips
from pathsnap.metrics import full_eval

warnings.filterwarnings("ignore")
SEED = 42
SUB = 3  # use every 3rd GPS observation for training (1 Hz points are highly autocorrelated)
D = os.path.join(_path.RES, "data")
facts = json.load(open(os.path.join(_path.RES, "eda_facts.json")))
ANOM = facts["anomalous_trips"]


def load_train():
    c = pd.read_parquet(f"{D}/cand_train.parquet"); o = pd.read_parquet(f"{D}/obs_train.parquet")
    o = o[~o.trip.isin(ANOM) & (o.has_cand == 1) & (o.obs % SUB == 0)]
    c = c.merge(o[["trip", "obs"]], on=["trip", "obs"])
    return c.reset_index(drop=True), o.reset_index(drop=True)


def arcs_for(part):
    return {t: np.loadtxt(os.path.join(_path.DATA, part, t, f"{t}.arcs"), dtype=int) for t in list_trips(os.path.join(_path.DATA, part))}


class MLPEpochs:
    """sklearn MLP trained epoch-by-epoch with partial_fit so train/validation loss curves are recorded."""
    def __init__(self, hidden=(64, 32), alpha=1e-4, lr=1e-3, epochs=25, batch=512):
        self.kw = dict(hidden=hidden, alpha=alpha, lr=lr, epochs=epochs, batch=batch)
        self.hidden, self.alpha, self.lr, self.epochs, self.batch = hidden, alpha, lr, epochs, batch
        self.history = []
    def get_params(self, deep=False): return dict(self.kw)
    def fit(self, X, y, Xv=None, yv=None):
        self.sc = StandardScaler().fit(X); Xs = self.sc.transform(X)
        self.m = MLPClassifier(self.hidden, alpha=self.alpha, learning_rate_init=self.lr, batch_size=self.batch, random_state=SEED)
        rng = np.random.RandomState(SEED); self.history = []
        for e in range(self.epochs):
            perm = rng.permutation(len(Xs))
            for s in range(0, len(Xs), 20000):
                b = perm[s:s + 20000]; self.m.partial_fit(Xs[b], y[b], classes=[0, 1])
            h = dict(epoch=e + 1, train_loss=log_loss(y, self.m.predict_proba(Xs)[:, 1]))
            if Xv is not None: h["val_loss"] = log_loss(yv, self.predict_proba(Xv)[:, 1])
            self.history.append(h)
        return self
    def predict_proba(self, X): return self.m.predict_proba(self.sc.transform(X))


def make_models():
    return {
        "LogisticRegression": ("Linear / probabilistic", "C=1.0, L2, StandardScaler",
                               lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))),
        "GaussianNB": ("Probabilistic (generative)", "var_smoothing=1e-9", lambda: GaussianNB()),
        "KNN": ("Instance-based", "k=31, distance-weighted, StandardScaler",
                lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(31, weights="distance", n_jobs=2))),
        "RandomForest": ("Bagging ensemble", "200 trees, min_samples_leaf=5, max_features=sqrt",
                         lambda: RandomForestClassifier(200, min_samples_leaf=5, n_jobs=2, random_state=SEED)),
        "LightGBM": ("Gradient boosting", "300 rounds, lr=0.05, num_leaves=31",
                     lambda: lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=SEED, n_jobs=2, verbose=-1)),
        "XGBoost": ("Gradient boosting", "300 rounds, lr=0.1, max_depth=6, hist",
                    lambda: xgb.XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6, tree_method="hist", n_jobs=2, random_state=SEED, eval_metric="logloss")),
        "MLP": ("Neural network", "64-32 ReLU, Adam lr=1e-3, alpha=1e-4, 25 epochs, StandardScaler",
                lambda: MLPEpochs()),
    }


def cv_eval(make, c, o, arcs_tr, n_splits=5):
    gkf = GroupKFold(n_splits)
    trips = o.trip.values
    res = []
    utrips = np.array(sorted(o.trip.unique()))
    for k, (tri, vai) in enumerate(gkf.split(utrips, groups=utrips)):
        vt = set(utrips[vai])
        mtr = ~c.trip.isin(vt).values
        m = make().fit(c.loc[mtr, FEATURES].values, c.loc[mtr, "y"].values)
        cv = c.loc[~mtr].copy(); cv["p"] = m.predict_proba(cv[FEATURES].values)[:, 1]
        met, _ = full_eval(cv, o[o.trip.isin(vt)], arcs_tr)
        met["fold"] = k; res.append(met)
    return pd.DataFrame(res)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Train/compare models. Use --models to split models across team laptops.")
    ap.add_argument("--models", default="all", help="comma-separated subset, e.g. LightGBM,XGBoost (default: all 7)")
    a = ap.parse_args()
    ALL = list(make_models())
    sel = ALL if a.models == "all" else [x.strip() for x in a.models.split(",")]
    bad = [x for x in sel if x not in ALL]
    if bad:
        raise SystemExit(f"unknown models {bad}; choose from {ALL}")
    tag = "" if a.models == "all" else "_" + "-".join(sel)
    c, o = load_train()
    ct = pd.read_parquet(f"{D}/cand_student_test.parquet"); ot = pd.read_parquet(f"{D}/obs_student_test.parquet")
    arcs_tr, arcs_te = arcs_for("train"), arcs_for("student_test")
    print("train rows", c.shape, "obs", len(o), "pos rate", c.y.mean())
    X, y = c[FEATURES].values, c["y"].values
    rows, folds = [], []
    os.makedirs(os.path.join(_path.ART, "models"), exist_ok=True)
    os.makedirs(os.path.join(D, "preds"), exist_ok=True)
    for name, (fam, cfg, make) in make_models().items():
        if name not in sel:
            continue
        t0 = time.time(); cvr = cv_eval(make, c, o, arcs_tr); t_cv = time.time() - t0
        cvr["model"] = name; folds.append(cvr)
        t0 = time.time(); m = make()
        if name == "MLP":
            m.fit(X, y, ct[FEATURES].values, ct["y"].values)
            pd.DataFrame(m.history).to_csv(os.path.join(_path.RES, "history_MLP.csv"), index=False)
        elif name in ("LightGBM", "XGBoost"):
            if name == "LightGBM":
                m.fit(X, y, eval_set=[(X, y), (ct[FEATURES].values, ct["y"].values)], eval_metric="binary_logloss")
                ev = m.evals_result_; h = pd.DataFrame({"iteration": np.arange(1, 301), "train_loss": ev["training"]["binary_logloss"], "val_loss": ev["valid_1"]["binary_logloss"]})
            else:
                m.fit(X, y, eval_set=[(X, y), (ct[FEATURES].values, ct["y"].values)], verbose=False)
                ev = m.evals_result(); h = pd.DataFrame({"iteration": np.arange(1, 301), "train_loss": ev["validation_0"]["logloss"], "val_loss": ev["validation_1"]["logloss"]})
            h.to_csv(os.path.join(_path.RES, f"history_{name}.csv"), index=False)
        else:
            m.fit(X, y)
        t_fit = time.time() - t0
        t0 = time.time(); p = m.predict_proba(ct[FEATURES].values)[:, 1]; t_inf = time.time() - t0
        ct2 = ct.copy(); ct2["p"] = p
        met, otop = full_eval(ct2, ot, arcs_te)
        ct2[["trip", "obs", "arc", "y", "p"]].to_parquet(os.path.join(D, "preds", f"student_test_{name}.parquet"))
        cm = confusion_matrix(ct2.y, (p >= .5).astype(int))
        joblib.dump(m, os.path.join(_path.ART, "models", f"{name}.joblib"), compress=3)
        # training-fit score (for over/underfitting evidence)
        idx = np.random.RandomState(SEED).choice(len(c), min(150000, len(c)), replace=False)
        ptr = m.predict_proba(X[idx])[:, 1]
        from pathsnap.metrics import candidate_metrics
        trm = candidate_metrics(y[idx], ptr)
        row = dict(model=name, family=fam, config=cfg,
                   cv_macro_f1=cvr.macro_f1.mean(), cv_macro_f1_std=cvr.macro_f1.std(), cv_top1=cvr.top1_acc.mean(), cv_top1_std=cvr.top1_acc.std(),
                   cv_log_loss=cvr.log_loss.mean(), train_macro_f1=trm["macro_f1"], train_log_loss=trm["log_loss"],
                   test_top1=met["top1_acc"], test_macro_f1=met["macro_f1"], test_precision=met["macro_precision"], test_recall=met["macro_recall"],
                   test_log_loss=met["log_loss"], test_route_continuity=met["route_continuity"], test_candidate_miss=met["candidate_miss_rate"],
                   fit_time_s=t_fit, cv_time_s=t_cv, inference_ms_per_1k_obs=1000 * t_inf / len(ot) * 1000,
                   tn=cm[0, 0], fp=cm[0, 1], fn=cm[1, 0], tp=cm[1, 1])
        rows.append(row); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(_path.RES, f"model_comparison{tag}.csv"), index=False)
        pd.concat(folds).to_csv(os.path.join(_path.RES, f"model_cv_folds{tag}.csv"), index=False)
