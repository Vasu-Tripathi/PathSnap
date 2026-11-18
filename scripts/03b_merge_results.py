"""Merge per-member model results (results/model_comparison_<models>.csv from `03_train_compare.py --models ...`)
into results/model_comparison.csv and results/model_cv_folds.csv, in a fixed model order."""
import os, glob, importlib
import pandas as pd
import _path

ORDER = list(importlib.import_module("03_train_compare").make_models())
comp = [pd.read_csv(f) for f in glob.glob(os.path.join(_path.RES, "model_comparison_*.csv"))]
fold = [pd.read_csv(f) for f in glob.glob(os.path.join(_path.RES, "model_cv_folds_*.csv"))]
if not comp:
    raise SystemExit("no per-member result files found (results/model_comparison_*.csv)")
comp = pd.concat(comp).drop_duplicates("model", keep="last")
missing = [m for m in ORDER if m not in set(comp.model)]
if missing:
    print("WARNING: models not yet trained by anyone:", missing)
comp["_o"] = comp.model.map({m: i for i, m in enumerate(ORDER)})
comp.sort_values("_o").drop(columns="_o").to_csv(os.path.join(_path.RES, "model_comparison.csv"), index=False)
fold = pd.concat(fold).drop_duplicates(["model", "fold"], keep="last")
fold["_o"] = fold.model.map({m: i for i, m in enumerate(ORDER)})
fold.sort_values(["_o", "fold"]).drop(columns="_o").to_csv(os.path.join(_path.RES, "model_cv_folds.csv"), index=False)
print(comp.sort_values("_o")[["model", "cv_macro_f1", "test_top1", "test_macro_f1", "test_log_loss"]].round(4).to_string(index=False))
