"""Evaluation metrics for candidate classification and route matching."""
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, log_loss


def top1(cand: pd.DataFrame, obs: pd.DataFrame, score="p"):
    """Observation-level top-1 arc per observation. Observations with no candidates -> arc -1."""
    idx = cand.groupby(["trip", "obs"])[score].idxmax()
    best = cand.loc[idx, ["trip", "obs", "arc"]].rename(columns={"arc": "pred_arc"})
    o = obs.merge(best, on=["trip", "obs"], how="left")
    o["pred_arc"] = o["pred_arc"].fillna(-1).astype(int)
    return o


def continuity(arc_seq, arcs):
    """Fraction of consecutive (de-duplicated) predicted arc pairs that are connected head-to-tail."""
    s = [a for a in arc_seq if a >= 0]
    s = [a for i, a in enumerate(s) if i == 0 or a != s[i - 1]]
    if len(s) < 2:
        return 1.0
    s = np.array(s)
    return float(np.mean(arcs[s[:-1], 1] == arcs[s[1:], 0]))


def route_prf(pred_route, ref_route):
    P, R = set(int(a) for a in pred_route if a >= 0), set(int(a) for a in ref_route)
    tp = len(P & R)
    p = tp / max(len(P), 1); r = tp / max(len(R), 1)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def candidate_metrics(y, p):
    yhat = (p >= 0.5).astype(int)
    return dict(macro_f1=f1_score(y, yhat, average="macro"),
                macro_precision=precision_score(y, yhat, average="macro", zero_division=0),
                macro_recall=recall_score(y, yhat, average="macro", zero_division=0),
                log_loss=log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]))


def full_eval(cand, obs, arcs_by_trip, score="p"):
    m = candidate_metrics(cand["y"].values, cand[score].values)
    o = top1(cand, obs, score)
    m["top1_acc"] = float((o.pred_arc == o.label_arc).mean())
    m["candidate_miss_rate"] = float(1 - obs.has_cand.mean())
    m["route_continuity"] = float(np.mean([continuity(g.sort_values("obs").pred_arc.values, arcs_by_trip[t])
                                           for t, g in o.groupby("trip")]))
    return m, o
