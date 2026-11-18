"""HMM/Viterbi decoder: tune (alpha, beta) on out-of-fold TRAIN predictions, then compare classifier-only vs
classifier+HMM on the student_test partition, for the final model and for every compared model."""
import os, time, json, itertools
import numpy as np, pandas as pd
import _path
from pathsnap.core import load_trip, list_trips
from pathsnap.hmm import viterbi, RoadGraph, reconstruct_route, DEFAULTS
from pathsnap.metrics import continuity, route_prf

D = os.path.join(_path.RES, "data")
ANOM = json.load(open(os.path.join(_path.RES, "eda_facts.json")))["anomalous_trips"]


class LazyGraphs(dict):
    def __init__(self, part): super().__init__(); self.part = part
    def __missing__(self, tid):
        tr = load_trip(os.path.join(_path.DATA, self.part, tid)); v = (tr, RoadGraph(tr)); self[tid] = v; return v


def evaluate(part, cand, obs, params, trips, graphs):
    rows = []
    for tid in trips:
        tr, g = graphs[tid]
        cc = cand[cand.trip == tid]
        lab = obs[obs.trip == tid].set_index("obs").label_arc
        top = cc.loc[cc.groupby("obs").p.idxmax()].set_index("obs").arc.reindex(lab.index).fillna(-1).astype(int)
        t0 = time.time()
        ob, path, br = viterbi(tr, cc[["obs", "arc", "p", "proj_t"]], params, g)
        dec_t = time.time() - t0
        hm = pd.Series(path, ob).reindex(lab.index).fillna(-1).astype(int)
        r_clf, unf_c = reconstruct_route(tr, top.values, g)
        r_hmm, unf_h = reconstruct_route(tr, hm.values, g)
        pc, rc, fc = route_prf(top.values, tr.route)
        ph, rh, fh = route_prf(r_hmm, tr.route)
        rows.append(dict(trip=tid, n_obs=len(lab), clf_top1=(top == lab).mean(), hmm_top1=(hm == lab).mean(),
                         clf_continuity=continuity(top.values, tr.arcs), hmm_continuity=continuity(hm.values, tr.arcs),
                         hmm_route_continuity=continuity(r_hmm, tr.arcs), clf_route_P=pc, clf_route_R=rc, clf_route_F1=fc,
                         hmm_route_P=ph, hmm_route_R=rh, hmm_route_F1=fh, hmm_breaks=br, unfilled_gaps=unf_h,
                         decode_s=dec_t))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # ---------- 1. decoder tuning on OOF train predictions
    oof = pd.read_parquet(f"{D}/preds/train_oof_LightGBM_tuned.parquet")
    otr = pd.read_parquet(f"{D}/obs_train.parquet")
    pool = [t for t in sorted(oof.trip.unique()) if t not in ANOM]
    rng = np.random.RandomState(42)
    sizes = otr.groupby("trip").size()
    pool = [t for t in pool if sizes[t] <= 4000]
    tune_trips = sorted(rng.choice(pool, 16, replace=False))
    gtr = LazyGraphs("train")
    grid = []
    for a, b in itertools.product([0.5, 1.0, 2.0], [2.0, 5.0, 10.0, 25.0]):
        r = evaluate("train", oof, otr, dict(alpha=a, beta=b), tune_trips, gtr)
        grid.append(dict(alpha=a, beta=b, hmm_top1=r.hmm_top1.mean(), clf_top1=r.clf_top1.mean(),
                         hmm_route_F1=r.hmm_route_F1.mean(), hmm_continuity=r.hmm_continuity.mean()))
        print(grid[-1], flush=True)
    grid = pd.DataFrame(grid); grid.to_csv(os.path.join(_path.RES, "hmm_decoder_grid_train.csv"), index=False)
    bst = grid.sort_values("hmm_top1", ascending=False).iloc[0]
    params = dict(DEFAULTS); params.update(alpha=float(bst.alpha), beta=float(bst.beta))
    json.dump(dict(decoder=params, tuned_on=f"OOF LightGBM-tuned predictions, {len(tune_trips)} training trips",
                   tune_trips=list(map(str, tune_trips))), open(os.path.join(_path.ART, "hmm_config.json"), "w"), indent=2)

    # ---------- 2. student_test comparison
    ct = pd.read_parquet(f"{D}/cand_student_test.parquet"); ot = pd.read_parquet(f"{D}/obs_student_test.parquet")
    trips = list_trips(os.path.join(_path.DATA, "student_test"))
    gte = LazyGraphs("student_test")
    summ = []
    for f in sorted(os.listdir(f"{D}/preds")):
        if not f.startswith("student_test_"):
            continue
        name = f[len("student_test_"):-8]
        p = pd.read_parquet(f"{D}/preds/{f}")
        cc = ct.copy(); cc["p"] = p.p.values
        r = evaluate("student_test", cc, ot, params, trips, gte)
        r.to_csv(os.path.join(_path.RES, f"hmm_student_test_{name}.csv"), index=False)
        w = r.n_obs / r.n_obs.sum()
        summ.append(dict(model=name, clf_top1=(r.clf_top1 * w).sum(), hmm_top1=(r.hmm_top1 * w).sum(),
                         clf_continuity=r.clf_continuity.mean(), hmm_continuity=r.hmm_continuity.mean(),
                         hmm_route_continuity=r.hmm_route_continuity.mean(), clf_route_F1=r.clf_route_F1.mean(),
                         hmm_route_F1=r.hmm_route_F1.mean(), hmm_route_P=r.hmm_route_P.mean(), hmm_route_R=r.hmm_route_R.mean(),
                         breaks=r.hmm_breaks.sum(), decode_s_per_trip=r.decode_s.mean()))
        print(summ[-1], flush=True)
    pd.DataFrame(summ).to_csv(os.path.join(_path.RES, "hmm_comparison_student_test.csv"), index=False)
