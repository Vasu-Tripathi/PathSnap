"""PathSnap FastAPI backend (Laptop 1).

Run:  uvicorn api.app:app --host 0.0.0.0 --port 8000
Env:  PATHSNAP_EVAL_DIR  folder containing the evaluation trip folders (default: data/eval_test)

The prediction pipeline only reads .track/.nodes/.arcs. The .route file (reference) is read separately,
after prediction, for server-side scoring; it is never returned to the client.
"""
import os, sys, json, time, logging
import numpy as np, pandas as pd, joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from pathsnap.core import load_trip, list_trips, build_candidates, align_route, FEATURES, _read  # noqa: E402
from pathsnap.hmm import viterbi, RoadGraph, reconstruct_route  # noqa: E402
from pathsnap.metrics import continuity, route_prf  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pathsnap")

EVAL_DIR = os.path.abspath(os.environ.get("PATHSNAP_EVAL_DIR", os.path.join(ROOT, "data", "eval_test")))
import lightgbm as lgb  # noqa: E402


class BoosterModel:
    """Portable LightGBM text model (artifacts/final_model.txt), independent of sklearn/pickle versions."""
    def __init__(self, path):
        self.b = lgb.Booster(model_file=path)
    def predict_proba(self, X):
        p = self.b.predict(X)
        return np.column_stack([1 - p, p])


MODEL = BoosterModel(os.path.join(ROOT, "artifacts", "final_model.txt"))
HMM = json.load(open(os.path.join(ROOT, "artifacts", "hmm_config.json")))["decoder"]
log.info("Loaded model %s | HMM params %s | eval dir %s", "LightGBM (tuned)", HMM, EVAL_DIR)

app = FastAPI(title="PathSnap API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    trip_id: str


def lonlat(trip, xy):
    lon, lat = trip.to_lonlat.transform(xy[:, 0], xy[:, 1])
    return np.column_stack([lat, lon])


def arc_lines(trip, arc_ids):
    return [[[float(trip.nodes[trip.arcs[a, 0], 1]), float(trip.nodes[trip.arcs[a, 0], 0])],
             [float(trip.nodes[trip.arcs[a, 1], 1]), float(trip.nodes[trip.arcs[a, 1], 0])]] for a in arc_ids]


def route_polyline(trip, route):
    pts = []
    for a in route:
        u, v = trip.arcs[a]
        for n in (u, v):
            p = [float(trip.nodes[n, 1]), float(trip.nodes[n, 0])]
            if not pts or pts[-1] != p:
                pts.append(p)
    return pts


@app.get("/health")
def health():
    return {"status": "ok", "model": "LightGBM (tuned)", "eval_dir": EVAL_DIR}


@app.get("/trips")
def trips():
    if not os.path.isdir(EVAL_DIR):
        raise HTTPException(500, f"eval dir not found: {EVAL_DIR}")
    ids = list_trips(EVAL_DIR)
    log.info("GET /trips -> %d trips", len(ids))
    return {"trips": ids, "count": len(ids)}


@app.post("/predict")
def predict(req: PredictRequest, request: Request):
    tid = req.trip_id
    d = os.path.join(EVAL_DIR, tid)
    if tid not in list_trips(EVAL_DIR):
        raise HTTPException(404, f"unknown trip {tid}")
    log.info("POST /predict trip=%s from %s", tid, request.client.host if request.client else "?")
    t0 = time.time()
    # ---------------- prediction pipeline (no reference data)
    trip = load_trip(d, with_route=False)
    cand = build_candidates(trip)
    cand["p"] = MODEL.predict_proba(cand[FEATURES].values)[:, 1]
    t_clf = time.time() - t0
    g = RoadGraph(trip)
    obs_ids, path, breaks = viterbi(trip, cand[["obs", "arc", "p", "proj_t"]], HMM, g)
    route, unfilled = reconstruct_route(trip, path, g)
    t_pred = time.time() - t0
    top = cand.loc[cand.groupby("obs").p.idxmax()].set_index("obs").arc
    pred_obs = pd.Series(path, obs_ids).reindex(range(len(trip.xy))).fillna(-1).astype(int)
    clf_obs = top.reindex(range(len(trip.xy))).fillna(-1).astype(int)
    # ---------------- server-side evaluation (reference route read only here)
    ref = _read(os.path.join(d, f"{tid}.route"), dtype=np.int64, ndmin=1)
    metrics = {}
    if ref is not None:
        trip.route = ref
        lab, _, _ = align_route(trip)
        p_, r_, f_ = route_prf(route, ref)
        metrics = dict(
            observation_top1_accuracy=float((pred_obs.values == lab).mean()),
            classifier_only_top1_accuracy=float((clf_obs.values == lab).mean()),
            route_precision=p_, route_recall=r_, route_f1=f_,
            route_continuity=continuity(route, trip.arcs),
        )
        trip.route = None
    resp = dict(
        trip_id=tid, n_gps=int(len(trip.xy)), n_candidates=int(len(cand)), hmm_breaks=int(breaks),
        unfilled_gaps=int(unfilled), route_arcs=int(len(route)),
        timing_s=dict(classifier=round(t_clf, 3), total_prediction=round(t_pred, 3), total=round(time.time() - t0, 3)),
        metrics=metrics,
        gps=[[float(a), float(b)] for a, b in trip.track[:, [1, 0]]],
        predicted_route=route_polyline(trip, route),
        candidate_network=arc_lines(trip, np.unique(cand.arc.values)),
    )
    log.info("trip=%s n_gps=%d top1=%.4f clf_top1=%.4f route_f1=%.4f continuity=%.4f time=%.2fs",
             tid, len(trip.xy), metrics.get("observation_top1_accuracy", float("nan")),
             metrics.get("classifier_only_top1_accuracy", float("nan")), metrics.get("route_f1", float("nan")),
             metrics.get("route_continuity", float("nan")), time.time() - t0)
    return resp
