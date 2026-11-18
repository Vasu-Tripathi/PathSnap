"""PathSnap core: data loading, projection, reference-label alignment, candidate generation, features.

All geometry is done in a per-trip local Azimuthal Equidistant projection (metres) centred on the
trip's GPS centroid. Trips are spread across the globe, so a single UTM zone is not appropriate.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

# ---------------------------------------------------------------- configuration
CAND_RADIUS_M = 50.0     # candidate search radius
CAND_MAX_K = 8           # max candidates per GPS observation
CAND_FALLBACK_K = 3      # if nothing within radius, keep this many nearest arcs (up to FALLBACK_RADIUS)
FALLBACK_RADIUS_M = 300.0
HEADING_WIN = 2          # heading computed from point i-WIN to i+WIN
CONTEXT_WIN = 3          # neighbour window for context distance feature

FEATURES = [
    "dist", "dist_rank", "dist_minus_min", "dist_ratio_min", "n_cand",
    "bearing_diff", "cos_bearing", "speed", "arc_len", "proj_t",
    "ctx_dist_mean", "ctx_dist_max", "ctx_bearing_diff", "is_twoway",
    "out_deg_to", "in_deg_from", "gps_jump",
]


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray            # (N,3) lon, lat, t
    nodes: np.ndarray            # (M,2) lon, lat
    arcs: np.ndarray             # (A,2) from, to
    route: np.ndarray | None     # (R,) arc ids (reference) or None
    xy: np.ndarray = field(default=None)       # projected GPS (N,2)
    nxy: np.ndarray = field(default=None)      # projected nodes (M,2)
    to_lonlat: Transformer = field(default=None, repr=False)


def _read(path, dtype=float, ndmin=2):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    return np.loadtxt(path, dtype=dtype, ndmin=ndmin)


def load_trip(trip_dir: str, with_route: bool = True) -> Trip:
    tid = os.path.basename(os.path.normpath(trip_dir))
    track = _read(f"{trip_dir}/{tid}.track")
    nodes = _read(f"{trip_dir}/{tid}.nodes")
    arcs = _read(f"{trip_dir}/{tid}.arcs", dtype=np.int64)
    route = _read(f"{trip_dir}/{tid}.route", dtype=np.int64, ndmin=1) if with_route else None
    # basic cleaning: drop non-finite rows, sort by time, drop duplicate timestamps
    ok = np.isfinite(track).all(1) & (np.abs(track[:, 0]) <= 180) & (np.abs(track[:, 1]) <= 90)
    track = track[ok]
    track = track[np.argsort(track[:, 2], kind="stable")]
    _, keep = np.unique(track[:, 2], return_index=True)
    track = track[np.sort(keep)]
    trip = Trip(tid, track, nodes, arcs, route)
    lon0, lat0 = np.mean(track[:, 0]), np.mean(track[:, 1])
    proj = f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m"
    fwd = Transformer.from_crs("EPSG:4326", proj, always_xy=True)
    trip.to_lonlat = Transformer.from_crs(proj, "EPSG:4326", always_xy=True)
    trip.xy = np.column_stack(fwd.transform(track[:, 0], track[:, 1]))
    trip.nxy = np.column_stack(fwd.transform(nodes[:, 0], nodes[:, 1]))
    return trip


def list_trips(root: str):
    return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))


# ---------------------------------------------------------------- geometry helpers
def arc_geometry(trip: Trip):
    a = trip.nxy[trip.arcs[:, 0]]
    b = trip.nxy[trip.arcs[:, 1]]
    d = b - a
    length = np.hypot(d[:, 0], d[:, 1])
    bearing = np.degrees(np.arctan2(d[:, 0], d[:, 1])) % 360  # 0 = north, clockwise
    return a, b, length, bearing


def point_seg(p, a, b):
    """Distance and projection parameter t of points p onto segments a-b (broadcast, (...,2))."""
    d = b - a
    L2 = (d ** 2).sum(-1)
    t = np.where(L2 > 0, ((p - a) * d).sum(-1) / np.where(L2 > 0, L2, 1), 0.0)
    t = np.clip(t, 0, 1)
    q = a + t[..., None] * d
    return np.linalg.norm(p - q, axis=-1), t


def ang_diff(a, b):
    d = np.abs((a - b + 180) % 360 - 180)
    return d


def gps_heading_speed(trip: Trip):
    xy, t = trip.xy, trip.track[:, 2]
    n = len(xy)
    i0 = np.clip(np.arange(n) - HEADING_WIN, 0, n - 1)
    i1 = np.clip(np.arange(n) + HEADING_WIN, 0, n - 1)
    d = xy[i1] - xy[i0]
    dt = np.maximum(t[i1] - t[i0], 1e-6)
    speed = np.hypot(d[:, 0], d[:, 1]) / dt
    heading = np.degrees(np.arctan2(d[:, 0], d[:, 1])) % 360
    step = np.r_[0, np.hypot(*np.diff(xy, axis=0).T)]
    return heading, speed, step


# ---------------------------------------------------------------- reference labels
def align_route(trip: Trip):
    """Assign each GPS observation to an arc of the reference route with a monotone (order-preserving)
    alignment that minimises total point-to-segment distance (dynamic programming).
    Returns (label_arc_ids, label_dist_m)."""
    r = trip.route
    a, b, _, _ = arc_geometry(trip)
    A, B = a[r], b[r]
    N, K = len(trip.xy), len(r)
    D = np.empty((N, K))
    for i in range(N):
        D[i], _ = point_seg(np.broadcast_to(trip.xy[i], A.shape), A, B)
    # dp[i,k] = D[i,k] + min_{k'<=k} dp[i-1,k']
    back = np.empty((N, K), dtype=np.int32)
    prev = D[0].copy()
    back[0] = np.arange(K)
    for i in range(1, N):
        cm = np.minimum.accumulate(prev)
        idx = np.where(prev <= cm, np.arange(K), 0)
        back[i] = np.maximum.accumulate(idx)
        prev = D[i] + cm
    k = int(np.argmin(prev))
    path = np.empty(N, dtype=np.int64)
    for i in range(N - 1, -1, -1):
        path[i] = k
        k = back[i, k]
    return r[path], D[np.arange(N), path], path


# ---------------------------------------------------------------- candidates + features
def build_candidates(trip: Trip, labels: np.ndarray | None = None):
    """Return DataFrame with one row per (observation, candidate arc) and engineered features."""
    a, b, length, bearing = arc_geometry(trip)
    # densify arcs into sample points for KD-tree lookup
    step = 10.0
    nseg = np.maximum(1, np.ceil(length / step).astype(int))
    rep = np.repeat(np.arange(len(a)), nseg + 1)
    frac = np.concatenate([np.linspace(0, 1, k + 1) for k in nseg])
    samp = a[rep] + frac[:, None] * (b[rep] - a[rep])
    tree = cKDTree(samp)
    heading, speed, gstep = gps_heading_speed(trip)
    N = len(trip.xy)

    obs_idx, arc_idx = [], []
    for i in range(N):
        hits = np.unique(rep[tree.query_ball_point(trip.xy[i], CAND_RADIUS_M + step)])
        if len(hits):
            d, _ = point_seg(np.broadcast_to(trip.xy[i], a[hits].shape), a[hits], b[hits])
            m = d <= CAND_RADIUS_M
            hits, d = hits[m], d[m]
        if len(hits) == 0:  # fallback: nearest arcs within a wider radius
            hits = np.unique(rep[tree.query_ball_point(trip.xy[i], FALLBACK_RADIUS_M)])
            if len(hits) == 0:
                continue
            d, _ = point_seg(np.broadcast_to(trip.xy[i], a[hits].shape), a[hits], b[hits])
            o = np.argsort(d)[:CAND_FALLBACK_K]
        else:
            o = np.argsort(d)[:CAND_MAX_K]
        obs_idx.append(np.full(len(o), i))
        arc_idx.append(hits[o])
    obs_idx = np.concatenate(obs_idx)
    arc_idx = np.concatenate(arc_idx)

    P = trip.xy[obs_idx]
    dist, t = point_seg(P, a[arc_idx], b[arc_idx])
    df = pd.DataFrame({"obs": obs_idx, "arc": arc_idx, "dist": dist, "proj_t": t})
    g = df.groupby("obs")["dist"]
    df["dist_rank"] = g.rank(method="first")
    dmin = g.transform("min")
    df["dist_minus_min"] = df["dist"] - dmin
    df["dist_ratio_min"] = (df["dist"] + 1) / (dmin + 1)
    df["n_cand"] = g.transform("size")
    bd = ang_diff(heading[obs_idx], bearing[arc_idx])
    df["bearing_diff"] = bd
    df["cos_bearing"] = np.cos(np.radians(bd))
    df["speed"] = speed[obs_idx]
    df["arc_len"] = length[arc_idx]
    # context: distances of neighbouring observations to the same arc
    ctx = []
    for off in range(-CONTEXT_WIN, CONTEXT_WIN + 1):
        j = np.clip(obs_idx + off, 0, N - 1)
        dd, _ = point_seg(trip.xy[j], a[arc_idx], b[arc_idx])
        ctx.append(dd)
    ctx = np.vstack(ctx)
    df["ctx_dist_mean"] = ctx.mean(0)
    df["ctx_dist_max"] = ctx.max(0)
    # heading over a wider window vs arc bearing
    j0 = np.clip(obs_idx - 5, 0, N - 1)
    j1 = np.clip(obs_idx + 5, 0, N - 1)
    dv = trip.xy[j1] - trip.xy[j0]
    wide_head = np.degrees(np.arctan2(dv[:, 0], dv[:, 1])) % 360
    df["ctx_bearing_diff"] = ang_diff(wide_head, bearing[arc_idx])
    # network topology
    pairs = set(map(tuple, trip.arcs.tolist()))
    rev = np.array([(int(trip.arcs[k, 1]), int(trip.arcs[k, 0])) in pairs for k in range(len(trip.arcs))])
    df["is_twoway"] = rev[arc_idx].astype(float)
    outdeg = np.bincount(trip.arcs[:, 0], minlength=len(trip.nodes))
    indeg = np.bincount(trip.arcs[:, 1], minlength=len(trip.nodes))
    df["out_deg_to"] = outdeg[trip.arcs[arc_idx, 1]]
    df["in_deg_from"] = indeg[trip.arcs[arc_idx, 0]]
    df["gps_jump"] = gstep[obs_idx]
    df["trip"] = trip.trip_id
    if labels is not None:
        df["y"] = (labels[obs_idx] == arc_idx).astype(int)
    return df
