"""HMM + Viterbi route reconstruction on top of classifier candidate probabilities.

Hidden states   : candidate road arcs of each GPS observation (from the candidate generator).
Emission        : log P_e = alpha * log(p_clf + eps)   (p_clf = calibrated classifier probability).
Transition      : Newson & Krumm style  log P_t = -|d_route - d_gps| / beta, where d_route is the
                  network distance from the projection on arc a (obs i-1) to the projection on arc b (obs i)
                  following arc direction, and d_gps is the straight-line distance between the GPS fixes.
                  Unreachable pairs (no directed path within the search limit) get -inf.
Breaks          : if dt > GAP_S or all transitions into step i are -inf, the chain is broken and restarted
                  from the emissions only (segment-wise Viterbi).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .core import Trip, arc_geometry

DEFAULTS = dict(alpha=1.0, beta=5.0, gap_s=60.0, limit_m=2000.0, eps=1e-4, backtrack_tol_m=15.0)


class RoadGraph:
    """Directed graph restricted to the trajectory bounding box (+margin) for fast shortest paths."""

    def __init__(self, trip: Trip, margin=1500.0):
        a, b, L, _ = arc_geometry(trip)
        self.trip = trip
        self.L = L
        lo, hi = trip.xy.min(0) - margin, trip.xy.max(0) + margin
        inbox = lambda p: (p[:, 0] > lo[0]) & (p[:, 0] < hi[0]) & (p[:, 1] > lo[1]) & (p[:, 1] < hi[1])
        keep = np.where(inbox(a) | inbox(b))[0]
        self.arc_ids = keep
        nodes = np.unique(trip.arcs[keep].ravel())
        self.nid = {int(n): i for i, n in enumerate(nodes)}
        self.nodes = nodes
        u = np.array([self.nid[int(x)] for x in trip.arcs[keep, 0]])
        v = np.array([self.nid[int(x)] for x in trip.arcs[keep, 1]])
        w = np.maximum(L[keep], 1e-3)
        # keep the shortest arc for parallel edges
        order = np.argsort(-w)
        self.edge = {}
        for k in order:
            self.edge[(u[k], v[k])] = int(keep[k])
        uu = np.array([k[0] for k in self.edge]); vv = np.array([k[1] for k in self.edge])
        ww = np.array([max(L[e], 1e-3) for e in self.edge.values()])
        self.G = csr_matrix((ww, (uu, vv)), shape=(len(nodes), len(nodes)))
        self.dist, self.tidx = {}, {}

    def prepare(self, sources, targets, limit):
        """Compact distance table: self.dist[src] = distances from src to every node in `targets` (local ids)."""
        new_t = [t for t in targets if t not in self.tidx]
        if new_t:  # target set changed -> rebuild
            self.tidx = {t: i for i, t in enumerate(sorted(set(targets) | set(self.tidx)))}
            self.dist = {}
        cols = np.array(sorted(self.tidx, key=self.tidx.get))
        need = [s for s in sources if s not in self.dist]
        for i in range(0, len(need), 64):
            chunk = need[i:i + 64]
            D = dijkstra(self.G, directed=True, indices=chunk, limit=limit)
            for j, s in enumerate(chunk):
                self.dist[s] = D[j, cols].astype(np.float32)

    def path_arcs(self, n_from, n_to, limit):
        """Arc ids of shortest directed path between global node ids, or None."""
        if n_from not in self.nid or n_to not in self.nid:
            return None
        s, t = self.nid[n_from], self.nid[n_to]
        if s == t:
            return []
        D, P = dijkstra(self.G, directed=True, indices=s, limit=limit, return_predecessors=True)
        if not np.isfinite(D[t]):
            return None
        out, cur = [], t
        while cur != s:
            pr = P[cur]
            if pr < 0:
                return None
            out.append(self.edge[(pr, cur)]); cur = pr
        return out[::-1]


def viterbi(trip: Trip, cand: pd.DataFrame, params=None, graph: RoadGraph | None = None):
    """cand: rows for this trip with columns obs, arc, p, proj_t. Returns (obs_ids, decoded_arc, n_breaks)."""
    prm = dict(DEFAULTS); prm.update(params or {})
    g = graph or RoadGraph(trip)
    _, _, L, _ = arc_geometry(trip)
    arcs = trip.arcs
    cand = cand.sort_values(["obs", "p"], ascending=[True, False])
    groups = [(o, gdf) for o, gdf in cand.groupby("obs", sort=True)]
    obs_ids = np.array([o for o, _ in groups])
    C = [gdf.arc.values for _, gdf in groups]
    T = [gdf.proj_t.values for _, gdf in groups]
    E = [prm["alpha"] * np.log(gdf.p.values + prm["eps"]) for _, gdf in groups]
    xy, tt = trip.xy, trip.track[:, 2]
    # shortest paths from head nodes of all candidate arcs (local indices) to tail nodes
    allc = np.concatenate(C)
    src = [g.nid[int(h)] for h in np.unique(arcs[allc, 1]) if int(h) in g.nid]
    tgt = [g.nid[int(h)] for h in np.unique(arcs[allc, 0]) if int(h) in g.nid]
    g.prepare(src, tgt, prm["limit_m"])

    n = len(groups)
    score = [None] * n
    back = [None] * n
    breaks = 0
    score[0] = E[0]
    for i in range(1, n):
        o0, o1 = obs_ids[i - 1], obs_ids[i]
        dt = tt[o1] - tt[o0]
        dgps = float(np.hypot(*(xy[o1] - xy[o0])))
        a, b = C[i - 1], C[i]
        ta, tb = T[i - 1], T[i]
        # network distance matrix |a| x |b|
        dr = np.full((len(a), len(b)), np.inf)
        for ia, arc_a in enumerate(a):
            h = int(arcs[arc_a, 1])
            if h not in g.nid:
                continue
            Drow = g.dist[g.nid[h]]
            rem = (1 - ta[ia]) * L[arc_a]
            for ib, arc_b in enumerate(b):
                if arc_b == arc_a:
                    fwd = (tb[ib] - ta[ia]) * L[arc_a]
                    if fwd >= -prm["backtrack_tol_m"]:
                        dr[ia, ib] = abs(fwd)
                        continue
                tl = int(arcs[arc_b, 0])
                if tl in g.nid:
                    d = Drow[g.tidx[g.nid[tl]]]
                    if np.isfinite(d):
                        dr[ia, ib] = rem + d + tb[ib] * L[arc_b]
        logT = -np.abs(dr - dgps) / prm["beta"]
        tot = score[i - 1][:, None] + logT
        best = tot.max(0)
        if dt > prm["gap_s"] or not np.isfinite(best).any():
            breaks += 1
            back[i] = None  # chain break
            score[i] = E[i] + score[i - 1].max()
            continue
        back[i] = tot.argmax(0)
        best = np.where(np.isfinite(best), best, score[i - 1].max() - 1e4)  # isolated state: heavy penalty
        score[i] = best + E[i]
        score[i] = score[i] - score[i].max()  # normalise to avoid drift
    # backtrack
    path = np.empty(n, dtype=np.int64)
    k = int(np.argmax(score[-1]))
    for i in range(n - 1, -1, -1):
        path[i] = C[i][k]
        if i > 0:
            if back[i] is None:
                k = int(np.argmax(score[i - 1]))
            else:
                k = int(back[i][k])
    return obs_ids, path, breaks


def reconstruct_route(trip: Trip, arc_seq, graph: RoadGraph, limit=2000.0):
    """Collapse consecutive duplicates and fill gaps between non-adjacent arcs with shortest paths."""
    s = [int(a) for i, a in enumerate(arc_seq) if a >= 0 and (i == 0 or a != arc_seq[i - 1])]
    if not s:
        return [], 0
    route, unfilled = [s[0]], 0
    for b in s[1:]:
        a = route[-1]
        if a == b:
            continue
        if trip.arcs[a, 1] != trip.arcs[b, 0]:
            p = graph.path_arcs(int(trip.arcs[a, 1]), int(trip.arcs[b, 0]), limit)
            if p is None:
                unfilled += 1
            else:
                route.extend(p)
        route.append(b)
    return route, unfilled
