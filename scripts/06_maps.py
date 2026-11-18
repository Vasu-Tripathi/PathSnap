"""Folium maps: EDA view of a training trip, and GPS vs reference vs predicted (classifier-only and HMM) routes
for student_test trips, using the final tuned model and the tuned HMM decoder."""
import os, json
import numpy as np, pandas as pd, folium, joblib
import _path
from pathsnap.core import load_trip, build_candidates, FEATURES

OUT = os.path.join(_path.RES, "maps"); os.makedirs(OUT, exist_ok=True)


def seg(trip, a):
    u, v = trip.arcs[a]
    return [(trip.nodes[u, 1], trip.nodes[u, 0]), (trip.nodes[v, 1], trip.nodes[v, 0])]


def poly(trip, route):
    pts = []
    for a in route:
        for p in seg(trip, a):
            if not pts or pts[-1] != p:
                pts.append(p)
    return pts


def base_map(trip, net_arcs, tiles=True):
    m = folium.Map(tiles=None, control_scale=True)
    if tiles:
        folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    net = folium.FeatureGroup("Road network (nearby arcs)")
    for a in net_arcs:
        folium.PolyLine(seg(trip, a), color="#98a2b3", weight=2, opacity=.7).add_to(net)
    net.add_to(m)
    return m


def add_gps(m, trip, step=1, icons=True):
    fg = folium.FeatureGroup("Raw GPS observations")
    for i in range(0, len(trip.track), step):
        folium.CircleMarker((trip.track[i, 1], trip.track[i, 0]), radius=2, color="#1570ef", fill=True, fill_opacity=.8, weight=0,
                            tooltip=f"obs {i}, t={trip.track[i,2]:.0f}s").add_to(fg)
    fg.add_to(m)
    if icons:
        folium.Marker((trip.track[0, 1], trip.track[0, 0]), tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(m)
        folium.Marker((trip.track[-1, 1], trip.track[-1, 0]), tooltip="End", icon=folium.Icon(color="black", icon="stop")).add_to(m)
    else:  # offline rendering (no icon CDN)
        folium.CircleMarker((trip.track[0, 1], trip.track[0, 0]), radius=8, color="#12b76a", fill=True, fill_opacity=1, tooltip="Start").add_to(m)
        folium.CircleMarker((trip.track[-1, 1], trip.track[-1, 0]), radius=8, color="#101828", fill=True, fill_opacity=1, tooltip="End").add_to(m)


def legend(m, items):
    html = '<div style="position:fixed;bottom:25px;left:25px;z-index:9999;background:white;padding:8px 12px;border-radius:6px;font:13px sans-serif;box-shadow:0 1px 4px rgba(0,0,0,.3)">'
    html += "".join(f'<div><span style="display:inline-block;width:22px;height:4px;background:{c};margin-right:6px;vertical-align:middle"></span>{t}</div>' for t, c in items)
    m.get_root().html.add_child(folium.Element(html + "</div>"))


def render(part, tid, tiles=True, suffix="", zoom=None):
    from pathsnap.hmm import viterbi, RoadGraph, reconstruct_route
    model = joblib.load(os.path.join(_path.ART, "final_model.joblib"))
    hmm = json.load(open(os.path.join(_path.ART, "hmm_config.json")))["decoder"]
    trip = load_trip(os.path.join(_path.DATA, part, tid))
    cand = build_candidates(trip)
    cand["p"] = model.predict_proba(cand[FEATURES].values)[:, 1]
    g = RoadGraph(trip)
    top = cand.loc[cand.groupby("obs").p.idxmax()].sort_values("obs").arc.values
    ob, path, _ = viterbi(trip, cand[["obs", "arc", "p", "proj_t"]], hmm, g)
    r_hmm, _ = reconstruct_route(trip, path, g)
    m = base_map(trip, np.unique(cand.arc.values), tiles)
    ref = folium.FeatureGroup("Reference route (.route)")
    folium.PolyLine(poly(trip, trip.route), color="#12b76a", weight=9, opacity=.6).add_to(ref); ref.add_to(m)
    clf = folium.FeatureGroup("Classifier-only top-1 arcs", show=False)
    for a in np.unique(top):
        folium.PolyLine(seg(trip, a), color="#f79009", weight=4).add_to(clf)
    clf.add_to(m)
    add_gps(m, trip, step=max(1, len(trip.track) // 1500), icons=tiles)
    pr = folium.FeatureGroup("Predicted route (classifier + HMM)")
    folium.PolyLine(poly(trip, r_hmm), color="#d92d20", weight=4, opacity=.9).add_to(pr); pr.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    legend(m, [("Reference route", "#12b76a"), ("Predicted route (HMM)", "#d92d20"), ("Classifier-only arcs", "#f79009"),
               ("Raw GPS", "#1570ef"), ("Road network", "#98a2b3")])
    tk = trip.track if zoom is None else trip.track[int(zoom[0] * len(trip.track)):int(zoom[1] * len(trip.track))]
    m.fit_bounds([[tk[:, 1].min(), tk[:, 0].min()], [tk[:, 1].max(), tk[:, 0].max()]])
    f = os.path.join(OUT, f"{part}_{tid}_prediction{suffix}{'_zoom' if zoom else ''}.html"); m.save(f); return f


def render_eda(part, tid, tiles=True, suffix=""):
    trip = load_trip(os.path.join(_path.DATA, part, tid))
    from pathsnap.core import arc_geometry
    a, b, _, _ = arc_geometry(trip)
    from scipy.spatial import cKDTree
    d, _ = cKDTree(trip.xy).query((a + b) / 2)
    inb = np.where(d < 250)[0]  # arcs whose midpoint is within 250 m of the trajectory
    m = base_map(trip, inb, tiles)
    ref = folium.FeatureGroup("Reference route (.route)")
    folium.PolyLine(poly(trip, trip.route), color="#12b76a", weight=6, opacity=.8).add_to(ref); ref.add_to(m)
    add_gps(m, trip, step=max(1, len(trip.track) // 1500), icons=tiles)
    folium.LayerControl(collapsed=False).add_to(m)
    legend(m, [("Reference route", "#12b76a"), ("Raw GPS", "#1570ef"), ("Road network", "#98a2b3")])
    m.fit_bounds([[trip.track[:, 1].min(), trip.track[:, 0].min()], [trip.track[:, 1].max(), trip.track[:, 0].max()]])
    f = os.path.join(OUT, f"eda_{part}_{tid}{suffix}.html"); m.save(f); return f


if __name__ == "__main__":
    import sys
    tiles = "--notiles" not in sys.argv
    sfx = "" if tiles else "_notiles"
    print(render_eda("train", "00000048", tiles, sfx))
    print(render_eda("train", "00000054", tiles, sfx))
    if "--eda-only" in sys.argv or not os.path.exists(os.path.join(_path.ART, "hmm_config.json")):
        sys.exit(0)
    for tid in ["00000000", "00000040", "00000076", "00000009", "00000077"]:
        print(render("student_test", tid, tiles, sfx))
    for tid in ["00000077", "00000009"]:
        print(render("student_test", tid, tiles, sfx, zoom=(0.45, 0.55)))
