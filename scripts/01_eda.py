"""Exploratory data analysis on the train and student_test partitions (evaluator data is never read)."""
import os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import _path
from pathsnap.core import load_trip, list_trips, arc_geometry

rows = []
for part in ["train", "student_test"]:
    for tid in list_trips(os.path.join(_path.DATA, part)):
        d = os.path.join(_path.DATA, part, tid)
        raw = np.loadtxt(f"{d}/{tid}.track")
        tr = load_trip(d)
        dt = np.diff(raw[:, 2])
        M, A = len(tr.nodes), len(tr.arcs)
        g = coo_matrix((np.ones(A), (tr.arcs[:, 0], tr.arcs[:, 1])), shape=(M, M))
        nw, lab_w = connected_components(g, directed=True, connection="weak")
        ns, lab_s = connected_components(g, directed=True, connection="strong")
        deg = np.bincount(tr.arcs.ravel(), minlength=M)
        _, _, L, _ = arc_geometry(tr)
        r = tr.route
        cont = np.mean(tr.arcs[r[:-1], 1] == tr.arcs[r[1:], 0]) if len(r) > 1 else np.nan
        step = np.hypot(*np.diff(tr.xy, axis=0).T)
        rows.append(dict(
            partition=part, trip=tid, n_gps=len(raw), n_nan=int((~np.isfinite(raw)).any(1).sum()),
            n_dup_t=int((dt == 0).sum()), n_backwards_t=int((dt < 0).sum()),
            dt_median=np.median(dt), dt_max=dt.max(), frac_dt_1=np.mean(dt == 1), n_gaps_gt5s=int((dt > 5).sum()),
            duration_s=raw[-1, 2] - raw[0, 2], path_len_km=step.sum() / 1000, max_step_m=step.max(),
            lon_min=raw[:, 0].min(), lon_max=raw[:, 0].max(), lat_min=raw[:, 1].min(), lat_max=raw[:, 1].max(),
            n_nodes=M, n_arcs=A, isolated_nodes=int((deg == 0).sum()), weak_components=nw,
            largest_weak_frac=np.bincount(lab_w).max() / M, strong_components=ns,
            largest_strong_frac=np.bincount(lab_s).max() / M, median_arc_len_m=np.median(L),
            route_len=len(r), route_unique=len(set(r.tolist())), route_contiguous_frac=cont,
            file_kb_track=os.path.getsize(f"{d}/{tid}.track") / 1024,
        ))
inv = pd.DataFrame(rows)
inv.to_csv(os.path.join(_path.RES, "eda_inventory.csv"), index=False)

obs = pd.concat([pd.read_parquet(os.path.join(_path.RES, "data", f"obs_{p}.parquet")) for p in ["train", "student_test"]])
lab = obs.groupby(["part", "trip"]).agg(label_dist_median=("label_dist", "median"),
                                         label_dist_p95=("label_dist", lambda x: x.quantile(.95)),
                                         candidate_miss=("has_cand", lambda x: 1 - x.mean())).reset_index()
lab.to_csv(os.path.join(_path.RES, "eda_label_alignment.csv"), index=False)
anomalous = lab[lab.label_dist_median > 50].trip.tolist()

summ = inv.groupby("partition").agg(trips=("trip", "size"), gps_total=("n_gps", "sum"), gps_mean=("n_gps", "mean"),
                                    gps_min=("n_gps", "min"), gps_max=("n_gps", "max"), nan_rows=("n_nan", "sum"),
                                    dup_timestamps=("n_dup_t", "sum"), gaps_gt5s=("n_gaps_gt5s", "sum"),
                                    nodes_mean=("n_nodes", "mean"), arcs_mean=("n_arcs", "mean"),
                                    isolated_nodes=("isolated_nodes", "sum"), route_contiguous=("route_contiguous_frac", "mean"))
summ.to_csv(os.path.join(_path.RES, "eda_summary.csv"))
dts = []
for part in ["train", "student_test"]:
    for tid in list_trips(os.path.join(_path.DATA, part)):
        dts.append(np.diff(np.loadtxt(os.path.join(_path.DATA, part, tid, f"{tid}.track"))[:, 2]))
dts = np.concatenate(dts)
facts = dict(anomalous_trips=anomalous, frac_dt_eq_1=float(np.mean(dts == 1)), frac_dt_eq_2=float(np.mean(dts == 2)),
             frac_dt_gt_5=float(np.mean(dts > 5)), max_dt=float(dts.max()),
             label_dist_median_all=float(obs.label_dist.median()),
             label_dist_p95_all=float(obs.label_dist.quantile(.95)),
             candidate_miss_train=float(1 - obs[obs.part == "train"].has_cand.mean()),
             candidate_miss_train_excl_anom=float(1 - obs[(obs.part == "train") & ~obs.trip.isin(anomalous)].has_cand.mean()),
             candidate_miss_student_test=float(1 - obs[obs.part == "student_test"].has_cand.mean()))
# nearest-arc baseline (no learning): pick the closest candidate arc for each student_test observation
_c = pd.read_parquet(os.path.join(_path.RES, "data", "cand_student_test.parquet"))
_b = _c[_c.dist_rank == 1][["trip", "obs", "arc"]].merge(obs[obs.part == "student_test"], on=["trip", "obs"], how="right")
facts["nearest_arc_top1_student_test"] = float((_b.arc == _b.label_arc).mean())
json.dump(facts, open(os.path.join(_path.RES, "eda_facts.json"), "w"), indent=2)
print(summ.T); print(facts)

# ------------------------------------------------------------- plots
F = _path.FIG
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for p, c in [("train", "tab:blue"), ("student_test", "tab:orange")]:
    ax[0].hist(inv[inv.partition == p].n_gps, bins=30, alpha=.6, label=p, color=c)
ax[0].set(title="GPS observations per trip", xlabel="observations", ylabel="trips", xscale="linear"); ax[0].legend()
v, c = np.unique(np.clip(dts, 0, 10), return_counts=True)
ax[1].bar(v, c); ax[1].set(yscale="log", title="Sampling interval (s), clipped at 10", xlabel="Δt (s)", ylabel="count")
ax[2].hist(np.clip(obs.label_dist, 0, 100), bins=50); ax[2].set(title="GPS → aligned reference arc distance (m)", xlabel="metres (clipped 100)")
plt.tight_layout(); plt.savefig(f"{F}/eda_distributions.png", dpi=130); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].scatter(inv.n_nodes, inv.n_arcs, c=(inv.partition == "train").map({True: "tab:blue", False: "tab:orange"}), s=15)
ax[0].set(title="Road-network size per trip", xlabel="nodes", ylabel="arcs")
ax[1].hist(inv.largest_strong_frac, bins=20); ax[1].set(title="Largest strongly-connected component (fraction of nodes)")
ax[2].hist(inv.path_len_km, bins=30); ax[2].set(title="GPS trajectory length (km)")
plt.tight_layout(); plt.savefig(f"{F}/eda_network.png", dpi=130); plt.close()

fig, ax = plt.subplots(figsize=(11, 5.5))
for p, c in [("train", "tab:blue"), ("student_test", "tab:orange")]:
    s = inv[inv.partition == p]
    ax.scatter((s.lon_min + s.lon_max) / 2, (s.lat_min + s.lat_max) / 2, s=25, c=c, label=p)
ax.set(title="Trip locations (centroid lon/lat)", xlabel="longitude", ylabel="latitude", xlim=(-180, 180), ylim=(-60, 75)); ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{F}/eda_trip_locations.png", dpi=130); plt.close()

# trajectory + network for a few trips
fig, axs = plt.subplots(2, 3, figsize=(15, 9))
for ax, (part, tid) in zip(axs.ravel(), [("train", "00000048"), ("train", "00000014"), ("train", "00000054"),
                                          ("student_test", "00000000"), ("student_test", "00000040"), ("student_test", "00000076")]):
    tr = load_trip(os.path.join(_path.DATA, part, tid))
    a, b, _, _ = arc_geometry(tr)
    pad = 300
    x0, y0 = tr.xy.min(0) - pad; x1, y1 = tr.xy.max(0) + pad
    m = ((a[:, 0] > x0) & (a[:, 0] < x1) & (a[:, 1] > y0) & (a[:, 1] < y1))
    for p_, q_ in zip(a[m], b[m]):
        ax.plot([p_[0], q_[0]], [p_[1], q_[1]], color="0.8", lw=.6)
    r = tr.route
    for k in r:
        ax.plot([a[k, 0], b[k, 0]], [a[k, 1], b[k, 1]], color="green", lw=2)
    ax.plot(tr.xy[:, 0], tr.xy[:, 1], ".", ms=1.5, color="tab:blue")
    ax.set(title=f"{part}/{tid}", aspect="equal", xlim=(x0, x1), ylim=(y0, y1)); ax.set_xticks([]); ax.set_yticks([])
plt.suptitle("GPS (blue), road network (grey), reference route (green) — local AEQD metres")
plt.tight_layout(); plt.savefig(f"{F}/eda_trajectories.png", dpi=130); plt.close()
