"""Build candidate/feature tables for the train and student_test partitions.
Reference per-observation labels come from monotone alignment of GPS points to the .route arcs."""
import os, time, json
from multiprocessing import Pool
import numpy as np, pandas as pd
import _path
from pathsnap.core import load_trip, align_route, build_candidates, list_trips


def proc(args):
    part, tid = args
    tr = load_trip(os.path.join(_path.DATA, part, tid))
    lab, ld, _ = align_route(tr)
    df = build_candidates(tr, lab)
    has = df.groupby("obs")["y"].max()
    obs = pd.DataFrame({"trip": tid, "obs": np.arange(len(lab)), "label_arc": lab, "label_dist": ld,
                        "t": tr.track[:, 2]})
    obs["has_cand"] = obs["obs"].map(has).fillna(0).astype(int)
    df["part"] = part
    obs["part"] = part
    return df, obs


def shard_trips(part, k, n):
    """Deterministic size-balanced split of a partition's trips into n shards (greedy by .track size)."""
    root = os.path.join(_path.DATA, part)
    trips = list_trips(root)
    size = {t: os.path.getsize(os.path.join(root, t, f"{t}.track")) for t in trips}
    load, assign = [0] * n, {i: [] for i in range(n)}
    for t in sorted(trips, key=lambda t: (-size[t], t)):
        i = min(range(n), key=lambda j: (load[j], j)); assign[i].append(t); load[i] += size[t]
    return sorted(assign[k])


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build candidate/feature tables. Use --shard/--nshards to split the work across team laptops.")
    ap.add_argument("--shard", type=int, default=None, help="shard index 0..nshards-1 (omit = process everything)")
    ap.add_argument("--nshards", type=int, default=4)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    out_dir = os.path.join(_path.RES, "data") if a.shard is None else os.path.join(_path.RES, "data", "shards")
    os.makedirs(out_dir, exist_ok=True)
    for part in ["train", "student_test"]:
        t = time.time()
        trips = list_trips(os.path.join(_path.DATA, part)) if a.shard is None else shard_trips(part, a.shard, a.nshards)
        with Pool(a.workers) as p:
            out = p.map(proc, [(part, tid) for tid in trips])
        cand = pd.concat([o[0] for o in out], ignore_index=True)
        obs = pd.concat([o[1] for o in out], ignore_index=True)
        tag = part if a.shard is None else f"{part}_shard{a.shard}of{a.nshards}"
        cand.to_parquet(os.path.join(out_dir, f"cand_{tag}.parquet"))
        obs.to_parquet(os.path.join(out_dir, f"obs_{tag}.parquet"))
        print(tag, len(trips), "trips", cand.shape, obs.shape, "miss rate", round(1 - obs.has_cand.mean(), 4), f"{time.time()-t:.1f}s")
