"""Merge the per-member shards produced by `02_build_dataset.py --shard k` into the full tables
results/data/{cand,obs}_{train,student_test}.parquet and check that every trip is present exactly once."""
import os, glob, argparse
import pandas as pd
import _path
from pathsnap.core import list_trips

ap = argparse.ArgumentParser(); ap.add_argument("--nshards", type=int, default=4); a = ap.parse_args()
S, D = os.path.join(_path.RES, "data", "shards"), os.path.join(_path.RES, "data")
for part in ["train", "student_test"]:
    tabs = {}
    for kind in ["cand", "obs"]:
        files = [os.path.join(S, f"{kind}_{part}_shard{k}of{a.nshards}.parquet") for k in range(a.nshards)]
        missing = [f for f in files if not os.path.exists(f)]
        if missing:
            raise SystemExit(f"missing shard files: {missing}")
        tabs[kind] = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    got = sorted(tabs["obs"].trip.unique()); want = list_trips(os.path.join(_path.DATA, part))
    assert got == want, f"{part}: trips mismatch ({len(got)} vs {len(want)})"
    assert not tabs["obs"].duplicated(["trip", "obs"]).any(), "duplicate observations across shards"
    # same ordering as a single-machine run (sorted trips)
    for kind in ["cand", "obs"]:
        t = tabs[kind]
        t["_o"] = t.trip.map({tr: i for i, tr in enumerate(want)})
        t = t.sort_values("_o", kind="stable").drop(columns="_o").reset_index(drop=True)
        t.to_parquet(os.path.join(D, f"{kind}_{part}.parquet"))
        print(part, kind, t.shape)
