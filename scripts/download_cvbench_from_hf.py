#!/usr/bin/env python
"""Mirror CV-Bench (nyu-visionx/CV-Bench) into the local evaluation store.

CV-Bench ships as two parquet shards with the images EMBEDDED (there is no external image source), so
it cannot go through `vlm_eval/tasks/download.py`, which only knows zip/tar/gz URLs. This mirrors the
`download_refcoco_from_hf.py` precedent: fetch from the Hub, materialise images on disk, and write a
flat `examples.jsonl` that `build_cvbench_indices` turns into the index.

    python scripts/download_cvbench_from_hf.py --root_dir /path/to/vlm_eval_data

Run it on a COMPUTE node: login1 SIGKILLs sustained downloads (see CLAUDE.md).

Note on the upstream helper: the repo ships a `build_img.py` that writes `f"{row['id']:06}.png"`, but
the schema has no `id` column (it is `idx`), so it silently falls back to the DataFrame index and
produces names that do NOT match the `filename` field every example references. We key on `filename`.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image

REPO_ID = "nyu-visionx/CV-Bench"
SHARDS = ("test_2d.parquet", "test_3d.parquet")
EXPECTED_TOTAL = 2638
EXPECTED_TASKS = {"Count": 788, "Relation": 650, "Depth": 600, "Distance": 600}


def _jsonable(value):
    """Recursively convert numpy scalars/arrays to plain Python.

    A single `.tolist()` is not enough: `bbox` and `target_size` come back as dtype=object arrays whose
    `.tolist()` yields a list still containing ndarrays, and `json.dumps` then dies on the inner one.
    """
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root_dir", type=Path,
                    default=Path("/lustre/nvwulf/projects/MilderGroup-nvwulf/skuo/vlm_eval_data"))
    ap.add_argument("--force", action="store_true", help="re-extract images even if they exist")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.root_dir / "download" / "cv-bench"
    out_dir.mkdir(parents=True, exist_ok=True)

    records, n_written, n_skipped = [], 0, 0
    for shard in SHARDS:
        print(f"[cv-bench] fetching {shard} ...", flush=True)
        local = hf_hub_download(repo_id=REPO_ID, filename=shard, repo_type="dataset")
        df = pd.read_parquet(local)
        print(f"[cv-bench] {shard}: {len(df)} rows, columns={list(df.columns)}", flush=True)

        for row in df.itertuples(index=False):
            rec = {c: getattr(row, c) for c in df.columns if c != "image"}
            rel = rec["filename"]                       # e.g. "img/2D/count/ade20k_10.png"
            dst = out_dir / rel
            if args.force or not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                img = getattr(row, "image")
                raw = img["bytes"] if isinstance(img, dict) else img
                Image.open(io.BytesIO(raw)).convert("RGB").save(dst)
                n_written += 1
            else:
                n_skipped += 1
            rec = {k: _jsonable(v) for k, v in rec.items()}
            records.append(rec)

    tasks = pd.Series([r["task"] for r in records]).value_counts().to_dict()
    print(f"[cv-bench] images written={n_written} reused={n_skipped}; examples={len(records)}")
    print(f"[cv-bench] tasks={tasks}")
    print(f"[cv-bench] sources={pd.Series([r['source'] for r in records]).value_counts().to_dict()}")
    print(f"[cv-bench] choice counts={pd.Series([len(r['choices']) for r in records]).value_counts().to_dict()}")

    if len(records) != EXPECTED_TOTAL:
        print(f"[cv-bench] WARNING expected {EXPECTED_TOTAL} examples, got {len(records)}")
    for task, n in EXPECTED_TASKS.items():
        if tasks.get(task) != n:
            print(f"[cv-bench] WARNING task {task}: expected {n}, got {tasks.get(task)}")

    dest = out_dir / "examples.jsonl"
    with open(dest, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    print(f"[cv-bench] wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
