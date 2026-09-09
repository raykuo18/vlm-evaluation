#!/usr/bin/env python
"""Re-score RefCOCO/+/g and OCID-Ref for every run (optionally just one)."""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from vlm_eval.conf import DatasetRegistry

DATASET_TARGETS: Dict[str, Dict[str, str]] = {
    "refcoco-full": {
        "family": "refcoco",
        "dataset_id": "refcoco-full",
        "registry": DatasetRegistry.REFCOCO_FULL_V2.dataset_id,
    },
    "ocid-ref-full": {
        "family": "ocid-ref",
        "dataset_id": "ocid-ref-full",
        "registry": DatasetRegistry.OCIDREF_FULL_V2.dataset_id,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("../vlm_runs").resolve(),
        help="Directory containing Prismatic run folders (defaults to ../vlm_runs)",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/lustre/nvwulf/projects/MilderGroup-nvwulf/skuo/vlm_eval_data"),
        help="Root with prepared evaluation datasets",
    )
    parser.add_argument(
        "--python",
        type=str,
        default="python",
        help="Python executable to invoke score.py",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without executing",
    )
    parser.add_argument(
        "--try-one",
        type=Path,
        default=None,
        help="Only process this run directory (absolute or relative path)",
    )
    parser.add_argument(
        "--shard",
        type=int,
        default=0,
        help="0-based index of this shard (use with --num-shards to parallelise across array tasks)",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="total number of shards; run folders are dealt round-robin so shards are balanced",
    )
    parser.add_argument(
        "--metrics-suffix",
        type=str,
        default="v2",
        help="Suffix for the corrected metrics file (default: v2 => metrics_v2.json)",
    )
    return parser.parse_args()


def iter_localization_dirs(run_dir: Path) -> Iterator[tuple[Path, Path, Path, str]]:
    """Yield (tag_dir, group_dir, model_dir, registry_choice)."""
    eval_root = run_dir / "eval_results"
    if not eval_root.exists():
        return
    for tag_dir in sorted(p for p in eval_root.iterdir() if p.is_dir()):
        for dataset_group, meta in DATASET_TARGETS.items():
            group_dir = tag_dir / dataset_group
            if not group_dir.exists():
                continue
            family_dir = group_dir / meta["family"]
            dataset_dir = family_dir / meta["dataset_id"]
            if not dataset_dir.exists():
                continue
            for model_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
                # Scoring reads `results+rank-*.json`; 8 result dirs corpus-wide are empty shells from
                # evals that never produced output (13B / cross configs). Without this guard score.py
                # hits its `expected_examples` assert ("found 0") and, with check=True, killed the whole
                # shard -- 4 of 16 array tasks died this way, silently skipping their remaining runs.
                if not any(model_dir.glob("results+rank*.json")):
                    print(f"[SKIP] no per-rank results in {model_dir} -- nothing to re-score")
                    continue
                yield tag_dir, group_dir, model_dir, meta["registry"]


def resolve_config_path(run_dir: Path) -> Optional[Path]:
    for candidate in [run_dir / "config.yaml", run_dir / "config.json"]:
        if candidate.exists():
            return candidate
    return None


def restore_original(metrics_backup: Optional[Path], model_dir: Path) -> None:
    if metrics_backup and metrics_backup.exists():
        target = model_dir / "metrics.json"
        if target.exists():
            target.unlink()
        metrics_backup.rename(target)


def main() -> None:
    args = parse_args()
    runs_root = args.runs_root
    if args.try_one is not None:
        try_path = args.try_one
        if not try_path.is_absolute():
            try_path = (runs_root / try_path).resolve()
        runs = [try_path]
    else:
        runs = sorted(p.resolve() for p in runs_root.iterdir() if p.is_dir())
        if args.num_shards > 1:
            # Round-robin, not contiguous blocks: run cost varies with how many eval dirs a run has,
            # and neighbouring folders are near-duplicates (seed variants), so blocks would be lumpy.
            runs = runs[args.shard :: args.num_shards]
            print(f"[SHARD] {args.shard + 1}/{args.num_shards} -> {len(runs)} run folders")

    repo_root = Path(__file__).resolve().parent.parent
    commands: List[List[str]] = []
    work_items: List[tuple[List[str], Path]] = []

    for run_dir in runs:
        if not run_dir.exists():
            print(f"[WARN] Run directory {run_dir} does not exist; skipping")
            continue
        config_path = resolve_config_path(run_dir)
        if config_path is None:
            print(f"[WARN] Missing config for {run_dir}; skipping")
            continue
        for tag_dir, group_dir, model_dir, registry_choice in iter_localization_dirs(run_dir):
            cmd = [
                args.python,
                "scripts/score.py",
                "--dataset.type",
                registry_choice,
                "--dataset.root_dir",
                str(args.dataset_root),
                "--results_dir",
                str(group_dir),
                "--model_id",
                model_dir.name,
                "--config_yaml",
                str(config_path),
            ]
            commands.append(cmd)
            work_items.append((cmd, model_dir))
            print(f"[PLAN] {run_dir.name} | {group_dir.name} | {model_dir.name}")

    if args.dry_run:
        print(f"[DRY-RUN] Prepared {len(commands)} score commands")
        return

    failures: List[Tuple[str, str]] = []
    for cmd, model_dir in work_items:
        metrics = model_dir / "metrics.json"
        backup: Optional[Path] = None
        if metrics.exists():
            # WRITE-ONCE pristine backup. The old code used `metrics_old.json` and unlinked it first,
            # so a second run of this script overwrote the pristine numbers with already-rescored ones
            # and the original was gone for good. Keep the first backup forever; later runs copy to a
            # throwaway so failure can still restore.
            pristine = model_dir / "metrics_pre_geomfix.json"
            if not pristine.exists():
                shutil.copy2(metrics, pristine)
            backup = model_dir / "metrics_old.json"
            if backup.exists():
                backup.unlink()
            metrics.rename(backup)
        try:
            print(f"[RUN] {' '.join(cmd)}")
            subprocess.run(cmd, check=True, cwd=repo_root)
        except subprocess.CalledProcessError as exc:
            # Restore this run and KEEP GOING. Aborting the process abandons every remaining run in the
            # shard, which is far more damage than one unscoreable result dir.
            restore_original(backup, model_dir)
            print(f"[FAIL] {model_dir} -> {exc}")
            failures.append((str(model_dir), str(exc)))
            continue
        else:
            if backup is None:
                continue
            new_metrics = model_dir / "metrics.json"
            if not new_metrics.exists():
                continue

    print(f"[DONE] Re-scored {len(commands) - len(failures)}/{len(commands)} localization results")
    if failures:
        print(f"[DONE] {len(failures)} FAILED (originals restored):")
        for path, err in failures:
            print(f"    {path}\n        {err}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
