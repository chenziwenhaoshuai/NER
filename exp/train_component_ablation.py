"""Train and materialize every neural branch used in the component ablation.

This is the full-data counterpart of ``component_ablation.py``.  It starts
from the common backbone/temporal-calibration exports, trains the three
label-free neural scorers, and materializes all ablation predictions with the
same candidate pools and alert budgets.

The script never edits the supplied backbone exports.  Each stage writes to a
separate subdirectory under ``--output-dir``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
ALL_PAIRS = "all"


def command_text(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def run(command: list[str], dry_run: bool) -> None:
    print("+", command_text(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrain the three NER experts and materialize the component ablation."
    )
    parser.add_argument(
        "--exp-dir",
        type=Path,
        required=True,
        help="Directory containing rescue/{dataset}_{backbone}_seed2021_predictions.npz.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root containing the preprocessed SMD, MSL, SMAP, PSM, and SWaT folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/full_component_ablation",
    )
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--score-batch-size", type=int, default=4096)
    parser.add_argument("--self-epochs", type=int, default=30)
    parser.add_argument("--geometry-epochs", type=int, default=8)
    parser.add_argument("--geometry-windows", type=int, default=40000)
    parser.add_argument("--augmented-epochs", type=int, default=4)
    parser.add_argument("--augmented-windows", type=int, default=12000)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the exact commands without training.",
    )
    args = parser.parse_args()

    exp_dir = args.exp_dir.resolve()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    self_dir = output_dir / "self_event_ranker"
    geometry_dir = output_dir / "geometry_convae"
    augmented_dir = output_dir / "score_augmented_convae"
    router_dir = output_dir / "neural_router_v7"

    if not args.dry_run:
        rescue_dir = exp_dir / "rescue"
        if not rescue_dir.is_dir():
            raise FileNotFoundError(
                f"missing {rescue_dir}; full training needs the common rescue exports"
            )
        if not data_root.is_dir():
            raise FileNotFoundError(data_root)
        output_dir.mkdir(parents=True, exist_ok=True)

    common = [
        "--exp_dir",
        str(exp_dir),
        "--data_root",
        str(data_root),
        "--pairs",
        ALL_PAIRS,
        "--seed",
        str(args.seed),
        "--batch_size",
        str(args.batch_size),
        "--score_batch_size",
        str(args.score_batch_size),
    ]

    run(
        [
            sys.executable,
            str(SRC / "experiment_self_trained_event_ranker.py"),
            *common,
            "--output_dir",
            str(self_dir),
            "--epochs",
            str(args.self_epochs),
        ],
        args.dry_run,
    )
    run(
        [
            sys.executable,
            str(SRC / "experiment_convae_candidate_scorer.py"),
            *common,
            "--output_dir",
            str(geometry_dir),
            "--epochs",
            str(args.geometry_epochs),
            "--train_windows",
            str(args.geometry_windows),
            "--feature_mode",
            "geometry",
        ],
        args.dry_run,
    )
    run(
        [
            sys.executable,
            str(SRC / "experiment_augmented_convae_candidate_scorer.py"),
            *common,
            "--self_dir",
            str(self_dir),
            "--output_dir",
            str(augmented_dir),
            "--epochs",
            str(args.augmented_epochs),
            "--train_windows",
            str(args.augmented_windows),
        ],
        args.dry_run,
    )
    run(
        [
            sys.executable,
            str(SRC / "materialize_neural_router_v7.py"),
            "--exp_dir",
            str(exp_dir),
            "--self_dir",
            str(self_dir),
            "--geometry_ae_dir",
            str(geometry_dir),
            "--augmented_ae_dir",
            str(augmented_dir),
            "--output_dir",
            str(router_dir),
        ],
        args.dry_run,
    )

    if not args.dry_run:
        config = {
            "exp_dir": str(exp_dir),
            "data_root": str(data_root),
            "output_dir": str(output_dir),
            "seed": args.seed,
            "batch_size": args.batch_size,
            "score_batch_size": args.score_batch_size,
            "self_epochs": args.self_epochs,
            "geometry_epochs": args.geometry_epochs,
            "geometry_windows": args.geometry_windows,
            "augmented_epochs": args.augmented_epochs,
            "augmented_windows": args.augmented_windows,
        }
        (output_dir / "training_config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )
        print(f"Full component-ablation artifacts: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
