#!/usr/bin/env python3
"""Merge independently executed confirmation seeds into one checked dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (1, 2, 3, 4, 5)
MODES = ("replay", "naive", "no_update")
METRICS = (
    "current_brier",
    "current_error",
    "retention_brier",
    "retention_error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Directory containing output_confirm_seed1 through seed5",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "source_data" / "plasticity_retention_confirmation",
    )
    return parser.parse_args()


def checked_seed_directory(input_root: Path, seed: int) -> Path:
    directory = input_root / f"output_confirm_seed{seed}"
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing confirmation directory: {directory}")
    return directory


def main() -> None:
    args = parse_args()
    result_frames: list[pd.DataFrame] = []
    initial_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []

    for seed in SEEDS:
        directory = checked_seed_directory(args.input_root, seed)
        results = pd.read_csv(directory / "Split_CIFAR10_candidate_results.csv")
        initial = pd.read_csv(directory / "Split_CIFAR10_initial_models.csv")
        summary = json.loads(
            (directory / "Split_CIFAR10_summary.json").read_text(encoding="utf-8")
        )

        if len(results) != len(MODES):
            raise ValueError(f"Seed {seed} has {len(results)} candidate rows")
        if set(results["seed"]) != {seed}:
            raise ValueError(f"Candidate rows are mislabeled for seed {seed}")
        if set(results["candidate_mode"]) != set(MODES):
            raise ValueError(f"Candidate modes are incomplete for seed {seed}")
        if set(results["evaluation_split"]) != {"confirmation"}:
            raise ValueError(f"Seed {seed} does not use the confirmation split")
        if len(initial) != 1 or set(initial["seed"]) != {seed}:
            raise ValueError(f"Initial-model row is invalid for seed {seed}")
        if set(initial["evaluation_split"]) != {"confirmation"}:
            raise ValueError(f"Initial model for seed {seed} uses another split")

        configuration = summary["configuration"]
        if configuration["seeds"] != [seed]:
            raise ValueError(f"Summary seed is invalid for seed {seed}")
        if configuration["evaluation_split"] != "confirmation":
            raise ValueError(f"Summary split is invalid for seed {seed}")
        if float(configuration["replay_ratio"]) != 3.25:
            raise ValueError(f"Replay ratio changed for seed {seed}")
        if float(configuration["adaptation_learning_rate"]) != 0.003:
            raise ValueError(f"Learning rate changed for seed {seed}")

        for mode in MODES:
            for metric in METRICS:
                name = f"split_cifar10_{mode}_{metric}_seed_{seed}.npy"
                if not (directory / name).is_file():
                    raise FileNotFoundError(f"Missing paired sequence: {name}")

        result_frames.append(results)
        initial_frames.append(initial)
        summaries.append(summary)

    reference_configuration = dict(summaries[0]["configuration"])
    reference_configuration["seeds"] = list(SEEDS)
    for seed, summary in zip(SEEDS[1:], summaries[1:], strict=True):
        comparison = dict(summary["configuration"])
        comparison["seeds"] = list(SEEDS)
        if comparison != reference_configuration:
            raise ValueError(f"Configuration differs for seed {seed}")

    combined_results = pd.concat(result_frames, ignore_index=True).sort_values(
        ["seed", "candidate_mode"]
    )
    combined_initial = pd.concat(initial_frames, ignore_index=True).sort_values(
        "seed"
    )
    combined_summary = {
        "configuration": reference_configuration,
        "initial_models": combined_initial.to_dict(orient="records"),
        "runs": combined_results.to_dict(orient="records"),
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    combined_results.to_csv(
        args.output_dir / "Split_CIFAR10_candidate_results.csv",
        index=False,
    )
    combined_initial.to_csv(
        args.output_dir / "Split_CIFAR10_initial_models.csv",
        index=False,
    )
    (args.output_dir / "Split_CIFAR10_summary.json").write_text(
        json.dumps(combined_summary, indent=2),
        encoding="utf-8",
    )
    for seed in SEEDS:
        directory = checked_seed_directory(args.input_root, seed)
        for mode in MODES:
            for metric in METRICS:
                name = f"split_cifar10_{mode}_{metric}_seed_{seed}.npy"
                shutil.copy2(directory / name, args.output_dir / name)

    print(args.output_dir)


if __name__ == "__main__":
    main()
