#!/usr/bin/env python3
"""Evaluate a plasticity-retention promotion gate on Split CIFAR-10.

Each candidate is generated from model-fitting data and then frozen before the
promotion monitor starts. The monitor uses held-out observations only.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18


ROOT = Path(__file__).resolve().parents[1]
CHANNEL_MEAN = (0.4914, 0.4822, 0.4465)
CHANNEL_STD = (0.2470, 0.2435, 0.2616)
CANDIDATE_MODES = ("replay", "naive", "no_update")
ALPHA = 0.05
EVIDENCE_SCALE = 0.5
LAMBDA_GRID = np.linspace(0.0, 1.9, 17, dtype=np.float64)[1:]


def make_model() -> nn.Module:
    model = resnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(
        3,
        64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    model.maxpool = nn.Identity()
    return model


def multiclass_brier(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    one_hot = torch.nn.functional.one_hot(targets, num_classes=10)
    squared_error = (probabilities - one_hot.to(probabilities.dtype)).square()
    return squared_error.sum(dim=1) / 2.0


def subexponential_psi(values: np.ndarray) -> np.ndarray:
    scaled = EVIDENCE_SCALE * values
    return (-np.log1p(-scaled) - scaled) / EVIDENCE_SCALE**2


def promotion_summary(differences: np.ndarray) -> tuple[int | None, float]:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Differences must be a nonempty one-dimensional array")
    if not np.isfinite(values).all() or np.max(np.abs(values)) > 1.0 + 1e-12:
        raise ValueError("Differences must be finite and lie in [-1, 1]")

    penalties = subexponential_psi(LAMBDA_GRID)
    threshold = math.log(1.0 / ALPHA)
    first_crossing: int | None = None
    final_log_evidence = -math.inf
    cumulative_difference = 0.0
    intrinsic_time = 0.0

    for index, difference in enumerate(values, start=1):
        previous_average = (
            cumulative_difference / (index - 1) if index > 1 else 0.0
        )
        predictable_center = float(
            np.clip(
                previous_average,
                -EVIDENCE_SCALE / 2.0,
                EVIDENCE_SCALE / 2.0,
            )
        )
        cumulative_difference += difference
        intrinsic_time += (difference - predictable_center) ** 2
        log_components = (
            LAMBDA_GRID * cumulative_difference
            - penalties * intrinsic_time
        )
        maximum = float(np.max(log_components))
        final_log_evidence = maximum + math.log(
            float(np.exp(log_components - maximum).sum())
        ) - math.log(len(LAMBDA_GRID))
        if first_crossing is None and final_log_evidence >= threshold:
            first_crossing = index

    return first_crossing, final_log_evidence


def train_initial_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
) -> None:
    model.train()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.05,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
    )
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for features, targets in loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            criterion(model(features), targets).backward()
            optimizer.step()
        scheduler.step()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / ".data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "source_data" / "plasticity_retention_stream",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--initial-epochs", type=int, default=20)
    parser.add_argument("--adaptation-epochs", type=int, default=5)
    parser.add_argument("--adaptation-size", type=int, default=15_000)
    parser.add_argument("--evaluation-size", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--adaptation-learning-rate", type=float, default=0.001)
    parser.add_argument("--replay-size", type=int, default=15_000)
    parser.add_argument("--replay-ratio", type=float, default=1.0)
    parser.add_argument(
        "--candidate-modes",
        nargs="+",
        choices=CANDIDATE_MODES,
        default=list(CANDIDATE_MODES),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Optional directory for deterministic initial-model checkpoints",
    )
    parser.add_argument("--retention-margin-brier", type=float, default=0.05)
    parser.add_argument("--retention-margin-error", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--evaluation-split",
        choices=("development", "confirmation"),
        default="confirmation",
        help=(
            "Use one of two disjoint 5,000-image-per-task evaluation partitions. "
            "Neither partition is available to model fitting."
        ),
    )
    parser.add_argument(
        "--partition-seed",
        type=int,
        default=20260723,
        help="Fixed seed for model-fitting, development, and confirmation partitions",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def class_indices(targets: list[int], allowed: set[int]) -> np.ndarray:
    return np.asarray(
        [index for index, target in enumerate(targets) if target in allowed],
        dtype=np.int64,
    )


def actionable_boundary(
    crossing: int | None,
    batch_size: int,
    horizon: int,
) -> int | None:
    if crossing is None:
        return None
    return min(int(math.ceil(crossing / batch_size) * batch_size), horizon)


def load_replay_memory(
    dataset: datasets.CIFAR10,
    indices: np.ndarray,
    batch_size: int,
    workers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        persistent_workers=workers > 0,
    )
    features: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for batch_features, batch_targets in loader:
        features.append(batch_features)
        targets.append(batch_targets)
    return torch.cat(features), torch.cat(targets)


def model_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    losses: list[np.ndarray] = []
    errors: list[np.ndarray] = []
    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            probabilities = model(features).softmax(dim=1)
            losses.append(
                multiclass_brier(probabilities, targets)
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            errors.append(
                (probabilities.argmax(dim=1) != targets)
                .cpu()
                .numpy()
                .astype(np.float64)
            )
    return float(np.concatenate(losses).mean()), float(np.concatenate(errors).mean())


def scaled_noninferiority_difference(
    incumbent_loss: np.ndarray,
    candidate_loss: np.ndarray,
    margin: float,
) -> np.ndarray:
    if not 0 < margin < 1:
        raise ValueError("The retention margin must lie in (0, 1)")
    return (incumbent_loss - candidate_loss + margin) / (1.0 + margin)


def adapt_candidate(
    initial_model: nn.Module,
    adaptation_loader: DataLoader,
    replay_memory: tuple[torch.Tensor, torch.Tensor],
    mode: str,
    seed: int,
    adaptation_epochs: int,
    adaptation_learning_rate: float,
    replay_ratio: float,
    device: torch.device,
) -> nn.Module:
    if mode not in CANDIDATE_MODES:
        raise ValueError(f"Unknown candidate mode: {mode}")

    candidate = copy.deepcopy(initial_model)
    if mode == "no_update":
        return candidate.eval()
    if replay_ratio <= 0:
        raise ValueError("Replay ratio must be positive")

    optimizer = torch.optim.SGD(
        candidate.parameters(),
        lr=adaptation_learning_rate,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True,
    )
    criterion = nn.CrossEntropyLoss()
    replay_features, replay_targets = replay_memory
    replay_generator = torch.Generator().manual_seed(seed + 20_000)

    for _ in range(adaptation_epochs):
        candidate.train()
        for current_features, current_targets in adaptation_loader:
            current_features = current_features.to(device, non_blocking=True)
            current_targets = current_targets.to(device, non_blocking=True)

            if mode == "replay":
                replay_batch_size = max(
                    1,
                    int(math.ceil(len(current_targets) * replay_ratio)),
                )
                replay_index = torch.randint(
                    len(replay_targets),
                    size=(replay_batch_size,),
                    generator=replay_generator,
                )
                replay_batch_features = replay_features[replay_index].to(
                    device,
                    non_blocking=True,
                )
                replay_batch_targets = replay_targets[replay_index].to(
                    device,
                    non_blocking=True,
                )
                training_features = torch.cat(
                    [current_features, replay_batch_features],
                    dim=0,
                )
                training_targets = torch.cat(
                    [current_targets, replay_batch_targets],
                    dim=0,
                )
            else:
                training_features = current_features
                training_targets = current_targets

            optimizer.zero_grad(set_to_none=True)
            criterion(candidate(training_features), training_targets).backward()
            optimizer.step()
    return candidate.eval()


def evaluate_frozen_candidate(
    incumbent: nn.Module,
    candidate: nn.Module,
    current_loader: DataLoader,
    retention_loader: DataLoader,
    mode: str,
    seed: int,
    evaluation_split: str,
    batch_size: int,
    retention_margin_brier: float,
    retention_margin_error: float,
    device: torch.device,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    current_brier: list[np.ndarray] = []
    current_error: list[np.ndarray] = []
    retention_brier: list[np.ndarray] = []
    retention_error: list[np.ndarray] = []
    incumbent_current_brier_values: list[np.ndarray] = []
    candidate_current_brier_values: list[np.ndarray] = []
    incumbent_current_error_values: list[np.ndarray] = []
    candidate_current_error_values: list[np.ndarray] = []
    incumbent_retention_brier_values: list[np.ndarray] = []
    candidate_retention_brier_values: list[np.ndarray] = []
    incumbent_retention_error_values: list[np.ndarray] = []
    candidate_retention_error_values: list[np.ndarray] = []

    for (current_features, current_targets), (
        probe_features,
        probe_targets,
    ) in zip(current_loader, retention_loader, strict=True):
        current_features = current_features.to(device, non_blocking=True)
        current_targets = current_targets.to(device, non_blocking=True)
        probe_features = probe_features.to(device, non_blocking=True)
        probe_targets = probe_targets.to(device, non_blocking=True)

        with torch.no_grad():
            incumbent_current = incumbent(current_features).softmax(dim=1)
            candidate_current = candidate(current_features).softmax(dim=1)
            incumbent_probe = incumbent(probe_features).softmax(dim=1)
            candidate_probe = candidate(probe_features).softmax(dim=1)

            incumbent_current_brier = multiclass_brier(
                incumbent_current,
                current_targets,
            )
            candidate_current_brier = multiclass_brier(
                candidate_current,
                current_targets,
            )
            incumbent_probe_brier = multiclass_brier(
                incumbent_probe,
                probe_targets,
            )
            candidate_probe_brier = multiclass_brier(
                candidate_probe,
                probe_targets,
            )

        incumbent_current_brier_np = (
            incumbent_current_brier.cpu().numpy().astype(np.float64)
        )
        candidate_current_brier_np = (
            candidate_current_brier.cpu().numpy().astype(np.float64)
        )
        incumbent_probe_brier_np = (
            incumbent_probe_brier.cpu().numpy().astype(np.float64)
        )
        candidate_probe_brier_np = (
            candidate_probe_brier.cpu().numpy().astype(np.float64)
        )
        incumbent_current_error = (
            (incumbent_current.argmax(dim=1) != current_targets)
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        candidate_current_error = (
            (candidate_current.argmax(dim=1) != current_targets)
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        incumbent_probe_error = (
            (incumbent_probe.argmax(dim=1) != probe_targets)
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        candidate_probe_error = (
            (candidate_probe.argmax(dim=1) != probe_targets)
            .cpu()
            .numpy()
            .astype(np.float64)
        )

        incumbent_current_brier_values.append(incumbent_current_brier_np)
        candidate_current_brier_values.append(candidate_current_brier_np)
        incumbent_current_error_values.append(incumbent_current_error)
        candidate_current_error_values.append(candidate_current_error)
        incumbent_retention_brier_values.append(incumbent_probe_brier_np)
        candidate_retention_brier_values.append(candidate_probe_brier_np)
        incumbent_retention_error_values.append(incumbent_probe_error)
        candidate_retention_error_values.append(candidate_probe_error)
        current_brier.append(incumbent_current_brier_np - candidate_current_brier_np)
        current_error.append(incumbent_current_error - candidate_current_error)
        retention_brier.append(
            scaled_noninferiority_difference(
                incumbent_probe_brier_np,
                candidate_probe_brier_np,
                retention_margin_brier,
            )
        )
        retention_error.append(
            scaled_noninferiority_difference(
                incumbent_probe_error,
                candidate_probe_error,
                retention_margin_error,
            )
        )

    arrays = {
        "current_brier": np.concatenate(current_brier),
        "current_error": np.concatenate(current_error),
        "retention_brier": np.concatenate(retention_brier),
        "retention_error": np.concatenate(retention_error),
    }
    absolute_metrics = {
        "incumbent_current_brier_loss": float(
            np.concatenate(incumbent_current_brier_values).mean()
        ),
        "candidate_current_brier_loss": float(
            np.concatenate(candidate_current_brier_values).mean()
        ),
        "incumbent_current_error_rate": float(
            np.concatenate(incumbent_current_error_values).mean()
        ),
        "candidate_current_error_rate": float(
            np.concatenate(candidate_current_error_values).mean()
        ),
        "incumbent_retention_brier_loss": float(
            np.concatenate(incumbent_retention_brier_values).mean()
        ),
        "candidate_retention_brier_loss": float(
            np.concatenate(candidate_retention_brier_values).mean()
        ),
        "incumbent_retention_error_rate": float(
            np.concatenate(incumbent_retention_error_values).mean()
        ),
        "candidate_retention_error_rate": float(
            np.concatenate(candidate_retention_error_values).mean()
        ),
    }
    horizon = len(arrays["current_brier"])
    raw_crossings: dict[str, int | None] = {}
    final_log_evidence: dict[str, float] = {}
    for metric, values in arrays.items():
        crossing, final_log_value = promotion_summary(values)
        raw_crossings[metric] = crossing
        final_log_evidence[metric] = final_log_value

    current_raw = (
        max(raw_crossings["current_brier"], raw_crossings["current_error"])
        if raw_crossings["current_brier"] is not None
        and raw_crossings["current_error"] is not None
        else None
    )
    full_raw = (
        max(value for value in raw_crossings.values() if value is not None)
        if all(value is not None for value in raw_crossings.values())
        else None
    )
    current_actionable = actionable_boundary(current_raw, batch_size, horizon)
    full_actionable = actionable_boundary(full_raw, batch_size, horizon)

    row: dict[str, object] = {
        "seed": seed,
        "candidate_mode": mode,
        "evaluation_split": evaluation_split,
        "n_current_observations": horizon,
        "n_retention_probes": len(arrays["retention_brier"]),
        "current_mean_brier_advantage": float(arrays["current_brier"].mean()),
        "current_mean_error_advantage": float(arrays["current_error"].mean()),
        "retention_mean_brier_difference": float(
            (
                arrays["retention_brier"] * (1.0 + retention_margin_brier)
                - retention_margin_brier
            ).mean()
        ),
        "retention_mean_error_difference": float(
            (
                arrays["retention_error"] * (1.0 + retention_margin_error)
                - retention_margin_error
            ).mean()
        ),
        "current_only_promotion_time": current_actionable,
        "plasticity_retention_promotion_time": full_actionable,
        **absolute_metrics,
    }
    for metric in arrays:
        row[f"{metric}_crossing_index"] = raw_crossings[metric]
        row[f"{metric}_final_log_evidence"] = final_log_evidence[metric]
    return row, arrays


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.checkpoint_dir is not None:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if args.evaluation_size > 5_000:
        raise ValueError("Each evaluation partition has 5,000 observations per task")
    if args.adaptation_size > 15_000:
        raise ValueError("The new-task model-fitting partition has 15,000 images")
    if args.replay_size > 15_000:
        raise ValueError("The old-task model-fitting partition has 15,000 images")

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CHANNEL_MEAN, CHANNEL_STD),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CHANNEL_MEAN, CHANNEL_STD),
        ]
    )
    train_augmented = datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )
    train_evaluation = datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        download=True,
        transform=evaluation_transform,
    )
    old_indices = class_indices(train_augmented.targets, set(range(5)))
    new_indices = class_indices(train_augmented.targets, set(range(5, 10)))
    partition_rng = np.random.default_rng(args.partition_seed)
    old_partition = partition_rng.permutation(old_indices)
    new_partition = partition_rng.permutation(new_indices)
    old_train_indices = old_partition[:15_000]
    new_train_indices = new_partition[:15_000]
    evaluation_partitions = {
        "development": (
            new_partition[15_000:20_000],
            old_partition[15_000:20_000],
        ),
        "confirmation": (
            new_partition[20_000:25_000],
            old_partition[20_000:25_000],
        ),
    }
    current_indices, retention_indices = evaluation_partitions[
        args.evaluation_split
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows: list[dict[str, object]] = []
    initial_rows: list[dict[str, object]] = []
    for seed in args.seeds:
        set_seed(seed)
        rng = np.random.default_rng(seed)
        initial_order = rng.permutation(old_train_indices)
        adaptation_order = rng.permutation(new_train_indices)[
            : args.adaptation_size
        ]
        current_order = rng.permutation(current_indices)[
            : args.evaluation_size
        ]
        probe_order = rng.permutation(retention_indices)[: args.evaluation_size]
        replay_order = rng.permutation(old_train_indices)[: args.replay_size]

        common_loader_options = {
            "batch_size": args.batch_size,
            "num_workers": args.workers,
            "pin_memory": device.type == "cuda",
            "persistent_workers": args.workers > 0,
        }
        initial_loader = DataLoader(
            Subset(train_augmented, initial_order.tolist()),
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
            **common_loader_options,
        )
        current_loader = DataLoader(
            Subset(train_evaluation, current_order.tolist()),
            shuffle=False,
            **common_loader_options,
        )
        retention_loader = DataLoader(
            Subset(train_evaluation, probe_order.tolist()),
            shuffle=False,
            **common_loader_options,
        )
        initial_model = make_model().to(device)
        checkpoint_path = (
            args.checkpoint_dir
            / f"split_cifar10_train15k_initial_seed_{seed}.pt"
            if args.checkpoint_dir is not None
            else None
        )
        if checkpoint_path is not None and checkpoint_path.exists():
            initial_model.load_state_dict(
                torch.load(
                    checkpoint_path,
                    map_location=device,
                    weights_only=True,
                )
            )
        else:
            train_initial_model(
                initial_model,
                initial_loader,
                device,
                args.initial_epochs,
            )
            if checkpoint_path is not None:
                torch.save(initial_model.state_dict(), checkpoint_path)
        initial_brier, initial_error = model_metrics(
            initial_model,
            retention_loader,
            device,
        )
        initial_rows.append(
            {
                "seed": seed,
                "old_task_brier_loss": initial_brier,
                "old_task_error_rate": initial_error,
                "evaluation_split": args.evaluation_split,
            }
        )

        replay_memory = load_replay_memory(
            train_augmented,
            replay_order,
            args.batch_size,
            args.workers,
        )
        for mode in args.candidate_modes:
            adaptation_loader = DataLoader(
                Subset(train_augmented, adaptation_order.tolist()),
                shuffle=True,
                generator=torch.Generator().manual_seed(seed + 10_000),
                **common_loader_options,
            )
            candidate = adapt_candidate(
                initial_model=initial_model,
                adaptation_loader=adaptation_loader,
                replay_memory=replay_memory,
                mode=mode,
                seed=seed,
                adaptation_epochs=args.adaptation_epochs,
                adaptation_learning_rate=args.adaptation_learning_rate,
                replay_ratio=args.replay_ratio,
                device=device,
            )
            row, arrays = evaluate_frozen_candidate(
                incumbent=initial_model.eval(),
                candidate=candidate,
                current_loader=current_loader,
                retention_loader=retention_loader,
                mode=mode,
                seed=seed,
                evaluation_split=args.evaluation_split,
                batch_size=args.batch_size,
                retention_margin_brier=args.retention_margin_brier,
                retention_margin_error=args.retention_margin_error,
                device=device,
            )
            rows.append(row)
            for metric, values in arrays.items():
                np.save(
                    args.output_dir
                    / f"split_cifar10_{mode}_{metric}_seed_{seed}.npy",
                    values,
                )
            print(json.dumps(row), flush=True)

    for filename, output_rows in (
        ("Split_CIFAR10_candidate_results.csv", rows),
        ("Split_CIFAR10_initial_models.csv", initial_rows),
    ):
        with (args.output_dir / filename).open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
            writer.writeheader()
            writer.writerows(output_rows)

    summary = {
        "configuration": {
            "dataset": "Split CIFAR-10",
            "old_task_classes": [0, 1, 2, 3, 4],
            "new_task_classes": [5, 6, 7, 8, 9],
            "architecture": "ResNet-18 with CIFAR stem",
            "initial_epochs": args.initial_epochs,
            "adaptation_epochs": args.adaptation_epochs,
            "adaptation_size": args.adaptation_size,
            "evaluation_size": args.evaluation_size,
            "model_fitting_size_per_task": 15_000,
            "development_size_per_task": 5_000,
            "confirmation_size_per_task": 5_000,
            "evaluation_split": args.evaluation_split,
            "partition_seed": args.partition_seed,
            "batch_size": args.batch_size,
            "adaptation_learning_rate": args.adaptation_learning_rate,
            "replay_size": args.replay_size,
            "replay_ratio": args.replay_ratio,
            "retention_margin_brier": args.retention_margin_brier,
            "retention_margin_error": args.retention_margin_error,
            "candidate_modes": list(args.candidate_modes),
            "seeds": args.seeds,
            "device": str(device),
            "decision_rule": (
                "promotion requires current-task Brier and error superiority "
                "and old-task Brier and error noninferiority"
            ),
            "evidence_policy": (
                "candidates are frozen before monitoring; the development "
                "partition is held out from model fitting and is used only for "
                "configuration selection"
                if args.evaluation_split == "development"
                else
                "candidates are frozen before monitoring; the confirmation "
                "partition is held out from model fitting and configuration "
                "selection"
            ),
        },
        "initial_models": initial_rows,
        "runs": rows,
    }
    (args.output_dir / "Split_CIFAR10_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
