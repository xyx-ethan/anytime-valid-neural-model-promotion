#!/usr/bin/env python3
"""Evaluate anytime-valid promotion for an online ResNet-18 on CIFAR-10."""

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


ALPHA = 0.05
LAMBDA_GRID = np.linspace(0.0, 1.0, 17, dtype=np.float64)[1:]
ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / ".data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "source_data" / "cifar10_stream",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--warmup-size", type=int, default=15_000)
    parser.add_argument("--warmup-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--online-learning-rate", type=float, default=0.005)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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


def promotion_summary(differences: np.ndarray) -> tuple[int | None, float]:
    log_components = np.zeros(len(LAMBDA_GRID), dtype=np.float64)
    threshold = math.log(1.0 / ALPHA)
    first_crossing: int | None = None
    final_log_evidence = -math.inf

    for index, difference in enumerate(differences, start=1):
        with np.errstate(divide="ignore", invalid="ignore"):
            log_components += np.log1p(LAMBDA_GRID * difference)
        finite = np.isfinite(log_components)
        if finite.any():
            maximum = float(np.max(log_components[finite]))
            final_log_evidence = maximum + math.log(
                float(np.exp(log_components[finite] - maximum).sum())
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


def evaluate_seed(
    dataset: datasets.CIFAR10,
    seed: int,
    warmup_size: int,
    warmup_epochs: int,
    batch_size: int,
    online_learning_rate: float,
    workers: int,
    device: torch.device,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    set_seed(seed)
    permutation = np.random.default_rng(seed).permutation(len(dataset))
    warmup_indices = permutation[:warmup_size].tolist()
    stream_indices = permutation[warmup_size:].tolist()
    loader_generator = torch.Generator().manual_seed(seed)
    common_loader_options = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
    }
    warmup_loader = DataLoader(
        Subset(dataset, warmup_indices),
        shuffle=True,
        generator=loader_generator,
        **common_loader_options,
    )
    stream_loader = DataLoader(
        Subset(dataset, stream_indices),
        shuffle=False,
        **common_loader_options,
    )

    initial_model = make_model().to(device)
    train_initial_model(initial_model, warmup_loader, device, warmup_epochs)
    incumbent = copy.deepcopy(initial_model).eval()
    candidate = copy.deepcopy(initial_model).train()
    optimizer = torch.optim.SGD(
        candidate.parameters(),
        lr=online_learning_rate,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True,
    )
    criterion = nn.CrossEntropyLoss()

    incumbent_losses: list[np.ndarray] = []
    candidate_losses: list[np.ndarray] = []
    brier_differences: list[np.ndarray] = []
    error_differences: list[np.ndarray] = []
    for features, targets in stream_loader:
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.no_grad():
            incumbent_probabilities = incumbent(features).softmax(dim=1)
            candidate.eval()
            candidate_probabilities = candidate(features).softmax(dim=1)
            incumbent_loss = multiclass_brier(
                incumbent_probabilities,
                targets,
            )
            candidate_loss = multiclass_brier(
                candidate_probabilities,
                targets,
            )

        incumbent_numpy = incumbent_loss.cpu().numpy().astype(np.float64)
        candidate_numpy = candidate_loss.cpu().numpy().astype(np.float64)
        incumbent_losses.append(incumbent_numpy)
        candidate_losses.append(candidate_numpy)
        brier_differences.append(incumbent_numpy - candidate_numpy)
        incumbent_error = (
            incumbent_probabilities.argmax(dim=1) != targets
        ).cpu().numpy().astype(np.float64)
        candidate_error = (
            candidate_probabilities.argmax(dim=1) != targets
        ).cpu().numpy().astype(np.float64)
        error_differences.append(incumbent_error - candidate_error)

        candidate.train()
        optimizer.zero_grad(set_to_none=True)
        criterion(candidate(features), targets).backward()
        optimizer.step()

    incumbent_array = np.concatenate(incumbent_losses)
    candidate_array = np.concatenate(candidate_losses)
    brier_difference_array = np.concatenate(brier_differences)
    error_difference_array = np.concatenate(error_differences)
    brier_promotion_time, brier_final_log_evidence = promotion_summary(
        brier_difference_array
    )
    error_promotion_time, error_final_log_evidence = promotion_summary(
        error_difference_array
    )
    promotion_time = (
        max(brier_promotion_time, error_promotion_time)
        if brier_promotion_time is not None and error_promotion_time is not None
        else None
    )
    row: dict[str, object] = {
        "dataset": "CIFAR-10",
        "architecture": "ResNet-18",
        "seed": seed,
        "warmup_size": warmup_size,
        "warmup_epochs": warmup_epochs,
        "update_batch_size": batch_size,
        "online_learning_rate": online_learning_rate,
        "n_post_warmup": len(brier_difference_array),
        "incumbent_mean_brier": float(incumbent_array.mean()),
        "candidate_mean_brier": float(candidate_array.mean()),
        "mean_brier_gap": float(brier_difference_array.mean()),
        "mean_error_gap": float(error_difference_array.mean()),
        "brier_promotion_time": brier_promotion_time,
        "error_promotion_time": error_promotion_time,
        "promotion_time": promotion_time,
        "brier_final_log_evidence": brier_final_log_evidence,
        "error_final_log_evidence": error_final_log_evidence,
    }
    return row, brier_difference_array, error_difference_array


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ]
    )
    dataset = datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict[str, object]] = []
    for seed in args.seeds:
        row, brier_differences, error_differences = evaluate_seed(
            dataset=dataset,
            seed=seed,
            warmup_size=args.warmup_size,
            warmup_epochs=args.warmup_epochs,
            batch_size=args.batch_size,
            online_learning_rate=args.online_learning_rate,
            workers=args.workers,
            device=device,
        )
        rows.append(row)
        np.save(
            args.output_dir / f"cifar10_loss_differences_seed_{seed}.npy",
            brier_differences,
        )
        np.save(
            args.output_dir / f"cifar10_error_differences_seed_{seed}.npy",
            error_differences,
        )
        print(json.dumps(row), flush=True)

    csv_path = args.output_dir / "CIFAR10_neural_stream_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "configuration": {
            "dataset": "CIFAR-10 training set",
            "architecture": "ResNet-18 with CIFAR stem",
            "warmup_size": args.warmup_size,
            "warmup_epochs": args.warmup_epochs,
            "update_batch_size": args.batch_size,
            "online_learning_rate": args.online_learning_rate,
            "seeds": args.seeds,
            "initial_optimizer": "SGD",
            "online_optimizer": "SGD",
            "metrics": [
                "multiclass Brier loss divided by two",
                "zero-one classification error",
            ],
            "promotion_rule": (
                "both metric-specific evidence processes cross at alpha"
            ),
            "alpha": ALPHA,
            "lambda_grid": LAMBDA_GRID.tolist(),
            "device": str(device),
        },
        "runs": rows,
    }
    (args.output_dir / "CIFAR10_neural_stream_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
