#!/usr/bin/env python3
"""Evaluate neural-model promotion on labeled CIFAR-10-C streams."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from run_cifar_stream import (
    ROOT,
    make_model,
    multiclass_brier,
    promotion_summary,
    train_initial_model,
)


CORRUPTIONS = (
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
)
CHANNEL_MEAN = (0.4914, 0.4822, 0.4465)
CHANNEL_STD = (0.2470, 0.2435, 0.2616)
IMAGES_PER_SEVERITY = 2_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / ".data")
    parser.add_argument(
        "--cifar10c-dir",
        type=Path,
        default=ROOT / ".data" / "CIFAR-10-C",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "source_data" / "cifar10c_stream",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--online-learning-rate", type=float, default=0.001)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class CIFAR10CStream(Dataset):
    """Use each base test image once as corruption severity increases."""

    def __init__(
        self,
        cifar10c_dir: Path,
        corruption: str,
        base_order: np.ndarray,
    ) -> None:
        self.images = np.load(
            cifar10c_dir / f"{corruption}.npy",
            mmap_mode="r",
        )
        self.labels = np.load(cifar10c_dir / "labels.npy", mmap_mode="r")
        self.base_order = np.asarray(base_order, dtype=np.int64)
        if len(self.base_order) != 10_000:
            raise ValueError("CIFAR-10-C streams require 10,000 base images")
        self.normalize = transforms.Normalize(CHANNEL_MEAN, CHANNEL_STD)

    def __len__(self) -> int:
        return len(self.base_order)

    def __getitem__(self, position: int) -> tuple[torch.Tensor, int, int]:
        base_index = int(self.base_order[position])
        severity = position // IMAGES_PER_SEVERITY
        array_index = severity * 10_000 + base_index
        image = np.array(self.images[array_index], copy=True)
        tensor = torch.from_numpy(image).permute(2, 0, 1).float().div_(255.0)
        label_index = array_index if len(self.labels) == 50_000 else base_index
        target = int(self.labels[label_index])
        return self.normalize(tensor), target, severity + 1


def clean_test_metrics(
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


def evaluate_corruption(
    initial_model: nn.Module,
    stream: Dataset,
    seed: int,
    corruption: str,
    batch_size: int,
    online_learning_rate: float,
    workers: int,
    device: torch.device,
) -> tuple[dict[str, object], np.ndarray, np.ndarray, list[dict[str, object]]]:
    loader = DataLoader(
        stream,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
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

    brier_differences: list[np.ndarray] = []
    error_differences: list[np.ndarray] = []
    severities: list[np.ndarray] = []
    for features, targets, severity in loader:
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.no_grad():
            incumbent_probabilities = incumbent(features).softmax(dim=1)
            candidate.eval()
            candidate_probabilities = candidate(features).softmax(dim=1)
            incumbent_brier = multiclass_brier(
                incumbent_probabilities,
                targets,
            )
            candidate_brier = multiclass_brier(
                candidate_probabilities,
                targets,
            )
        brier_differences.append(
            (incumbent_brier - candidate_brier)
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        incumbent_error = incumbent_probabilities.argmax(dim=1) != targets
        candidate_error = candidate_probabilities.argmax(dim=1) != targets
        error_differences.append(
            (incumbent_error.to(torch.float64) - candidate_error.to(torch.float64))
            .cpu()
            .numpy()
        )
        severities.append(severity.numpy().astype(np.int64))

        candidate.train()
        optimizer.zero_grad(set_to_none=True)
        criterion(candidate(features), targets).backward()
        optimizer.step()

    brier_array = np.concatenate(brier_differences)
    error_array = np.concatenate(error_differences)
    severity_array = np.concatenate(severities)
    brier_time, brier_log_evidence = promotion_summary(brier_array)
    error_time, error_log_evidence = promotion_summary(error_array)
    joint_time = (
        max(brier_time, error_time)
        if brier_time is not None and error_time is not None
        else None
    )

    severity_rows: list[dict[str, object]] = []
    for severity in range(1, 6):
        selected = severity_array == severity
        severity_rows.append(
            {
                "seed": seed,
                "corruption": corruption,
                "severity": severity,
                "n_observations": int(selected.sum()),
                "mean_brier_advantage": float(brier_array[selected].mean()),
                "mean_error_advantage": float(error_array[selected].mean()),
            }
        )

    row: dict[str, object] = {
        "dataset": "CIFAR-10-C",
        "architecture": "ResNet-18",
        "seed": seed,
        "corruption": corruption,
        "n_stream_observations": len(brier_array),
        "online_learning_rate": online_learning_rate,
        "mean_brier_advantage": float(brier_array.mean()),
        "mean_error_advantage": float(error_array.mean()),
        "brier_promotion_time": brier_time,
        "error_promotion_time": error_time,
        "promotion_time": joint_time,
        "brier_final_log_evidence": brier_log_evidence,
        "error_final_log_evidence": error_log_evidence,
    }
    return row, brier_array, error_array, severity_rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    missing = [
        name
        for name in (*CORRUPTIONS, "labels")
        if not (args.cifar10c_dir / f"{name}.npy").exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing CIFAR-10-C arrays in {args.cifar10c_dir}: {missing}"
        )

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CHANNEL_MEAN, CHANNEL_STD),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CHANNEL_MEAN, CHANNEL_STD),
        ]
    )
    train_dataset = datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )
    test_dataset = datasets.CIFAR10(
        root=args.data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows: list[dict[str, object]] = []
    severity_rows: list[dict[str, object]] = []
    clean_rows: list[dict[str, object]] = []
    for seed in args.seeds:
        set_seed(seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
        )
        clean_test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
        )
        initial_model = make_model().to(device)
        train_initial_model(
            initial_model,
            train_loader,
            device,
            args.warmup_epochs,
        )
        clean_brier, clean_error = clean_test_metrics(
            initial_model,
            clean_test_loader,
            device,
        )
        clean_rows.append(
            {
                "seed": seed,
                "clean_brier_loss": clean_brier,
                "clean_error_rate": clean_error,
            }
        )

        base_order = np.random.default_rng(seed).permutation(10_000)
        for corruption in CORRUPTIONS:
            stream = CIFAR10CStream(
                args.cifar10c_dir,
                corruption,
                base_order,
            )
            row, brier, error, per_severity = evaluate_corruption(
                initial_model=initial_model,
                stream=stream,
                seed=seed,
                corruption=corruption,
                batch_size=args.batch_size,
                online_learning_rate=args.online_learning_rate,
                workers=args.workers,
                device=device,
            )
            rows.append(row)
            severity_rows.extend(per_severity)
            np.save(
                args.output_dir
                / f"cifar10c_{corruption}_brier_seed_{seed}.npy",
                brier,
            )
            np.save(
                args.output_dir
                / f"cifar10c_{corruption}_error_seed_{seed}.npy",
                error,
            )
            print(json.dumps(row), flush=True)

    for filename, output_rows in (
        ("CIFAR10C_stream_results.csv", rows),
        ("CIFAR10C_severity_results.csv", severity_rows),
        ("CIFAR10_clean_starting_models.csv", clean_rows),
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
            "dataset": "CIFAR-10-C",
            "corruptions": list(CORRUPTIONS),
            "architecture": "ResNet-18 with CIFAR stem",
            "clean_training_images": 50_000,
            "warmup_epochs": args.warmup_epochs,
            "stream_images_per_corruption": 10_000,
            "unique_base_images_per_severity": IMAGES_PER_SEVERITY,
            "batch_size": args.batch_size,
            "online_learning_rate": args.online_learning_rate,
            "seeds": args.seeds,
            "device": str(device),
        },
        "clean_starting_models": clean_rows,
        "runs": rows,
        "severity_blocks": severity_rows,
    }
    (args.output_dir / "CIFAR10C_stream_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
