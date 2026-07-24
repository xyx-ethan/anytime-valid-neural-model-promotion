#!/usr/bin/env python3
"""Reproduce the simulations and tabular neural-stream analyses."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from river import datasets
from scipy.stats import binom
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "source_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20260723
RUNS = 8000
HORIZON = 1000
ALPHA = 0.05
LOOKS = np.arange(50, HORIZON + 1, 50)
LAMBDA_GRID = np.linspace(0.0, 1.0, 17)[1:]
NEURAL_WARMUP = 600
NEURAL_BATCH_SIZE = 32
NEURAL_SEEDS = tuple(range(20))
NEURAL_ARCHITECTURES = {
    "8": (8,),
    "16": (16,),
    "16-8": (16, 8),
}

def paired_binary_losses(
    rng: np.random.Generator,
    incumbent_error: float,
    candidate_error: float,
) -> np.ndarray:
    """Generate paired 0-1 loss differences for independent model errors."""
    incumbent = rng.random((RUNS, HORIZON)) < incumbent_error
    candidate = rng.random((RUNS, HORIZON)) < candidate_error
    return incumbent.astype(np.int8) - candidate.astype(np.int8)


def dependent_null_losses(rng: np.random.Generator) -> np.ndarray:
    """Generate equal-performance losses with persistent shared difficulty."""
    differences = np.empty((RUNS, HORIZON), dtype=np.int8)
    difficult = rng.random(RUNS) < 0.5
    for time in range(HORIZON):
        switch = rng.random(RUNS) < 0.03
        difficult = np.logical_xor(difficult, switch)
        error_probability = np.where(difficult, 0.45, 0.05)
        incumbent = rng.random(RUNS) < error_probability
        shared_error = rng.random(RUNS) < 0.75
        independent_candidate = rng.random(RUNS) < error_probability
        candidate = np.where(shared_error, incumbent, independent_candidate)
        differences[:, time] = (
            incumbent.astype(np.int8) - candidate.astype(np.int8)
        )
    return differences


def mixture_log_evidence(differences: np.ndarray) -> np.ndarray:
    """Return the log of the equally weighted betting mixture at every time."""
    log_mixture = np.full(differences.shape, -np.inf, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        for betting_fraction in LAMBDA_GRID:
            log_factors = np.log1p(betting_fraction * differences)
            component = np.cumsum(log_factors, axis=1)
            log_mixture = np.logaddexp(log_mixture, component)
    return log_mixture - math.log(len(LAMBDA_GRID))


def one_stream_log_evidence(differences: np.ndarray) -> np.ndarray:
    """Return the mixture log evidence for one observed loss sequence."""
    sequence = np.asarray(differences, dtype=np.float64).reshape(1, -1)
    return mixture_log_evidence(sequence)[0]


def first_crossing(log_evidence: np.ndarray) -> np.ndarray:
    """Return one-based crossing times, or HORIZON + 1 when no crossing occurs."""
    crossed = log_evidence >= math.log(1.0 / ALPHA)
    return np.where(crossed.any(axis=1), crossed.argmax(axis=1) + 1, HORIZON + 1)


def scheduled_sign_tests(differences: np.ndarray) -> dict[str, np.ndarray]:
    """Run exact one-sided paired sign tests at 20 prespecified looks."""
    positive = np.cumsum(differences == 1, axis=1)[:, LOOKS - 1]
    discordant = np.cumsum(differences != 0, axis=1)[:, LOOKS - 1]
    p_values = binom.sf(positive - 1, discordant, 0.5)

    results: dict[str, np.ndarray] = {}
    thresholds = {
        "unadjusted 20-look test": ALPHA,
        "Bonferroni 20-look test": ALPHA / len(LOOKS),
    }
    for method, threshold in thresholds.items():
        crossed = p_values <= threshold
        results[method] = np.where(
            crossed.any(axis=1),
            LOOKS[crossed.argmax(axis=1)],
            HORIZON + 1,
        )
    return results


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return center - half_width, center + half_width


def evaluate_simulations() -> tuple[pd.DataFrame, dict[str, object]]:
    """Evaluate false promotion, power, and a conjunctive promotion gate."""
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "seed": SEED,
        "runs": RUNS,
        "horizon": HORIZON,
        "alpha": ALPHA,
        "looks": LOOKS.tolist(),
        "null": {},
        "dependent_null": {},
        "gain": {},
        "conjunctive_null": {},
    }

    for equal_error in (0.10, 0.20, 0.30):
        differences = paired_binary_losses(rng, equal_error, equal_error)
        crossings = {
            "e-process": first_crossing(mixture_log_evidence(differences)),
            **scheduled_sign_tests(differences),
        }
        scenario_summary: dict[str, object] = {}
        for method, crossing_time in crossings.items():
            successes = int(np.sum(crossing_time <= HORIZON))
            rate = successes / RUNS
            low, high = wilson_interval(successes, RUNS)
            rows.append(
                {
                    "panel": "false_promotion",
                    "scenario": f"equal error {equal_error:.2f}",
                    "method": method,
                    "x": equal_error,
                    "y": rate,
                    "lower": low,
                    "upper": high,
                    "n": RUNS,
                }
            )
            scenario_summary[method] = {
                "rate": rate,
                "ci95": [low, high],
            }
        summary["null"][f"{equal_error:.2f}"] = scenario_summary

    differences = dependent_null_losses(np.random.default_rng(SEED + 1))
    crossings = {
        "e-process": first_crossing(mixture_log_evidence(differences)),
        **scheduled_sign_tests(differences),
    }
    for method, crossing_time in crossings.items():
        successes = int(np.sum(crossing_time <= HORIZON))
        rate = successes / RUNS
        low, high = wilson_interval(successes, RUNS)
        summary["dependent_null"][method] = {
            "rate": rate,
            "ci95": [low, high],
        }
        rows.append(
            {
                "panel": "dependent_false_promotion",
                "scenario": "persistent shared difficulty",
                "method": method,
                "x": np.nan,
                "y": rate,
                "lower": low,
                "upper": high,
                "n": RUNS,
            }
        )

    representative_crossings: dict[str, np.ndarray] | None = None
    for gap in (0.05, 0.10, 0.20):
        differences = paired_binary_losses(rng, 0.30, 0.30 - gap)
        crossings = {
            "e-process": first_crossing(mixture_log_evidence(differences)),
            **scheduled_sign_tests(differences),
        }
        scenario_summary = {}
        for method, crossing_time in crossings.items():
            detected = crossing_time <= HORIZON
            successes = int(np.sum(detected))
            rate = successes / RUNS
            low, high = wilson_interval(successes, RUNS)
            median_time = float(np.median(crossing_time[detected])) if successes else None
            scenario_summary[method] = {
                "promotion_rate": rate,
                "ci95": [low, high],
                "median_time_among_promoted": median_time,
            }
            rows.append(
                {
                    "panel": "promotion_summary",
                    "scenario": f"loss gap {gap:.2f}",
                    "method": method,
                    "x": gap,
                    "y": rate,
                    "lower": low,
                    "upper": high,
                    "n": RUNS,
                }
            )
        summary["gain"][f"{gap:.2f}"] = scenario_summary
        if math.isclose(gap, 0.10):
            representative_crossings = crossings

    if representative_crossings is None:
        raise RuntimeError("Representative loss-gap simulation was not generated")

    favorable_differences = paired_binary_losses(rng, 0.30, 0.20)
    null_differences = paired_binary_losses(rng, 0.20, 0.20)
    favorable_crossing = first_crossing(
        mixture_log_evidence(favorable_differences)
    )
    null_crossing = first_crossing(mixture_log_evidence(null_differences))
    conjunctive_crossing = np.maximum(favorable_crossing, null_crossing)
    favorable_successes = int(np.sum(favorable_crossing <= HORIZON))
    conjunctive_successes = int(np.sum(conjunctive_crossing <= HORIZON))
    conjunctive_low, conjunctive_high = wilson_interval(
        conjunctive_successes,
        RUNS,
    )
    summary["conjunctive_null"] = {
        "description": (
            "criterion 1 has a 0.10 loss advantage and criterion 2 has equal "
            "error probabilities of 0.20"
        ),
        "criterion_1_promotion_rate": favorable_successes / RUNS,
        "conjunctive_promotion_rate": conjunctive_successes / RUNS,
        "conjunctive_ci95": [conjunctive_low, conjunctive_high],
    }

    time_grid = np.arange(0, HORIZON + 1, 10)
    for method, crossing_time in representative_crossings.items():
        for time in time_grid:
            rows.append(
                {
                    "panel": "promotion_curve",
                    "scenario": "loss gap 0.10",
                    "method": method,
                    "x": int(time),
                    "y": float(np.mean(crossing_time <= time)),
                    "lower": np.nan,
                    "upper": np.nan,
                    "n": RUNS,
                }
            )

    return pd.DataFrame(rows), summary


def numeric_stream(stream_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Materialize a River stream as a fixed numeric matrix and binary target."""
    observations = list(getattr(datasets, stream_name)())
    feature_names = tuple(sorted(observations[0][0]))
    for features, _ in observations:
        if tuple(sorted(features)) != feature_names:
            raise ValueError(f"{stream_name} does not have a fixed numeric schema")
    features = np.asarray(
        [
            [float(observation[name]) for name in feature_names]
            for observation, _ in observations
        ],
        dtype=np.float64,
    )
    targets = np.asarray([int(target) for _, target in observations], dtype=np.int8)
    if set(np.unique(targets)) != {0, 1}:
        raise ValueError(f"{stream_name} is not a binary classification stream")
    return features, targets


def neural_stream_loss_differences(
    features: np.ndarray,
    targets: np.ndarray,
    hidden_layers: tuple[int, ...],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compare frozen and continually updated MLPs using prequential Brier loss."""
    scaler = StandardScaler().fit(features[:NEURAL_WARMUP])
    standardized = scaler.transform(features)
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="sgd",
        alpha=1e-4,
        learning_rate="constant",
        learning_rate_init=0.01,
        momentum=0.9,
        nesterovs_momentum=True,
        max_iter=1,
        shuffle=False,
        random_state=seed,
    )

    first_update = True
    classes = np.asarray([0, 1], dtype=np.int8)
    for lower in range(0, NEURAL_WARMUP, NEURAL_BATCH_SIZE):
        upper = min(NEURAL_WARMUP, lower + NEURAL_BATCH_SIZE)
        model.partial_fit(
            standardized[lower:upper],
            targets[lower:upper],
            classes=classes if first_update else None,
        )
        first_update = False

    incumbent = copy.deepcopy(model)
    candidate = copy.deepcopy(model)
    incumbent_losses: list[float] = []
    candidate_losses: list[float] = []
    for lower in range(NEURAL_WARMUP, len(targets), NEURAL_BATCH_SIZE):
        upper = min(len(targets), lower + NEURAL_BATCH_SIZE)
        batch_features = standardized[lower:upper]
        batch_targets = targets[lower:upper]
        incumbent_probabilities = incumbent.predict_proba(batch_features)[:, 1]
        candidate_probabilities = candidate.predict_proba(batch_features)[:, 1]
        incumbent_losses.extend((incumbent_probabilities - batch_targets) ** 2)
        candidate_losses.extend((candidate_probabilities - batch_targets) ** 2)
        candidate.partial_fit(batch_features, batch_targets)

    incumbent_array = np.asarray(incumbent_losses, dtype=np.float64)
    candidate_array = np.asarray(candidate_losses, dtype=np.float64)
    return incumbent_array - candidate_array, incumbent_array, candidate_array


def evaluate_neural_streams() -> tuple[pd.DataFrame, dict[str, object]]:
    """Evaluate promotion of continually learning MLPs across initializations."""
    rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "configuration": {
            "warmup": NEURAL_WARMUP,
            "update_batch_size": NEURAL_BATCH_SIZE,
            "seeds": list(NEURAL_SEEDS),
            "architectures": {
                label: list(hidden_layers)
                for label, hidden_layers in NEURAL_ARCHITECTURES.items()
            },
            "loss": "binary Brier loss",
            "optimizer": "SGD",
            "learning_rate": 0.01,
            "momentum": 0.9,
            "l2_penalty": 1e-4,
        },
        "streams": {},
    }
    threshold = math.log(1.0 / ALPHA)

    for stream_name in ("Elec2", "Phishing", "Bananas"):
        features, targets = numeric_stream(stream_name)
        stream_summary: dict[str, object] = {}
        for architecture, hidden_layers in NEURAL_ARCHITECTURES.items():
            architecture_rows: list[dict[str, object]] = []
            for seed in NEURAL_SEEDS:
                differences, incumbent_losses, candidate_losses = (
                    neural_stream_loss_differences(
                        features,
                        targets,
                        hidden_layers,
                        seed,
                    )
                )
                log_evidence = one_stream_log_evidence(differences)
                crossings = np.flatnonzero(log_evidence >= threshold)
                promotion_time = int(crossings[0] + 1) if crossings.size else None
                row = {
                    "stream": stream_name,
                    "architecture": architecture,
                    "seed": seed,
                    "n_post_warmup": int(len(differences)),
                    "incumbent_mean_brier": float(np.mean(incumbent_losses)),
                    "candidate_mean_brier": float(np.mean(candidate_losses)),
                    "mean_brier_gap": float(np.mean(differences)),
                    "promotion_time": promotion_time,
                    "final_log_evidence": float(log_evidence[-1]),
                }
                rows.append(row)
                architecture_rows.append(row)

            gaps = np.asarray(
                [row["mean_brier_gap"] for row in architecture_rows],
                dtype=np.float64,
            )
            promotion_times = np.asarray(
                [
                    row["promotion_time"]
                    for row in architecture_rows
                    if row["promotion_time"] is not None
                ],
                dtype=np.float64,
            )
            incumbent_brier = np.asarray(
                [row["incumbent_mean_brier"] for row in architecture_rows],
                dtype=np.float64,
            )
            candidate_brier = np.asarray(
                [row["candidate_mean_brier"] for row in architecture_rows],
                dtype=np.float64,
            )
            stream_summary[architecture] = {
                "runs": len(architecture_rows),
                "promoted": int(len(promotion_times)),
                "mean_brier_gap_median": float(np.median(gaps)),
                "mean_brier_gap_range": [float(np.min(gaps)), float(np.max(gaps))],
                "promotion_time_median": (
                    float(np.median(promotion_times))
                    if len(promotion_times)
                    else None
                ),
                "promotion_time_range": (
                    [int(np.min(promotion_times)), int(np.max(promotion_times))]
                    if len(promotion_times)
                    else None
                ),
                "incumbent_mean_brier_median": float(np.median(incumbent_brier)),
                "candidate_mean_brier_median": float(np.median(candidate_brier)),
            }
        summary["streams"][stream_name] = stream_summary

    return pd.DataFrame(rows), summary


def main() -> None:
    simulation_data, simulation_summary = evaluate_simulations()
    neural_data, neural_summary = evaluate_neural_streams()
    simulation_data.to_csv(DATA_DIR / "Figure1_source_data.csv", index=False)
    neural_data.to_csv(DATA_DIR / "Neural_stream_results.csv", index=False)

    summary = {
        "simulation": simulation_summary,
        "neural_streams": neural_summary,
    }
    (DATA_DIR / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
