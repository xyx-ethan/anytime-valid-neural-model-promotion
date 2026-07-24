#!/usr/bin/env python3
"""Reproduce component-crossing control and detection simulations."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "source_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20260723
RUNS = 8000
HORIZON = 1000
ALPHA = 0.05
LOOKS = np.arange(50, HORIZON + 1, 50)
EVIDENCE_SCALE = 0.5
LAMBDA_GRID = np.linspace(0.0, 1.9, 17)[1:]
STRONG_LAMBDA_GRID = np.linspace(0.0, 1.0, 17)[1:]

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


def subexponential_psi(values: np.ndarray) -> np.ndarray:
    """Return the sub-exponential CGF-like function."""
    scaled = EVIDENCE_SCALE * values
    return (-np.log1p(-scaled) - scaled) / EVIDENCE_SCALE**2


def mixture_log_evidence(differences: np.ndarray) -> np.ndarray:
    """Return weak-null mixture log evidence at every time."""
    values = np.asarray(differences, dtype=np.float64)
    cumulative = np.cumsum(values, axis=1)
    previous_average = np.zeros_like(values)
    previous_average[:, 1:] = (
        cumulative[:, :-1] / np.arange(1, values.shape[1])
    )
    predictable_center = np.clip(
        previous_average,
        -EVIDENCE_SCALE / 2.0,
        EVIDENCE_SCALE / 2.0,
    )
    intrinsic_time = np.cumsum(
        (values - predictable_center) ** 2,
        axis=1,
    )

    log_mixture = np.full(values.shape, -np.inf, dtype=np.float64)
    for mixture_parameter, penalty in zip(
        LAMBDA_GRID,
        subexponential_psi(LAMBDA_GRID),
        strict=True,
    ):
        component = (
            mixture_parameter * cumulative - penalty * intrinsic_time
        )
        log_mixture = np.logaddexp(log_mixture, component)
    return log_mixture - math.log(len(LAMBDA_GRID))


def strong_null_log_evidence(differences: np.ndarray) -> np.ndarray:
    """Return product-mixture evidence for the stepwise strong null."""
    log_mixture = np.full(differences.shape, -np.inf, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        for betting_fraction in STRONG_LAMBDA_GRID:
            log_factors = np.log1p(betting_fraction * differences)
            component = np.cumsum(log_factors, axis=1)
            log_mixture = np.logaddexp(log_mixture, component)
    return log_mixture - math.log(len(STRONG_LAMBDA_GRID))


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
    """Evaluate component crossing, power, and a conjunctive promotion gate."""
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
                    "panel": "false_component_crossing",
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
                "panel": "dependent_false_component_crossing",
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


def main() -> None:
    simulation_data, simulation_summary = evaluate_simulations()
    simulation_data.to_csv(DATA_DIR / "Figure1_source_data.csv", index=False)

    summary = {"simulation": simulation_summary}
    (DATA_DIR / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
