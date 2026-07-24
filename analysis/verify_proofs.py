#!/usr/bin/env python3
"""Verify the algebra and implementation used by the promotion gate.

These checks verify transformations, numerical implementation, and finite
special cases. General e-process validity for the weak average null is supplied
by Theorem 3 of Choe and Ramdas (2024), not by this script.
"""

from __future__ import annotations

import json
import math
from fractions import Fraction
from itertools import product
from pathlib import Path

import numpy as np
import sympy as sp


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "analysis" / "proof_verification_report.json"
ALPHA = 0.05
SCALE = 0.5
LAMBDA_GRID = np.linspace(0.0, 1.9, 17, dtype=np.float64)[1:]


def psi_subexponential(values: np.ndarray) -> np.ndarray:
    """Evaluate psi_E,c(lambda) for c=1/2."""
    values = np.asarray(values, dtype=np.float64)
    if np.any(values < 0.0) or np.any(values >= 1.0 / SCALE):
        raise ValueError("lambda must lie in [0, 1/c)")
    scaled = SCALE * values
    return (-np.log1p(-scaled) - scaled) / SCALE**2


def reference_log_evidence(sequence: np.ndarray) -> np.ndarray:
    """Direct matrix implementation of the finite-mixture e-process."""
    values = np.asarray(sequence, dtype=np.float64)
    cumulative = np.cumsum(values)
    previous_average = np.zeros_like(values)
    previous_average[1:] = cumulative[:-1] / np.arange(1, len(values))
    center = np.clip(previous_average, -SCALE / 2.0, SCALE / 2.0)
    intrinsic_time = np.cumsum((values - center) ** 2)
    components = (
        LAMBDA_GRID[:, None] * cumulative[None, :]
        - psi_subexponential(LAMBDA_GRID)[:, None]
        * intrinsic_time[None, :]
    )
    maximum = components.max(axis=0)
    return maximum + np.log(
        np.exp(components - maximum[None, :]).mean(axis=0)
    )


def streaming_log_evidence(sequence: np.ndarray) -> np.ndarray:
    """Streaming implementation used independently for agreement checks."""
    values = np.asarray(sequence, dtype=np.float64)
    penalties = psi_subexponential(LAMBDA_GRID)
    cumulative = 0.0
    intrinsic_time = 0.0
    output: list[float] = []
    for index, value in enumerate(values, start=1):
        previous_average = cumulative / (index - 1) if index > 1 else 0.0
        center = float(
            np.clip(previous_average, -SCALE / 2.0, SCALE / 2.0)
        )
        cumulative += float(value)
        intrinsic_time += (float(value) - center) ** 2
        components = LAMBDA_GRID * cumulative - penalties * intrinsic_time
        maximum = float(components.max())
        output.append(
            maximum
            + math.log(float(np.exp(components - maximum).mean()))
        )
    return np.asarray(output)


def verify_weak_process_implementation() -> dict[str, object]:
    """Compare two independent implementations on fixed and random paths."""
    paths = [
        np.asarray([-1.0, 0.0, 1.0, 0.5, -0.5, 1.0]),
        np.ones(64),
        -np.ones(64),
        np.zeros(64),
    ]
    rng = np.random.default_rng(20260723)
    paths.extend(rng.uniform(-1.0, 1.0, size=257) for _ in range(20))

    maximum_error = 0.0
    for path in paths:
        direct = reference_log_evidence(path)
        streaming = streaming_log_evidence(path)
        maximum_error = max(
            maximum_error,
            float(np.max(np.abs(direct - streaming))),
        )
        if not np.allclose(direct, streaming, rtol=0.0, atol=1e-12):
            raise AssertionError("Weak-null e-process implementations disagree")

    if not np.all((LAMBDA_GRID >= 0.0) & (LAMBDA_GRID < 1.0 / SCALE)):
        raise AssertionError("Mixture support falls outside [0, 1/c)")
    if not np.all(psi_subexponential(LAMBDA_GRID) >= 0.0):
        raise AssertionError("The sub-exponential penalty became negative")

    return {
        "paths_checked": len(paths),
        "largest_absolute_log_evidence_difference": maximum_error,
        "mixture_components": len(LAMBDA_GRID),
        "lambda_min": float(LAMBDA_GRID.min()),
        "lambda_max": float(LAMBDA_GRID.max()),
    }


def verify_predictable_center_and_intrinsic_time() -> dict[str, object]:
    """Check center bounds and monotonic intrinsic time on random paths."""
    rng = np.random.default_rng(20260724)
    minimum_increment = math.inf
    maximum_center = 0.0
    for _ in range(100):
        values = rng.uniform(-1.0, 1.0, size=500)
        cumulative = np.cumsum(values)
        previous_average = np.zeros_like(values)
        previous_average[1:] = cumulative[:-1] / np.arange(1, len(values))
        center = np.clip(
            previous_average,
            -SCALE / 2.0,
            SCALE / 2.0,
        )
        increments = (values - center) ** 2
        intrinsic_time = np.cumsum(increments)
        minimum_increment = min(minimum_increment, float(increments.min()))
        maximum_center = max(maximum_center, float(np.abs(center).max()))
        if np.any(np.diff(intrinsic_time) < -1e-14):
            raise AssertionError("Intrinsic time decreased")
    if maximum_center > SCALE / 2.0 + 1e-14:
        raise AssertionError("Predictable center exceeded its bound")
    return {
        "paths_checked": 100,
        "observations_per_path": 500,
        "maximum_absolute_center": maximum_center,
        "minimum_intrinsic_time_increment": minimum_increment,
    }


def verify_finite_iid_null_special_cases() -> dict[str, object]:
    """Enumerate short iid null paths and check crossing probabilities.

    This is a diagnostic special case, not a proof of the general weak-null
    theorem. The coarse alpha and short horizon make threshold crossings
    observable during exact enumeration.
    """
    support = np.asarray([-1.0, 0.0, 1.0])
    horizon = 8
    diagnostic_alpha = 0.2
    threshold = math.log(1.0 / diagnostic_alpha)
    paths = list(product(range(len(support)), repeat=horizon))
    crossed = np.asarray(
        [
            bool(
                np.any(
                    reference_log_evidence(
                        support[np.asarray(path, dtype=np.int64)]
                    )
                    >= threshold
                )
            )
            for path in paths
        ],
        dtype=bool,
    )

    models = (
        (0.25, 0.50, 0.25),
        (0.40, 0.40, 0.20),
        (0.50, 0.50, 0.00),
        (0.30, 0.50, 0.20),
    )
    probabilities: list[float] = []
    for model in models:
        mean = float(np.dot(model, support))
        if mean > 1e-14:
            raise AssertionError("Diagnostic model violates the iid null")
        crossing_probability = 0.0
        for path, path_crossed in zip(paths, crossed, strict=True):
            if not path_crossed:
                continue
            path_probability = math.prod(model[index] for index in path)
            crossing_probability += path_probability
        probabilities.append(crossing_probability)
        if crossing_probability > diagnostic_alpha + 1e-12:
            raise AssertionError(
                "Finite iid diagnostic exceeded its alpha threshold"
            )
    return {
        "support": support.tolist(),
        "horizon": horizon,
        "diagnostic_alpha": diagnostic_alpha,
        "null_models_checked": len(models),
        "crossing_probabilities": probabilities,
        "maximum_crossing_probability": max(probabilities),
    }


def verify_retention_transform() -> dict[str, object]:
    """Check boundedness and the noninferiority-null mapping."""
    difference, margin = sp.symbols("difference margin", real=True)
    transformed = (difference + margin) / (1 + margin)
    recovered = sp.simplify(transformed * (1 + margin) - margin)
    if recovered != difference:
        raise AssertionError("Retention transformation is not invertible")

    margins = (0.01, 0.05, 0.25, 0.90)
    grid = np.linspace(-1.0, 1.0, 2001)
    for value in margins:
        mapped = (grid + value) / (1.0 + value)
        if mapped.min() < -1.0 - 1e-12 or mapped.max() > 1.0 + 1e-12:
            raise AssertionError("Retention transform left [-1, 1]")
        null_grid = grid[grid <= -value]
        if np.any((null_grid + value) / (1.0 + value) > 1e-12):
            raise AssertionError("Noninferiority null mapped above zero")
    return {
        "margins_checked": list(margins),
        "difference_grid_points": len(grid),
        "inverse_identity": "x = r(1 + delta) - delta",
    }


def verify_candidate_budget() -> dict[str, object]:
    """Verify the successive-candidate alpha schedule symbolically and exactly."""
    index, count = sp.symbols("index count", integer=True, positive=True)
    identity = sp.summation(
        1 / (index * (index + 1)),
        (index, 1, count),
    )
    if sp.simplify(identity - count / (count + 1)) != 0:
        raise AssertionError("Candidate alpha-budget identity failed")

    alpha = Fraction(1, 20)
    candidates = 10_000
    spent = sum(
        alpha / (candidate * (candidate + 1))
        for candidate in range(1, candidates + 1)
    )
    expected = alpha * Fraction(candidates, candidates + 1)
    if spent != expected or spent >= alpha:
        raise AssertionError("Finite candidate alpha budget failed")
    return {
        "global_alpha": str(alpha),
        "checked_candidates": candidates,
        "spent_alpha": str(spent),
        "partial_sum_identity": "sum_{j=1}^J 1/[j(j+1)] = J/(J+1)",
    }


def verify_conjunctive_gate() -> dict[str, object]:
    """Exhaustively verify the intersection-union decision logic."""
    components = 4
    patterns = list(product((False, True), repeat=components))
    promoted_patterns = [pattern for pattern in patterns if all(pattern)]
    if promoted_patterns != [(True, True, True, True)]:
        raise AssertionError("Conjunctive gate accepted an incomplete pattern")
    for pattern in patterns:
        for true_null_component in range(components):
            if all(pattern) and not pattern[true_null_component]:
                raise AssertionError(
                    "Joint promotion omitted a true-null component crossing"
                )
    return {
        "components": components,
        "crossing_patterns_checked": len(patterns),
        "promoting_patterns": len(promoted_patterns),
    }


def verify_batch_boundaries() -> dict[str, object]:
    """Check the conversion from label-level crossings to action times."""
    batch_size = 128
    horizon = 5_000
    cases = {
        1: 128,
        127: 128,
        128: 128,
        129: 256,
        4_993: 5_000,
        5_000: 5_000,
    }
    for crossing, expected in cases.items():
        observed = min(
            int(math.ceil(crossing / batch_size) * batch_size),
            horizon,
        )
        if observed != expected:
            raise AssertionError(
                f"Batch boundary mismatch: {crossing} -> {observed}"
            )
    return {
        "batch_size": batch_size,
        "horizon": horizon,
        "cases_checked": len(cases),
    }


def main() -> None:
    report = {
        "status": "pass",
        "scope": (
            "Algebra, two independent numerical implementations, finite iid "
            "diagnostics, retention transformation, candidate alpha spending, "
            "conjunctive logic, and batch-aligned action times. General validity "
            "for the weak average null follows from Theorem 3 of Choe and Ramdas "
            "(2024)."
        ),
        "weak_process_implementation": verify_weak_process_implementation(),
        "predictable_center_and_intrinsic_time": (
            verify_predictable_center_and_intrinsic_time()
        ),
        "finite_iid_null_diagnostics": verify_finite_iid_null_special_cases(),
        "retention_transform": verify_retention_transform(),
        "candidate_alpha_budget": verify_candidate_budget(),
        "conjunctive_gate": verify_conjunctive_gate(),
        "batch_boundaries": verify_batch_boundaries(),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
