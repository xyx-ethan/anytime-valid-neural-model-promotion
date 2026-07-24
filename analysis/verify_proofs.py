#!/usr/bin/env python3
"""Machine-check the algebraic core and finite cases used in the proofs."""

from __future__ import annotations

import json
from fractions import Fraction
from itertools import product
from pathlib import Path
from typing import Iterator

import sympy as sp


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "analysis" / "proof_verification_report.json"


def compositions(total: int, parts: int) -> Iterator[tuple[int, ...]]:
    """Yield nonnegative integer tuples of fixed length and sum."""
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in compositions(total - first, parts - 1):
            yield (first, *rest)


def verify_symbolic_identities() -> dict[str, str]:
    """Check the conditional-update and alpha-budget identities."""
    previous, betting_fraction, conditional_mean = sp.symbols(
        "previous betting_fraction conditional_mean",
        nonnegative=True,
        real=True,
    )
    update_remainder = sp.expand(
        previous * (1 + betting_fraction * conditional_mean)
        - previous
        - previous * betting_fraction * conditional_mean
    )
    if update_remainder != 0:
        raise AssertionError("One-step update identity failed")

    mean = sp.symbols("mean", real=True)
    weights = sp.symbols("w0:4", nonnegative=True)
    capitals = sp.symbols("m0:4", nonnegative=True)
    fractions = sp.symbols("l0:4", nonnegative=True)
    updated_mixture = sum(
        weight * capital * (1 + fraction * mean)
        for weight, capital, fraction in zip(
            weights,
            capitals,
            fractions,
            strict=True,
        )
    )
    previous_mixture = sum(
        weight * capital
        for weight, capital in zip(weights, capitals, strict=True)
    )
    expected_change = mean * sum(
        weight * capital * fraction
        for weight, capital, fraction in zip(
            weights,
            capitals,
            fractions,
            strict=True,
        )
    )
    if sp.simplify(updated_mixture - previous_mixture - expected_change) != 0:
        raise AssertionError("Mixture update identity failed")

    index, count = sp.symbols("index count", integer=True, positive=True)
    partial_budget = sp.summation(1 / (index * (index + 1)), (index, 1, count))
    if sp.simplify(partial_budget - count / (count + 1)) != 0:
        raise AssertionError("Successive-candidate budget identity failed")

    return {
        "one_step_update": "M_prev * (1 + lambda * mu)",
        "mixture_change": "mu * sum(w_i * M_i * lambda_i)",
        "partial_alpha_budget": "sum_{j=1}^J 1/[j(j+1)] = J/(J+1)",
    }


def verify_one_step_conditions() -> dict[str, int]:
    """Exhaustively check exact finite distributions on a bounded support."""
    support = (
        Fraction(-1),
        Fraction(-1, 2),
        Fraction(0),
        Fraction(1, 2),
        Fraction(1),
    )
    betting_grid = tuple(Fraction(index, 16) for index in range(1, 17))
    denominator = 8
    null_distributions = 0
    checked_updates = 0

    for counts in compositions(denominator, len(support)):
        probabilities = tuple(Fraction(value, denominator) for value in counts)
        conditional_mean = sum(
            probability * outcome
            for probability, outcome in zip(
                probabilities,
                support,
                strict=True,
            )
        )
        if conditional_mean > 0:
            continue
        null_distributions += 1
        for betting_fraction in betting_grid:
            factors = tuple(1 + betting_fraction * outcome for outcome in support)
            if min(factors) < 0:
                raise AssertionError("A betting factor became negative")
            expected_factor = sum(
                probability * factor
                for probability, factor in zip(
                    probabilities,
                    factors,
                    strict=True,
                )
            )
            if expected_factor != 1 + betting_fraction * conditional_mean:
                raise AssertionError("Conditional expectation identity failed")
            if expected_factor > 1:
                raise AssertionError("Supermartingale update condition failed")
            checked_updates += 1

    return {
        "null_distributions": null_distributions,
        "betting_fractions_per_distribution": len(betting_grid),
        "exact_updates": checked_updates,
    }


def path_crosses(
    sequence: tuple[Fraction, ...],
    betting_grid: tuple[Fraction, ...],
    alpha: Fraction,
) -> bool:
    """Return whether a finite betting mixture crosses its threshold."""
    capitals = [Fraction(1) for _ in betting_grid]
    weight = Fraction(1, len(betting_grid))
    threshold = 1 / alpha
    for outcome in sequence:
        capitals = [
            capital * (1 + betting_fraction * outcome)
            for capital, betting_fraction in zip(
                capitals,
                betting_grid,
                strict=True,
            )
        ]
        evidence = weight * sum(capitals)
        if evidence >= threshold:
            return True
    return False


def verify_finite_horizon_ville() -> dict[str, object]:
    """Compute exact crossing probabilities for finite iid null models."""
    support = (Fraction(-1), Fraction(0), Fraction(1))
    betting_grid = (
        Fraction(1, 4),
        Fraction(1, 2),
        Fraction(3, 4),
        Fraction(1),
    )
    alpha = Fraction(1, 5)
    horizon = 8
    sequences = tuple(product(support, repeat=horizon))
    crossing = {
        sequence: path_crosses(sequence, betting_grid, alpha)
        for sequence in sequences
    }

    denominator = 6
    checked_models = 0
    maximum_probability = Fraction(0)
    maximizing_counts: tuple[int, ...] | None = None
    for counts in compositions(denominator, len(support)):
        probabilities = tuple(Fraction(value, denominator) for value in counts)
        mean = sum(
            probability * outcome
            for probability, outcome in zip(
                probabilities,
                support,
                strict=True,
            )
        )
        if mean > 0:
            continue
        checked_models += 1
        crossing_probability = Fraction(0)
        for sequence in sequences:
            if not crossing[sequence]:
                continue
            path_probability = Fraction(1)
            for outcome in sequence:
                path_probability *= probabilities[support.index(outcome)]
            crossing_probability += path_probability
        if crossing_probability > alpha:
            raise AssertionError("Finite-horizon crossing probability exceeded alpha")
        if crossing_probability > maximum_probability:
            maximum_probability = crossing_probability
            maximizing_counts = counts

    return {
        "support": ["-1", "0", "1"],
        "horizon": horizon,
        "alpha": str(alpha),
        "iid_null_models": checked_models,
        "paths_per_model": len(sequences),
        "maximum_crossing_probability": str(maximum_probability),
        "maximizing_probability_counts_out_of_6": maximizing_counts,
    }


def verify_component_threshold() -> dict[str, int]:
    """Check that one crossing component forces the mixture to cross."""
    support = (Fraction(-1), Fraction(0), Fraction(1))
    betting_grid = (
        Fraction(1, 4),
        Fraction(1, 2),
        Fraction(3, 4),
        Fraction(1),
    )
    selected_index = 1
    weight = Fraction(1, len(betting_grid))
    alpha = Fraction(1, 5)
    component_threshold = 1 / (alpha * weight)
    mixture_threshold = 1 / alpha
    triggering_paths = 0

    for sequence in product(support, repeat=8):
        capitals = [Fraction(1) for _ in betting_grid]
        for outcome in sequence:
            capitals = [
                capital * (1 + betting_fraction * outcome)
                for capital, betting_fraction in zip(
                    capitals,
                    betting_grid,
                    strict=True,
                )
            ]
        if capitals[selected_index] < component_threshold:
            continue
        triggering_paths += 1
        if weight * sum(capitals) < mixture_threshold:
            raise AssertionError("Component crossing did not imply mixture crossing")

    if triggering_paths == 0:
        raise AssertionError("Threshold implication test had no triggering path")
    return {
        "enumerated_paths": len(support) ** 8,
        "triggering_paths": triggering_paths,
    }


def verify_conjunctive_logic() -> dict[str, int]:
    """Check that joint promotion implies crossing of every component."""
    components = 4
    checked_cases = 0
    for crossing_pattern in product((False, True), repeat=components):
        joint_promotion = all(crossing_pattern)
        for null_component in range(components):
            if joint_promotion and not crossing_pattern[null_component]:
                raise AssertionError(
                    "Joint promotion did not imply a null-component crossing"
                )
            checked_cases += 1
    return {
        "components": components,
        "logical_cases": checked_cases,
    }


def main() -> None:
    report = {
        "status": "pass",
        "scope": (
            "Exact algebra and finite discrete cases. Ville's inequality, the "
            "positive-drift hitting-time result, and Wald's identity remain cited "
            "external theorems."
        ),
        "symbolic_identities": verify_symbolic_identities(),
        "one_step_conditions": verify_one_step_conditions(),
        "finite_horizon_ville": verify_finite_horizon_ville(),
        "component_threshold": verify_component_threshold(),
        "conjunctive_gate": verify_conjunctive_logic(),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
