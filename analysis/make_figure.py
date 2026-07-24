#!/usr/bin/env python3
"""Create the three submission figures from saved numerical results."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "source_data"
CONFIRMATION_DIR = DATA_DIR / "plasticity_retention_confirmation"
FIGURE_DIR = ROOT / "figures"

WIDTH_MM = 183.0
ALPHA = 0.05
HORIZON = 5_000
RETENTION_MARGIN = 0.05

METHOD_ORDER = (
    "e-process",
    "unadjusted 20-look test",
    "Bonferroni 20-look test",
)
METHOD_LABELS = {
    "e-process": "Anytime-valid evidence",
    "unadjusted 20-look test": "Repeated test, unadjusted",
    "Bonferroni 20-look test": "Repeated test, Bonferroni",
}
METHOD_COLORS = {
    "e-process": "#315F78",
    "unadjusted 20-look test": "#9A5A5A",
    "Bonferroni 20-look test": "#7A858C",
}
METHOD_MARKERS = {
    "e-process": "o",
    "unadjusted 20-look test": "s",
    "Bonferroni 20-look test": "^",
}

MODE_ORDER = ("replay", "naive", "no_update")
MODE_LABELS = {
    "replay": "Replay",
    "naive": "Naive",
    "no_update": "No update",
}
MODE_COLORS = {
    "replay": "#315F78",
    "naive": "#9A5A5A",
    "no_update": "#7A858C",
}
MODE_SEED_COLORS = {
    "replay": "#7893A2",
    "naive": "#C08E8E",
    "no_update": "#AAB1B5",
}

INK = "#20272C"
MUTED = "#66737B"
GRID = "#D8DEE2"
LIGHT = "#F4F6F7"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8.2,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "axes.linewidth": 0.7,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "lines.linewidth": 1.1,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.10,
        1.01,
        label,
        transform=axis.transAxes,
        fontsize=9.0,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=INK,
    )


def save_figure(figure: plt.Figure, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    base = FIGURE_DIR / name
    figure.savefig(base.with_suffix(".pdf"))
    figure.savefig(base.with_suffix(".svg"))
    figure.savefig(base.with_suffix(".eps"))
    figure.savefig(
        base.with_suffix(".tiff"),
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    figure.savefig(base.with_suffix(".png"), dpi=600)
    figure.savefig(FIGURE_DIR / f"{name}_preview.png", dpi=300)


def figure_one(simulation: pd.DataFrame) -> None:
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(WIDTH_MM / 25.4, 82.0 / 25.4),
        gridspec_kw={"wspace": 0.30},
    )

    independent = simulation.loc[
        simulation["panel"] == "false_component_crossing"
    ].copy()
    dependent = simulation.loc[
        simulation["panel"] == "dependent_false_component_crossing"
    ].copy()
    dependent["scenario"] = "serial dependence"
    null_data = pd.concat([independent, dependent], ignore_index=True)
    scenario_order = (
        "equal error 0.10",
        "equal error 0.20",
        "equal error 0.30",
        "serial dependence",
    )
    scenario_labels = ("0.10", "0.20", "0.30", "Dependent")
    x_base = np.arange(len(scenario_order), dtype=float)
    offsets = (-0.18, 0.0, 0.18)

    for method, offset in zip(METHOD_ORDER, offsets, strict=True):
        values = (
            null_data.loc[null_data["method"] == method]
            .set_index("scenario")
            .loc[list(scenario_order)]
        )
        y = values["y"].to_numpy(dtype=float)
        lower = values["lower"].to_numpy(dtype=float)
        upper = values["upper"].to_numpy(dtype=float)
        axes[0].errorbar(
            x_base + offset,
            y,
            yerr=np.vstack([y - lower, upper - y]),
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            linestyle="none",
            markersize=4.5,
            capsize=2.0,
            capthick=0.7,
            label=METHOD_LABELS[method],
            zorder=3,
        )
    axes[0].axhline(
        ALPHA,
        color=INK,
        linewidth=0.8,
        linestyle=(0, (3, 2)),
        zorder=1,
    )
    axes[0].axhspan(0.0, ALPHA, color="#EAF0F2", zorder=0)
    axes[0].text(
        3.28,
        ALPHA + 0.006,
        r"$\alpha=0.05$",
        ha="right",
        va="bottom",
        color=MUTED,
    )
    axes[0].set_xticks(x_base, scenario_labels)
    axes[0].set_xlabel("Null setting")
    axes[0].set_ylabel("False-crossing probability")
    axes[0].set_ylim(0.0, 0.215)
    axes[0].grid(axis="y", color=GRID, linewidth=0.5)
    axes[0].set_title(
        "False crossings under the null",
        loc="left",
        fontsize=8.5,
        fontweight="bold",
        pad=6,
    )
    panel_label(axes[0], "a")

    promotion = simulation.loc[
        simulation["panel"] == "promotion_summary"
    ].copy()
    for method, offset in zip(METHOD_ORDER, (-0.004, 0.0, 0.004), strict=True):
        values = promotion.loc[promotion["method"] == method].sort_values("x")
        x = values["x"].to_numpy(dtype=float) + offset
        y = values["y"].to_numpy(dtype=float)
        lower = values["lower"].to_numpy(dtype=float)
        upper = values["upper"].to_numpy(dtype=float)
        axes[1].errorbar(
            x,
            y,
            yerr=np.vstack([y - lower, upper - y]),
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            linestyle="-",
            markersize=4.5,
            capsize=2.0,
            capthick=0.7,
            label=METHOD_LABELS[method],
        )
    axes[1].set_xlim(0.035, 0.215)
    axes[1].set_ylim(0.0, 1.04)
    axes[1].set_xticks([0.05, 0.10, 0.20])
    axes[1].set_xlabel("True loss advantage")
    axes[1].set_ylabel("Detection probability by 1,000")
    axes[1].grid(axis="y", color=GRID, linewidth=0.5)
    axes[1].set_title(
        "Detection under a true advantage",
        loc="left",
        fontsize=8.5,
        fontweight="bold",
        pad=6,
    )
    panel_label(axes[1], "b")

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        columnspacing=1.4,
        handletextpad=0.5,
    )
    figure.subplots_adjust(left=0.09, right=0.985, bottom=0.20, top=0.78)
    save_figure(figure, "Fig1")
    plt.close(figure)


def draw_performance_panel(
    axis: plt.Axes,
    results: pd.DataFrame,
    specifications: tuple[tuple[str, str, str], ...],
    xlabel: str,
    xlim: tuple[float, float],
    reference: float,
    acceptable_region: tuple[float, float] | None = None,
) -> None:
    metric_offsets = (-0.11, 0.11)
    for mode_index, mode in enumerate(MODE_ORDER):
        mode_data = results.loc[
            results["candidate_mode"] == mode
        ].sort_values("seed")
        for metric_index, (value_column, _, marker) in enumerate(specifications):
            values = mode_data[value_column].to_numpy(dtype=float)
            y_position = mode_index + metric_offsets[metric_index]
            seed_jitter = np.linspace(-0.028, 0.028, len(values))
            axis.hlines(
                y_position,
                values.min(),
                values.max(),
                color=MODE_COLORS[mode],
                linewidth=1.0,
                zorder=2,
            )
            axis.scatter(
                values,
                np.full(len(values), y_position) + seed_jitter,
                s=17,
                marker=marker,
                color=MODE_SEED_COLORS[mode],
                edgecolor="white",
                linewidth=0.35,
                zorder=3,
            )
            axis.scatter(
                np.median(values),
                y_position,
                marker=marker,
                s=47,
                color=MODE_COLORS[mode],
                edgecolor="white",
                linewidth=0.7,
                zorder=4,
            )
    if acceptable_region is not None:
        axis.axvspan(
            acceptable_region[0],
            acceptable_region[1],
            color="#EAF0F2",
            zorder=0,
        )
    axis.axvline(
        reference,
        color=INK,
        linewidth=0.8,
        linestyle=(0, (3, 2)),
        zorder=1,
    )
    axis.set_xlim(*xlim)
    axis.set_ylim(len(MODE_ORDER) - 0.52, -0.52)
    axis.set_yticks(
        range(len(MODE_ORDER)),
        [MODE_LABELS[mode] for mode in MODE_ORDER],
    )
    axis.set_xlabel(xlabel)
    axis.grid(axis="x", color=GRID, linewidth=0.5)


def figure_two(results: pd.DataFrame) -> None:
    required = {
        "current_mean_brier_advantage",
        "current_mean_error_advantage",
        "retention_mean_brier_difference",
        "retention_mean_error_difference",
    }
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Confirmation results lack columns: {sorted(missing)}")

    plot_data = results.copy()
    plot_data["old_task_brier_increase"] = -plot_data[
        "retention_mean_brier_difference"
    ]
    plot_data["old_task_error_increase"] = -plot_data[
        "retention_mean_error_difference"
    ]
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(WIDTH_MM / 25.4, 88.0 / 25.4),
        gridspec_kw={"wspace": 0.34},
    )
    new_task_metrics = (
        (
            "current_mean_brier_advantage",
            "Brier loss",
            "o",
        ),
        (
            "current_mean_error_advantage",
            "Classification error",
            "s",
        ),
    )
    old_task_metrics = (
        (
            "old_task_brier_increase",
            "Brier loss",
            "o",
        ),
        (
            "old_task_error_increase",
            "Classification error",
            "s",
        ),
    )
    draw_performance_panel(
        axes[0],
        plot_data,
        new_task_metrics,
        "New-task improvement (higher is better)",
        (-0.05, 0.86),
        0.0,
    )
    axes[0].set_title(
        "Learning the new task",
        loc="left",
        fontsize=8.5,
        fontweight="bold",
        pad=6,
    )
    panel_label(axes[0], "a")

    draw_performance_panel(
        axes[1],
        plot_data,
        old_task_metrics,
        "Old-task loss increase (lower is better)",
        (-0.05, 0.98),
        RETENTION_MARGIN,
        acceptable_region=(-0.05, RETENTION_MARGIN),
    )
    axes[1].text(
        RETENTION_MARGIN + 0.012,
        2.35,
        "0.05 margin",
        color=MUTED,
        ha="left",
        va="center",
    )
    axes[1].set_title(
        "Retaining the old task",
        loc="left",
        fontsize=8.5,
        fontweight="bold",
        pad=6,
    )
    panel_label(axes[1], "b")

    metric_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color=INK,
            linestyle="none",
            markersize=5.5,
            label=label,
        )
        for _, label, marker in new_task_metrics
    ]
    figure.legend(
        handles=metric_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=2,
        columnspacing=1.6,
        handletextpad=0.5,
    )
    figure.subplots_adjust(left=0.105, right=0.985, bottom=0.19, top=0.77)
    save_figure(figure, "Fig2")
    plt.close(figure)


def figure_three(results: pd.DataFrame) -> None:
    component_columns = (
        "current_brier_crossing_index",
        "current_error_crossing_index",
        "retention_brier_crossing_index",
        "retention_error_crossing_index",
    )
    component_labels = (
        "Brier",
        "Error",
        "Brier",
        "Error",
    )
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(WIDTH_MM / 25.4, 91.0 / 25.4),
        gridspec_kw={"width_ratios": (1.18, 1.0), "wspace": 0.40},
    )

    for mode_index, mode in enumerate(MODE_ORDER):
        mode_data = results.loc[
            results["candidate_mode"] == mode
        ].sort_values("seed")
        promotion = mode_data["plasticity_retention_promotion_time"]
        promoted = promotion.notna().to_numpy()
        y = np.full(len(mode_data), mode_index, dtype=float)
        y += np.linspace(-0.12, 0.12, len(mode_data))
        if promoted.any():
            axes[0].scatter(
                promotion.to_numpy(dtype=float)[promoted],
                y[promoted],
                color=MODE_COLORS[mode],
                s=26,
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
        if (~promoted).any():
            axes[0].scatter(
                np.full((~promoted).sum(), HORIZON),
                y[~promoted],
                color=MODE_COLORS[mode],
                marker="x",
                linewidth=1.0,
                s=31,
                zorder=3,
            )
    axes[0].axvline(
        HORIZON,
        color=MUTED,
        linewidth=0.8,
        linestyle=(0, (3, 2)),
        zorder=1,
    )
    axes[0].text(
        HORIZON,
        -0.36,
        "not met by horizon",
        ha="right",
        va="center",
        color=MUTED,
        fontsize=7.7,
    )
    axes[0].set_xlim(0, HORIZON * 1.06)
    axes[0].set_xticks([0, 1_000, 2_000, 3_000, 4_000, 5_000])
    axes[0].set_yticks(
        range(len(MODE_ORDER)),
        [MODE_LABELS[m] for m in MODE_ORDER],
    )
    axes[0].set_ylim(len(MODE_ORDER) - 0.5, -0.5)
    axes[0].set_xlabel("Labeled observations per stream")
    axes[0].grid(axis="x", color=GRID, linewidth=0.5)
    axes[0].set_title(
        "When the full gate was met",
        loc="left",
        fontsize=8.5,
        fontweight="bold",
        pad=6,
    )
    panel_label(axes[0], "a")

    pass_counts = np.asarray(
        [
            [
                int(
                    results.loc[
                        results["candidate_mode"] == mode,
                        column,
                    ].notna().sum()
                )
                for column in component_columns
            ]
            for mode in MODE_ORDER
        ],
        dtype=float,
    )
    axes[1].set_xlim(-0.5, 3.5)
    axes[1].set_ylim(2.5, -0.5)
    for row in range(pass_counts.shape[0]):
        for column in range(pass_counts.shape[1]):
            value = int(pass_counts[row, column])
            if value == 5:
                face_color = "#315F78"
                text_color = "white"
            elif value == 4:
                face_color = "#7F9AA8"
                text_color = "white"
            else:
                face_color = "#F1F3F4"
                text_color = "#8B4F4F"
            axes[1].add_patch(
                Rectangle(
                    (column - 0.47, row - 0.42),
                    0.94,
                    0.84,
                    facecolor=face_color,
                    edgecolor="white",
                    linewidth=1.2,
                )
            )
            axes[1].text(
                column,
                row,
                f"{value}/5",
                ha="center",
                va="center",
                color=text_color,
                fontweight="bold",
            )
    axes[1].axvline(1.5, color=INK, linewidth=0.7)
    axes[1].set_xticks(range(4), component_labels)
    axes[1].set_yticks(
        range(len(MODE_ORDER)),
        [MODE_LABELS[m] for m in MODE_ORDER],
    )
    axes[1].tick_params(length=0)
    for spine in axes[1].spines.values():
        spine.set_visible(False)
    axes[1].text(
        0.5,
        1.08,
        "NEW TASK",
        transform=axes[1].get_xaxis_transform(),
        ha="center",
        va="bottom",
        color=MUTED,
        fontsize=7.7,
        fontweight="bold",
    )
    axes[1].text(
        2.5,
        1.08,
        "OLD TASK",
        transform=axes[1].get_xaxis_transform(),
        ha="center",
        va="bottom",
        color=MUTED,
        fontsize=7.7,
        fontweight="bold",
    )
    axes[1].set_xlabel("Seeds meeting each requirement")
    axes[1].set_title(
        "Which requirements were met",
        loc="left",
        fontsize=8.5,
        fontweight="bold",
        pad=23,
    )
    panel_label(axes[1], "b")

    figure.subplots_adjust(left=0.10, right=0.985, bottom=0.20, top=0.78)
    save_figure(figure, "Fig3")
    plt.close(figure)


def validate_confirmation_results(results: pd.DataFrame) -> None:
    if len(results) != 15:
        raise ValueError(f"Expected 15 confirmation rows, found {len(results)}")
    if set(results["seed"]) != {1, 2, 3, 4, 5}:
        raise ValueError("Confirmation results must contain seeds 1-5")
    if set(results["candidate_mode"]) != set(MODE_ORDER):
        raise ValueError("Confirmation results must contain all candidate modes")
    if set(results["evaluation_split"]) != {"confirmation"}:
        raise ValueError("Only confirmation-partition results may be plotted")


def main() -> None:
    simulation = pd.read_csv(DATA_DIR / "Figure1_source_data.csv")
    confirmation = pd.read_csv(
        CONFIRMATION_DIR / "Split_CIFAR10_candidate_results.csv"
    )
    validate_confirmation_results(confirmation)
    figure_one(simulation)
    figure_two(confirmation)
    figure_three(confirmation)


if __name__ == "__main__":
    main()
