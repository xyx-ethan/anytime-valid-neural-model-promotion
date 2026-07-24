#!/usr/bin/env python3
"""Create the three submission figures from saved numerical results."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


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
    "e-process": "Anytime-valid process",
    "unadjusted 20-look test": "Unadjusted repeated test",
    "Bonferroni 20-look test": "Bonferroni repeated test",
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
        -0.13,
        1.04,
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
        simulation["panel"] == "false_promotion"
    ].copy()
    dependent = simulation.loc[
        simulation["panel"] == "dependent_false_promotion"
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
    axes[0].set_ylabel("False-promotion probability")
    axes[0].set_ylim(0.0, 0.215)
    axes[0].grid(axis="y", color=GRID, linewidth=0.5)
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
    axes[1].set_ylabel("Promotion probability by 1,000")
    axes[1].grid(axis="y", color=GRID, linewidth=0.5)
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
    figure.subplots_adjust(left=0.09, right=0.985, bottom=0.20, top=0.82)
    save_figure(figure, "Fig1")
    plt.close(figure)


def draw_seed_summary(
    axis: plt.Axes,
    results: pd.DataFrame,
    value_column: str,
    ylabel: str,
    reference: float,
) -> None:
    for mode_index, mode in enumerate(MODE_ORDER):
        values = (
            results.loc[results["candidate_mode"] == mode]
            .sort_values("seed")[value_column]
            .to_numpy(dtype=float)
        )
        jitter = np.linspace(-0.045, 0.045, len(values))
        axis.scatter(
            np.full(len(values), mode_index) + jitter,
            values,
            s=22,
            color=MODE_COLORS[mode],
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        axis.vlines(
            mode_index,
            values.min(),
            values.max(),
            color=MODE_COLORS[mode],
            linewidth=1.0,
            zorder=2,
        )
        axis.scatter(
            mode_index,
            np.median(values),
            marker="D",
            s=40,
            color=MODE_COLORS[mode],
            edgecolor="white",
            linewidth=0.65,
            zorder=4,
        )
    axis.axhline(
        reference,
        color=INK,
        linewidth=0.8,
        linestyle=(0, (3, 2)),
        zorder=1,
    )
    axis.set_xticks(range(len(MODE_ORDER)), [MODE_LABELS[m] for m in MODE_ORDER])
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", color=GRID, linewidth=0.5)


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
        2,
        2,
        figsize=(WIDTH_MM / 25.4, 126.0 / 25.4),
        gridspec_kw={"hspace": 0.42, "wspace": 0.30},
    )
    specifications = (
        (
            "current_mean_brier_advantage",
            "New-task Brier advantage",
            0.0,
        ),
        (
            "current_mean_error_advantage",
            "New-task error advantage",
            0.0,
        ),
        (
            "old_task_brier_increase",
            "Old-task Brier increase",
            RETENTION_MARGIN,
        ),
        (
            "old_task_error_increase",
            "Old-task error increase",
            RETENTION_MARGIN,
        ),
    )
    for label, axis, spec in zip(
        ("a", "b", "c", "d"),
        axes.flat,
        specifications,
        strict=True,
    ):
        value_column, ylabel, reference = spec
        draw_seed_summary(
            axis,
            plot_data,
            value_column,
            ylabel,
            reference,
        )
        panel_label(axis, label)
    figure.subplots_adjust(left=0.10, right=0.985, bottom=0.10, top=0.96)
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
        "New Brier",
        "New error",
        "Old Brier",
        "Old error",
    )
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(WIDTH_MM / 25.4, 84.0 / 25.4),
        gridspec_kw={"width_ratios": (1.15, 1.0), "wspace": 0.38},
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
                np.full((~promoted).sum(), HORIZON * 1.045),
                y[~promoted],
                facecolor="white",
                edgecolor=MODE_COLORS[mode],
                marker=">",
                linewidth=1.0,
                s=34,
                zorder=3,
            )
    axes[0].set_xlim(0, HORIZON * 1.10)
    axes[0].set_xticks([0, 1_000, 2_000, 3_000, 4_000, 5_000])
    axes[0].set_yticks(
        range(len(MODE_ORDER)),
        [MODE_LABELS[m] for m in MODE_ORDER],
    )
    axes[0].set_ylim(len(MODE_ORDER) - 0.5, -0.5)
    axes[0].set_xlabel("Observations per stream at promotion")
    axes[0].grid(axis="x", color=GRID, linewidth=0.5)
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
    cmap = LinearSegmentedColormap.from_list(
        "neutral_blue",
        ["#F4F6F7", "#AFC2CC", "#315F78"],
    )
    axes[1].imshow(
        pass_counts,
        cmap=cmap,
        vmin=0,
        vmax=5,
        interpolation="nearest",
        aspect="auto",
    )
    for row in range(pass_counts.shape[0]):
        for column in range(pass_counts.shape[1]):
            value = int(pass_counts[row, column])
            text_color = "white" if value >= 4 else INK
            axes[1].text(
                column,
                row,
                f"{value}/5",
                ha="center",
                va="center",
                color=text_color,
                fontweight="bold",
            )
    axes[1].set_xticks(range(4), component_labels, rotation=20, ha="right")
    axes[1].set_yticks(
        range(len(MODE_ORDER)),
        [MODE_LABELS[m] for m in MODE_ORDER],
    )
    axes[1].tick_params(length=0)
    for spine in axes[1].spines.values():
        spine.set_visible(False)
    axes[1].set_xlabel("Component requirements crossed")
    panel_label(axes[1], "b")

    figure.subplots_adjust(left=0.10, right=0.985, bottom=0.21, top=0.95)
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
