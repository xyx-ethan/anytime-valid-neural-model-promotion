#!/usr/bin/env python3
"""Build the submission figure from the saved numerical source data."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "source_data"
FIGURE_DIR = ROOT / "figures"
FIG_WIDTH_MM = 183.0
FIG_HEIGHT_MM = 80.0
FIG_HEIGHT_TALL_MM = 165.0
ALPHA = 0.05
HORIZON = 1000
CIFAR_C_HORIZON = 10_000

CORRUPTION_LABELS = {
    "gaussian_noise": "Gaussian noise",
    "shot_noise": "Shot noise",
    "impulse_noise": "Impulse noise",
    "defocus_blur": "Defocus blur",
    "glass_blur": "Glass blur",
    "motion_blur": "Motion blur",
    "zoom_blur": "Zoom blur",
    "snow": "Snow",
    "frost": "Frost",
    "fog": "Fog",
    "brightness": "Brightness",
    "contrast": "Contrast",
    "elastic_transform": "Elastic transform",
    "pixelate": "Pixelate",
    "jpeg_compression": "JPEG compression",
}

COLORS = {
    "e-process": "#315F78",
    "unadjusted 20-look test": "#A65353",
    "Bonferroni 20-look test": "#737F87",
    "ink": "#20272C",
    "muted": "#65727A",
    "line": "#D2D8DC",
    "light": "#F4F6F7",
}

MARKERS = {
    "e-process": "o",
    "unadjusted 20-look test": "s",
    "Bonferroni 20-look test": "^",
}

LINESTYLES = {
    "e-process": "-",
    "unadjusted 20-look test": "--",
    "Bonferroni 20-look test": ":",
}

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
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


def draw_false_promotion(axis: plt.Axes, source: pd.DataFrame) -> None:
    panel = source[source["panel"] == "false_promotion"]
    offsets = {
        "e-process": -0.012,
        "unadjusted 20-look test": 0.0,
        "Bonferroni 20-look test": 0.012,
    }
    for method, offset in offsets.items():
        data = panel[panel["method"] == method].sort_values("x")
        x = data["x"].to_numpy(dtype=float) + offset
        y = data["y"].to_numpy(dtype=float)
        lower = data["lower"].to_numpy(dtype=float)
        upper = data["upper"].to_numpy(dtype=float)
        axis.errorbar(
            x,
            y,
            yerr=np.vstack([y - lower, upper - y]),
            color=COLORS[method],
            marker=MARKERS[method],
            linestyle=LINESTYLES[method],
            linewidth=1.15,
            markersize=4.2,
            capsize=2.0,
            capthick=0.7,
            label=method,
            zorder=3,
        )
    axis.axhline(ALPHA, color=COLORS["ink"], linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
    axis.text(0.305, ALPHA + 0.004, "$\\alpha=0.05$", ha="right", va="bottom", color=COLORS["muted"])
    axis.set_xlim(0.075, 0.325)
    axis.set_ylim(0.0, 0.29)
    axis.set_xticks([0.10, 0.20, 0.30])
    axis.set_xlabel("Equal model error rate")
    axis.set_ylabel("Probability of false promotion")
    axis.grid(axis="y", color=COLORS["line"], linewidth=0.5)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        columnspacing=1.2,
        handlelength=2.2,
    )


def draw_promotion_curve(axis: plt.Axes, source: pd.DataFrame) -> None:
    panel = source[source["panel"] == "promotion_curve"]
    for method in ("e-process", "unadjusted 20-look test", "Bonferroni 20-look test"):
        data = panel[panel["method"] == method].sort_values("x")
        axis.step(
            data["x"],
            data["y"],
            where="post",
            color=COLORS[method],
            linestyle=LINESTYLES[method],
            linewidth=1.25,
            label=method,
        )
    axis.set_xlim(0, HORIZON)
    axis.set_ylim(0.0, 1.02)
    axis.set_xlabel("Observations")
    axis.set_ylabel("Cumulative promotion probability")
    axis.grid(axis="y", color=COLORS["line"], linewidth=0.5)
    axis.text(
        0.03,
        0.96,
        "True loss advantage = 0.10",
        transform=axis.transAxes,
        color=COLORS["muted"],
        ha="left",
        va="top",
    )
    axis.legend(loc="lower right", handlelength=2.4)


def draw_neural_promotion(axis: plt.Axes, source: pd.DataFrame) -> None:
    architecture_styles = {
        "8": ("#315F78", "#B7C7D0", "o", -0.18, "MLP 8"),
        "16": ("#737F87", "#C5CACD", "s", 0.0, "MLP 16"),
        "16-8": ("#A65353", "#DDBABA", "^", 0.18, "MLP 16, 8"),
        "ResNet-18": ("#536B52", "#C0CBBF", "D", 0.0, "ResNet-18"),
    }
    streams = ("Elec2", "Phishing", "Bananas", "CIFAR-10")
    legend_added: set[str] = set()
    for architecture, (
        color,
        light_color,
        marker,
        offset,
        display_label,
    ) in architecture_styles.items():
        for stream_index, stream_name in enumerate(streams):
            data = source[
                (source["stream"] == stream_name)
                & (source["architecture"] == architecture)
            ]
            if data["promotion_time"].isna().any():
                raise ValueError(
                    f"Missing promotion time for {stream_name}, {architecture}"
                )
            promotion_time = data["promotion_time"].to_numpy(dtype=float)
            if len(promotion_time) == 0:
                continue
            y = np.full(len(promotion_time), stream_index + offset)
            y += np.linspace(-0.035, 0.035, num=len(y))
            axis.scatter(
                promotion_time,
                y,
                color=light_color,
                marker=marker,
                edgecolor="none",
                s=18,
                zorder=2,
            )
            median = float(np.median(promotion_time))
            lower = float(np.min(promotion_time))
            upper = float(np.max(promotion_time))
            axis.hlines(
                stream_index + offset,
                lower,
                upper,
                color=color,
                linewidth=1.0,
                zorder=1,
            )
            axis.scatter(
                [median],
                [stream_index + offset],
                color=color,
                marker=marker,
                edgecolor="white",
                linewidth=0.7,
                s=38,
                zorder=3,
                label=display_label if architecture not in legend_added else None,
            )
            legend_added.add(architecture)
    axis.set_xscale("log")
    axis.set_xlim(80, 20_000)
    axis.set_xticks([100, 300, 1000, 3000, 10_000])
    axis.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    axis.set_yticks(range(len(streams)), streams)
    axis.set_ylim(len(streams) - 0.25, -0.62)
    axis.set_xlabel("Post-warm-up observations to promotion")
    axis.grid(axis="x", color=COLORS["line"], linewidth=0.5)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.53, 1.01),
        ncol=4,
        columnspacing=1.0,
        handletextpad=0.4,
    )


def draw_corruption_promotion(axis: plt.Axes, source: pd.DataFrame) -> None:
    observed = set(source["corruption"])
    expected = set(CORRUPTION_LABELS)
    if observed != expected:
        raise ValueError(
            "CIFAR-10-C corruption mismatch: "
            f"missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )

    color = "#536B52"
    light_color = "#C0CBBF"
    ordered_corruptions = sorted(
        CORRUPTION_LABELS,
        key=lambda name: float(
            source.loc[source["corruption"] == name, "promotion_time"]
            .fillna(CIFAR_C_HORIZON)
            .median()
        ),
    )

    for row, corruption in enumerate(ordered_corruptions):
        data = (
            source.loc[source["corruption"] == corruption]
            .sort_values("seed")
            .reset_index(drop=True)
        )
        y = row + np.linspace(-0.13, 0.13, num=len(data))
        promoted = data["promotion_time"].notna().to_numpy()
        values = data["promotion_time"].to_numpy(dtype=float)

        if promoted.any():
            axis.scatter(
                values[promoted],
                y[promoted],
                color=light_color,
                marker="o",
                edgecolor=color,
                linewidth=0.45,
                s=24,
                zorder=2,
            )
        if (~promoted).any():
            axis.scatter(
                np.full((~promoted).sum(), CIFAR_C_HORIZON),
                y[~promoted],
                facecolor="white",
                edgecolor=color,
                marker=">",
                linewidth=0.9,
                s=32,
                zorder=3,
            )

        if promoted.all():
            axis.hlines(
                row,
                float(np.min(values)),
                float(np.max(values)),
                color=color,
                linewidth=1.0,
                zorder=1,
            )
            axis.scatter(
                [float(np.median(values))],
                [row],
                color=color,
                marker="D",
                edgecolor="white",
                linewidth=0.65,
                s=34,
                zorder=4,
            )

    axis.scatter(
        [],
        [],
        color=light_color,
        edgecolor=color,
        linewidth=0.45,
        marker="o",
        s=24,
        label="individual run",
    )
    if source["promotion_time"].isna().any():
        axis.scatter(
            [],
            [],
            facecolor="white",
            edgecolor=color,
            marker=">",
            linewidth=0.9,
            s=32,
            label="no promotion by 10,000",
        )
    axis.scatter(
        [],
        [],
        color=color,
        edgecolor="white",
        linewidth=0.65,
        marker="D",
        s=34,
        label="median and range",
    )
    axis.set_xscale("log")
    axis.set_xlim(250, 13_000)
    axis.set_xticks([300, 1000, 3000, 10_000])
    axis.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    axis.set_yticks(
        range(len(ordered_corruptions)),
        [CORRUPTION_LABELS[name] for name in ordered_corruptions],
    )
    axis.set_ylim(len(ordered_corruptions) - 0.45, -0.55)
    axis.set_xlabel("Labeled observations to promotion on both metrics")
    axis.grid(axis="x", color=COLORS["line"], linewidth=0.5)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.55, 1.06),
        ncol=3 if source["promotion_time"].isna().any() else 2,
        columnspacing=1.2,
        handletextpad=0.4,
    )


def make_figure(
    source: pd.DataFrame,
    neural_source: pd.DataFrame,
    corruption_source: pd.DataFrame,
) -> None:
    figure_specs = (
        ("Fig1", draw_false_promotion, 0.11, FIG_HEIGHT_MM),
        ("Fig2", draw_promotion_curve, 0.11, FIG_HEIGHT_MM),
        (
            "Fig3",
            lambda axis, _: draw_neural_promotion(axis, neural_source),
            0.10,
            FIG_HEIGHT_MM,
        ),
        (
            "Fig4",
            lambda axis, _: draw_corruption_promotion(
                axis,
                corruption_source,
            ),
            0.19,
            FIG_HEIGHT_TALL_MM,
        ),
    )
    for name, draw, left_margin, height_mm in figure_specs:
        figure, axis = plt.subplots(
            figsize=(FIG_WIDTH_MM / 25.4, height_mm / 25.4),
            constrained_layout=False,
        )
        figure.subplots_adjust(
            left=left_margin,
            right=0.985,
            bottom=0.12 if name == "Fig4" else 0.20,
            top=0.96 if name == "Fig4" else 0.95,
        )
        draw(axis, source)
        save_figure(figure, name)
        plt.close(figure)


def save_figure(figure: plt.Figure, name: str) -> None:
    output_base = FIGURE_DIR / name
    figure.savefig(output_base.with_suffix(".pdf"))
    figure.savefig(output_base.with_suffix(".svg"))
    figure.savefig(output_base.with_suffix(".eps"))
    figure.savefig(
        output_base.with_suffix(".tiff"),
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    figure.savefig(output_base.with_suffix(".png"), dpi=600)
    figure.savefig(FIGURE_DIR / f"{name}_preview.png", dpi=300)


def main() -> None:
    source = pd.read_csv(DATA_DIR / "Figure1_source_data.csv")
    neural_source = pd.read_csv(DATA_DIR / "Neural_stream_results.csv")
    image_source = pd.read_csv(
        DATA_DIR / "cifar10_stream" / "CIFAR10_neural_stream_results.csv"
    ).rename(columns={"dataset": "stream"})
    corruption_source = pd.read_csv(
        DATA_DIR / "cifar10c_stream" / "CIFAR10C_stream_results.csv"
    )
    neural_source = pd.concat([neural_source, image_source], ignore_index=True)
    make_figure(source, neural_source, corruption_source)


if __name__ == "__main__":
    main()
