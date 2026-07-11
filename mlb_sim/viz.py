"""Matplotlib output figures.

Follows the dataviz method: form chosen by the data's job, one hue per series
role, recessive grid/axes, thin marks, direct labels, text in ink tokens
(never the series color), a legend whenever two series share a plot.

Colors come from the validated reference categorical palette (slot 1 blue for
the base case / single-series charts, slot 6 red for the stress case) on the
light chart surface.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import OUTPUT_DIR
from .sensitivity import ScenarioOutcome

# Reference palette (dataviz skill, validated set) — light mode.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"     # categorical slot 1 — base / single series
RED = "#e34948"      # categorical slot 6 — stress scenario

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "text.color": INK,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_2,
    "xtick.color": MUTED,
    "ytick.color": INK_2,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def _style_barh_axis(ax: plt.Axes) -> None:
    """Recessive chrome: hairline value grid, baseline only, no tick marks."""
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _barh_with_labels(ax: plt.Axes, labels: list[str], values: np.ndarray,
                      highlight: str | None = None) -> None:
    """Thin horizontal bars, best at top, direct value labels in ink."""
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, height=0.58, color=BLUE, edgecolor="none")
    ax.set_yticks(y, labels)
    for yi, v, lab in zip(y, values, labels):
        ax.text(v + max(values) * 0.015, yi, f"{v:.1f}%", va="center",
                fontsize=8.5, color=INK_2)
        if highlight and lab == highlight:
            ax.get_yticklabels()[list(y).index(yi)].set_fontweight("bold")
    ax.set_xlim(0, max(values) * 1.14)


def plot_season_outlook(summary: pd.DataFrame, n_sims: int,
                        season: int, highlight_abbrev: str = "LAD") -> str:
    """Ranking dashboard: World Series odds + playoff odds, single hue."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 7.4))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.86, bottom=0.08,
                        wspace=0.30)

    top = summary.head(12)
    _barh_with_labels(ax1, top["abbrev"].tolist(),
                      top["ws_pct"].to_numpy(), highlight_abbrev)
    ax1.set_title("World Series championship odds — top 12",
                  fontsize=11, color=INK, loc="left", pad=10)
    _style_barh_axis(ax1)

    po = summary.sort_values("playoff_pct", ascending=False)
    _barh_with_labels(ax2, po["abbrev"].tolist(),
                      po["playoff_pct"].to_numpy(), highlight_abbrev)
    ax2.set_title("Playoff odds — all 30 clubs", fontsize=11, color=INK,
                  loc="left", pad=10)
    ax2.tick_params(axis="y", labelsize=7.5)
    _style_barh_axis(ax2)

    fig.suptitle(f"{season} MLB season outlook — hybrid Elo + roster model",
                 fontsize=14, color=INK, x=0.07, ha="left")
    fig.text(0.07, 0.895, f"{n_sims:,} Monte Carlo trials of the remaining "
             "schedule and full postseason bracket; completed games locked in",
             fontsize=9.5, color=INK_2)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "season_outlook.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def plot_ohtani_sensitivity(a: ScenarioOutcome, b: ScenarioOutcome,
                            season: int) -> str:
    """LAD win distribution under both health states + October odds strip."""
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(12.6, 5.9), gridspec_kw={"width_ratios": [1.9, 1.0]})
    fig.subplots_adjust(left=0.06, right=0.965, top=0.80, bottom=0.11,
                        wspace=0.26)

    lo = int(min(a.lad_wins.min(), b.lad_wins.min()))
    hi = int(max(a.lad_wins.max(), b.lad_wins.max()))
    bins = np.arange(lo, hi + 2) - 0.5
    mean_a, mean_b = a.lad_wins.mean(), b.lad_wins.mean()
    for outcome, color, mean in ((a, BLUE, mean_a), (b, RED, mean_b)):
        counts, edges = np.histogram(outcome.lad_wins, bins=bins, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        ax.plot(centers, counts * 100, color=color, linewidth=2,
                label=outcome.label)
        ax.fill_between(centers, counts * 100, color=color, alpha=0.14,
                        linewidth=0)
        ax.axvline(mean, color=color, linewidth=1.2, linestyle=(0, (4, 3)))
        # Label each mean on the outer side of its line so they never collide.
        outer_right = mean >= (mean_a + mean_b) / 2
        ax.text(mean + (0.25 if outer_right else -0.25),
                ax.get_ylim()[1] * 0.02, f"{mean:.1f}", fontsize=9,
                color=INK_2, ha="left" if outer_right else "right")

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.set_xlabel("Regular-season wins")
    ax.set_ylabel("Share of trials (%)")
    ax.set_title("Dodgers win distribution", fontsize=11, color=INK,
                 loc="left", pad=10)
    leg = ax.legend(frameon=False, fontsize=9.5, loc="upper left",
                    labelcolor=INK_2)
    for h in leg.legend_handles:
        h.set_linewidth(3)

    # Right panel: October odds under each scenario, paired bars.
    metrics = [("Win NL West", "lad_division_pct"),
               ("Make playoffs", "lad_playoff_pct"),
               ("Win pennant", "lad_pennant_pct"),
               ("Win World Series", "lad_ws_pct")]
    y = np.arange(len(metrics))[::-1]
    off = 0.19
    va = [getattr(a, attr) for _, attr in metrics]
    vb = [getattr(b, attr) for _, attr in metrics]
    ax2.barh(y + off, va, height=0.34, color=BLUE, edgecolor="none")
    ax2.barh(y - off, vb, height=0.34, color=RED, edgecolor="none")
    ax2.set_yticks(y, [m for m, _ in metrics])
    xmax = max(*va, *vb)
    for yi, v in zip(y + off, va):
        ax2.text(v + xmax * 0.02, yi, f"{v:.1f}%", va="center", fontsize=8.5,
                 color=INK_2)
    for yi, v in zip(y - off, vb):
        ax2.text(v + xmax * 0.02, yi, f"{v:.1f}%", va="center", fontsize=8.5,
                 color=INK_2)
    ax2.set_xlim(0, xmax * 1.2)
    ax2.set_title("October odds by scenario", fontsize=11, color=INK,
                  loc="left", pad=10)
    _style_barh_axis(ax2)

    fig.suptitle(f"Sensitivity analysis — Shohei Ohtani health, "
                 f"{season} Dodgers", fontsize=14, color=INK, x=0.06,
                 ha="left")
    fig.text(0.06, 0.865, "Scenario B moves his innings to a replacement-"
             "level arm (league FIP + 1.00) while his DH bat stays in the "
             "lineup; identical seeds isolate the injury effect",
             fontsize=9.5, color=INK_2)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "ohtani_sensitivity.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)
