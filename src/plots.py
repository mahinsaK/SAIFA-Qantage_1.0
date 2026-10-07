"""Publication-ready result figures."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .results import ScaleResult


BLUE = "#173A63"
ORANGE = "#E5892E"
GREEN = "#27836D"
GRID = "#D8E0E8"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "figure.dpi": 140,
        }
    )


def plot_tail_dependence(
    results: Sequence[ScaleResult],
    path: Path,
    tail_quantile: float,
) -> None:
    _style()
    x = np.arange(len(results))
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.plot(
        x,
        [result.empirical_tail for result in results],
        marker="o",
        lw=2,
        color=BLUE,
        label=f"Empirical {100 * tail_quantile:g}% co-exceedance",
    )
    ax.plot(
        x,
        [result.t_tail for result in results],
        marker="s",
        lw=2,
        color=ORANGE,
        label="Fitted t-copula asymptotic tail",
    )
    ax.axhline(0, color="#666", lw=0.8)
    ax.set_xticks(x, [result.scale for result in results], rotation=20, ha="right")
    ax.set_ylabel("Average lower-tail dependence")
    ax.set_title("Dependence in losses changes across wavelet horizons")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_risk_comparison(
    results: Sequence[ScaleResult],
    path: Path,
    confidence_level: float,
) -> None:
    _style()
    x = np.arange(len(results))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    t_es = np.array([result.t_risk.expected_shortfall for result in results])
    gaussian_es = np.array(
        [result.gaussian_risk.expected_shortfall for result in results]
    )
    ax.bar(x - width / 2, 100 * t_es, width, color=BLUE, label="Wavelet + t-copula")
    ax.bar(x + width / 2, 100 * gaussian_es, width, color=ORANGE, label="Single Gaussian dependence")
    ax.set_xticks(x, [result.scale for result in results], rotation=20, ha="right")
    confidence_pct = 100 * confidence_level
    ax.set_ylabel(f"{confidence_pct:g}% Expected Shortfall (% return)")
    ax.set_title("Ignoring horizon-specific dependence changes measured tail risk")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_backtest(
    results: Sequence[ScaleResult],
    path: Path,
    confidence_level: float,
) -> None:
    _style()
    x = np.arange(len(results))
    width = 0.36
    target = 1 - confidence_level
    confidence_pct = 100 * confidence_level
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    t_rates = np.array([result.t_backtest.exception_rate for result in results])
    gaussian_rates = np.array(
        [result.gaussian_backtest.exception_rate for result in results]
    )
    ax.bar(x - width / 2, 100 * t_rates, width, color=BLUE, label="Wavelet + t-copula")
    ax.bar(x + width / 2, 100 * gaussian_rates, width, color=ORANGE, label="Single Gaussian dependence")
    ax.axhline(
        100 * target,
        color=GREEN,
        ls="--",
        lw=1.8,
        label=f"Nominal {100 * target:g}%",
    )
    plotted_max = max(
        float((100 * t_rates).max()),
        float((100 * gaussian_rates).max()),
        100 * target,
    )
    ax.set_ylim(0, 1.30 * plotted_max)
    ax.set_xticks(x, [result.scale for result in results], rotation=20, ha="right")
    ax.set_ylabel("Holdout VaR exception rate (%)")
    ax.set_title(f"Out-of-sample {confidence_pct:g}% VaR coverage")
    ax.legend(frameon=False, ncol=3, fontsize=7.2, loc="upper center")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
