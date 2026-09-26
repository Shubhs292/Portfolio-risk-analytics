"""Figures for the risk analytics pipeline."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def plot_rolling_var(bt: pd.DataFrame, title: str, path: str, alpha: float):
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.plot(bt.index, bt["return"] * 100, lw=0.5, color="#9aa5b1",
            label="Daily P&L")
    ax.plot(bt.index, -bt["VaR"] * 100, lw=1.2, color="#d62728",
            label=f"VaR {alpha:.0%} ({title})")
    viol = bt[bt["violation"]]
    ax.scatter(viol.index, viol["return"] * 100, color="#d62728", s=14,
               zorder=5, label=f"Violations ({len(viol)})")
    ax.set_ylabel("Daily return (%)")
    ax.set_title(f"1-day {alpha:.0%} VaR backtest — {title}")
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_var_comparison(var_table: pd.DataFrame, path: str):
    fig, ax = plt.subplots(figsize=(8.5, 4))
    (var_table * 100).plot(kind="bar", ax=ax, width=0.75)
    ax.set_ylabel("Loss (% of portfolio, 1 day)")
    ax.set_title("VaR and Expected Shortfall by estimation method")
    plt.xticks(rotation=15, ha="right")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_return_distribution(r: pd.Series, path: str, alpha: float, var_hist: float,
                             es_hist: float):
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.hist(r * 100, bins=120, density=True, alpha=0.55, color="#1f77b4",
            label="Empirical")
    x = np.linspace(r.min(), r.max(), 400)
    ax.plot(x * 100, stats.norm.pdf(x, r.mean(), r.std()) / 100, "k--", lw=1,
            label="Gaussian fit")
    nu, loc, scale = stats.t.fit(r)
    ax.plot(x * 100, stats.t.pdf(x, nu, loc, scale) / 100, color="#2ca02c", lw=1.2,
            label=f"Student-t fit (ν={nu:.1f})")
    ax.axvline(-var_hist * 100, color="#d62728", lw=1.2,
               label=f"Hist. VaR {alpha:.0%} = {var_hist:.2%}")
    ax.axvline(-es_hist * 100, color="#d62728", ls=":", lw=1.2,
               label=f"Hist. ES {alpha:.0%} = {es_hist:.2%}")
    ax.set_xlabel("Daily return (%)")
    ax.set_title("Portfolio return distribution: fat tails vs. the Gaussian")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_stress(hist: pd.Series, hypo: pd.Series, path: str):
    both = pd.concat([hist.rename("Historical replay"),
                      hypo.rename("Hypothetical shock")])
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    colors = ["#1f77b4"] * len(hist) + ["#9467bd"] * len(hypo)
    ax.barh(both.index, both.values * 100, color=colors, alpha=0.85)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Portfolio P&L over the scenario (%)")
    ax.set_title("Stress tests: historical replay (blue) and hypothetical shocks (purple)")
    for i, v in enumerate(both.values):
        ax.text(v * 100 + (0.4 if v >= 0 else -0.4), i, f"{v:.1%}",
                va="center", ha="left" if v >= 0 else "right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
