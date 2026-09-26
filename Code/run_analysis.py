"""End-to-end risk pipeline: portfolio -> VaR/ES -> formal backtests -> stress tests.

Usage
-----
    python run_analysis.py                # real data via yfinance
    python run_analysis.py --synthetic    # offline demo with simulated data
    python run_analysis.py --alpha 0.95   # different confidence level
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src import data as dat
from src import plots
from src.risk import (VAR_METHODS, backtest_summary, expected_shortfall,
                      stress_historical, stress_hypothetical, var_historical)

FIG = Path("figures")
RES = Path("results")

# A diversified risk-based allocation (HRP-style output of the companion
# repo `portfolio-allocation-strategies`). Override via weights.json if present.
DEFAULT_WEIGHTS = {
    "SPY": 0.12, "VGK": 0.08, "EEM": 0.06, "IEF": 0.34,
    "LQD": 0.22, "GLD": 0.12, "DBC": 0.06,
}


def main(synthetic: bool, alpha: float):
    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)

    prices = dat.load_prices(synthetic=synthetic)
    asset_rets = dat.to_returns(prices)

    wfile = Path("weights.json")
    weights = pd.Series(json.loads(wfile.read_text()) if wfile.exists()
                        else DEFAULT_WEIGHTS)
    weights = weights / weights.sum()
    port = (asset_rets @ weights.reindex(asset_rets.columns).fillna(0)).rename("portfolio")
    print(f"[run] portfolio: {len(port)} daily returns "
          f"({port.index[0].date()} -> {port.index[-1].date()}), alpha={alpha:.0%}")

    # ---- point-in-time VaR & ES on the last 500 days
    recent = port.iloc[-500:]
    var_table = pd.DataFrame({
        "VaR": {name: fn(recent, alpha) for name, fn in VAR_METHODS.items()},
    })
    var_table.loc["Historical", "ES"] = expected_shortfall(recent, alpha, "historical")
    var_table.loc["Parametric (Gaussian)", "ES"] = expected_shortfall(recent, alpha, "parametric")
    var_table.to_csv(RES / "var_es_estimates.csv")
    print("\nPoint-in-time estimates (last 500 days):")
    print((var_table * 100).round(2).to_string())

    # ---- rolling backtest + Kupiec / Christoffersen
    print("\n[run] rolling 1-day-ahead backtest (250d window)...")
    tests, series = backtest_summary(port, alpha=alpha, window=250)
    tests.to_csv(RES / "backtest_tests.csv")
    fmt = tests.copy()
    for c in fmt.columns:
        fmt[c] = fmt[c].map(lambda x: f"{x:.4f}" if isinstance(x, float) else x)
    (RES / "backtest_tests.md").write_text(fmt.to_markdown())
    print(tests.round(4).to_string())

    # ---- stress tests
    hist = stress_historical(asset_rets, weights)
    hypo = stress_hypothetical(weights)
    pd.concat([hist.rename("historical"), hypo.rename("hypothetical")]) \
        .to_csv(RES / "stress_tests.csv")
    if hist.empty:
        print("\n[run] note: dataset does not cover the historical crisis windows "
              "(synthetic run?) — only hypothetical shocks reported.")

    # ---- figures
    plots.plot_return_distribution(
        recent, FIG / "return_distribution.png", alpha,
        var_hist=var_historical(recent, alpha),
        es_hist=expected_shortfall(recent, alpha, "historical"),
    )
    plots.plot_var_comparison(var_table, FIG / "var_comparison.png")
    plots.plot_rolling_var(series["Historical"], "Historical",
                           FIG / "rolling_var_historical.png", alpha)
    plots.plot_rolling_var(series["Parametric (Gaussian)"], "Gaussian",
                           FIG / "rolling_var_gaussian.png", alpha)
    plots.plot_stress(hist, hypo, FIG / "stress_tests.png")
    print(f"\n[run] figures saved to {FIG}/, tables to {RES}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--alpha", type=float, default=0.99)
    args = p.parse_args()
    main(args.synthetic, args.alpha)
