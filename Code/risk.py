"""Market-risk engine for a portfolio of assets.

Implements:
- Value-at-Risk: parametric (Gaussian + Cornish-Fisher), historical, Monte Carlo (Student-t)
- Expected Shortfall (CVaR) for each method
- Formal VaR backtesting: Kupiec POF test, Christoffersen independence and
  conditional-coverage tests
- Stress testing: historical scenario replay + hypothetical shocks
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


# ------------------------------------------------------------- VaR estimators
def var_parametric(r: pd.Series, alpha: float = 0.99, cornish_fisher: bool = False) -> float:
    """Gaussian VaR, optionally with the Cornish-Fisher expansion which
    adjusts the quantile for the empirical skewness and excess kurtosis."""
    mu, sigma = r.mean(), r.std()
    z = stats.norm.ppf(1 - alpha)
    if cornish_fisher:
        s, k = stats.skew(r), stats.kurtosis(r)  # excess kurtosis
        z = (z + (z**2 - 1) * s / 6
             + (z**3 - 3 * z) * k / 24
             - (2 * z**3 - 5 * z) * s**2 / 36)
    return -(mu + z * sigma)


def var_historical(r: pd.Series, alpha: float = 0.99) -> float:
    """Empirical quantile of the historical P&L distribution."""
    return -np.quantile(r, 1 - alpha)


def var_monte_carlo(
    r: pd.Series, alpha: float = 0.99, n_sims: int = 100_000, seed: int = 0
) -> float:
    """Monte Carlo VaR under a fitted Student-t distribution (fat tails)."""
    nu, loc, scale = stats.t.fit(r)
    rng = np.random.default_rng(seed)
    sims = stats.t.rvs(nu, loc=loc, scale=scale, size=n_sims, random_state=rng)
    return -np.quantile(sims, 1 - alpha)


def expected_shortfall(r: pd.Series, alpha: float = 0.99, method: str = "historical") -> float:
    """ES / CVaR: expected loss conditional on exceeding the VaR."""
    if method == "historical":
        var = var_historical(r, alpha)
        tail = r[r <= -var]
        return -tail.mean() if len(tail) else var
    if method == "parametric":
        mu, sigma = r.mean(), r.std()
        z = stats.norm.ppf(1 - alpha)
        return -(mu - sigma * stats.norm.pdf(z) / (1 - alpha))
    raise ValueError(method)


VAR_METHODS = {
    "Parametric (Gaussian)": lambda r, a: var_parametric(r, a),
    "Parametric (Cornish-Fisher)": lambda r, a: var_parametric(r, a, cornish_fisher=True),
    "Historical": var_historical,
    "Monte Carlo (Student-t)": lambda r, a: var_monte_carlo(r, a, n_sims=20_000),
}


# ------------------------------------------------------------ rolling backtest
def rolling_var(
    returns: pd.Series, method, alpha: float = 0.99, window: int = 250
) -> pd.DataFrame:
    """One-day-ahead rolling VaR: estimate on the trailing window, compare with
    the *next* day's realized return (strictly out-of-sample)."""
    var_vals, dates = [], []
    vals = returns.values
    for i in range(window, len(returns)):
        var_vals.append(method(pd.Series(vals[i - window : i]), alpha))
        dates.append(returns.index[i])
    out = pd.DataFrame({"VaR": var_vals}, index=pd.Index(dates))
    out["return"] = returns.loc[out.index]
    out["violation"] = out["return"] < -out["VaR"]
    return out


# --------------------------------------------------------------- Kupiec & co.
def kupiec_pof(violations: pd.Series, alpha: float = 0.99) -> dict:
    """Kupiec (1995) Proportion-of-Failures test.

    H0: the observed violation frequency equals the expected 1-alpha.
    LR_pof ~ chi2(1) under H0.
    """
    n = len(violations)
    x = int(violations.sum())
    p = 1 - alpha
    pi_hat = x / n if n else np.nan
    if x in (0, n):
        lr = -2 * (n * np.log(1 - p) if x == 0 else n * np.log(p))
    else:
        lr = -2 * (
            (n - x) * np.log((1 - p) / (1 - pi_hat)) + x * np.log(p / pi_hat)
        )
    return {
        "violations": x,
        "expected": n * p,
        "violation_rate": pi_hat,
        "LR_pof": lr,
        "p_value": 1 - stats.chi2.cdf(lr, df=1),
    }


def christoffersen(violations: pd.Series, alpha: float = 0.99) -> dict:
    """Christoffersen (1998) independence and conditional-coverage tests.

    Independence H0: violations are not clustered (a violation today does not
    change the probability of one tomorrow). LR_ind ~ chi2(1).
    Conditional coverage: LR_cc = LR_pof + LR_ind ~ chi2(2).
    """
    v = violations.astype(int).values
    # transition counts
    n00 = n01 = n10 = n11 = 0
    for prev, curr in zip(v[:-1], v[1:]):
        if prev == 0 and curr == 0: n00 += 1
        elif prev == 0 and curr == 1: n01 += 1
        elif prev == 1 and curr == 0: n10 += 1
        else: n11 += 1

    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    def _ll(p, zeros, ones):
        if p in (0.0, 1.0):
            return 0.0
        return zeros * np.log(1 - p) + ones * np.log(p)

    ll_null = _ll(pi, n00 + n10, n01 + n11)
    ll_alt = _ll(pi01, n00, n01) + _ll(pi11, n10, n11)
    lr_ind = -2 * (ll_null - ll_alt)

    pof = kupiec_pof(violations, alpha)
    lr_cc = pof["LR_pof"] + lr_ind
    return {
        "LR_ind": lr_ind,
        "p_value_ind": 1 - stats.chi2.cdf(lr_ind, df=1),
        "LR_cc": lr_cc,
        "p_value_cc": 1 - stats.chi2.cdf(lr_cc, df=2),
    }


def backtest_summary(
    returns: pd.Series, alpha: float = 0.99, window: int = 250
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Run the rolling backtest + statistical tests for every VaR method."""
    rows, series = {}, {}
    for name, method in VAR_METHODS.items():
        bt = rolling_var(returns, method, alpha, window)
        pof = kupiec_pof(bt["violation"], alpha)
        chr_ = christoffersen(bt["violation"], alpha)
        rows[name] = {
            "Violations": pof["violations"],
            "Expected": round(pof["expected"], 1),
            "Viol. rate": pof["violation_rate"],
            "Kupiec p-value": pof["p_value"],
            "Christoffersen p-value (ind.)": chr_["p_value_ind"],
            "Cond. coverage p-value": chr_["p_value_cc"],
        }
        series[name] = bt
    return pd.DataFrame(rows).T, series


# ---------------------------------------------------------------- stress tests
HISTORICAL_SCENARIOS = {
    "GFC (Sep-Nov 2008)": ("2008-09-01", "2008-11-30"),
    "COVID crash (Feb-Mar 2020)": ("2020-02-19", "2020-03-23"),
    "2022 rate shock (Jan-Oct 2022)": ("2022-01-01", "2022-10-31"),
}

# Hypothetical instantaneous shocks (used when history doesn't cover a scenario)
HYPOTHETICAL_SHOCKS = {
    "Equity crash -30%": {"SPY": -0.30, "VGK": -0.32, "EEM": -0.38, "IEF": 0.06,
                          "LQD": -0.05, "GLD": 0.08, "DBC": -0.15},
    "Rates +200bp": {"SPY": -0.12, "VGK": -0.12, "EEM": -0.15, "IEF": -0.12,
                     "LQD": -0.14, "GLD": -0.05, "DBC": 0.05},
    "Stagflation": {"SPY": -0.20, "VGK": -0.22, "EEM": -0.18, "IEF": -0.08,
                    "LQD": -0.10, "GLD": 0.15, "DBC": 0.25},
    "USD liquidity crunch": {"SPY": -0.15, "VGK": -0.18, "EEM": -0.25, "IEF": 0.03,
                             "LQD": -0.08, "GLD": -0.06, "DBC": -0.12},
}


def stress_historical(asset_returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Replay historical crisis windows through today's portfolio weights."""
    out = {}
    for name, (start, end) in HISTORICAL_SCENARIOS.items():
        window = asset_returns.loc[start:end]
        if len(window) < 5:
            continue  # data does not cover this period
        cum = (1 + window @ weights).prod() - 1
        out[name] = cum
    return pd.Series(out, dtype=float)


def stress_hypothetical(weights: pd.Series) -> pd.Series:
    out = {}
    for name, shocks in HYPOTHETICAL_SHOCKS.items():
        out[name] = sum(weights.get(k, 0.0) * v for k, v in shocks.items())
    return pd.Series(out, dtype=float)
