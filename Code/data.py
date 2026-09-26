"""Data layer: downloads real market data via yfinance, or generates a
realistic synthetic dataset (multivariate Student-t returns calibrated to
long-run asset-class parameters) for offline/demo runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Multi-asset investable universe (all liquid ETFs)
ASSETS = {
    "SPY": "US Equity (S&P 500)",
    "VGK": "Europe Equity",
    "EEM": "EM Equity",
    "IEF": "US Treasuries 7-10y",
    "LQD": "US IG Corporate Bonds",
    "GLD": "Gold",
    "DBC": "Broad Commodities",
}

# Long-run calibration used for the synthetic generator
# (annualised mean, annualised vol) -- rough historical orders of magnitude
_SYN_PARAMS = {
    "SPY": (0.095, 0.16),
    "VGK": (0.070, 0.18),
    "EEM": (0.065, 0.21),
    "IEF": (0.030, 0.065),
    "LQD": (0.040, 0.08),
    "GLD": (0.055, 0.15),
    "DBC": (0.030, 0.17),
}

_SYN_CORR = np.array([
    #  SPY   VGK   EEM   IEF   LQD   GLD   DBC
    [1.00, 0.85, 0.75, -0.20, 0.25, 0.05, 0.35],  # SPY
    [0.85, 1.00, 0.78, -0.18, 0.28, 0.10, 0.38],  # VGK
    [0.75, 0.78, 1.00, -0.12, 0.30, 0.18, 0.45],  # EEM
    [-0.20, -0.18, -0.12, 1.00, 0.60, 0.25, -0.10],  # IEF
    [0.25, 0.28, 0.30, 0.60, 1.00, 0.20, 0.10],  # LQD
    [0.05, 0.10, 0.18, 0.25, 0.20, 1.00, 0.30],  # GLD
    [0.35, 0.38, 0.45, -0.10, 0.10, 0.30, 1.00],  # DBC
])

TRADING_DAYS = 252


def download_prices(start: str = "2007-01-01", end: str | None = None) -> pd.DataFrame:
    """Download adjusted close prices for the universe via yfinance."""
    import yfinance as yf  # optional dependency, only needed for real data

    data = yf.download(list(ASSETS), start=start, end=end, auto_adjust=True)
    prices = data["Close"].dropna(how="any")
    return prices[list(ASSETS)]


def synthetic_prices(
    n_years: int = 15, seed: int = 42, nu: float = 6.0, start: str = "2010-01-01"
) -> pd.DataFrame:
    """Simulate daily prices from a multivariate Student-t model.

    A Student-t with ``nu`` degrees of freedom reproduces the fat tails of
    daily financial returns far better than a Gaussian. Marginals are scaled
    so that annualised means/vols match ``_SYN_PARAMS``.
    """
    rng = np.random.default_rng(seed)
    tickers = list(ASSETS)
    n = n_years * TRADING_DAYS

    mu = np.array([_SYN_PARAMS[t][0] for t in tickers]) / TRADING_DAYS
    vol = np.array([_SYN_PARAMS[t][1] for t in tickers]) / np.sqrt(TRADING_DAYS)

    # multivariate t = gaussian / sqrt(chi2/nu), rescaled to unit variance
    L = np.linalg.cholesky(_SYN_CORR)
    z = rng.standard_normal((n, len(tickers))) @ L.T
    chi = rng.chisquare(nu, size=(n, 1)) / nu
    t_shocks = z / np.sqrt(chi) * np.sqrt((nu - 2) / nu)

    rets = mu + vol * t_shocks
    dates = pd.bdate_range(start=start, periods=n)
    prices = 100.0 * np.exp(np.cumsum(np.log1p(rets), axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers)


def load_prices(synthetic: bool = False, **kwargs) -> pd.DataFrame:
    if synthetic:
        return synthetic_prices(**kwargs)
    try:
        return download_prices(**kwargs)
    except Exception as exc:  # graceful fallback (no network, etc.)
        print(f"[data] yfinance download failed ({exc}); using synthetic data.")
        return synthetic_prices()


def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="any")
