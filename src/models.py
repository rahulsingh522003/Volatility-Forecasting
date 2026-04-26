from typing import Optional

import numpy as np
import pandas as pd


def compute_ewma_variance(
    returns: pd.Series,
    lam: float = 0.94,
    initial_variance: Optional[float] = None
) -> pd.Series:
    """
    Compute EWMA conditional variance series from a return series.

    Parameters
    ----------
    returns : pd.Series
        Time series of returns.
    lam : float
        Decay factor (lambda), usually between 0 and 1.
    initial_variance : Optional[float]
        Initial variance value. If None, use sample variance of returns.

    Returns
    -------
    pd.Series
        EWMA conditional variance series aligned with returns index.
    """
    if not 0 < lam < 1:
        raise ValueError("lam must be between 0 and 1.")

    returns = returns.dropna().copy()

    if len(returns) == 0:
        raise ValueError("returns series is empty after dropping NaNs.")

    ewma_var = np.zeros(len(returns))

    if len(returns) < 30 and initial_variance is None:
        initial_variance = returns.var()
    elif initial_variance is None:
        initial_variance = returns.iloc[:30].var()

    ewma_var[0] = initial_variance

    for t in range(1, len(returns)):
        ewma_var[t] = lam * ewma_var[t - 1] + (1 - lam) * (returns.iloc[t - 1] ** 2)

    return pd.Series(ewma_var, index=returns.index, name="ewma_variance")


def compute_ewma_volatility(
    returns: pd.Series,
    lam: float = 0.94,
    initial_variance: Optional[float] = None,
    annualize: bool = False,
    trading_days: int = 252
) -> pd.Series:
    """
    Compute EWMA conditional volatility from a return series.

    Parameters
    ----------
    returns : pd.Series
        Time series of returns.
    lam : float
        Decay factor.
    initial_variance : Optional[float]
        Initial variance value.
    annualize : bool
        If True, annualize volatility using sqrt(trading_days).
    trading_days : int
        Number of trading days in a year.

    Returns
    -------
    pd.Series
        EWMA volatility series.
    """
    ewma_var = compute_ewma_variance(returns, lam=lam, initial_variance=initial_variance)
    ewma_vol = np.sqrt(ewma_var)

    if annualize:
        ewma_vol = ewma_vol * np.sqrt(trading_days)

    ewma_vol.name = "ewma_volatility"
    return ewma_vol


def forecast_ewma_variance(
    last_variance: float,
    last_return: float,
    lam: float = 0.94
) -> float:
    """
    One-step-ahead EWMA variance forecast.

    Parameters
    ----------
    last_variance : float
        Latest estimated variance.
    last_return : float
        Latest observed return.
    lam : float
        Decay factor.

    Returns
    -------
    float
        Next-step forecast variance.
    """
    if not 0 < lam < 1:
        raise ValueError("lam must be between 0 and 1.")

    return lam * last_variance + (1 - lam) * (last_return ** 2)






"""
    GARCH model
"""

from arch import arch_model


def compute_garch_volatility(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    dist: str = "normal",
    annualize: bool = False,
    trading_days: int = 252
) -> pd.Series:
    """
    Fit GARCH(p, q) model and return conditional volatility.
    """

    returns = returns.dropna().copy()

    # GARCH expects returns in percentage (important!)
    returns_scaled = returns * 100

    model = arch_model(
        returns_scaled,
        vol="GARCH",
        p=p,
        q=q,
        dist=dist
    )

    res = model.fit(disp="off")

    cond_vol = res.conditional_volatility / 100  # scale back

    if annualize:
        cond_vol = cond_vol * np.sqrt(trading_days)

    return pd.Series(cond_vol, index=returns.index, name="garch_volatility")