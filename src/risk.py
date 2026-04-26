import numpy as np
import pandas as pd
from scipy.stats import norm


def compute_var(
    volatility: pd.Series,
    confidence_level: float = 0.95
) -> pd.Series:
    """
    Compute Value at Risk (VaR) assuming normal distribution.

    VaR_t = z * sigma_t
    """

    z_score = norm.ppf(1 - confidence_level)

    var = z_score * volatility

    return pd.Series(var, index=volatility.index, name=f"VaR_{int(confidence_level*100)}")


def compute_var_violations(
    returns: pd.Series,
    var: pd.Series
) -> float:
    """
    Compute violation rate: how often actual returns breach VaR.
    """

    df = pd.concat([returns, var], axis=1).dropna()
    df.columns = ["returns", "var"]

    violations = df["returns"] < df["var"]

    violation_rate = violations.mean()

    return violation_rate