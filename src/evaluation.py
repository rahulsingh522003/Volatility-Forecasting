import numpy as np
import pandas as pd


def compute_realized_volatility(
    returns: pd.Series,
    horizon: int = 5,
    annualize: bool = False,
    trading_days: int = 252
) -> pd.Series:
    """
    Compute forward realized volatility.

    RV_t = sqrt(sum_{i=1}^{horizon} r_{t+i}^2)
    """

    returns = returns.dropna().copy()

    squared_returns = returns.pow(2)

    realized_var = np.full(len(returns), np.nan)

    for t in range(len(returns) - horizon):
        realized_var[t] = squared_returns.iloc[t + 1 : t + 1 + horizon].sum()

    realized_vol = np.sqrt(realized_var)

    if annualize:
        realized_vol = realized_vol * np.sqrt(trading_days / horizon)

    return pd.Series(realized_vol, index=returns.index, name=f"realized_vol_{horizon}d")


def compute_mae(pred: pd.Series, actual: pd.Series) -> float:
    """
    Mean Absolute Error
    """
    df = pd.concat([pred, actual], axis=1).dropna()
    return (df.iloc[:, 0] - df.iloc[:, 1]).abs().mean()


def compute_rmse(pred: pd.Series, actual: pd.Series) -> float:
    """
    Root Mean Squared Error
    """
    df = pd.concat([pred, actual], axis=1).dropna()
    return np.sqrt(((df.iloc[:, 0] - df.iloc[:, 1]) ** 2).mean())


def compute_qlike(pred_var: pd.Series, actual_vol: pd.Series) -> float:
    df = pd.concat([pred_var, actual_vol], axis=1).dropna()

    pred_var = df.iloc[:, 0]
    actual_vol = df.iloc[:, 1]

    actual_var = actual_vol ** 2

    eps = 1e-8
    pred_var = np.maximum(pred_var, eps)
    actual_var = np.maximum(actual_var, eps)

    ratio = actual_var / pred_var

    qlike = ratio - np.log(ratio) - 1

    return qlike.mean()