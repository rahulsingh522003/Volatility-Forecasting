from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from src.models import compute_ewma_variance
from src.evaluation import compute_realized_volatility, compute_qlike


def tune_ewma_lambda_for_one_stock(
    returns: pd.Series,
    lambdas: List[float] = [0.90, 0.92, 0.94, 0.96, 0.98, 0.985, 0.99, 0.995],
    horizon: int = 5
) -> Tuple[float, pd.DataFrame]:
    """
    Tune EWMA lambda for a single stock by minimizing QLIKE.

    Parameters
    ----------
    returns : pd.Series
        Validation return series for one stock.
    lambdas : List[float]
        Candidate lambda values to evaluate.
    horizon : int
        Realized volatility horizon.

    Returns
    -------
    best_lambda : float
        Lambda with lowest QLIKE.
    results_df : pd.DataFrame
        Table of lambda vs QLIKE.
    """
    returns = returns.dropna().copy()

    realized_vol = compute_realized_volatility(
        returns,
        horizon=horizon,
        annualize=False
    )

    results = []

    for lam in lambdas:
        ewma_var = compute_ewma_variance(returns, lam=lam)
        qlike = compute_qlike(ewma_var, realized_vol)

        results.append({
            "lambda": lam,
            "qlike": qlike
        })

    results_df = pd.DataFrame(results).sort_values("qlike").reset_index(drop=True)
    best_lambda = results_df.loc[0, "lambda"]

    return best_lambda, results_df


def tune_ewma_lambda_for_all_stocks(
    returns_df: pd.DataFrame,
    lambdas: List[float] = [0.90, 0.92, 0.94, 0.96, 0.98],
    horizon: int = 5
) -> pd.DataFrame:
    """
    Tune EWMA lambda for all stocks independently using QLIKE.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Validation return DataFrame with one column per stock.
    lambdas : List[float]
        Candidate lambda values.
    horizon : int
        Realized volatility horizon.

    Returns
    -------
    pd.DataFrame
        Per-stock best lambda and best QLIKE.
    """
    all_results = []

    for stock in returns_df.columns:
        stock_returns = returns_df[stock].dropna()

        try:
            best_lambda, results_df = tune_ewma_lambda_for_one_stock(
                stock_returns,
                lambdas=lambdas,
                horizon=horizon
            )

            best_qlike = results_df.loc[0, "qlike"]

            all_results.append({
                "stock": stock,
                "best_lambda": best_lambda,
                "best_qlike": best_qlike
            })

        except Exception as e:
            print(f"Skipping {stock} due to error: {e}")

    return pd.DataFrame(all_results).sort_values("stock").reset_index(drop=True)


def summarize_lambda_tuning(results_df: pd.DataFrame) -> pd.Series:
    """
    Summarize tuned lambda results across all stocks.
    """
    summary = {
        "num_stocks": len(results_df),
        "avg_best_lambda": results_df["best_lambda"].mean(),
        "median_best_lambda": results_df["best_lambda"].median(),
        "min_best_lambda": results_df["best_lambda"].min(),
        "max_best_lambda": results_df["best_lambda"].max(),
    }

    return pd.Series(summary)