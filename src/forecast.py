from typing import List

import numpy as np
import pandas as pd

from src.models import compute_ewma_variance, compute_garch_volatility
from src.evaluation import (
    compute_realized_volatility,
    compute_mae,
    compute_rmse,
    compute_qlike,
)


def evaluate_models_for_one_stock(
    returns: pd.Series,
    stock_name: str,
    horizon: int = 5,
    ewma_lambda: float = 0.94
) -> dict:
    """
    Evaluate EWMA and GARCH for a single stock return series.
    Returns a dictionary of metrics.
    """
    returns = returns.dropna().copy()

    # EWMA
    ewma_var = compute_ewma_variance(returns, lam=ewma_lambda)
    ewma_vol = np.sqrt(ewma_var)

    # GARCH
    garch_vol = compute_garch_volatility(returns)
    garch_var = garch_vol ** 2

    # Realized volatility
    realized_vol = compute_realized_volatility(returns, horizon=horizon, annualize=False)

    # Metrics
    ewma_mae = compute_mae(ewma_vol, realized_vol)
    ewma_rmse = compute_rmse(ewma_vol, realized_vol)
    ewma_qlike = compute_qlike(ewma_var, realized_vol)

    garch_mae = compute_mae(garch_vol, realized_vol)
    garch_rmse = compute_rmse(garch_vol, realized_vol)
    garch_qlike = compute_qlike(garch_var, realized_vol)

    return {
        "stock": stock_name,
        "ewma_mae": ewma_mae,
        "ewma_rmse": ewma_rmse,
        "ewma_qlike": ewma_qlike,
        "garch_mae": garch_mae,
        "garch_rmse": garch_rmse,
        "garch_qlike": garch_qlike,
        "mae_improvement": ewma_mae - garch_mae,
        "rmse_improvement": ewma_rmse - garch_rmse,
        "qlike_improvement": ewma_qlike - garch_qlike,
    }


def evaluate_models_for_all_stocks(
    returns_df: pd.DataFrame,
    horizon: int = 5,
    ewma_lambda: float = 0.94
) -> pd.DataFrame:
    """
    Evaluate EWMA and GARCH across all stocks in a return DataFrame.
    Returns a results DataFrame.
    """
    results = []

    for stock in returns_df.columns:
        stock_returns = returns_df[stock].dropna()

        try:
            stock_result = evaluate_models_for_one_stock(
                stock_returns,
                stock_name=stock,
                horizon=horizon,
                ewma_lambda=ewma_lambda
            )
            results.append(stock_result)
        except Exception as e:
            print(f"Skipping {stock} due to error: {e}")

    results_df = pd.DataFrame(results)

    if not results_df.empty:
        results_df["garch_better_mae"] = results_df["garch_mae"] < results_df["ewma_mae"]
        results_df["garch_better_rmse"] = results_df["garch_rmse"] < results_df["ewma_rmse"]
        results_df["garch_better_qlike"] = results_df["garch_qlike"] < results_df["ewma_qlike"]

    return results_df


def summarize_results(results_df: pd.DataFrame) -> pd.Series:
    """
    Produce a compact summary of model comparison results.
    """
    summary = {
        "num_stocks": len(results_df),
        "avg_ewma_mae": results_df["ewma_mae"].mean(),
        "avg_garch_mae": results_df["garch_mae"].mean(),
        "avg_ewma_rmse": results_df["ewma_rmse"].mean(),
        "avg_garch_rmse": results_df["garch_rmse"].mean(),
        "avg_ewma_qlike": results_df["ewma_qlike"].mean(),
        "avg_garch_qlike": results_df["garch_qlike"].mean(),
        "garch_better_mae_count": results_df["garch_better_mae"].sum(),
        "garch_better_rmse_count": results_df["garch_better_rmse"].sum(),
        "garch_better_qlike_count": results_df["garch_better_qlike"].sum(),
    }

    return pd.Series(summary)


def evaluate_models_multiple_horizons(
    returns_df: pd.DataFrame,
    horizons: list = [1, 5, 10],
    ewma_lambda: float = 0.94
) -> pd.DataFrame:
    """
    Evaluate EWMA and GARCH across multiple horizons and stocks.
    """
    all_results = []

    for horizon in horizons:
        print(f"Running for horizon = {horizon}")

        results_df = evaluate_models_for_all_stocks(
            returns_df,
            horizon=horizon,
            ewma_lambda=ewma_lambda
        )

        results_df["horizon"] = horizon

        all_results.append(results_df)

    final_df = pd.concat(all_results, ignore_index=True)

    return final_df