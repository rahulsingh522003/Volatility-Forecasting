import os
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import yfinance as yf


TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "TSLA", "JPM", "BAC", "GS",
    "WMT", "COST", "PG", "KO", "PEP",
    "XOM", "CVX", "UNH", "JNJ", "PFE",
    "HD", "CAT", "BA", "AMD", "NFLX"
]


def download_price_data(
    tickers: List[str],
    start_date: str = "2018-01-01",
    end_date: str = "2025-01-01"
) -> pd.DataFrame:
    """
    Download adjusted close prices for the given tickers.
    Returns a DataFrame with dates as index and tickers as columns.
    """
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        raise ValueError("No data downloaded. Check tickers or date range.")

    # yfinance may return MultiIndex columns
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            prices = data["Close"].copy()
        else:
            prices = data["Adj Close"].copy()
    else:
        prices = data.copy()

    prices = prices.sort_index()
    prices = prices.dropna(axis=1, how="all")
    return prices


def clean_price_data(prices: pd.DataFrame, max_missing_frac: float = 0.05) -> pd.DataFrame:
    """
    Remove tickers with too many missing values, then forward-fill and drop remaining missing rows.
    """
    missing_frac = prices.isna().mean()
    keep_cols = missing_frac[missing_frac <= max_missing_frac].index
    cleaned = prices[keep_cols].copy()

    cleaned = cleaned.ffill().dropna()
    return cleaned


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns.
    """
    log_returns = np.log(prices / prices.shift(1))
    return log_returns.dropna()


def split_time_series(
    data: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15
) -> Dict[str, pd.DataFrame]:
    """
    Split time-series data into train / validation / test by time order.
    """
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

    n = len(data)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    train = data.iloc[:train_end].copy()
    val = data.iloc[train_end:val_end].copy()
    test = data.iloc[val_end:].copy()

    return {
        "train": train,
        "val": val,
        "test": test
    }


def save_dataframes(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    splits: Dict[str, pd.DataFrame],
    output_dir: str = "data"
) -> None:
    """
    Save prices, returns, and splits to CSV files in project root /data folder.
    """
    # Get project root (one level above src/)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # Create correct data path
    output_path = os.path.join(project_root, output_dir)

    os.makedirs(output_path, exist_ok=True)

    prices.to_csv(os.path.join(output_path, "prices.csv"))
    returns.to_csv(os.path.join(output_path, "log_returns.csv"))

    splits["train"].to_csv(os.path.join(output_path, "train_returns.csv"))
    splits["val"].to_csv(os.path.join(output_path, "val_returns.csv"))
    splits["test"].to_csv(os.path.join(output_path, "test_returns.csv"))


def build_dataset(
    tickers: List[str] = TICKERS,
    start_date: str = "2018-01-01",
    end_date: str = "2025-01-01",
    output_dir: str = "data"
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Full pipeline:
    1. Download prices
    2. Clean data
    3. Compute log returns
    4. Split into train / val / test
    5. Save all outputs
    """
    prices = download_price_data(tickers, start_date, end_date)
    prices = clean_price_data(prices)
    returns = compute_log_returns(prices)
    splits = split_time_series(returns)

    save_dataframes(prices, returns, splits, output_dir)
    return prices, returns, splits


if __name__ == "__main__":
    prices_df, returns_df, split_dict = build_dataset()

    print("Price data shape:", prices_df.shape)
    print("Return data shape:", returns_df.shape)
    print("Train shape:", split_dict["train"].shape)
    print("Validation shape:", split_dict["val"].shape)
    print("Test shape:", split_dict["test"].shape)
    print("Tickers used:", list(prices_df.columns))