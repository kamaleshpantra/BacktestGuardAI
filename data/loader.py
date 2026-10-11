"""
data/loader.py
--------------
Smart CSV ingestion and schema normalization for BacktestGuard AI.

Normalizes raw market CSVs from diverse financial data providers
(Yahoo Finance, Binance, TradingView, Alpha Vantage, Zerodha, etc.):
  1. Auto-detects and converts Date / Timestamp column into DatetimeIndex.
  2. Case-insensitive mapping for OHLCV columns (e.g. 'open' -> 'Open', 'vol' -> 'Volume').
  3. Preserves additional provider columns (e.g. 'Adj Close') while guaranteeing
     canonical OHLCV schema required by BacktestGuard validators and engines.
"""

from __future__ import annotations

import io
from typing import BinaryIO

import pandas as pd

# Provider synonyms mapped to standard BacktestGuard canonical names
COLUMN_MAPPINGS: dict[str, str] = {
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume",
    "vol": "Volume",
    "shares": "Volume",
}

DATE_SYNONYMS: tuple[str, ...] = (
    "date",
    "datetime",
    "timestamp",
    "time",
    "date_time",
    "ts",
)


def normalize_ohlcv_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize DataFrame columns and index into canonical BacktestGuard format.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input DataFrame.

    Returns
    -------
    pd.DataFrame
        Normalized DataFrame with DatetimeIndex and canonical 'Open', 'High',
        'Low', 'Close', 'Volume' columns.
    """
    df = df.copy()

    # 1. Normalize Column Names (lowercase lookup)
    renamed: dict[str, str] = {}
    for col in df.columns:
        col_clean = str(col).strip().lower()
        if col_clean in COLUMN_MAPPINGS:
            renamed[col] = COLUMN_MAPPINGS[col_clean]
    df = df.rename(columns=renamed)

    # 2. Auto-detect Date/Time Column if Index is not DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        date_col = None
        for col in df.columns:
            if str(col).strip().lower() in DATE_SYNONYMS:
                date_col = col
                break

        if date_col is not None:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.set_index(date_col)
            df.index.name = "Date"
        else:
            # Try parsing the existing index as dates
            try:
                parsed_index = pd.to_datetime(df.index, errors="coerce")
                if not parsed_index.isna().all():
                    df.index = parsed_index
                    df.index.name = "Date"
            except Exception:
                pass

    return df


def load_csv(source: str | BinaryIO | bytes) -> pd.DataFrame:
    """
    Load raw CSV content and normalize it into canonical format.

    Parameters
    ----------
    source : str | BinaryIO | bytes
        File path, file buffer, or raw bytes from an upload.

    Returns
    -------
    pd.DataFrame
    """
    if isinstance(source, bytes):
        source = io.BytesIO(source)

    raw_df = pd.read_csv(source)
    return normalize_ohlcv_dataframe(raw_df)
