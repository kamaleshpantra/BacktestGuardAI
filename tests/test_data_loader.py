"""
tests/test_data_loader.py
-------------------------
Unit tests for the smart CSV normalization loader.
"""

import pandas as pd

from data.loader import load_csv


class TestDataLoader:

    def test_lowercase_headers(self):
        csv_text = """date,open,high,low,close,volume
2024-01-01,100,105,95,102,500000
2024-01-02,102,107,101,106,600000
"""
        df = load_csv(csv_text.encode("utf-8"))
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert isinstance(df.index, pd.DatetimeIndex)
        assert len(df) == 2

    def test_timestamp_column_as_datetime(self):
        csv_text = """timestamp,Open,High,Low,Close,Vol,Adj Close
2024-01-01 09:30:00,10,12,9,11,1000,11
2024-01-02 09:30:00,11,13,10,12,1200,12
"""
        df = load_csv(csv_text.encode("utf-8"))
        assert "Volume" in df.columns
        assert "Adj Close" in df.columns  # preserved
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.name == "Date"

    def test_custom_casing_and_spaces(self):
        csv_text = """ Time , OPEN , HIGH , LOW , CLOSE , SHARES
2024-01-01,50,55,48,52,25000
"""
        df = load_csv(csv_text.encode("utf-8"))
        assert {"Open", "High", "Low", "Close", "Volume"}.issubset(set(df.columns))
        assert isinstance(df.index, pd.DatetimeIndex)
