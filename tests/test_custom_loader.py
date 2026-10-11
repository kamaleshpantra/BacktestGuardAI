"""
tests/test_custom_loader.py
---------------------------
Unit tests for the SafeCustomStrategy dynamic loader.
"""

import pandas as pd
import pytest

from strategies.custom_loader import SafeCustomStrategy, SecurityViolationError


@pytest.fixture
def clean_df() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    prices = [100.0 + i for i in range(30)]
    return pd.DataFrame(
        {
            "Open": prices,
            "High": [p + 1.0 for p in prices],
            "Low": [p - 1.0 for p in prices],
            "Close": prices,
            "Volume": 100_000,
        },
        index=dates,
    )


class TestSafeCustomStrategy:

    def test_valid_custom_strategy(self, clean_df):
        code = """
def generate_signals(df):
    ma = df['Close'].rolling(5).mean()
    raw = (df['Close'] > ma).astype(float)
    return raw.shift(1).fillna(0)
"""
        s = SafeCustomStrategy(code)
        signals = s.generate_signals(clean_df)
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(clean_df)
        assert set(signals.unique()).issubset({0.0, 1.0})

    def test_security_violation_import_os(self):
        code = """
import os
def generate_signals(df):
    return df['Close']
"""
        with pytest.raises(SecurityViolationError, match="prohibited"):
            SafeCustomStrategy(code)

    def test_security_violation_builtins(self):
        code = """
def generate_signals(df):
    eval("1 + 1")
    return df['Close']
"""
        with pytest.raises(SecurityViolationError, match="prohibited"):
            SafeCustomStrategy(code)

    def test_lookahead_bias_rejection(self):
        code = """
def generate_signals(df):
    # Prohibited negative shift
    return df['Close'].shift(-1)
"""
        with pytest.raises(ValueError, match="Look-Ahead Bias"):
            SafeCustomStrategy(code)
