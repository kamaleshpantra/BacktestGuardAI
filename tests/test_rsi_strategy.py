"""
tests/test_rsi_strategy.py
--------------------------
Unit tests for the RSIMeanReversion strategy.
"""

import numpy as np
import pandas as pd
import pytest

from strategies.rsi import RSIMeanReversion


@pytest.fixture
def sample_data() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=50, freq="B")
    # Synthetic oscillating price path
    prices = [100.0 + 5.0 * np.sin(i / 3.0) for i in range(50)]
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


class TestRSIStrategy:

    def test_rsi_config_valid(self):
        s = RSIMeanReversion(period=14, oversold_threshold=30.0, overbought_threshold=70.0)
        assert s.validate_config() == []

    def test_rsi_config_invalid_thresholds(self):
        s = RSIMeanReversion(oversold_threshold=80.0, overbought_threshold=20.0)
        assert len(s.validate_config()) > 0

    def test_rsi_config_negative_period(self):
        s = RSIMeanReversion(period=0)
        assert len(s.validate_config()) > 0

    def test_rsi_signals_generation(self, sample_data):
        s = RSIMeanReversion(period=10, oversold_threshold=35.0, overbought_threshold=65.0)
        signals = s.generate_signals(sample_data)

        assert isinstance(signals, pd.Series)
        assert len(signals) == len(sample_data)
        assert set(signals.unique()).issubset({0.0, 1.0})
        # Verifies 1-bar execution lag: the very first row must be 0
        assert signals.iloc[0] == 0.0
