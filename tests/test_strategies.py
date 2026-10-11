"""
tests/test_strategies.py
-------------------------
Tests for the Strategy base class and MovingAverageCrossover implementation.

Test categories:
  - Configuration validation (valid and invalid parameters)
  - Signal shape and value constraints
  - Look-ahead bias prevention (the shift(1) rule)
  - Crossover direction correctness
  - Behaviour at insufficient data
  - Edge cases: single crossover, equal MAs, minimum rows
"""

import pandas as pd
import pytest

from strategies.moving_average import MovingAverageCrossover

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_strategy() -> MovingAverageCrossover:
    """Strategy with default windows: short=20, long=50."""
    return MovingAverageCrossover(short_window=20, long_window=50)


@pytest.fixture
def small_strategy() -> MovingAverageCrossover:
    """Strategy with small windows for use with compact test DataFrames."""
    return MovingAverageCrossover(short_window=3, long_window=5)


def make_df(prices: list[float]) -> pd.DataFrame:
    """
    Build a minimal OHLCV DataFrame from a list of closing prices.
    Open, High, Low are set equal to Close for simplicity.
    """
    dates = pd.date_range("2024-01-01", periods=len(prices), freq="B")
    return pd.DataFrame(
        {
            "Open":   prices,
            "High":   prices,
            "Low":    prices,
            "Close":  prices,
            "Volume": 1_000_000,
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# Test: validate_config — valid configurations
# ---------------------------------------------------------------------------

class TestValidateConfigValid:

    def test_default_config_is_valid(self, default_strategy):
        assert default_strategy.validate_config() == []

    def test_custom_valid_windows(self):
        s = MovingAverageCrossover(short_window=5, long_window=20)
        assert s.validate_config() == []

    def test_open_price_column(self):
        s = MovingAverageCrossover(price_column="Open")
        assert s.validate_config() == []

    def test_high_price_column(self):
        s = MovingAverageCrossover(price_column="High")
        assert s.validate_config() == []


# ---------------------------------------------------------------------------
# Test: validate_config — invalid configurations
# ---------------------------------------------------------------------------

class TestValidateConfigInvalid:

    def test_short_window_zero(self):
        errors = MovingAverageCrossover(short_window=0, long_window=50).validate_config()
        assert any("short_window" in e for e in errors)

    def test_short_window_negative(self):
        errors = MovingAverageCrossover(short_window=-5, long_window=50).validate_config()
        assert any("short_window" in e for e in errors)

    def test_long_window_zero(self):
        errors = MovingAverageCrossover(short_window=5, long_window=0).validate_config()
        assert any("long_window" in e for e in errors)

    def test_short_equals_long(self):
        errors = MovingAverageCrossover(short_window=20, long_window=20).validate_config()
        assert any("short_window" in e and "long_window" in e for e in errors)

    def test_short_greater_than_long(self):
        errors = MovingAverageCrossover(short_window=50, long_window=20).validate_config()
        assert any("short_window" in e and "long_window" in e for e in errors)

    def test_invalid_price_column(self):
        errors = MovingAverageCrossover(price_column="Adj Close").validate_config()
        assert any("price_column" in e for e in errors)

    def test_multiple_errors_collected(self):
        # Both short >= long AND invalid price column.
        errors = MovingAverageCrossover(
            short_window=50, long_window=20, price_column="Volume"
        ).validate_config()
        assert len(errors) >= 2


# ---------------------------------------------------------------------------
# Test: strategy metadata
# ---------------------------------------------------------------------------

class TestStrategyMetadata:

    def test_name_contains_windows(self, default_strategy):
        assert "20" in default_strategy.name
        assert "50" in default_strategy.name

    def test_min_required_rows(self, default_strategy):
        # long_window (50) + 1 for the shift = 51
        assert default_strategy.min_required_rows == 51

    def test_min_required_rows_small(self, small_strategy):
        # long_window (5) + 1 for the shift = 6
        assert small_strategy.min_required_rows == 6


# ---------------------------------------------------------------------------
# Test: generate_signals raises on bad inputs
# ---------------------------------------------------------------------------

class TestGenerateSignalsErrors:

    def test_raises_on_invalid_config(self):
        bad = MovingAverageCrossover(short_window=50, long_window=20)
        df = make_df([100.0] * 100)
        with pytest.raises(ValueError, match="configuration is invalid"):
            bad.generate_signals(df)

    def test_raises_when_too_few_rows(self, small_strategy):
        # small_strategy needs 6 rows; give it 5.
        df = make_df([100.0] * 5)
        with pytest.raises(ValueError, match="requires at least"):
            small_strategy.generate_signals(df)

    def test_raises_on_missing_price_column(self, small_strategy):
        df = make_df([100.0] * 20)
        df = df.drop(columns=["Close"])
        with pytest.raises(KeyError):
            small_strategy.generate_signals(df)


# ---------------------------------------------------------------------------
# Test: signal shape and value constraints
# ---------------------------------------------------------------------------

class TestSignalShapeAndValues:

    def test_signal_length_matches_df(self, small_strategy):
        df = make_df([100.0] * 20)
        signal = small_strategy.generate_signals(df)
        assert len(signal) == len(df)

    def test_signal_index_matches_df(self, small_strategy):
        df = make_df([100.0] * 20)
        signal = small_strategy.generate_signals(df)
        assert signal.index.equals(df.index)

    def test_signal_values_are_in_valid_set(self, small_strategy):
        df = make_df([float(100 + i % 7) for i in range(30)])
        signal = small_strategy.generate_signals(df)
        assert set(signal.unique()).issubset({-1.0, 0.0, 1.0})

    def test_signal_name_is_signal(self, small_strategy):
        df = make_df([100.0] * 20)
        signal = small_strategy.generate_signals(df)
        assert signal.name == "signal"

    def test_signal_has_no_nan(self, small_strategy):
        df = make_df([100.0] * 20)
        signal = small_strategy.generate_signals(df)
        assert signal.isna().sum() == 0


# ---------------------------------------------------------------------------
# Test: look-ahead bias prevention (the shift rule)
# ---------------------------------------------------------------------------

class TestLookAheadBiasPrevention:
    """
    These tests verify that the signal at row T uses MA values from row T-1,
    not row T. This is the core correctness guarantee of the strategy.

    We do this by constructing a price series where the MA crossover happens
    at a known row, and then checking that the signal switches one row LATER.
    """

    def test_first_rows_are_zero(self, small_strategy):
        # Before the long MA has enough data (first long_window rows),
        # signals must be 0. After the shift, the first non-zero signal
        # appears at row long_window + 1.
        df = make_df([100.0] * 30)
        signal = small_strategy.generate_signals(df)
        # Rows 0 through long_window (inclusive) should be 0.
        # long_window = 5, so indices 0..5 should be 0.
        assert (signal.iloc[: small_strategy.long_window + 1] == 0).all()

    def test_signal_at_row_T_reflects_mas_from_T_minus_1(self):
        """
        Use short=2, long=3 with a manually computable price series.

        Prices:    [10, 10, 10, 20, 20, 20, 20]
        short MA:  [ -,  10,  10,  15,  20,  20,  20]  (window=2)
        long MA:   [ -,   -,  10, 13.3, 16.7, 20,  20]  (window=3)
        raw signal: 0, 0, 0, 1, 1, 1, 1   (short > long from index 3)
        after shift(1): 0, 0, 0, 0, 1, 1, 1

        So the signal at row 4 (index 4) should be 1, not row 3.
        """
        strategy = MovingAverageCrossover(short_window=2, long_window=3)
        prices = [10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 20.0]
        df = make_df(prices)
        signal = strategy.generate_signals(df)

        # Row 3: raw MA comparison is first 1, but shift means signal is still 0
        assert signal.iloc[3] == 0.0, (
            "Signal at row 3 should be 0 — the crossover happened at row 3's "
            "close, so we can only act at row 4's open."
        )
        # Row 4: signal should now be 1
        assert signal.iloc[4] == 1.0, (
            "Signal at row 4 should be 1 — this is when we can trade on "
            "the crossover that happened at row 3's close."
        )


# ---------------------------------------------------------------------------
# Test: crossover direction
# ---------------------------------------------------------------------------

class TestCrossoverDirection:

    def _make_bullish_then_bearish(self) -> pd.DataFrame:
        """
        Returns a price series that rises then falls so that:
          - Short MA crosses above long MA early on (bullish)
          - Short MA crosses below long MA later (bearish)
        """
        rising = [float(100 + i * 3) for i in range(20)]
        falling = [float(160 - i * 3) for i in range(20)]
        return make_df(rising + falling)

    def test_bullish_signal_is_positive(self):
        strategy = MovingAverageCrossover(short_window=5, long_window=10)
        df = self._make_bullish_then_bearish()
        signal = strategy.generate_signals(df)
        # Somewhere in the rising phase there should be a +1 signal.
        assert (signal > 0).any(), "Expected at least one bullish (+1) signal."

    def test_bearish_signal_is_negative(self):
        strategy = MovingAverageCrossover(short_window=5, long_window=10)
        df = self._make_bullish_then_bearish()
        signal = strategy.generate_signals(df)
        # Somewhere in the falling phase there should be a -1 signal.
        assert (signal < 0).any(), "Expected at least one bearish (-1) signal."

    def test_flat_prices_produce_zero_signal(self, small_strategy):
        # When prices are perfectly flat, both MAs are equal — signal is 0.
        df = make_df([100.0] * 30)
        signal = small_strategy.generate_signals(df)
        # After the warm-up period, all signals should be 0.
        assert (signal.iloc[small_strategy.long_window + 1 :] == 0.0).all()

    def test_exact_minimum_rows(self, small_strategy):
        # Exactly min_required_rows rows should not raise.
        df = make_df([100.0] * small_strategy.min_required_rows)
        signal = small_strategy.generate_signals(df)
        assert len(signal) == small_strategy.min_required_rows
