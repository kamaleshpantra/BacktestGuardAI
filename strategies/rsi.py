"""
strategies/rsi.py
-----------------
Relative Strength Index (RSI) Mean-Reversion Strategy.

Concept
-------
RSI is an oscillator that measures the speed and change of price movements
on a scale from 0 to 100.
  - RSI < lower_bound (typically 30): The asset is "oversold" (cheap). Mean-reversion BUY signal.
  - RSI > upper_bound (typically 70): The asset is "overbought" (expensive). Exit to cash (SELL signal).

Execution Timing & Look-Ahead Prevention
----------------------------------------
Like all BacktestGuard strategies, signals calculated at the close of bar T
are shifted by 1 bar (.shift(1)) so trades execute strictly at bar T+1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.base import Strategy


class RSIMeanReversion(Strategy):
    """
    RSI-based mean-reversion strategy.

    Parameters
    ----------
    period : int
        RSI lookback period (default: 14 bars).
    oversold_threshold : float
        RSI level below which to buy (default: 30.0).
    overbought_threshold : float
        RSI level above which to exit/sell (default: 70.0).
    price_column : str
        OHLCV column to compute RSI on (default: 'Close').
    """

    def __init__(
        self,
        period: int = 14,
        oversold_threshold: float = 30.0,
        overbought_threshold: float = 70.0,
        price_column: str = "Close",
    ) -> None:
        self.period = period
        self.oversold_threshold = oversold_threshold
        self.overbought_threshold = overbought_threshold
        self.price_column = price_column

    @property
    def name(self) -> str:
        return f"RSI_MeanReversion_{self.period}_{int(self.oversold_threshold)}_{int(self.overbought_threshold)}"

    @property
    def min_required_rows(self) -> int:
        # Needs at least period + 2 rows to compute price change, rolling average, and 1-bar lag
        return self.period + 2

    def validate_config(self) -> list[str]:
        errors: list[str] = []

        if not isinstance(self.period, int) or self.period <= 1:
            errors.append(f"period must be an integer > 1, got {self.period!r}")

        if not (0.0 < self.oversold_threshold < 100.0):
            errors.append(
                f"oversold_threshold must be between 0 and 100, got {self.oversold_threshold!r}"
            )

        if not (0.0 < self.overbought_threshold < 100.0):
            errors.append(
                f"overbought_threshold must be between 0 and 100, got {self.overbought_threshold!r}"
            )

        if self.oversold_threshold >= self.overbought_threshold:
            errors.append(
                f"oversold_threshold ({self.oversold_threshold}) must be strictly less than "
                f"overbought_threshold ({self.overbought_threshold})"
            )

        if self.price_column not in ("Open", "High", "Low", "Close"):
            errors.append(f"Invalid price_column: {self.price_column!r}")

        return errors

    def _calculate_rsi(self, series: pd.Series) -> pd.Series:
        """Standard Wilder's / Exponential RSI formula."""
        delta = series.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)

        # Exponential moving average for RSI smoothing
        avg_gain = gain.ewm(alpha=1.0 / self.period, min_periods=self.period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / self.period, min_periods=self.period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        # If avg_loss is 0, RSI is 100; if avg_gain is 0, RSI is 0
        rsi = rsi.fillna(100.0 * (avg_gain > 0))
        return rsi

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        errors = self.validate_config()
        if errors:
            raise ValueError(f"Invalid RSI config: {errors}")

        if len(df) < self.min_required_rows:
            raise ValueError(
                f"{self.name} requires at least {self.min_required_rows} rows, got {len(df)}"
            )

        if self.price_column not in df.columns:
            raise KeyError(f"Price column '{self.price_column}' not found in DataFrame.")

        prices = df[self.price_column]
        rsi = self._calculate_rsi(prices)

        # Raw signals: 1 when oversold (buy), 0 when overbought (sell/flat), NaN in-between (hold position)
        raw_signal = pd.Series(index=df.index, dtype=float)
        raw_signal.loc[rsi < self.oversold_threshold] = 1.0
        raw_signal.loc[rsi > self.overbought_threshold] = 0.0

        # Forward fill so position is held until an opposite signal triggers
        stateful_signal = raw_signal.ffill().fillna(0.0)

        # Shift by 1 bar to strictly eliminate look-ahead bias
        final_signal = stateful_signal.shift(1).fillna(0.0)
        final_signal.name = "signal"
        return final_signal
