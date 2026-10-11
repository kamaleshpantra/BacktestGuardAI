"""
strategies/moving_average.py
-----------------------------
A simple moving-average crossover strategy.

How it works
------------
Two rolling averages of the closing price are calculated:
  - Short moving average (e.g. 20 days)
  - Long  moving average (e.g. 50 days)

When the short MA rises above the long MA, it suggests the recent price
trend is stronger than the longer-term trend — a bullish signal.
When the short MA falls below the long MA, it suggests weakening — bearish.

This is one of the oldest and most studied technical indicators. It is used
here as the reference strategy because:
  - It is simple enough to verify by hand
  - It requires a configurable minimum data length (long_window rows)
  - It has a natural shift(1) requirement that makes look-ahead bias
    easy to demonstrate and teach

Look-ahead bias prevention
--------------------------
The raw MA comparison at day T uses today's closing price.
If we acted on that signal TODAY, we would be trading using today's
close — a price we cannot know until the market closes.

We prevent this by calling .shift(1) on the raw signal Series.
The signal at row T now reflects the MA comparison from row T-1.
This means: "at the open of tomorrow, take the position that yesterday's
moving averages suggested."

This is a deliberate simplification. Real execution happens at various
prices (VWAP, next open, etc.). The assumption here is documented and
consistent, which is what matters for a reliable backtest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.base import Strategy

# Columns this strategy can produce signals from.
SUPPORTED_PRICE_COLUMNS = ("Open", "High", "Low", "Close")


class MovingAverageCrossover(Strategy):
    """
    Long/flat strategy based on a short-over-long moving average crossover.

    Parameters
    ----------
    short_window : int
        Number of periods for the short moving average. Default 20.
    long_window : int
        Number of periods for the long moving average. Default 50.
    price_column : str
        Which OHLCV column to compute averages on. Default 'Close'.

    Signals
    -------
    short_ma > long_ma  →  1  (long)
    short_ma < long_ma  → -1  (short / out of market — see note)
    short_ma == long_ma →  0  (flat)

    Note: short-selling (-1) is included in signal generation but the
    backtesting engine in Phase 4 will only support long and flat positions
    initially. The -1 signal will be treated as flat (0) until short-selling
    is explicitly enabled.
    """

    def __init__(
        self,
        short_window: int = 20,
        long_window: int = 50,
        price_column: str = "Close",
    ) -> None:
        self.short_window = short_window
        self.long_window = long_window
        self.price_column = price_column

    @property
    def name(self) -> str:
        return f"MA_Crossover_{self.short_window}_{self.long_window}"

    @property
    def min_required_rows(self) -> int:
        # We need long_window rows to compute the first valid long MA,
        # plus 1 more row because the signal is shifted by one period.
        return self.long_window + 1

    def validate_config(self) -> list[str]:
        """
        Check that the window sizes and price column make sense.

        Returns
        -------
        list[str]
            Error descriptions, or an empty list if config is valid.
        """
        errors: list[str] = []

        if not isinstance(self.short_window, int) or self.short_window <= 0:
            errors.append(
                f"short_window must be a positive integer, got {self.short_window!r}"
            )

        if not isinstance(self.long_window, int) or self.long_window <= 0:
            errors.append(
                f"long_window must be a positive integer, got {self.long_window!r}"
            )

        # Only compare windows if both are valid ints
        if (
            isinstance(self.short_window, int)
            and isinstance(self.long_window, int)
            and self.short_window > 0
            and self.long_window > 0
            and self.short_window >= self.long_window
        ):
            errors.append(
                f"short_window ({self.short_window}) must be strictly less than "
                f"long_window ({self.long_window}). "
                "A crossover requires two different time horizons."
            )

        if self.price_column not in SUPPORTED_PRICE_COLUMNS:
            errors.append(
                f"price_column must be one of {SUPPORTED_PRICE_COLUMNS}, "
                f"got {self.price_column!r}"
            )

        return errors

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute a signal Series from the OHLCV DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Validated OHLCV DataFrame with a DatetimeIndex.
            Must have at least min_required_rows rows.

        Returns
        -------
        pd.Series
            Signal Series aligned to df.index, values in {-1, 0, 1}.
            Named 'signal'.

        Raises
        ------
        ValueError
            If the strategy configuration is invalid or the DataFrame has
            fewer rows than min_required_rows.
        KeyError
            If price_column is not present in df.
        """
        config_errors = self.validate_config()
        if config_errors:
            raise ValueError(
                f"Strategy configuration is invalid: {config_errors}"
            )

        if len(df) < self.min_required_rows:
            raise ValueError(
                f"{self.name} requires at least {self.min_required_rows} rows "
                f"(long_window={self.long_window} + 1 for the signal shift), "
                f"but the DataFrame has only {len(df)} rows."
            )

        if self.price_column not in df.columns:
            raise KeyError(
                f"price_column '{self.price_column}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )

        prices = df[self.price_column]

        # Step 1: compute rolling means.
        # rolling(N).mean() at position T uses rows [T-N+1, T].
        # The first N-1 values are NaN because there is not enough history.
        short_ma = prices.rolling(window=self.short_window).mean()
        long_ma = prices.rolling(window=self.long_window).mean()

        # Step 2: generate raw signal by comparing the two averages.
        # np.select is used to handle the three cases cleanly.
        conditions = [short_ma > long_ma, short_ma < long_ma]
        choices = [1.0, -1.0]
        raw = pd.Series(
            np.select(conditions, choices, default=0.0),
            index=df.index,
            dtype=float,
        )

        # Step 3: shift by 1 to prevent look-ahead bias.
        # The signal at row T now uses the MA values from row T-1.
        # Rows where the long MA had not yet formed (NaN) become 0 after fillna.
        signal = raw.shift(1).fillna(0.0)
        signal.name = "signal"

        return signal
