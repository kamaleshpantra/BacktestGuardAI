"""
strategies/base.py
------------------
Defines the Strategy abstract base class.

Every strategy in BacktestGuard must inherit from this class and implement
its two required methods: generate_signals() and validate_config().

Why use an abstract base class instead of just a plain class?
--------------------------------------------------------------
An abstract base class (ABC) enforces a contract. If you write a strategy
but forget to implement generate_signals(), Python will raise a clear error
the moment you try to instantiate it — not buried somewhere in the backtesting
engine after it's already running. This makes errors surface early and with
clear messages.

We use a class here (rather than a plain function) because a strategy
needs to carry configuration (window sizes, price column, etc.) alongside
the logic that uses that configuration. Grouping them together is exactly
what classes are for.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    """
    Base class for all BacktestGuard trading strategies.

    Subclasses must implement:
      - name         (property) — a short human-readable identifier
      - generate_signals(df)   — produces a signal Series from OHLCV data
      - validate_config()      — checks that the strategy parameters are sane

    The signal convention used throughout this project:
      1  — buy / hold long position
     -1  — sell / hold short position  (short-selling not supported yet)
      0  — no position / stay flat

    Execution timing convention
    ---------------------------
    Signals generated at the close of day T represent the position to take
    at the open of day T+1. This is enforced by shifting signals by one
    period inside generate_signals(). Any strategy that does not apply this
    shift uses today's closing price to decide today's trade, which is
    look-ahead bias.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short human-readable strategy name (e.g. 'MA_Crossover_20_50')."""

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute a trading signal for each row in the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            A validated OHLCV DataFrame with a DatetimeIndex.

        Returns
        -------
        pd.Series
            Integer-valued Series aligned to df.index.
            Values must be one of: 1 (long), -1 (short), 0 (flat).
            The Series name should be 'signal'.
        """

    @abstractmethod
    def validate_config(self) -> list[str]:
        """
        Check that the strategy's parameters are internally consistent.

        Returns
        -------
        list[str]
            Error strings for every configuration problem found.
            An empty list means the configuration is valid.
        """

    @property
    @abstractmethod
    def min_required_rows(self) -> int:
        """
        Minimum number of DataFrame rows needed to produce at least one
        non-zero signal. Used by the backtesting engine to reject datasets
        that are too short for this strategy.
        """
