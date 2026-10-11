"""
backtesting/engine.py
---------------------
Controlled backtesting engine for BacktestGuard AI.

Design & Accounting Principles
-------------------------------
1. Cash & Asset Accounting:
   - Starts with `initial_capital`.
   - Portfolio value = Cash + (Shares * Current_Price).
2. Position States:
   - Currently supports Long (1) and Flat (0).
   - If a strategy emits -1 (short signal), it is treated as flat (0)
     to avoid unmodeled borrowing and margin constraints.
3. Realistic Execution & Look-Ahead Prevention:
   - Signals are generated at bar close (already lagged/shifted via strategy).
   - Execution price accounts for slippage:
       Buy execution price  = Price * (1 + slippage_pct)
       Sell execution price = Price * (1 - slippage_pct)
   - Transaction fee is charged as fixed fee + percentage fee on trade value.
4. Determinism & Reproducibility:
   - Strict bar-by-bar calculation or vectorization where equivalent.
   - Outputs a detailed ledger / equity curve.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from strategies.base import Strategy


@dataclass
class Trade:
    """Represents an executed trade."""
    timestamp: pd.Timestamp
    action: str          # 'BUY' or 'SELL'
    price: float         # Raw market price
    execution_price: float # Price after slippage
    shares: float
    cost: float          # Total trade value (shares * execution_price)
    fee: float           # Transaction fee paid


@dataclass
class BacktestResult:
    """
    Complete output of a backtest simulation.

    Attributes
    ----------
    equity_curve : pd.Series
        Portfolio value over time, indexed by timestamp.
    trades : list[Trade]
        Log of executed trades.
    initial_capital : float
        Starting balance.
    final_capital : float
        Ending portfolio value.
    total_fees_paid : float
        Cumulative transaction costs.
    positions : pd.Series
        Shares held at each point in time.
    signals : pd.Series
        Strategy signals used during execution.
    """
    equity_curve: pd.Series
    trades: list[Trade] = field(default_factory=list)
    initial_capital: float = 10_000.0
    final_capital: float = 10_000.0
    total_fees_paid: float = 0.0
    positions: pd.Series = field(default_factory=pd.Series)
    signals: pd.Series = field(default_factory=pd.Series)


class BacktestEngine:
    """
    Executes a strategy against validated OHLCV data.

    Parameters
    ----------
    initial_capital : float
        Starting capital in currency units. Default 10,000.
    fee_pct : float
        Brokerage commission as a percentage of trade value. Default 0.001 (0.1%).
    fixed_fee : float
        Fixed fee per trade order. Default 0.0.
    slippage_pct : float
        Estimated price slippage penalty per trade. Default 0.0005 (0.05%).
    execution_price_col : str
        Which bar price to execute at. Default 'Close'.
    """

    def __init__(
        self,
        initial_capital: float = 10_000.0,
        fee_pct: float = 0.001,
        fixed_fee: float = 0.0,
        slippage_pct: float = 0.0005,
        execution_price_col: str = "Close",
    ) -> None:
        if initial_capital <= 0:
            raise ValueError(f"initial_capital must be positive, got {initial_capital}")
        if fee_pct < 0:
            raise ValueError(f"fee_pct cannot be negative, got {fee_pct}")
        if fixed_fee < 0:
            raise ValueError(f"fixed_fee cannot be negative, got {fixed_fee}")
        if slippage_pct < 0:
            raise ValueError(f"slippage_pct cannot be negative, got {slippage_pct}")

        self.initial_capital = float(initial_capital)
        self.fee_pct = float(fee_pct)
        self.fixed_fee = float(fixed_fee)
        self.slippage_pct = float(slippage_pct)
        self.execution_price_col = execution_price_col

    def run(self, df: pd.DataFrame, strategy: Strategy) -> BacktestResult:
        """
        Run simulation for given strategy and dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Validated OHLCV DataFrame with DatetimeIndex.
        strategy : Strategy
            Configured strategy instance.

        Returns
        -------
        BacktestResult
        """
        if self.execution_price_col not in df.columns:
            raise KeyError(
                f"Execution price column '{self.execution_price_col}' missing from data."
            )

        signals = strategy.generate_signals(df)
        prices = df[self.execution_price_col]

        cash = self.initial_capital
        shares = 0.0
        total_fees = 0.0

        equity_list: list[float] = []
        positions_list: list[float] = []
        trades: list[Trade] = []

        # Iterate bar by bar to simulate realistic execution and order fills
        for ts, price in prices.items():
            sig = signals.loc[ts]

            # Desired position: long (1) -> 100% equity invested; flat (0 or -1) -> 0%
            target_state = 1 if sig == 1.0 else 0

            # State transition: Flat -> Long (BUY)
            if target_state == 1 and shares == 0.0:
                exec_price = price * (1.0 + self.slippage_pct)
                # Maximize whole/fractional shares after reserve for fees
                # cash_allocated = cash - fixed_fee
                # cash_allocated = shares * exec_price * (1 + fee_pct)
                available_cash = max(0.0, cash - self.fixed_fee)
                if available_cash > 0 and exec_price > 0:
                    trade_shares = available_cash / (exec_price * (1.0 + self.fee_pct))
                    trade_value = trade_shares * exec_price
                    fee = self.fixed_fee + (trade_value * self.fee_pct)

                    cost = trade_value + fee
                    if cost <= cash and trade_shares > 0:
                        shares = trade_shares
                        cash -= cost
                        total_fees += fee
                        trades.append(
                            Trade(
                                timestamp=ts,
                                action="BUY",
                                price=price,
                                execution_price=exec_price,
                                shares=trade_shares,
                                cost=trade_value,
                                fee=fee,
                            )
                        )

            # State transition: Long -> Flat (SELL)
            elif target_state == 0 and shares > 0.0:
                exec_price = price * (1.0 - self.slippage_pct)
                trade_value = shares * exec_price
                fee = self.fixed_fee + (trade_value * self.fee_pct)
                net_proceeds = trade_value - fee

                cash += net_proceeds
                total_fees += fee
                trades.append(
                    Trade(
                        timestamp=ts,
                        action="SELL",
                        price=price,
                        execution_price=exec_price,
                        shares=shares,
                        cost=trade_value,
                        fee=fee,
                    )
                )
                shares = 0.0

            current_portfolio_value = cash + (shares * price)
            equity_list.append(current_portfolio_value)
            positions_list.append(shares)

        equity_curve = pd.Series(equity_list, index=df.index, name="equity")
        positions_series = pd.Series(positions_list, index=df.index, name="positions")

        return BacktestResult(
            equity_curve=equity_curve,
            trades=trades,
            initial_capital=self.initial_capital,
            final_capital=equity_curve.iloc[-1] if len(equity_curve) > 0 else self.initial_capital,
            total_fees_paid=total_fees,
            positions=positions_series,
            signals=signals,
        )
