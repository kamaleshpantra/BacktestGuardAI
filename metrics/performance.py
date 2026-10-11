"""
metrics/performance.py
-----------------------
Performance evaluation calculations for backtest results.

Calculates:
  - Total Return
  - Annualized Return (CAGR) with explicit day assumptions (default: 252 trading days)
  - Annualized Volatility
  - Maximum Drawdown (MDD)
  - Sharpe Ratio (with explicit risk-free rate)
  - Win Rate & Number of Trades
  - Total Transaction Fees
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from backtesting.engine import BacktestResult


@dataclass
class PerformanceMetrics:
    """
    Summary metrics computed from a backtest run.
    """
    total_return_pct: float
    annualized_return_pct: float
    annualized_volatility_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    total_trades: int
    winning_trades: int
    win_rate_pct: float
    total_fees_paid: float
    observations: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "total_return_pct": self.total_return_pct,
            "annualized_return_pct": self.annualized_return_pct,
            "annualized_volatility_pct": self.annualized_volatility_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "win_rate_pct": self.win_rate_pct,
            "total_fees_paid": self.total_fees_paid,
            "observations": self.observations,
        }


def calculate_metrics(
    result: BacktestResult,
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252,
) -> PerformanceMetrics:
    """
    Computes performance metrics from BacktestResult.

    Parameters
    ----------
    result : BacktestResult
        Output from BacktestEngine.run().
    risk_free_rate : float
        Annual risk-free rate assumption (e.g. 0.04 for 4%). Default 0.0.
    annualization_factor : int
        Number of trading bars per year (252 for daily trading days).

    Returns
    -------
    PerformanceMetrics
    """
    equity = result.equity_curve
    n_bars = len(equity)

    if n_bars <= 1:
        return PerformanceMetrics(
            total_return_pct=0.0,
            annualized_return_pct=0.0,
            annualized_volatility_pct=0.0,
            max_drawdown_pct=0.0,
            sharpe_ratio=0.0,
            total_trades=len(result.trades),
            winning_trades=0,
            win_rate_pct=0.0,
            total_fees_paid=result.total_fees_paid,
            observations=n_bars,
        )

    # 1. Total Return
    initial = result.initial_capital
    final = result.final_capital
    total_return = (final - initial) / initial

    # 2. Annualized Return (CAGR)
    # (1 + total_return) ^ (factor / n_bars) - 1
    # Only compute if final capital > 0
    if final > 0 and n_bars > 0:
        ann_return = ((final / initial) ** (annualization_factor / n_bars)) - 1.0
    else:
        ann_return = -1.0

    # 3. Daily / Periodic Returns & Volatility
    bar_returns = equity.pct_change().dropna()
    if len(bar_returns) > 1 and bar_returns.std() > 0:
        ann_vol = bar_returns.std() * math.sqrt(annualization_factor)
        # 4. Sharpe Ratio
        rf_per_bar = risk_free_rate / annualization_factor
        excess_returns = bar_returns - rf_per_bar
        sharpe = (excess_returns.mean() / bar_returns.std()) * math.sqrt(annualization_factor)
    else:
        ann_vol = 0.0
        sharpe = 0.0

    # 5. Maximum Drawdown (Peak to trough)
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_drawdown = float(drawdown.min())  # negative or 0.0

    # 6. Trade Analysis
    # Group trades into round trips (BUY followed by SELL)
    trades = result.trades
    total_trades = len(trades)
    winning_trades = 0
    completed_pairs = 0

    # Compute realized profit on matched buy-sell
    for i in range(0, len(trades) - 1, 2):
        if trades[i].action == "BUY" and trades[i + 1].action == "SELL":
            buy_t = trades[i]
            sell_t = trades[i + 1]
            net_profit = (sell_t.cost - sell_t.fee) - (buy_t.cost + buy_t.fee)
            completed_pairs += 1
            if net_profit > 0:
                winning_trades += 1

    win_rate = (winning_trades / completed_pairs * 100.0) if completed_pairs > 0 else 0.0

    return PerformanceMetrics(
        total_return_pct=round(total_return * 100.0, 4),
        annualized_return_pct=round(ann_return * 100.0, 4),
        annualized_volatility_pct=round(ann_vol * 100.0, 4),
        max_drawdown_pct=round(max_drawdown * 100.0, 4),
        sharpe_ratio=round(float(sharpe), 4),
        total_trades=total_trades,
        winning_trades=winning_trades,
        win_rate_pct=round(win_rate, 2),
        total_fees_paid=round(result.total_fees_paid, 4),
        observations=n_bars,
    )
