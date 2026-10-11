"""
tests/test_backtest_and_metrics.py
----------------------------------
Unit and regression tests for BacktestEngine, PerformanceMetrics, and ReportGenerator.
"""

import pandas as pd
import pytest

from backtesting.engine import BacktestEngine, BacktestResult
from inspection.ast_inspector import inspect_strategy_source
from metrics.performance import PerformanceMetrics, calculate_metrics
from reporting.report_generator import generate_report
from strategies.moving_average import MovingAverageCrossover
from validate_ohlcv import validate_ohlcv


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Generate 60 days of prices suitable for testing."""
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    # Up-trend then down-trend
    prices = [100.0 + i for i in range(30)] + [130.0 - i for i in range(30)]
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


class TestBacktestEngine:

    def test_engine_initialization_defaults(self):
        engine = BacktestEngine()
        assert engine.initial_capital == 10_000.0
        assert engine.fee_pct == 0.001
        assert engine.slippage_pct == 0.0005

    def test_engine_invalid_inputs(self):
        with pytest.raises(ValueError):
            BacktestEngine(initial_capital=-500)
        with pytest.raises(ValueError):
            BacktestEngine(fee_pct=-0.01)

    def test_backtest_run_execution(self, sample_ohlcv):
        strategy = MovingAverageCrossover(short_window=5, long_window=15)
        engine = BacktestEngine(initial_capital=10_000.0, fee_pct=0.001)
        res = engine.run(sample_ohlcv, strategy)

        assert isinstance(res, BacktestResult)
        assert len(res.equity_curve) == len(sample_ohlcv)
        assert res.final_capital > 0
        assert len(res.trades) > 0
        assert res.total_fees_paid >= 0

    def test_zero_trades_on_flat_signal(self, sample_ohlcv):
        class FlatStrategy(MovingAverageCrossover):
            def generate_signals(self, df):
                return pd.Series(0.0, index=df.index, name="signal")

        engine = BacktestEngine()
        res = engine.run(sample_ohlcv, FlatStrategy())
        assert len(res.trades) == 0
        assert res.final_capital == 10_000.0
        assert (res.equity_curve == 10_000.0).all()


class TestPerformanceMetrics:

    def test_calculate_metrics_normal(self, sample_ohlcv):
        strategy = MovingAverageCrossover(short_window=5, long_window=15)
        engine = BacktestEngine()
        res = engine.run(sample_ohlcv, strategy)
        metrics = calculate_metrics(res)

        assert isinstance(metrics, PerformanceMetrics)
        assert isinstance(metrics.total_return_pct, float)
        assert isinstance(metrics.sharpe_ratio, float)
        assert metrics.observations == 60
        assert 0 <= metrics.win_rate_pct <= 100.0

    def test_short_series_metrics(self):
        # Empty or single-bar equity curve
        empty_res = BacktestResult(
            equity_curve=pd.Series([1000.0]),
            initial_capital=1000.0,
            final_capital=1000.0,
        )
        m = calculate_metrics(empty_res)
        assert m.total_return_pct == 0.0
        assert m.sharpe_ratio == 0.0


class TestReporting:

    def test_generate_report_flow(self, sample_ohlcv):
        data_res = validate_ohlcv(sample_ohlcv)
        code_res = inspect_strategy_source("x = df['Close'].rolling(10).mean()")
        strategy = MovingAverageCrossover(short_window=5, long_window=15)
        engine = BacktestEngine()
        backtest_res = engine.run(sample_ohlcv, strategy)
        metrics = calculate_metrics(backtest_res)

        report = generate_report(data_res, code_res, metrics)
        assert report.dataset_passed is True
        assert report.has_lookahead_risk is False
        assert report.backtest_executed is True
        assert report.metrics is not None

        # Check export
        json_str = report.to_json()
        assert "dataset_passed" in json_str
        md_str = report.summary_markdown()
        assert "BacktestGuard AI Validation Report" in md_str
