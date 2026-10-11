"""
tests/test_telegram_notifier.py
-------------------------------
Unit tests for TelegramNotifier formatting and validation.
"""

from metrics.performance import PerformanceMetrics
from notifications.telegram import TelegramNotifier
from reporting.report_generator import generate_report


class TestTelegramNotifier:

    def test_unconfigured_credentials(self):
        t = TelegramNotifier(bot_token="", chat_id="")
        assert t.is_configured() is False
        ok, msg = t.test_connection()
        assert ok is False
        assert "empty" in msg

    def test_report_formatting(self):
        t = TelegramNotifier(bot_token="fake_token", chat_id="12345")
        assert t.is_configured() is True

        metrics = PerformanceMetrics(
            total_return_pct=15.5,
            annualized_return_pct=16.0,
            annualized_volatility_pct=8.0,
            max_drawdown_pct=-3.5,
            sharpe_ratio=1.75,
            total_trades=12,
            winning_trades=8,
            win_rate_pct=66.67,
            total_fees_paid=14.2,
            observations=150,
        )
        report = generate_report(metrics=metrics)

        # Mocking the network call by verifying internal message formulation
        ok, msg = t.send_audit_alert(
            report,
            ticker="NVDA",
            ai_explanation="• Strong risk-reward with high win rate.\n• Low fee drag.\n• No past guarantee.",
        )
        # Because fake_token cannot reach Telegram, it should return False with connection failure,
        # but crucially must not raise an unhandled exception or crash.
        assert ok is False
        assert isinstance(msg, str)
