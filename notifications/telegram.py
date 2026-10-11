"""
notifications/telegram.py
-------------------------
Direct Telegram Bot API notification dispatcher for BacktestGuard AI.
Uses httpx with no third-party Telegram dependencies.
"""

from __future__ import annotations

import httpx

from reporting.report_generator import ValidationReport


class TelegramNotifier:
    """Dispatches formatted audit reports and alerts to Telegram chats."""

    def __init__(self, bot_token: str, chat_id: str, timeout: float = 10.0) -> None:
        self.bot_token = bot_token.strip()
        self.chat_id = str(chat_id).strip()
        self.timeout = timeout
        self.api_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"

    def is_configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    def test_connection(self) -> tuple[bool, str]:
        """Send a test handshake message to verify token and chat_id."""
        if not self.is_configured():
            return False, "Bot token or Chat ID is empty."

        text = (
            "🛡️ *BacktestGuard AI — Connection Successful!*\n\n"
            "Your Telegram bot is successfully connected and ready to receive "
            "daily strategy audits and look-ahead alerts."
        )
        return self.send_message(text)

    def send_message(self, text: str) -> tuple[bool, str]:
        """Send raw text message with markdown formatting."""
        if not self.is_configured():
            return False, "Telegram credentials not configured."

        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(
                    self.api_url,
                    json={
                        "chat_id": self.chat_id,
                        "text": text,
                        "parse_mode": "Markdown",
                        "disable_web_page_preview": True,
                    },
                )
                if res.status_code == 200:
                    return True, "Notification sent successfully."
                else:
                    return False, f"Telegram API error {res.status_code}: {res.text}"
        except Exception as exc:
            return False, f"Failed to connect to Telegram: {str(exc)}"

    def send_audit_alert(
        self,
        report: ValidationReport,
        ticker: str = "ASSET",
        ai_explanation: str | None = None,
    ) -> tuple[bool, str]:
        """Format a full validation report + AI explanation into a clean mobile alert."""
        status_emoji = "✅ PASSED" if report.dataset_passed and not report.has_lookahead_risk else "🚨 ALERT / FAILED"

        lines = [
            "🛡️ *BacktestGuard AI — Audit Report*",
            f"*Asset:* `{ticker}` | *Status:* {status_emoji}",
            f"*Timestamp:* `{report.timestamp[:19]}`",
            "─────────────────────────",
        ]

        # Data & Code findings
        if not report.dataset_passed:
            lines.append("❌ *Data Quality Issues:*")
            for f in report.data_findings[:3]:
                lines.append(f"• `{f.get('code')}`: {f.get('description')}")
        else:
            lines.append("• Data Integrity: Clean (0 errors) ✅")

        if report.has_lookahead_risk:
            lines.append("🚨 *LOOK-AHEAD BIAS DETECTED IN CODE!*")
            for f in report.code_findings[:2]:
                lines.append(f"• `{f.get('code')}` (Line {f.get('line_number')})")
        else:
            lines.append("• Code Look-Ahead Risk: Clean ✅")

        # Performance summary
        if report.metrics:
            m = report.metrics
            lines.extend([
                "─────────────────────────",
                "📊 *Performance Snapshot:*",
                f"• Return: `{m.get('total_return_pct')}%`",
                f"• Max Drawdown: `{m.get('max_drawdown_pct')}%`",
                f"• Sharpe Ratio: `{m.get('sharpe_ratio')}`",
                f"• Win Rate: `{m.get('win_rate_pct')}%` ({m.get('total_trades')} trades)",
            ])

        # AI Mentor Explanation (Mandatory section)
        lines.extend([
            "─────────────────────────",
            "🤖 *Local/Cloud AI Mentor Analysis:*",
        ])
        if ai_explanation:
            lines.append(ai_explanation.strip())
        else:
            lines.append("_AI analysis pending or currently offline._")

        return self.send_message("\n".join(lines))
