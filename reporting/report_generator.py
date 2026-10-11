"""
reporting/report_generator.py
-----------------------------
Generates unified, structured validation and backtest reports.

Aggregates:
  - Dataset validation findings (OHLCV checks)
  - Code inspection findings (AST checks)
  - Backtest simulation outcomes and trade counts
  - Quant performance metrics
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime

from inspection.ast_inspector import InspectionResult
from metrics.performance import PerformanceMetrics
from validate_ohlcv import OHLCVValidationResult


@dataclass
class ValidationReport:
    """
    Consolidated machine- and human-readable validation report.
    """
    timestamp: str
    dataset_passed: bool
    data_findings: list[dict]
    code_findings: list[dict]
    has_lookahead_risk: bool
    backtest_executed: bool
    metrics: dict | None = None
    execution_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "dataset_passed": self.dataset_passed,
            "data_findings": self.data_findings,
            "code_findings": self.code_findings,
            "has_lookahead_risk": self.has_lookahead_risk,
            "backtest_executed": self.backtest_executed,
            "metrics": self.metrics,
            "execution_notes": self.execution_notes,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def summary_markdown(self) -> str:
        """Render a readable summary for UI or CLI export."""
        lines = [
            "# BacktestGuard AI Validation Report",
            f"**Run Timestamp:** {self.timestamp}",
            f"**Data Validation:** {'✅ Passed' if self.dataset_passed else '❌ Failed'}",
            f"**Look-Ahead Risk Detected:** {'⚠️ YES' if self.has_lookahead_risk else '✅ NO'}",
            "",
            "## Data Validation Findings",
        ]
        if not self.data_findings:
            lines.append("No data anomalies identified.")
        else:
            for f in self.data_findings:
                lines.append(f"- **[{f['severity'].upper()}] {f['code']}**: {f['description']}")
                lines.append(f"  *Evidence:* `{f['evidence']}`")

        lines.extend(["", "## Strategy Code Findings"])
        if not self.code_findings:
            lines.append("No suspicious code patterns detected.")
        else:
            for f in self.code_findings:
                lines.append(f"- **[{f['certainty'].upper()}] {f['code']}** (Line {f['line_number']}): {f['description']}")
                lines.append(f"  *Evidence:* `{f['evidence']}`")

        lines.extend(["", "## Performance Summary"])
        if self.metrics:
            lines.append(f"- **Total Return:** {self.metrics.get('total_return_pct')}%")
            lines.append(f"- **Annualized Return:** {self.metrics.get('annualized_return_pct')}%")
            lines.append(f"- **Max Drawdown:** {self.metrics.get('max_drawdown_pct')}%")
            lines.append(f"- **Sharpe Ratio:** {self.metrics.get('sharpe_ratio')}")
            lines.append(f"- **Total Trades:** {self.metrics.get('total_trades')}")
            lines.append(f"- **Win Rate:** {self.metrics.get('win_rate_pct')}%")
            lines.append(f"- **Total Fees Paid:** ${self.metrics.get('total_fees_paid')}")
        else:
            lines.append("Backtest simulation was not executed due to data or code validation failures.")

        return "\n".join(lines)


def generate_report(
    data_result: OHLCVValidationResult | None = None,
    code_result: InspectionResult | None = None,
    metrics: PerformanceMetrics | None = None,
    notes: list[str] | None = None,
) -> ValidationReport:
    """
    Builds a unified ValidationReport.
    """
    data_findings = []
    dataset_passed = True
    if data_result:
        dataset_passed = data_result.passed
        for f in data_result.findings:
            data_findings.append({
                "code": f.code,
                "severity": f.severity,
                "description": f.description,
                "evidence": f.evidence,
                "recommendation": f.recommendation,
            })

    code_findings = []
    has_lookahead = False
    if code_result:
        has_lookahead = code_result.has_confirmed_violations
        for f in code_result.findings:
            code_findings.append({
                "code": f.code,
                "certainty": f.certainty,
                "description": f.description,
                "evidence": f.evidence,
                "line_number": f.line_number,
                "recommendation": f.recommendation,
            })

    return ValidationReport(
        timestamp=datetime.now().isoformat(),
        dataset_passed=dataset_passed,
        data_findings=data_findings,
        code_findings=code_findings,
        has_lookahead_risk=has_lookahead,
        backtest_executed=(metrics is not None),
        metrics=metrics.to_dict() if metrics else None,
        execution_notes=notes or [],
    )
