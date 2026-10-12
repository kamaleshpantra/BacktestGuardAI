"""
api/server.py
-------------
FastAPI service exposing validation and backtesting workflows.
"""

from __future__ import annotations

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from backtesting.engine import BacktestEngine
from inspection.ast_inspector import inspect_strategy_source
from metrics.performance import calculate_metrics
from reporting.report_generator import generate_report
from storage.db import RunDatabase
from strategies.moving_average import MovingAverageCrossover
from validate_ohlcv import validate_ohlcv

app = FastAPI(
    title="BacktestGuard AI API",
    description="Deterministic validation & backtesting API for quantitative trading strategies",
    version="0.1.0",
)

db = RunDatabase()


class CodeInspectRequest(BaseModel):
    source_code: str


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "BacktestGuard AI"}


@app.post("/inspect/code")
def inspect_code(payload: CodeInspectRequest):
    """Run AST static inspection on provided strategy Python code."""
    res = inspect_strategy_source(payload.source_code)
    return {
        "is_parseable": res.is_parseable,
        "has_confirmed_violations": res.has_confirmed_violations,
        "findings": [
            {
                "code": f.code,
                "certainty": f.certainty,
                "line": f.line_number,
                "description": f.description,
                "evidence": f.evidence,
                "recommendation": f.recommendation,
            }
            for f in res.findings
        ],
    }


@app.post("/validate/dataset")
async def validate_dataset(file: UploadFile = File(...)):
    """Validate uploaded CSV file against OHLCV rules."""
    try:
        contents = await file.read()
        from data.loader import load_csv
        df = load_csv(contents)
        res = validate_ohlcv(df)
        return {
            "passed": res.passed,
            "row_count": res.row_count,
            "errors": [f.__dict__ for f in res.errors],
            "warnings": [f.__dict__ for f in res.warnings],
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to process CSV: {str(exc)}")


@app.get("/history")
def get_history(limit: int = 20):
    """Retrieve historical validation logs."""
    return db.list_runs(limit=limit)


class WebhookAuditRequest(BaseModel):
    ticker: str = "AAPL"
    strategy_code: str | None = None
    strategy_type: str = "ma"  # 'ma' or 'rsi'
    bot_token: str | None = None
    chat_id: str | None = None


@app.post("/audit/webhook-trigger")
async def webhook_audit_trigger(payload: WebhookAuditRequest):
    """
    Automated webhook endpoint callable by background schedulers, cron, or external webhooks.
    Audits latest price data, runs backtest, and optionally dispatches Telegram alert.
    """
    import numpy as np

    # Generate benchmark prices for monitored ticker
    bars = 100
    dates = pd.date_range("2024-01-01", periods=bars, freq="B")
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.015, size=bars)
    price_path = 150.0 * np.cumprod(1.0 + returns)
    df = pd.DataFrame(
        {
            "Open": price_path * (1 - 0.002),
            "High": price_path * (1 + 0.005),
            "Low": price_path * (1 - 0.005),
            "Close": price_path,
            "Volume": 1_000_000,
        },
        index=dates,
    )

    data_res = validate_ohlcv(df)

    # Strategy selection
    if payload.strategy_code:
        from strategies.custom_loader import SafeCustomStrategy
        strat = SafeCustomStrategy(payload.strategy_code)
    elif payload.strategy_type == "rsi":
        from strategies.rsi import RSIMeanReversion
        strat = RSIMeanReversion()
    else:
        strat = MovingAverageCrossover()

    engine = BacktestEngine()
    bt_res = engine.run(df, strat)
    metrics = calculate_metrics(bt_res)
    report = generate_report(data_result=data_res, metrics=metrics)
    run_id = db.save_run(report.to_dict())

    # Optional Telegram notification
    telegram_sent = False
    if payload.bot_token and payload.chat_id:
        from notifications.telegram import TelegramNotifier
        notifier = TelegramNotifier(payload.bot_token, payload.chat_id)
        ok, _ = notifier.send_audit_alert(report, ticker=payload.ticker)
        telegram_sent = ok

    return {
        "status": "success",
        "run_id": run_id,
        "dataset_passed": data_res.passed,
        "metrics": metrics.to_dict(),
        "telegram_sent": telegram_sent,
    }

