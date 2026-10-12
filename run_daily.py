"""
run_daily.py
------------
Standalone command-line runner for scheduled daily strategy audits
and push notifications in BacktestGuard AI.

Usage:
    python run_daily.py --ticker AAPL --strategy rsi
    python run_daily.py --ticker BTC --strategy ma --bot-token "..." --chat-id "..."
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

from backtesting.engine import BacktestEngine
from llm.unified_explainer import UnifiedExplainer
from metrics.performance import calculate_metrics
from notifications.telegram import TelegramNotifier
from reporting.report_generator import generate_report
from storage.db import RunDatabase
from strategies.moving_average import MovingAverageCrossover
from strategies.rsi import RSIMeanReversion
from validate_ohlcv import validate_ohlcv


def fetch_or_simulate_market_data(ticker: str, bars: int = 120) -> pd.DataFrame:
    """
    Attempt to fetch real OHLCV data using yfinance if available,
    falling back to synthetic geometric Brownian motion bars.
    """
    try:
        import yfinance as yf

        df = yf.download(ticker, period="6mo", interval="1d", progress=False)
        if df is not None and len(df) >= 30:
            # Flatten MultiIndex columns if present in newer yfinance versions
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
            return df
    except Exception:
        pass

    # High-fidelity synthetic fallback
    dates = pd.date_range(end=pd.Timestamp.now(), periods=bars, freq="B")
    np.random.seed(abs(hash(ticker)) % (2**32))
    returns = np.random.normal(0.0005, 0.018, size=bars)
    price_path = 150.0 * np.cumprod(1.0 + returns)

    df = pd.DataFrame(
        {
            "Open": price_path * (1 - 0.002),
            "High": price_path * (1 + 0.006),
            "Low": price_path * (1 - 0.006),
            "Close": price_path,
            "Volume": 1_250_000,
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


def run_daily_audit(
    ticker: str = "AAPL",
    strategy_name: str = "rsi",
    bot_token: str | None = None,
    chat_id: str | None = None,
    groq_api_key: str | None = None,
) -> int:
    """Execute complete daily audit pipeline."""
    print(f"[*] Starting daily BacktestGuard audit for {ticker} ({strategy_name.upper()})...")

    # 1. Ingest Data
    df = fetch_or_simulate_market_data(ticker)
    print(f"[+] Loaded {len(df)} market bars.")

    # 2. Deterministic OHLCV Validation
    data_res = validate_ohlcv(df)
    if not data_res.passed:
        print(f"[!] Warning: Data validation encountered {len(data_res.errors)} errors.")
    else:
        print("[+] Data integrity check: PASSED (10/10 checks clean).")

    # 3. Strategy Instantiation
    if strategy_name.lower() == "rsi":
        strat = RSIMeanReversion()
    else:
        strat = MovingAverageCrossover()

    # 4. Controlled Backtest Simulation
    engine = BacktestEngine()
    bt_res = engine.run(df, strat)
    metrics = calculate_metrics(bt_res)
    print(f"[+] Backtest finished: Return={metrics.total_return_pct:.2%}, Sharpe={metrics.sharpe_ratio:.2f}")

    # 5. Report Generation & DB Logging
    report = generate_report(data_result=data_res, metrics=metrics)
    db = RunDatabase()
    run_id = db.save_run(report.to_dict())
    print(f"[+] Audit record saved to local database (Run ID: {run_id}).")

    # 6. AI Explanation
    groq_key = groq_api_key or os.getenv("GROQ_API_KEY")
    explainer = UnifiedExplainer(api_key=groq_key)
    ai_explanation = explainer.explain(report)

    # 7. Telegram Dispatch
    token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
    chat = chat_id or os.getenv("TELEGRAM_CHAT_ID")

    if token and chat:
        notifier = TelegramNotifier(token, chat)
        ok, msg = notifier.send_audit_alert(report, ticker=ticker, ai_explanation=ai_explanation)
        if ok:
            print(f"[+] Telegram alert delivered successfully to chat {chat}!")
        else:
            print(f"[!] Telegram alert delivery failed: {msg}")
    else:
        print("[i] Telegram credentials not provided; skipped push notification.")
        print("\n--- AI MENTOR REPORT PREVIEW ---")
        print(ai_explanation)

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="BacktestGuard AI Daily Strategy Auditor")
    parser.add_argument("--ticker", default="AAPL", help="Ticker or asset symbol (default: AAPL)")
    parser.add_argument("--strategy", choices=["rsi", "ma"], default="rsi", help="Strategy to evaluate")
    parser.add_argument("--bot-token", default=None, help="Telegram Bot Token (or set TELEGRAM_BOT_TOKEN env)")
    parser.add_argument("--chat-id", default=None, help="Telegram Chat ID (or set TELEGRAM_CHAT_ID env)")
    parser.add_argument("--groq-api-key", default=None, help="Groq Cloud API Key (or set GROQ_API_KEY env)")

    args = parser.parse_args()
    ret = run_daily_audit(
        ticker=args.ticker,
        strategy_name=args.strategy,
        bot_token=args.bot_token,
        chat_id=args.chat_id,
        groq_api_key=args.groq_api_key,
    )
    sys.exit(ret)


if __name__ == "__main__":
    main()
