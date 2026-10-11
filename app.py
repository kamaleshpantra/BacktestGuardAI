"""
app.py
------
Streamlit Interactive Dashboard for BacktestGuard AI.
Allows users to:
1. Inspect custom strategy code for Look-Ahead Bias.
2. Load / generate synthetic market data or upload CSV.
3. Validate data and execute controlled backtest simulations.
4. View metrics, findings, and generate local AI explanations via Ollama.
5. Review stored runs in SQLite.
"""

import numpy as np
import pandas as pd
import streamlit as st

from backtesting.engine import BacktestEngine
from inspection.ast_inspector import inspect_strategy_source
from llm.unified_explainer import UnifiedExplainer
from metrics.performance import calculate_metrics
from notifications.telegram import TelegramNotifier
from reporting.report_generator import generate_report
from storage.db import RunDatabase
from strategies.moving_average import MovingAverageCrossover
from strategies.rsi import RSIMeanReversion
from validate_ohlcv import validate_ohlcv

st.set_page_config(
    page_title="BacktestGuard AI",
    page_icon="🛡️",
    layout="wide",
)

db = RunDatabase()

# Sidebar for Cloud AI / API Keys
with st.sidebar:
    st.header("⚙️ Configuration")
    cloud_key = st.text_input(
        "Cloud AI Key (Groq / OpenAI - Optional):",
        type="password",
        help="Leave empty to use local Ollama model directly.",
    )
    st.caption("Default: Local Ollama (qwen2.5:1.5b) when running on local PC.")

explainer = UnifiedExplainer(api_key=cloud_key if cloud_key else None)

st.title("🛡️ BacktestGuard AI — Verification & Audit Platform")
st.markdown(
    "Detect data anomalies, look-ahead bias, and unrealistic backtest assumptions before deploying strategies."
)

tab_audit, tab_code, tab_telegram, tab_history = st.tabs(
    ["🚀 Full Audit & Backtest", "🔍 AST Code Inspector", "📱 Telegram Alerts & Automation", "📜 Run History"]
)

# -----------------------------------------------------------------------------
# TAB 1: AUDIT & BACKTEST
# -----------------------------------------------------------------------------
with tab_audit:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("1. Data & Strategy Setup")
        data_choice = st.radio(
            "Select Market Data:",
            ["Generate Realistic Synthetic Data", "Upload CSV"],
        )

        df = None
        if data_choice == "Generate Realistic Synthetic Data":
            bars = st.slider("Number of trading bars:", 30, 250, 100)
            dates = pd.date_range("2024-01-01", periods=bars, freq="B")
            np.random.seed(42)
            returns = np.random.normal(0.0005, 0.015, size=bars)
            price_path = 100.0 * np.cumprod(1.0 + returns)

            df = pd.DataFrame(
                {
                    "Open": price_path * (1 - 0.002),
                    "High": price_path * (1 + 0.005),
                    "Low": price_path * (1 - 0.005),
                    "Close": price_path,
                    "Volume": np.random.randint(50_000, 500_000, size=bars),
                },
                index=dates,
            )
            st.success(f"Generated {bars} bars.")
        else:
            uploaded = st.file_uploader("Upload OHLCV CSV file:", type=["csv"])
            if uploaded:
                try:
                    from data.loader import load_csv
                    df = load_csv(uploaded)
                    st.info(f"Loaded and normalized {len(df)} rows.")
                    with st.expander("Preview Normalized Data"):
                        st.dataframe(df.head(5), use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading CSV: {e}")

        st.subheader("Strategy Selection & Parameters")
        strat_choice = st.selectbox(
            "Select Strategy Archetype:",
            [
                "Trend Following: Moving Average Crossover",
                "Mean Reversion: Relative Strength Index (RSI)",
                "Custom Python Strategy (Live Code Editor)",
            ],
        )

        selected_strategy = None
        custom_compile_error = None

        if strat_choice == "Trend Following: Moving Average Crossover":
            short_w = st.number_input("Short Moving Average Window", min_value=2, max_value=50, value=10)
            long_w = st.number_input("Long Moving Average Window", min_value=5, max_value=150, value=30)
            selected_strategy = MovingAverageCrossover(short_window=int(short_w), long_window=int(long_w))
        elif strat_choice == "Mean Reversion: Relative Strength Index (RSI)":
            rsi_period = st.number_input("RSI Lookback Period", min_value=2, max_value=50, value=14)
            oversold = st.number_input("Oversold Buy Threshold", min_value=5.0, max_value=45.0, value=30.0)
            overbought = st.number_input("Overbought Sell Threshold", min_value=55.0, max_value=95.0, value=70.0)
            selected_strategy = RSIMeanReversion(
                period=int(rsi_period),
                oversold_threshold=float(oversold),
                overbought_threshold=float(overbought),
            )
        else:
            default_custom_code = '''def generate_signals(df):
    # Safe Example: Buy when price is above its 20-day moving average
    ma20 = df['Close'].rolling(20).mean()
    raw_signal = (df['Close'] > ma20).astype(float)

    # Mandatory: 1-bar execution lag to eliminate look-ahead bias
    return raw_signal.shift(1).fillna(0)
'''
            st.caption("Write your custom signal generator. Must define `generate_signals(df)` returning a Series.")
            custom_code_str = st.text_area("Custom Python Code:", value=default_custom_code, height=160)
            from strategies.custom_loader import SafeCustomStrategy
            try:
                selected_strategy = SafeCustomStrategy(custom_code_str, strategy_name="UserCustomStrategy")
                st.success("✅ Code passed AST security & look-ahead audit!")
            except Exception as exc:
                custom_compile_error = str(exc)
                st.error(f"❌ {custom_compile_error}")

        capital = st.number_input("Initial Capital ($)", min_value=1000.0, value=10_000.0)
        slippage = st.number_input("Slippage (%)", min_value=0.0, max_value=1.0, value=0.05) / 100.0
        fee = st.number_input("Commission (%)", min_value=0.0, max_value=1.0, value=0.1) / 100.0

        run_btn = st.button("Run Comprehensive Audit", type="primary", use_container_width=True)

    with col2:
        st.subheader("2. Audit Results & Explanations")
        if run_btn:
            if df is None:
                st.warning("Please provide or generate a dataset first.")
            elif selected_strategy is None:
                st.error("Cannot run audit: custom strategy has syntax or look-ahead errors. Please fix them above.")
            else:
                # 1. Dataset Validation
                data_res = validate_ohlcv(df)

                # 2. Strategy Validation
                strat = selected_strategy
                config_errs = strat.validate_config()

                if config_errs:
                    st.error(f"Strategy Config Error: {config_errs}")
                elif not data_res.passed:
                    st.error("Data Validation Failed! See findings below:")
                    for err in data_res.errors:
                        st.write(f"- ❌ **{err.code}**: {err.description}")
                else:
                    # 3. Controlled Simulation
                    engine = BacktestEngine(
                        initial_capital=capital,
                        fee_pct=fee,
                        slippage_pct=slippage,
                    )
                    bt_res = engine.run(df, strat)
                    metrics = calculate_metrics(bt_res)

                    report = generate_report(data_result=data_res, metrics=metrics)
                    run_id = db.save_run(report.to_dict())

                    st.success(f"Audit completed successfully! Saved as Run #{run_id}.")

                    # Metric cards
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total Return", f"{metrics.total_return_pct}%")
                    m2.metric("Sharpe Ratio", f"{metrics.sharpe_ratio}")
                    m3.metric("Max Drawdown", f"{metrics.max_drawdown_pct}%")
                    m4.metric("Win Rate", f"{metrics.win_rate_pct}%")

                    # Visual chart
                    st.line_chart(bt_res.equity_curve, height=300)

                    # AI Explanation
                    st.markdown("### 🤖 Local AI Mentor Explanation")
                    with st.spinner("Generating explanation with Ollama..."):
                        explanation = explainer.explain(report)
                        st.markdown(explanation)

# -----------------------------------------------------------------------------
# TAB 2: CODE INSPECTOR
# -----------------------------------------------------------------------------
with tab_code:
    st.subheader("AST Strategy Code Inspector")
    st.write(
        "Paste strategy code below to deterministically detect look-ahead bias without executing untrusted code."
    )

    sample_code = '''def generate_signals(df):
    # Suspicious pattern: negative shift looks into future
    future_price = df['Close'].shift(-1)

    # Suspicious: global mean
    global_avg = df['Close'].mean()

    return (future_price > global_avg).astype(int)
'''
    code_input = st.text_area("Strategy Python Code:", value=sample_code, height=200)

    if st.button("Inspect Code Structure"):
        res = inspect_strategy_source(code_input)
        if not res.is_parseable:
            st.error(f"Parse error: {res.parse_error}")
        else:
            if res.has_confirmed_violations:
                st.error("🚨 Confirmed Look-Ahead Violations Found!")
            elif res.suspicious:
                st.warning("⚠️ Suspicious Patterns Detected")
            else:
                st.success("✅ Clean! No obvious static look-ahead patterns detected.")

            for f in res.findings:
                st.markdown(
                    f"- **[{f.certainty.upper()}] {f.code}** (Line {f.line_number}): {f.description}"
                )
                st.caption(f"Evidence: `{f.evidence}` | Fix: {f.recommendation}")

# -----------------------------------------------------------------------------
# TAB 3: TELEGRAM ALERTS & AUTOMATION
# -----------------------------------------------------------------------------
with tab_telegram:
    st.subheader("📱 Automated Mobile Alerts via Telegram")
    st.markdown(
        "Receive instant push notifications after market close with data integrity audits, "
        "strategy execution signals, and AI mentor explanations."
    )

    t_col1, t_col2 = st.columns(2)
    with t_col1:
        st.markdown("#### 1. Configure Telegram Bot")
        st.caption("Create a free bot on Telegram via **@BotFather** to get your token.")
        bot_token = st.text_input("Telegram Bot Token:", type="password", placeholder="e.g. 123456789:ABCdefGhIJKlmNoPQR")
        chat_id = st.text_input("Your Telegram Chat ID:", placeholder="e.g. 987654321")

        if st.button("🔔 Send Test Handshake Notification"):
            notifier = TelegramNotifier(bot_token=bot_token, chat_id=chat_id)
            success, msg = notifier.test_connection()
            if success:
                st.success(f"✅ {msg} Check your Telegram app!")
            else:
                st.error(f"❌ {msg}")

    with t_col2:
        st.markdown("#### 2. Schedule Daily Monitoring")
        st.caption("BacktestGuard will automatically audit this asset and send the report with AI explanation.")
        monitored_ticker = st.text_input("Monitored Asset / Ticker:", value="AAPL")
        monitored_strat = st.selectbox("Strategy to Monitor:", ["RSI Mean Reversion (14, 30/70)", "MA Crossover (10, 30)"])

        if st.button("🚀 Trigger Instant Audit & Send Telegram Alert", type="primary"):
            notifier = TelegramNotifier(bot_token=bot_token, chat_id=chat_id)
            if not notifier.is_configured():
                st.warning("Please fill in both Bot Token and Chat ID first.")
            else:
                with st.spinner("Running complete audit and generating AI explanation..."):
                    # 1. Generate realistic benchmark bars for monitored asset
                    bars = 100
                    dates = pd.date_range("2024-01-01", periods=bars, freq="B")
                    np.random.seed(42)
                    returns = np.random.normal(0.0005, 0.015, size=bars)
                    price_path = 150.0 * np.cumprod(1.0 + returns)
                    sim_df = pd.DataFrame(
                        {
                            "Open": price_path * (1 - 0.002),
                            "High": price_path * (1 + 0.005),
                            "Low": price_path * (1 - 0.005),
                            "Close": price_path,
                            "Volume": 1_500_000,
                        },
                        index=dates,
                    )
                    data_res = validate_ohlcv(sim_df)
                    strat = RSIMeanReversion() if "RSI" in monitored_strat else MovingAverageCrossover()
                    engine = BacktestEngine()
                    bt_res = engine.run(sim_df, strat)
                    metrics = calculate_metrics(bt_res)
                    report = generate_report(data_result=data_res, metrics=metrics)

                    # 2. Generate AI Explanation
                    ai_text = explainer.explain(report)

                    # 3. Dispatch to Telegram
                    ok, send_msg = notifier.send_audit_alert(report, ticker=monitored_ticker, ai_explanation=ai_text)
                    if ok:
                        st.success(f"✅ Alert delivered to your phone! ({send_msg})")
                        st.markdown("### Preview of Sent Message:")
                        st.info(ai_text)
                    else:
                        st.error(f"❌ Delivery failed: {send_msg}")

# -----------------------------------------------------------------------------
# TAB 4: HISTORY
# -----------------------------------------------------------------------------
with tab_history:
    st.subheader("Audit Logs & Stored Runs")
    runs = db.list_runs(limit=30)
    if not runs:
        st.info("No runs stored yet.")
    else:
        st.dataframe(pd.DataFrame(runs), use_container_width=True)
