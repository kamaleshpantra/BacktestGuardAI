# BacktestGuard AI

BacktestGuard AI is a Python-based quantitative auditing framework and simulation engine. It inspects historical financial datasets, validates custom trading strategies against look-ahead bias using AST (Abstract Syntax Tree) analysis, executes realistic bar-by-bar backtests with execution lags and friction, and generates AI-driven diagnostic reports.

---

## The Problem It Solves

Backtesting simulations often present an illusion of alpha. In quantitative research and algorithmic trading, misleading backtest results typically stem from three root causes:

1. **Dirty Market Data:** Unsorted timestamps, non-positive prices, non-finite values (NaN / Inf), or corrupted OHLC relationships ($Low > High$).
2. **Look-Ahead Bias (Information Leakage):** Strategies calculating signals using future data (e.g., negative shifts like `df['Close'].shift(-1)`, centered moving windows, global dataset statistics applied to historic bars, or backward indexing).
3. **Frictionless & Zero-Lag Assumptions:** Fictitious performance caused by executing trades on the exact close bar where signals are generated, or omitting exchange fees and market slippage.

BacktestGuard AI enforces a **deterministic-first** auditing pipeline, guaranteeing that data integrity and code safety are systematically verified before any capital or confidence is committed.

---

## Key System Architecture

```
                               ┌────────────────────────────────┐
                               │   Raw Data (CSV / Synthetic)   │
                               └───────────────┬────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │  OHLCV & Price Data Validator    │
                              │  (10 Deterministic Checks)       │
                              └────────────────┬─────────────────┘
                                               │
                      ┌────────────────────────┴────────────────────────┐
                      ▼                                                 ▼
      ┌──────────────────────────────┐                ┌───────────────────────────────────┐
      │   Standard Built-in Logic    │                │    Custom Python Code Strategy    │
      │   (MA Crossover, RSI Reversion)               │    (Restricted AST Sandbox Loader)│
      └──────────────┬───────────────┘                └─────────────────┬─────────────────┘
                     │                                                  │
                     │                                ┌─────────────────┴─────────────────┐
                     │                                │   AST Static Security & Bias      │
                     │                                │   Inspector (Blocks malicious     │
                     │                                │   imports & future look-ahead)    │
                     │                                └─────────────────┬─────────────────┘
                     │                                                  │
                     └────────────────────────┬─────────────────────────┘
                                              │
                                              ▼
                              ┌──────────────────────────────────┐
                              │    Bar-by-Bar Backtest Engine    │
                              │  (1-Bar Lag, Fees, Slippage)     │
                              └───────────────┬──────────────────┘
                                              │
                                              ▼
                              ┌──────────────────────────────────┐
                              │    Performance Analytics Core    │
                              │  (Sharpe, Max DD, CAGR, Win Rate)│
                              └───────────────┬──────────────────┘
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     ▼                                                 ▼
     ┌───────────────────────────────┐                 ┌─────────────────────────────────┐
     │   Unified Explainer (Local    │                 │   Persistent Storage & Alerts   │
     │   Ollama / Cloud LLM Mentor)  │                 │   (SQLite Run Logs + Telegram)  │
     └───────────────────────────────┘                 └─────────────────────────────────┘
```

---

## Core Capabilities

### 1. Robust OHLCV & Price Integrity Engine
- **Price Validator:** Validates flat price series for invalid types, empty sets, negative/zero prices, and non-finite values.
- **OHLCV Validator:** Rigorously audits DataFrames for missing columns, duplicate or unsorted timestamps, nulls, invalid volume values, and geometric violations ($High < Low$, $High < Open$, $Low > Close$).
- Returns structured findings with explicit severity classification (`ERROR` vs `WARNING`), error codes, and row/column indices.

### 2. AST Static Code Inspector
- Analyzes Python strategy code without executing unsafe code or arbitrary scripts.
- Detects look-ahead patterns using Python's `ast` module:
  - **Negative Shifts (`shift(-k)`):** Accessing future bars.
  - **Centered Rolling Windows (`rolling(..., center=True)`):** Bilateral window centering.
  - **Global Aggregate Leakage:** Strategy decisions comparing against global dataset metrics (`df['Close'].mean()`, `max()`, `min()`).
  - **Negative Slicing / Indexing:** Suspicious index manipulation.

### 3. Safe Dynamic Strategy Sandbox
- Allows researchers to write custom signal logic directly in Python.
- Multi-tier execution sandbox: passes AST security scanning (blocks built-ins like `eval`, `exec`, `open`, `os`, `subprocess`) and bias checks before generating trades.
- Implements the strict `Strategy` abstract base class interface.

### 4. Controlled Backtesting & Metric Suite
- **Bar-by-Bar Accounting:** Signal generated at close of bar $T$ is filled at bar $T+1$ open.
- **Frictional Costs:** Configurable per-trade commission fees and slippage models.
- **Metrics Calculated:** Total Return, Compound Annual Growth Rate (CAGR), Annualized Volatility, Maximum Drawdown, Sharpe Ratio, and Win Rate.

### 5. Multi-Model AI Explainer (Local + Cloud)
- **Local Ollama Integration:** Runs entirely offline using local lightweight models (e.g. `qwen2.5:1.5b` or `qwen2.5-coder:7b`).
- **Cloud Fallback:** Supports free-tier cloud endpoints (Groq / OpenAI-compatible API) via sidebar configuration when deployed.
- Generates 3-point structured mentor summaries explaining data risks, strategy quality, and actionable engineering next steps.

### 6. Automated Alerts & Storage
- **Telegram Push Alerts:** Formatted Markdown reports sent directly to your phone via Telegram Bot API with performance metrics and AI explanations.
- **SQLite History:** Local auditing database storing run histories and audit reports.
- **FastAPI Endpoints:** Webhook triggers and REST endpoints for external systems and pipelines.

---

## Project Structure

```
BacktestGuardAI/
├── app.py                         # Streamlit interactive dashboard (4 tabs)
├── pyproject.toml                 # Dependencies, pytest, and Ruff configuration
├── requirements.txt               # Production dependencies for cloud deployment
├── validate_prices.py             # Flat price list validation
├── validate_ohlcv.py              # OHLCV DataFrame integrity validation
├── api/
│   └── server.py                  # FastAPI REST endpoints & webhook dispatcher
├── backtesting/
│   └── engine.py                  # Bar-by-bar backtest simulation engine
├── data/
│   └── loader.py                  # Smart CSV loader (multi-vendor column mapping)
├── inspection/
│   └── ast_inspector.py           # AST-based static look-ahead bias detector
├── llm/
│   ├── explainer.py               # Local Ollama explanation client
│   └── unified_explainer.py       # Hybrid local/cloud LLM router
├── metrics/
│   └── performance.py             # Quant performance metric calculators
├── notifications/
│   └── telegram.py                # Direct Telegram Bot API notifier
├── reporting/
│   └── report_generator.py        # Unified JSON & Markdown report generator
├── storage/
│   └── db.py                      # SQLite database repository for run logs
├── strategies/
│   ├── base.py                    # Strategy abstract base class (ABC)
│   ├── moving_average.py          # MA Crossover strategy
│   ├── rsi.py                     # RSI Mean-Reversion strategy
│   └── custom_loader.py           # Safe custom Python strategy sandbox
└── tests/
    ├── conftest.py                # Pytest configuration & sys.path setup
    ├── test_validate_prices.py    # 37 tests (price validation)
    ├── test_validate_ohlcv.py     # 60 tests (OHLCV DataFrame validation)
    ├── test_ast_inspector.py      # 49 tests (AST look-ahead detection)
    ├── test_strategies.py         # 25 tests (MA strategy logic)
    ├── test_backtest_and_metrics.py# 7 tests (engine, metrics, reporting)
    ├── test_rsi_strategy.py       # 4 tests (RSI indicator calculation)
    ├── test_data_loader.py        # 3 tests (CSV normalization)
    ├── test_telegram_notifier.py  # 2 tests (Telegram alert formatting)
    └── test_custom_loader.py      # 4 tests (sandbox security & execution)
```

---

## Test Suite & Verification

The test suite covers edge cases, data errors, code security, and quantitative accuracy across **191 unit and integration tests**:

```powershell
python -m pytest tests/ -v
```

```
======================= 191 passed, 1 warning in 9.31s ========================
```

Linting and code style are verified with Ruff:
```powershell
python -m ruff check .
```
```
All checks passed!
```

---

## Getting Started

### 1. Installation
Clone the repository and install requirements:
```powershell
git clone https://github.com/<your-username>/BacktestGuardAI.git
cd BacktestGuardAI
pip install -r requirements.txt
```

### 2. Running the Interactive Streamlit App
Launch the web UI using Python's module runner:
```powershell
python -m streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser:
- **Tab 1: Full Audit & Backtest:** Upload CSVs or generate synthetic market data, pick or write a custom strategy, run the backtest, view equity charts, and get AI mentor notes.
- **Tab 2: AST Code Inspector:** Paste raw Python strategy code to check for look-ahead violations.
- **Tab 3: Telegram Alerts:** Configure your Bot Token and Chat ID to receive instant audit summaries.
- **Tab 4: Run History:** View stored audit runs from the SQLite database.

### 3. Running the FastAPI Backend (Optional)
```powershell
uvicorn api.server:app --reload --port 8000
```
Visit [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive Swagger API documentation.

---

## Cloud Deployment (Streamlit Community Cloud)

1. Push your repository to **GitHub**.
2. Visit [share.streamlit.io](https://share.streamlit.io) and connect your repository.
3. Set the Main file path to `app.py`.
4. (Optional) In the dashboard sidebar, enter a free **Groq API Key** (e.g. `gsk_...` with `llama-3.1-8b-instant`) to enable cloud-based AI explanations without running local Ollama.

---

## Engineering Design Principles

1. **Deterministic Core:** Safety and validation rules do not depend on probabilistic LLM responses. Static code checks and price validations are 100% deterministic.
2. **AI as an Explainer, Not a Decision-Maker:** Language models explain identified risks and summarize metrics in plain English; they do not dictate whether code passes validation.
3. **Execution Lag Enforcement:** Signals generated at bar $T$ are executed at bar $T+1$ to mirror live trading realities.
4. **Zero Forced Dependencies:** The application works entirely offline with local models or pure rule-based reporting if no LLM is available.

---

## License

MIT License. Developed for research and developer tooling.
