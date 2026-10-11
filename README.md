# BacktestGuard AI

A Python tool that checks stock-trading backtests for common data quality issues, look-ahead bias, and unrealistic execution assumptions before you trust the simulated results.

---

## Why this exists

Backtests are simulations — you apply a trading strategy to historical price data to see how it would have performed. The problem is that a backtest can show great results even when the code is broken, the data is corrupt, or the strategy is accidentally "cheating" by looking at future prices.

Some common issues that produce misleading backtest results:

- Historical data with missing values, non-positive prices, or duplicate timestamps
- Strategies that accidentally use tomorrow's prices to make today's decision (look-ahead bias)
- Missing transaction costs — commissions and slippage that eat into real-world returns
- Unrealistic metrics and lack of reproducibility

BacktestGuard AI runs deterministic checks on historical market data and strategy code (using Python's AST), runs simulations through a controlled backtesting engine, and uses a local language model (via Ollama) to explain findings in plain language.

It is a developer and research tool. It does not provide financial advice, and passing its checks does not guarantee profitability in live markets.

---

## Current Architecture & Completed Features

| Feature Component | Implementation Details | Status |
|---|---|---|
| **Price & OHLCV Validation** | Validates flat prices and full OHLCV DataFrames (types, order, nulls, bounds, High/Low consistency) | ✅ Completed (97 tests) |
| **Strategy Contract & Crossover** | `Strategy` abstract base class, look-ahead safe moving average crossover with 1-bar execution lag | ✅ Completed (25 tests) |
| **AST Code Inspector** | Static syntax tree analysis detecting `.shift(-N)`, `.rolling(center=True)`, global stats, negative `iloc` | ✅ Completed (49 tests) |
| **Controlled Backtesting Engine** | Bar-by-bar cash & share accounting, configurable fees and slippage modeling | ✅ Completed (4 tests) |
| **Performance Analytics** | Return, CAGR, Annualized Volatility, Max Drawdown, Sharpe Ratio, Win Rate | ✅ Completed (2 tests) |
| **Unified Validation Reports** | JSON & Markdown report generation pairing deterministic checks with metrics | ✅ Completed (1 test) |
| **Local LLM Explainer** | Ollama integration (`qwen2.5:1.5b`) with grounded prompting & graceful offline fallback | ✅ Completed |
| **SQLite Run History** | Stores audit runs, metrics, and JSON payloads locally without external databases | ✅ Completed |
| **FastAPI REST API** | Endpoints for dataset validation, code inspection, and audit history (`api/server.py`) | ✅ Completed |
| **Streamlit Interactive UI** | Multi-tab dashboard for live backtesting, code inspection, and history review (`app.py`) | ✅ Completed |

---

## Tech Stack

- **Core & Data:** Python 3.11, pandas, NumPy
- **Quality & Testing:** pytest (178 passing tests), Ruff (linter/formatter)
- **Static Analysis:** Python `ast` module (AST-based static code analysis)
- **Local AI:** Ollama (`qwen2.5:1.5b` or `qwen2.5-coder:7b`) with `httpx`
- **Application & Storage:** FastAPI, Streamlit, SQLite

Everything runs locally on your machine with zero paid API subscriptions.

---

## Project Structure

```
BacktestGuardAI/
├── app.py                         # Streamlit interactive dashboard
├── pyproject.toml                 # Dependencies, pytest, and Ruff configuration
├── .gitignore                     # Caches, virtualenvs, local databases
├── validate_prices.py             # Validates flat price lists
├── validate_ohlcv.py              # Validates full OHLCV DataFrames
├── strategies/
│   ├── base.py                    # Strategy abstract base class
│   └── moving_average.py          # MA Crossover strategy
├── inspection/
│   └── ast_inspector.py           # AST look-ahead bias detector
├── backtesting/
│   └── engine.py                  # Simulation engine with fees & slippage
├── metrics/
│   └── performance.py             # Sharpe ratio, drawdown, CAGR, win rate
├── reporting/
│   └── report_generator.py        # Unified report builder
├── llm/
│   └── explainer.py               # Local Ollama explanation client
├── storage/
│   └── db.py                      # SQLite run logging repository
├── api/
│   └── server.py                  # FastAPI REST endpoints
└── tests/
    ├── conftest.py                # sys.path root configuration
    ├── test_validate_prices.py    # 37 tests
    ├── test_validate_ohlcv.py     # 60 tests
    ├── test_strategies.py         # 25 tests
    ├── test_ast_inspector.py      # 49 tests
    └── test_backtest_and_metrics.py # 7 tests
```

---

## Setup & Quickstart

### 1. Prerequisites
Python 3.11+ installed. (Optional: [Ollama](https://ollama.com) running for AI explanations).

### 2. Installation
```powershell
# Clone or navigate to the repository
cd C:\Users\91798\OneDrive\Desktop\BacktestGuardAI

# Run tests to verify the setup
python -m pytest tests/ -v
```

Expected result:
```
178 passed in ~1.5s
```

---

## Running the Application

### Option A: Interactive Streamlit Dashboard
```powershell
streamlit run app.py
```
Open `http://localhost:8501` to test strategy code in the AST inspector, generate synthetic data, run simulations, view equity curves, and receive local AI explanations.

### Option B: FastAPI Backend
```powershell
uvicorn api.server:app --reload --port 8000
```
Visit `http://localhost:8000/docs` to test the interactive Swagger REST API.

---

## Key Design Principles

1. **Deterministic First, AI Second:** Validation and look-ahead checks are 100% deterministic code rules. The language model only interprets and explains findings to the developer; it never overrides validation decisions.
2. **Safe Code Inspection:** Never execute untrusted or uploaded strategy Python files with `eval()` or `exec()`. We parse the code structure using Python's `ast` module.
3. **Execution Lag:** Signals generated at bar $T$ are executed at bar $T+1$ to ensure simulations do not trade on prices known only after the bar closes.
4. **Frictional Realism:** Slippage and commissions are explicitly accounted for to prevent false impressions of profitability.

---

## License

MIT
