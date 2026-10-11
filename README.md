# BacktestGuard AI

A Python tool that checks your stock-trading backtest for common mistakes before you trust the results.

---

## Why this exists

Backtests are simulations — you apply a trading strategy to historical price data to see how it would have performed. The problem is that a backtest can show great results even when the code is broken, the data is bad, or the strategy is accidentally "cheating" by looking at future prices.

Some common issues that produce misleading backtest results:

- Historical data with missing values, incorrect prices, or duplicate timestamps
- Strategies that accidentally use tomorrow's prices to make today's decision (look-ahead bias)
- Missing transaction costs — commissions and slippage that eat into real-world returns
- Performance metrics calculated incorrectly (e.g. Sharpe ratio with no risk-free rate adjustment)

BacktestGuard AI runs a set of deterministic checks on your data and strategy code, runs the backtest through a controlled engine, and then uses a local language model (via Ollama) to explain the findings in plain English.

It is a developer tool and a learning project. It does not give financial advice, and passing its checks does not mean a strategy will be profitable in live markets.

---

## What it does (current state)

| Feature | Status |
|---|---|
| Validate a list of price values | ✅ Done |
| Validate a full OHLCV DataFrame (Open, High, Low, Close, Volume) | ✅ Done |
| Detect missing values, bad timestamps, OHLC inconsistencies | ✅ Done |
| Structured validation findings with severity and evidence | ✅ Done |
| Strategy code inspection (AST-based) | 🔧 In progress |
| Backtesting engine (moving-average crossover) | 🔧 In progress |
| Performance metrics (return, drawdown, Sharpe ratio) | 🔧 In progress |
| AI explanation via local Ollama model | 🔧 In progress |
| FastAPI endpoint | 🔧 In progress |
| Streamlit dashboard | 🔧 In progress |

---

## Tech stack

| Tool | What it's used for |
|---|---|
| Python 3.11 | Core language |
| pandas | DataFrame manipulation and time-series data |
| NumPy | Numerical calculations |
| pytest | Automated tests |
| Ruff | Linting and formatting |
| Ollama (`qwen2.5:1.5b`) | Local LLM for explaining findings |
| FastAPI | REST API layer |
| Streamlit | Interactive dashboard |
| SQLite | Storing validation run history |

Everything runs locally. No paid API keys required.

---

## Project structure

```
BacktestGuardAI/
├── validate_prices.py        # Validates a flat list of price values
├── validate_ohlcv.py         # Validates a full OHLCV DataFrame
├── tests/
│   ├── conftest.py
│   ├── test_validate_prices.py   # 37 tests
│   └── test_validate_ohlcv.py    # 60 tests
├── data/
│   └── sample/               # Sample datasets for local testing
├── docs/                     # Architecture notes and design decisions
├── pyproject.toml            # Project config, dependencies, tool settings
└── .gitignore
```

---

## Setup

**Requirements:** Python 3.11+, and [Ollama](https://ollama.com) installed locally (for AI explanations — optional for the validation-only features).

```powershell
# Clone the repository
git clone https://github.com/your-username/BacktestGuardAI.git
cd BacktestGuardAI

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install pandas numpy pytest ruff
```

---

## Running the tests

```powershell
python -m pytest tests/ -v
```

Expected output:
```
97 passed, 1 warning in 1.7s
```

The one warning is a pandas FutureWarning in a test that deliberately inserts an infinite volume value. It does not affect correctness.

---

## How validation works

The core idea is simple: before running any simulation, check every assumption.

**`validate_prices(prices)`** takes a plain Python list and checks:
- Is it actually a list?
- Is it non-empty?
- Is every element a finite, positive number?

It returns a list of error strings. An empty list means all checks passed.

```python
from validate_prices import validate_prices

errors = validate_prices([100.0, -5.0, float('inf'), 102.0])
for e in errors:
    print(e)
# index 1: price must be positive, got -5.0
# index 2: price is not finite (inf or -inf or nan)
```

**`validate_ohlcv(df)`** takes a pandas DataFrame and runs 10 checks:

1. Input is a DataFrame
2. Required columns exist (Open, High, Low, Close, Volume)
3. Index is a DatetimeIndex
4. No duplicate timestamps
5. Timestamps are in ascending order
6. No missing (NaN) values
7. All prices are finite
8. All prices are strictly positive
9. Volume is non-negative
10. OHLC relationships are internally consistent (e.g. High ≥ Open, High ≥ Close)

Each finding is a structured object with a `code`, `severity` (`"error"` or `"warning"`), `description`, `evidence`, and `recommendation`.

```python
from validate_ohlcv import validate_ohlcv

result = validate_ohlcv(df)

if result.passed:
    print("Data looks clean.")
else:
    for finding in result.errors:
        print(finding)
        print("Evidence:", finding.evidence)
        print("Fix:", finding.recommendation)
```

---

## Design decisions worth knowing

**Why return a list of findings instead of raising an exception?**
Raising an exception on the first problem forces you to fix one issue at a time and re-run. Collecting all findings lets you see the full picture in one pass — which is more useful when you have real messy data.

**Why separate `error` and `warning` severity?**
Not every data issue should block a backtest. A dataset with 28 rows instead of 30 is worth flagging, but it should not prevent the user from experimenting. Errors are blockers; warnings are things to investigate.

**Why not execute the user's strategy code directly?**
Running arbitrary Python code (`exec()`) is a security risk. Instead, BacktestGuard uses AST (abstract syntax tree) analysis to inspect the code structure without running it, and a predefined strategy interface for the backtesting engine.

---

## Known limitations

- OHLCV validation only checks the data that is given to it. It cannot detect survivorship bias, adjusted vs. unadjusted prices, or corporate-action errors.
- The strategy inspector flags suspicious patterns but cannot guarantee the absence of look-ahead bias. Some forms of bias require domain knowledge to detect.
- The local LLM explains findings in natural language but cannot override the deterministic validation results. LLM output is informational only.
- This project is not production software. It is a portfolio project built for learning and demonstration.

---

## What's next

- Strategy code inspection using Python's `ast` module
- Moving-average crossover backtesting engine with transaction costs
- Performance metrics (total return, Sharpe ratio, max drawdown)
- Validation report combining all findings
- AI explanation via Ollama
- FastAPI + Streamlit interface
- SQLite run history

---

## License

MIT
