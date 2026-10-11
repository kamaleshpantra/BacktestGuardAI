"""
tests/test_ast_inspector.py
----------------------------
Tests for inspect_strategy_source() and the pattern-detection logic.

Each test class targets one specific pattern (NEGATIVE_SHIFT, CENTERED_ROLLING,
etc.). Source code strings are defined inline so tests are self-contained
and easy to read.

Important: the inspector analyses code structure only. It never executes
any of the source strings provided to it.
"""

import pytest

from inspection.ast_inspector import (
    InspectionFinding,
    InspectionResult,
    inspect_strategy_source,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def has_code(result: InspectionResult, code: str) -> bool:
    return any(f.code == code for f in result.findings)


def confirmed_codes(result: InspectionResult) -> list[str]:
    return [f.code for f in result.confirmed]


def suspicious_codes(result: InspectionResult) -> list[str]:
    return [f.code for f in result.suspicious]


# ---------------------------------------------------------------------------
# Test: clean code produces no findings
# ---------------------------------------------------------------------------

class TestCleanCode:

    CLEAN_SOURCE = """
def generate_signals(df):
    short_ma = df['Close'].rolling(20).mean()
    long_ma = df['Close'].rolling(50).mean()
    signal = (short_ma > long_ma).astype(float)
    return signal.shift(1).fillna(0)
"""

    def test_clean_strategy_has_no_findings(self):
        result = inspect_strategy_source(self.CLEAN_SOURCE)
        assert result.findings == []

    def test_clean_strategy_is_parseable(self):
        result = inspect_strategy_source(self.CLEAN_SOURCE)
        assert result.is_parseable is True

    def test_clean_strategy_has_no_confirmed_violations(self):
        result = inspect_strategy_source(self.CLEAN_SOURCE)
        assert result.has_confirmed_violations is False

    def test_rolling_mean_is_not_flagged_as_global_stat(self):
        # .rolling(20).mean() must NOT trigger GLOBAL_STATISTIC.
        result = inspect_strategy_source(self.CLEAN_SOURCE)
        assert "GLOBAL_STATISTIC" not in suspicious_codes(result)

    def test_positive_shift_is_not_flagged(self):
        source = """
def generate_signals(df):
    return df['Close'].shift(1)
"""
        result = inspect_strategy_source(source)
        assert "NEGATIVE_SHIFT" not in confirmed_codes(result)


# ---------------------------------------------------------------------------
# Test: NEGATIVE_SHIFT (confirmed)
# ---------------------------------------------------------------------------

class TestNegativeShift:

    def test_shift_minus_one(self):
        source = """
def generate_signals(df):
    future_close = df['Close'].shift(-1)
    return (future_close > df['Close']).astype(int)
"""
        result = inspect_strategy_source(source)
        assert "NEGATIVE_SHIFT" in confirmed_codes(result)

    def test_shift_minus_five(self):
        source = """
def generate_signals(df):
    future = df['Close'].shift(-5)
    return future
"""
        result = inspect_strategy_source(source)
        assert "NEGATIVE_SHIFT" in confirmed_codes(result)

    def test_certainty_is_confirmed(self):
        source = "x = prices.shift(-2)"
        result = inspect_strategy_source(source)
        negative_findings = [f for f in result.findings if f.code == "NEGATIVE_SHIFT"]
        assert len(negative_findings) == 1
        assert negative_findings[0].certainty == "confirmed"

    def test_line_number_is_recorded(self):
        source = "x = 1\ny = 2\nz = df.shift(-1)"
        result = inspect_strategy_source(source)
        finding = next(f for f in result.findings if f.code == "NEGATIVE_SHIFT")
        assert finding.line_number == 3

    def test_multiple_negative_shifts_all_detected(self):
        source = """
a = df['Open'].shift(-1)
b = df['Close'].shift(-2)
c = df['High'].shift(-3)
"""
        result = inspect_strategy_source(source)
        negative = [f for f in result.findings if f.code == "NEGATIVE_SHIFT"]
        assert len(negative) == 3

    def test_shift_zero_is_not_flagged(self):
        source = "x = df['Close'].shift(0)"
        result = inspect_strategy_source(source)
        assert "NEGATIVE_SHIFT" not in confirmed_codes(result)

    def test_shift_positive_is_not_flagged(self):
        source = "x = df['Close'].shift(3)"
        result = inspect_strategy_source(source)
        assert "NEGATIVE_SHIFT" not in confirmed_codes(result)

    def test_shift_no_args_is_not_flagged(self):
        # shift() with no argument defaults to 1 — safe.
        source = "x = df['Close'].shift()"
        result = inspect_strategy_source(source)
        assert "NEGATIVE_SHIFT" not in confirmed_codes(result)


# ---------------------------------------------------------------------------
# Test: CENTERED_ROLLING (confirmed)
# ---------------------------------------------------------------------------

class TestCenteredRolling:

    def test_center_true_detected(self):
        source = """
def generate_signals(df):
    ma = df['Close'].rolling(window=20, center=True).mean()
    return (df['Close'] > ma).astype(int)
"""
        result = inspect_strategy_source(source)
        assert "CENTERED_ROLLING" in confirmed_codes(result)

    def test_certainty_is_confirmed(self):
        source = "x = df.rolling(10, center=True).mean()"
        result = inspect_strategy_source(source)
        finding = next(f for f in result.findings if f.code == "CENTERED_ROLLING")
        assert finding.certainty == "confirmed"

    def test_center_false_is_not_flagged(self):
        source = "x = df['Close'].rolling(20, center=False).mean()"
        result = inspect_strategy_source(source)
        assert "CENTERED_ROLLING" not in confirmed_codes(result)

    def test_rolling_without_center_is_not_flagged(self):
        source = "x = df['Close'].rolling(20).mean()"
        result = inspect_strategy_source(source)
        assert "CENTERED_ROLLING" not in confirmed_codes(result)


# ---------------------------------------------------------------------------
# Test: GLOBAL_STATISTIC (suspicious)
# ---------------------------------------------------------------------------

class TestGlobalStatistic:

    @pytest.mark.parametrize("method", ["mean", "std", "var", "max", "min", "sum", "median"])
    def test_bare_aggregate_method_flagged(self, method):
        source = f"val = df['Close'].{method}()"
        result = inspect_strategy_source(source)
        assert "GLOBAL_STATISTIC" in suspicious_codes(result), (
            f".{method}() on full series should be flagged as suspicious"
        )

    def test_rolling_mean_not_flagged(self):
        source = "val = df['Close'].rolling(20).mean()"
        result = inspect_strategy_source(source)
        assert "GLOBAL_STATISTIC" not in suspicious_codes(result)

    def test_ewm_mean_not_flagged(self):
        source = "val = df['Close'].ewm(span=20).mean()"
        result = inspect_strategy_source(source)
        assert "GLOBAL_STATISTIC" not in suspicious_codes(result)

    def test_certainty_is_suspicious(self):
        source = "val = df['Close'].mean()"
        result = inspect_strategy_source(source)
        finding = next(f for f in result.findings if f.code == "GLOBAL_STATISTIC")
        assert finding.certainty == "suspicious"


# ---------------------------------------------------------------------------
# Test: EXPANDING_WINDOW (suspicious)
# ---------------------------------------------------------------------------

class TestExpandingWindow:

    def test_expanding_mean_flagged(self):
        source = "val = df['Close'].expanding().mean()"
        result = inspect_strategy_source(source)
        assert "EXPANDING_WINDOW" in suspicious_codes(result)

    def test_expanding_std_flagged(self):
        source = "val = df['Close'].expanding().std()"
        result = inspect_strategy_source(source)
        assert "EXPANDING_WINDOW" in suspicious_codes(result)

    def test_expanding_is_suspicious_not_confirmed(self):
        source = "val = df['Close'].expanding().mean()"
        result = inspect_strategy_source(source)
        assert "EXPANDING_WINDOW" not in confirmed_codes(result)
        assert "EXPANDING_WINDOW" in suspicious_codes(result)


# ---------------------------------------------------------------------------
# Test: NEGATIVE_ILOC (suspicious)
# ---------------------------------------------------------------------------

class TestNegativeIloc:

    def test_iloc_minus_one_flagged(self):
        source = "last_row = df.iloc[-1]"
        result = inspect_strategy_source(source)
        assert "NEGATIVE_ILOC" in suspicious_codes(result)

    def test_iloc_minus_n_flagged(self):
        source = "last_rows = df.iloc[-5]"
        result = inspect_strategy_source(source)
        assert "NEGATIVE_ILOC" in suspicious_codes(result)

    def test_iloc_positive_not_flagged(self):
        source = "first_row = df.iloc[0]"
        result = inspect_strategy_source(source)
        assert "NEGATIVE_ILOC" not in suspicious_codes(result)

    def test_iloc_is_suspicious_not_confirmed(self):
        source = "x = df.iloc[-1]"
        result = inspect_strategy_source(source)
        assert "NEGATIVE_ILOC" not in confirmed_codes(result)
        assert "NEGATIVE_ILOC" in suspicious_codes(result)


# ---------------------------------------------------------------------------
# Test: parse error handling
# ---------------------------------------------------------------------------

class TestParseErrors:

    def test_invalid_syntax_sets_parse_error(self):
        source = "def broken(: this is not valid python"
        result = inspect_strategy_source(source)
        assert result.parse_error is not None
        assert result.is_parseable is False

    def test_invalid_syntax_has_no_findings(self):
        source = "def f(: ..."
        result = inspect_strategy_source(source)
        assert result.findings == []

    def test_empty_source_is_parseable(self):
        result = inspect_strategy_source("")
        assert result.is_parseable is True
        assert result.findings == []

    def test_whitespace_only_is_parseable(self):
        result = inspect_strategy_source("   \n\n   ")
        assert result.is_parseable is True


# ---------------------------------------------------------------------------
# Test: InspectionResult properties
# ---------------------------------------------------------------------------

class TestInspectionResultProperties:

    def test_confirmed_property_filters_correctly(self):
        source = """
a = df['Close'].shift(-1)          # confirmed
b = df['Close'].mean()             # suspicious
"""
        result = inspect_strategy_source(source)
        assert all(f.certainty == "confirmed" for f in result.confirmed)
        assert all(f.certainty == "suspicious" for f in result.suspicious)

    def test_has_confirmed_violations_true(self):
        result = inspect_strategy_source("x = df.shift(-1)")
        assert result.has_confirmed_violations is True

    def test_has_confirmed_violations_false(self):
        result = inspect_strategy_source("x = df['Close'].mean()")
        assert result.has_confirmed_violations is False

    def test_source_line_count(self):
        source = "a = 1\nb = 2\nc = 3"
        result = inspect_strategy_source(source)
        assert result.source_line_count == 3


# ---------------------------------------------------------------------------
# Test: multiple issues in one source
# ---------------------------------------------------------------------------

class TestMultipleIssues:

    PROBLEMATIC_SOURCE = """
def generate_signals(df):
    # Problem 1: negative shift
    future = df['Close'].shift(-1)

    # Problem 2: centered rolling
    smooth = df['Close'].rolling(window=10, center=True).mean()

    # Problem 3: global mean
    global_avg = df['Close'].mean()

    return (future > global_avg).astype(int)
"""

    def test_all_three_issues_detected(self):
        result = inspect_strategy_source(self.PROBLEMATIC_SOURCE)
        codes = [f.code for f in result.findings]
        assert "NEGATIVE_SHIFT" in codes
        assert "CENTERED_ROLLING" in codes
        assert "GLOBAL_STATISTIC" in codes

    def test_confirmed_and_suspicious_both_present(self):
        result = inspect_strategy_source(self.PROBLEMATIC_SOURCE)
        assert len(result.confirmed) >= 2   # NEGATIVE_SHIFT + CENTERED_ROLLING
        assert len(result.suspicious) >= 1  # GLOBAL_STATISTIC


# ---------------------------------------------------------------------------
# Test: InspectionFinding structure
# ---------------------------------------------------------------------------

class TestInspectionFindingStructure:

    def test_finding_str_contains_certainty_and_code(self):
        f = InspectionFinding(
            code="NEGATIVE_SHIFT",
            certainty="confirmed",
            description="desc",
            evidence="ev",
            line_number=5,
            recommendation="rec",
        )
        s = str(f)
        assert "CONFIRMED" in s
        assert "NEGATIVE_SHIFT" in s
        assert "5" in s

    def test_finding_has_all_required_fields(self):
        source = "x = df.shift(-1)"
        result = inspect_strategy_source(source)
        f = result.findings[0]
        assert f.code
        assert f.certainty in ("confirmed", "suspicious")
        assert f.description
        assert f.evidence
        assert isinstance(f.line_number, int)
        assert f.recommendation
