"""
test_validate_ohlcv.py
----------------------
Automated tests for validate_ohlcv() and ValidationFinding.

Each test class targets one specific check so failures are easy to locate.
A shared fixture builds a clean 50-row OHLCV DataFrame that all tests can
start from and selectively corrupt.
"""


import pandas as pd
import pytest

from validate_ohlcv import (
    DEFAULT_MIN_ROWS,
    REQUIRED_COLUMNS,
    OHLCVValidationResult,
    ValidationFinding,
    validate_ohlcv,
)

# ---------------------------------------------------------------------------
# Shared fixture: a clean, valid OHLCV DataFrame
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_df() -> pd.DataFrame:
    """
    Return a minimal valid OHLCV DataFrame with 50 business-day rows.

    All prices are consistent: High >= Open/Close, Low <= Open/Close.
    Volume is positive. Index is a DatetimeIndex sorted ascending.
    """
    dates = pd.date_range("2024-01-01", periods=50, freq="B")
    return pd.DataFrame(
        {
            "Open":   100.0,
            "High":   102.0,
            "Low":     98.0,
            "Close":  101.0,
            "Volume": 1_000_000,
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def has_code(result: OHLCVValidationResult, code: str) -> bool:
    """True when any finding in result has the given code."""
    return any(f.code == code for f in result.findings)


def error_codes(result: OHLCVValidationResult) -> list[str]:
    return [f.code for f in result.errors]


def warning_codes(result: OHLCVValidationResult) -> list[str]:
    return [f.code for f in result.warnings]


# ---------------------------------------------------------------------------
# Test: clean data passes
# ---------------------------------------------------------------------------

class TestCleanDataPasses:

    def test_clean_df_passes(self, clean_df):
        result = validate_ohlcv(clean_df)
        assert result.passed is True
        assert result.findings == []

    def test_clean_df_row_count(self, clean_df):
        result = validate_ohlcv(clean_df)
        assert result.row_count == 50

    def test_clean_df_column_count(self, clean_df):
        result = validate_ohlcv(clean_df)
        assert result.column_count == 5

    def test_extra_columns_do_not_cause_failure(self, clean_df):
        clean_df["ExtraCol"] = 0.0
        result = validate_ohlcv(clean_df)
        assert result.passed is True


# ---------------------------------------------------------------------------
# Test: NOT_A_DATAFRAME
# ---------------------------------------------------------------------------

class TestNotADataFrame:

    @pytest.mark.parametrize("bad_input", [
        None, 42, "data", [1, 2, 3], {"Open": 100}
    ])
    def test_non_dataframe_input(self, bad_input):
        result = validate_ohlcv(bad_input)
        assert result.passed is False
        assert has_code(result, "NOT_A_DATAFRAME")

    def test_non_dataframe_returns_immediately(self):
        # Only one finding should be produced — no further checks run.
        result = validate_ohlcv(42)
        assert len(result.findings) == 1

    def test_non_dataframe_row_count_is_minus_one(self):
        result = validate_ohlcv(None)
        assert result.row_count == -1


# ---------------------------------------------------------------------------
# Test: MISSING_COLUMNS
# ---------------------------------------------------------------------------

class TestMissingColumns:

    def test_missing_one_column(self, clean_df):
        df = clean_df.drop(columns=["Volume"])
        result = validate_ohlcv(df)
        assert has_code(result, "MISSING_COLUMNS")
        assert result.passed is False

    def test_missing_all_ohlcv_columns(self):
        df = pd.DataFrame({"Price": [100.0]})
        result = validate_ohlcv(df)
        assert has_code(result, "MISSING_COLUMNS")

    def test_missing_columns_returns_early(self, clean_df):
        # When columns are missing we cannot check their content.
        # The result should contain MISSING_COLUMNS and nothing else.
        df = clean_df.drop(columns=["Open", "Close"])
        result = validate_ohlcv(df)
        assert len(result.findings) == 1
        assert result.findings[0].code == "MISSING_COLUMNS"


# ---------------------------------------------------------------------------
# Test: INVALID_INDEX_TYPE
# ---------------------------------------------------------------------------

class TestInvalidIndexType:

    def test_integer_index(self, clean_df):
        clean_df.index = range(len(clean_df))
        result = validate_ohlcv(clean_df)
        assert has_code(result, "INVALID_INDEX_TYPE")
        assert result.passed is False

    def test_string_index(self, clean_df):
        clean_df.index = [str(i) for i in range(len(clean_df))]
        result = validate_ohlcv(clean_df)
        assert has_code(result, "INVALID_INDEX_TYPE")


# ---------------------------------------------------------------------------
# Test: DUPLICATE_TIMESTAMPS
# ---------------------------------------------------------------------------

class TestDuplicateTimestamps:

    def test_duplicate_timestamps(self, clean_df):
        # Repeat the first date at position 5.
        dup_index = clean_df.index.tolist()
        dup_index[5] = dup_index[0]
        clean_df.index = pd.DatetimeIndex(dup_index)
        result = validate_ohlcv(clean_df)
        assert has_code(result, "DUPLICATE_TIMESTAMPS")
        assert result.passed is False

    def test_no_false_positive_on_unique_dates(self, clean_df):
        assert not has_code(validate_ohlcv(clean_df), "DUPLICATE_TIMESTAMPS")


# ---------------------------------------------------------------------------
# Test: UNSORTED_TIMESTAMPS
# ---------------------------------------------------------------------------

class TestUnsortedTimestamps:

    def test_reversed_index(self, clean_df):
        clean_df.index = clean_df.index[::-1]
        result = validate_ohlcv(clean_df)
        assert has_code(result, "UNSORTED_TIMESTAMPS")
        assert result.passed is False

    def test_single_swap(self, clean_df):
        idx = clean_df.index.tolist()
        idx[3], idx[4] = idx[4], idx[3]  # swap two adjacent dates
        clean_df.index = pd.DatetimeIndex(idx)
        result = validate_ohlcv(clean_df)
        assert has_code(result, "UNSORTED_TIMESTAMPS")

    def test_sorted_index_does_not_trigger(self, clean_df):
        assert not has_code(validate_ohlcv(clean_df), "UNSORTED_TIMESTAMPS")


# ---------------------------------------------------------------------------
# Test: MISSING_VALUES
# ---------------------------------------------------------------------------

class TestMissingValues:

    @pytest.mark.parametrize("col", REQUIRED_COLUMNS)
    def test_nan_in_each_column(self, clean_df, col):
        clean_df.loc[clean_df.index[10], col] = float("nan")
        result = validate_ohlcv(clean_df)
        assert has_code(result, "MISSING_VALUES")
        assert result.passed is False

    def test_nan_reports_correct_column(self, clean_df):
        clean_df.loc[clean_df.index[0], "Close"] = float("nan")
        result = validate_ohlcv(clean_df)
        nan_findings = [f for f in result.findings if f.code == "MISSING_VALUES"]
        assert any("Close" in f.description for f in nan_findings)

    def test_no_false_positive_clean_data(self, clean_df):
        assert not has_code(validate_ohlcv(clean_df), "MISSING_VALUES")


# ---------------------------------------------------------------------------
# Test: NON_FINITE_PRICES
# ---------------------------------------------------------------------------

class TestNonFinitePrices:

    @pytest.mark.parametrize("col", ["Open", "High", "Low", "Close"])
    def test_inf_in_price_column(self, clean_df, col):
        clean_df.loc[clean_df.index[2], col] = float("inf")
        result = validate_ohlcv(clean_df)
        assert has_code(result, "NON_FINITE_PRICES")

    def test_negative_inf(self, clean_df):
        clean_df.loc[clean_df.index[0], "Open"] = float("-inf")
        result = validate_ohlcv(clean_df)
        assert has_code(result, "NON_FINITE_PRICES")


# ---------------------------------------------------------------------------
# Test: NON_POSITIVE_PRICES
# ---------------------------------------------------------------------------

class TestNonPositivePrices:

    @pytest.mark.parametrize("col", ["Open", "High", "Low", "Close"])
    def test_zero_price(self, clean_df, col):
        clean_df.loc[clean_df.index[1], col] = 0.0
        result = validate_ohlcv(clean_df)
        assert has_code(result, "NON_POSITIVE_PRICES")

    @pytest.mark.parametrize("col", ["Open", "High", "Low", "Close"])
    def test_negative_price(self, clean_df, col):
        clean_df.loc[clean_df.index[1], col] = -1.0
        result = validate_ohlcv(clean_df)
        assert has_code(result, "NON_POSITIVE_PRICES")


# ---------------------------------------------------------------------------
# Test: volume checks
# ---------------------------------------------------------------------------

class TestVolumeChecks:

    def test_negative_volume(self, clean_df):
        clean_df.loc[clean_df.index[5], "Volume"] = -500
        result = validate_ohlcv(clean_df)
        assert has_code(result, "NEGATIVE_VOLUME")
        assert result.passed is False

    def test_zero_volume_is_allowed(self, clean_df):
        # Zero volume is valid (e.g. a holiday with no trades in some feeds).
        clean_df.loc[clean_df.index[5], "Volume"] = 0
        result = validate_ohlcv(clean_df)
        assert not has_code(result, "NEGATIVE_VOLUME")

    def test_infinite_volume(self, clean_df):
        clean_df.loc[clean_df.index[3], "Volume"] = float("inf")
        result = validate_ohlcv(clean_df)
        assert has_code(result, "NON_FINITE_VOLUME")


# ---------------------------------------------------------------------------
# Test: OHLC relationship checks
# ---------------------------------------------------------------------------

class TestOHLCRelationships:

    def test_high_below_open(self, clean_df):
        # High < Open violates: High must be the period's highest price.
        clean_df.loc[clean_df.index[0], "High"] = 50.0  # Open is 100
        result = validate_ohlcv(clean_df)
        assert has_code(result, "OHLC_HIGH_VIOLATION")

    def test_high_below_close(self, clean_df):
        clean_df.loc[clean_df.index[0], "High"] = 50.0  # Close is 101
        result = validate_ohlcv(clean_df)
        assert has_code(result, "OHLC_HIGH_VIOLATION")

    def test_low_above_open(self, clean_df):
        clean_df.loc[clean_df.index[0], "Low"] = 200.0  # Open is 100
        result = validate_ohlcv(clean_df)
        assert has_code(result, "OHLC_LOW_VIOLATION")

    def test_low_above_close(self, clean_df):
        clean_df.loc[clean_df.index[0], "Low"] = 200.0  # Close is 101
        result = validate_ohlcv(clean_df)
        assert has_code(result, "OHLC_LOW_VIOLATION")

    def test_high_below_low(self, clean_df):
        # This is physically impossible but can appear in corrupted data.
        clean_df.loc[clean_df.index[0], "High"] = 10.0
        clean_df.loc[clean_df.index[0], "Low"] = 200.0
        result = validate_ohlcv(clean_df)
        assert has_code(result, "HIGH_BELOW_LOW")

    def test_equal_ohlc_values_are_valid(self, clean_df):
        # e.g. circuit-breaker day: Open == High == Low == Close
        clean_df.loc[clean_df.index[0], ["Open", "High", "Low", "Close"]] = 100.0
        result = validate_ohlcv(clean_df)
        assert result.passed is True


# ---------------------------------------------------------------------------
# Test: INSUFFICIENT_ROWS
# ---------------------------------------------------------------------------

class TestInsufficientRows:

    def test_below_default_min_rows(self):
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        df = pd.DataFrame(
            {"Open": 100.0, "High": 102.0, "Low": 98.0,
             "Close": 101.0, "Volume": 1_000_000},
            index=dates,
        )
        result = validate_ohlcv(df)
        assert has_code(result, "INSUFFICIENT_ROWS")
        assert result.passed is True  # warning only, not an error

    def test_warning_not_error(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        df = pd.DataFrame(
            {"Open": 100.0, "High": 102.0, "Low": 98.0,
             "Close": 101.0, "Volume": 1_000_000},
            index=dates,
        )
        result = validate_ohlcv(df)
        assert "INSUFFICIENT_ROWS" in warning_codes(result)
        assert "INSUFFICIENT_ROWS" not in error_codes(result)

    def test_custom_min_rows(self, clean_df):
        # 50 rows is below 100 — should warn with min_rows=100
        result = validate_ohlcv(clean_df, min_rows=100)
        assert has_code(result, "INSUFFICIENT_ROWS")

    def test_exactly_at_min_rows_does_not_warn(self):
        dates = pd.date_range("2024-01-01", periods=DEFAULT_MIN_ROWS, freq="B")
        df = pd.DataFrame(
            {"Open": 100.0, "High": 102.0, "Low": 98.0,
             "Close": 101.0, "Volume": 1_000_000},
            index=dates,
        )
        result = validate_ohlcv(df)
        assert not has_code(result, "INSUFFICIENT_ROWS")


# ---------------------------------------------------------------------------
# Test: ValidationFinding structure
# ---------------------------------------------------------------------------

class TestValidationFindingStructure:

    def test_finding_has_required_fields(self):
        f = ValidationFinding(
            code="TEST_CODE",
            severity="error",
            description="Test description.",
            evidence="Some evidence.",
            recommendation="Do something.",
        )
        assert f.code == "TEST_CODE"
        assert f.severity == "error"
        assert f.description == "Test description."
        assert f.evidence == "Some evidence."
        assert f.recommendation == "Do something."

    def test_finding_str_contains_severity_and_code(self):
        f = ValidationFinding(
            code="MISSING_COLUMNS",
            severity="error",
            description="desc",
            evidence="ev",
            recommendation="rec",
        )
        s = str(f)
        assert "ERROR" in s
        assert "MISSING_COLUMNS" in s

    def test_result_errors_property(self, clean_df):
        clean_df.loc[clean_df.index[0], "Close"] = -1.0
        result = validate_ohlcv(clean_df)
        assert all(f.severity == "error" for f in result.errors)

    def test_result_warnings_property(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        df = pd.DataFrame(
            {"Open": 100.0, "High": 102.0, "Low": 98.0,
             "Close": 101.0, "Volume": 1_000_000},
            index=dates,
        )
        result = validate_ohlcv(df)
        assert all(f.severity == "warning" for f in result.warnings)


# ---------------------------------------------------------------------------
# Test: multiple simultaneous problems
# ---------------------------------------------------------------------------

class TestMultipleProblems:

    def test_collects_multiple_errors(self, clean_df):
        # Introduce three separate problems.
        clean_df.loc[clean_df.index[0], "Close"] = -5.0   # NON_POSITIVE_PRICES
        clean_df.loc[clean_df.index[1], "Open"] = float("nan")  # MISSING_VALUES
        clean_df.loc[clean_df.index[2], "High"] = 10.0    # OHLC_HIGH_VIOLATION
        result = validate_ohlcv(clean_df)
        assert result.passed is False
        assert len(result.errors) >= 3

    def test_warnings_and_errors_coexist(self):
        # 10 rows (warning) and a bad price (error).
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        df = pd.DataFrame(
            {"Open": 100.0, "High": 102.0, "Low": 98.0,
             "Close": -1.0, "Volume": 1_000_000},
            index=dates,
        )
        result = validate_ohlcv(df)
        assert len(result.errors) > 0
        assert len(result.warnings) > 0
        assert result.passed is False
