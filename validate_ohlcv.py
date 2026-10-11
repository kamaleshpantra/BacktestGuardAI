"""
validate_ohlcv.py
-----------------
Validates an OHLCV (Open, High, Low, Close, Volume) DataFrame before it
enters the backtesting engine.

Real market data arrives as a structured table, not a plain list of numbers.
This module checks the shape, content, and internal consistency of that table.

Two concepts are introduced here:

1. Structured findings
   Instead of returning plain error strings, each problem is represented as
   a ValidationFinding with a code, severity, description, evidence, and
   recommendation. This makes it easier to filter by severity, display
   findings in a report, or pass them to the LLM explainer later.

2. Severity levels
   - "error"   : the data cannot be used for a meaningful backtest.
   - "warning" : the data has a potential problem the user should investigate,
                 but the backtest could still run.

Checks performed
----------------
- Input is a pandas DataFrame.
- Required columns are present.
- Index is a DatetimeIndex (timestamps).
- No duplicate timestamps.
- Timestamps are sorted in ascending order.
- No missing (NaN) values in price and volume columns.
- All price values are finite and strictly positive.
- Volume values are finite and non-negative.
- OHLC relationships are internally consistent:
    High >= Open, High >= Close
    Low  <= Open, Low  <= Close
    High >= Low
- At least a minimum number of rows exist (configurable).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

# Columns that every OHLCV dataset must contain.
REQUIRED_COLUMNS: list[str] = ["Open", "High", "Low", "Close", "Volume"]

# Price-only columns (must be strictly positive).
PRICE_COLUMNS: list[str] = ["Open", "High", "Low", "Close"]

# Default minimum number of rows needed for a meaningful backtest.
# Fewer rows than this produces a warning, not an error, because the
# user may be deliberately testing with a tiny dataset.
DEFAULT_MIN_ROWS: int = 30


@dataclass
class ValidationFinding:
    """
    A single finding produced by the OHLCV validator.

    Attributes
    ----------
    code : str
        A short, stable identifier for this type of finding.
        Examples: "MISSING_COLUMNS", "DUPLICATE_TIMESTAMPS".
    severity : str
        Either "error" (data unusable) or "warning" (investigate further).
    description : str
        A plain-language explanation of the problem.
    evidence : str
        Specific details: which rows, which values, how many occurrences.
    recommendation : str
        A practical suggestion for fixing the problem.
    """

    code: str
    severity: str
    description: str
    evidence: str
    recommendation: str

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.code}: {self.description}"


@dataclass
class OHLCVValidationResult:
    """
    The complete output of validate_ohlcv().

    Attributes
    ----------
    passed : bool
        True when no "error"-severity findings were produced.
        A result with only warnings is still considered passed.
    findings : list[ValidationFinding]
        All findings, ordered by the check that produced them.
    row_count : int
        Number of rows in the dataset (-1 if the input was not a DataFrame).
    column_count : int
        Number of columns in the dataset (-1 if the input was not a DataFrame).
    """

    passed: bool
    findings: list[ValidationFinding] = field(default_factory=list)
    row_count: int = -1
    column_count: int = -1

    @property
    def errors(self) -> list[ValidationFinding]:
        """Findings with severity == 'error'."""
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> list[ValidationFinding]:
        """Findings with severity == 'warning'."""
        return [f for f in self.findings if f.severity == "warning"]


def validate_ohlcv(
    df: object,
    min_rows: int = DEFAULT_MIN_ROWS,
) -> OHLCVValidationResult:
    """
    Validate an OHLCV DataFrame and return a structured result.

    Parameters
    ----------
    df : object
        The value to validate. Expected to be a pandas DataFrame with columns
        Open, High, Low, Close, Volume and a DatetimeIndex.
    min_rows : int
        Minimum number of rows required. Fewer rows produce a warning.
        Default is 30.

    Returns
    -------
    OHLCVValidationResult
        Contains a passed flag, a list of findings, and basic shape info.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> dates = pd.date_range("2024-01-01", periods=50, freq="B")
    >>> df = pd.DataFrame({
    ...     "Open": 100.0, "High": 102.0, "Low": 99.0,
    ...     "Close": 101.0, "Volume": 1_000_000
    ... }, index=dates)
    >>> result = validate_ohlcv(df)
    >>> result.passed
    True
    """
    findings: list[ValidationFinding] = []

    # ------------------------------------------------------------------
    # Check 1: input must be a DataFrame
    # ------------------------------------------------------------------
    if not isinstance(df, pd.DataFrame):
        findings.append(
            ValidationFinding(
                code="NOT_A_DATAFRAME",
                severity="error",
                description=(
                    f"Expected a pandas DataFrame, got {type(df).__name__}."
                ),
                evidence=f"type(df) = {type(df).__name__}",
                recommendation=(
                    "Pass a pandas DataFrame with columns "
                    f"{REQUIRED_COLUMNS} and a DatetimeIndex."
                ),
            )
        )
        return OHLCVValidationResult(passed=False, findings=findings)

    row_count = len(df)
    column_count = len(df.columns)

    # ------------------------------------------------------------------
    # Check 2: required columns must be present
    # ------------------------------------------------------------------
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        findings.append(
            ValidationFinding(
                code="MISSING_COLUMNS",
                severity="error",
                description=(
                    f"{len(missing_cols)} required column(s) are absent."
                ),
                evidence=f"Missing: {missing_cols}",
                recommendation=(
                    f"Ensure the DataFrame contains all of: {REQUIRED_COLUMNS}."
                ),
            )
        )
        # Cannot check column contents without the columns — return early.
        return OHLCVValidationResult(
            passed=False,
            findings=findings,
            row_count=row_count,
            column_count=column_count,
        )

    # ------------------------------------------------------------------
    # Check 3: index must be a DatetimeIndex
    # ------------------------------------------------------------------
    if not isinstance(df.index, pd.DatetimeIndex):
        findings.append(
            ValidationFinding(
                code="INVALID_INDEX_TYPE",
                severity="error",
                description=(
                    "The DataFrame index must be a DatetimeIndex. "
                    f"Got {type(df.index).__name__} instead."
                ),
                evidence=f"type(df.index) = {type(df.index).__name__}",
                recommendation=(
                    "Convert the index with: "
                    "df.index = pd.to_datetime(df.index)"
                ),
            )
        )

    # ------------------------------------------------------------------
    # Check 4: duplicate timestamps
    # ------------------------------------------------------------------
    if isinstance(df.index, pd.DatetimeIndex):
        duplicate_mask = df.index.duplicated(keep=False)
        n_duplicates = int(duplicate_mask.sum())
        if n_duplicates > 0:
            sample = df.index[duplicate_mask][:5].tolist()
            findings.append(
                ValidationFinding(
                    code="DUPLICATE_TIMESTAMPS",
                    severity="error",
                    description=(
                        f"{n_duplicates} row(s) share a timestamp with "
                        "another row. Each timestamp must be unique."
                    ),
                    evidence=f"Sample duplicate timestamps: {sample}",
                    recommendation=(
                        "Remove or merge duplicate rows before running "
                        "the backtest. Check: df[df.index.duplicated()]"
                    ),
                )
            )

        # ------------------------------------------------------------------
        # Check 5: timestamps must be sorted ascending
        # ------------------------------------------------------------------
        if not df.index.is_monotonic_increasing:
            first_bad_pos = _first_unsorted_position(df.index)
            findings.append(
                ValidationFinding(
                    code="UNSORTED_TIMESTAMPS",
                    severity="error",
                    description=(
                        "Timestamps are not sorted in ascending order. "
                        "The backtesting engine assumes chronological data."
                    ),
                    evidence=(
                        f"First out-of-order position: index {first_bad_pos} "
                        f"({df.index[first_bad_pos]})"
                    ),
                    recommendation="Sort with: df = df.sort_index()",
                )
            )

    # ------------------------------------------------------------------
    # Check 6: missing values (NaN) in OHLCV columns
    # ------------------------------------------------------------------
    for col in REQUIRED_COLUMNS:
        n_missing = int(df[col].isna().sum())
        if n_missing > 0:
            sample_idx = df.index[df[col].isna()][:5].tolist()
            findings.append(
                ValidationFinding(
                    code="MISSING_VALUES",
                    severity="error",
                    description=(
                        f"Column '{col}' contains {n_missing} missing "
                        f"(NaN) value(s)."
                    ),
                    evidence=f"Sample affected timestamps: {sample_idx}",
                    recommendation=(
                        f"Drop rows with: df = df.dropna(subset=['{col}']) "
                        f"or fill forward with: df['{col}'] = "
                        f"df['{col}'].ffill()"
                    ),
                )
            )

    # ------------------------------------------------------------------
    # Check 7: price values must be finite and strictly positive
    # ------------------------------------------------------------------
    for col in PRICE_COLUMNS:
        series = df[col].dropna()  # NaN already reported above

        non_finite_mask = ~series.apply(math.isfinite)
        n_non_finite = int(non_finite_mask.sum())
        if n_non_finite > 0:
            sample_idx = series.index[non_finite_mask][:5].tolist()
            findings.append(
                ValidationFinding(
                    code="NON_FINITE_PRICES",
                    severity="error",
                    description=(
                        f"Column '{col}' contains {n_non_finite} non-finite "
                        "value(s) (inf or -inf)."
                    ),
                    evidence=f"Sample affected timestamps: {sample_idx}",
                    recommendation=(
                        f"Investigate the source of infinite values in '{col}'. "
                        "They usually indicate a division-by-zero in a "
                        "data-processing step."
                    ),
                )
            )

        non_positive_mask = series <= 0
        n_non_positive = int(non_positive_mask.sum())
        if n_non_positive > 0:
            sample_idx = series.index[non_positive_mask][:5].tolist()
            findings.append(
                ValidationFinding(
                    code="NON_POSITIVE_PRICES",
                    severity="error",
                    description=(
                        f"Column '{col}' contains {n_non_positive} value(s) "
                        "that are zero or negative. Stock prices must be "
                        "strictly positive."
                    ),
                    evidence=f"Sample affected timestamps: {sample_idx}",
                    recommendation=(
                        f"Remove or investigate rows where {col} <= 0."
                    ),
                )
            )

    # ------------------------------------------------------------------
    # Check 8: volume must be finite and non-negative
    # ------------------------------------------------------------------
    vol = df["Volume"].dropna()

    non_finite_vol = ~vol.apply(math.isfinite)
    if int(non_finite_vol.sum()) > 0:
        n = int(non_finite_vol.sum())
        findings.append(
            ValidationFinding(
                code="NON_FINITE_VOLUME",
                severity="error",
                description=f"Column 'Volume' contains {n} non-finite value(s).",
                evidence=f"Count: {n}",
                recommendation="Investigate and remove or replace infinite volume values.",
            )
        )

    negative_vol = vol < 0
    if int(negative_vol.sum()) > 0:
        n = int(negative_vol.sum())
        sample_idx = vol.index[negative_vol][:5].tolist()
        findings.append(
            ValidationFinding(
                code="NEGATIVE_VOLUME",
                severity="error",
                description=f"Column 'Volume' contains {n} negative value(s).",
                evidence=f"Sample affected timestamps: {sample_idx}",
                recommendation="Volume cannot be negative. Remove or correct these rows.",
            )
        )

    # ------------------------------------------------------------------
    # Check 9: OHLC internal consistency
    # ------------------------------------------------------------------
    # These checks only make sense when all four columns have usable data.
    ohlc_clean = df[PRICE_COLUMNS].dropna()
    ohlc_clean = ohlc_clean[ohlc_clean.apply(
        lambda row: all(math.isfinite(v) and v > 0 for v in row), axis=1
    )]

    if len(ohlc_clean) > 0:
        # High must be >= Open and >= Close
        high_lt_open = ohlc_clean["High"] < ohlc_clean["Open"]
        high_lt_close = ohlc_clean["High"] < ohlc_clean["Close"]
        high_violations = high_lt_open | high_lt_close
        n_high = int(high_violations.sum())
        if n_high > 0:
            sample_idx = ohlc_clean.index[high_violations][:5].tolist()
            findings.append(
                ValidationFinding(
                    code="OHLC_HIGH_VIOLATION",
                    severity="error",
                    description=(
                        f"{n_high} row(s) have a High price lower than "
                        "Open or Close. High must be the highest price of "
                        "the period."
                    ),
                    evidence=f"Sample affected timestamps: {sample_idx}",
                    recommendation=(
                        "Verify the data source. These rows may indicate "
                        "swapped columns or corrupted records."
                    ),
                )
            )

        # Low must be <= Open and <= Close
        low_gt_open = ohlc_clean["Low"] > ohlc_clean["Open"]
        low_gt_close = ohlc_clean["Low"] > ohlc_clean["Close"]
        low_violations = low_gt_open | low_gt_close
        n_low = int(low_violations.sum())
        if n_low > 0:
            sample_idx = ohlc_clean.index[low_violations][:5].tolist()
            findings.append(
                ValidationFinding(
                    code="OHLC_LOW_VIOLATION",
                    severity="error",
                    description=(
                        f"{n_low} row(s) have a Low price higher than "
                        "Open or Close. Low must be the lowest price of "
                        "the period."
                    ),
                    evidence=f"Sample affected timestamps: {sample_idx}",
                    recommendation=(
                        "Verify the data source. These rows may indicate "
                        "swapped columns or corrupted records."
                    ),
                )
            )

        # High must always be >= Low
        high_lt_low = ohlc_clean["High"] < ohlc_clean["Low"]
        n_hl = int(high_lt_low.sum())
        if n_hl > 0:
            sample_idx = ohlc_clean.index[high_lt_low][:5].tolist()
            findings.append(
                ValidationFinding(
                    code="HIGH_BELOW_LOW",
                    severity="error",
                    description=(
                        f"{n_hl} row(s) have High < Low, which is "
                        "physically impossible."
                    ),
                    evidence=f"Sample affected timestamps: {sample_idx}",
                    recommendation=(
                        "These rows are almost certainly corrupted. "
                        "Remove them before running any analysis."
                    ),
                )
            )

    # ------------------------------------------------------------------
    # Check 10: minimum row count
    # ------------------------------------------------------------------
    if row_count < min_rows:
        findings.append(
            ValidationFinding(
                code="INSUFFICIENT_ROWS",
                severity="warning",
                description=(
                    f"Dataset has only {row_count} row(s). "
                    f"At least {min_rows} are recommended for a "
                    "meaningful backtest."
                ),
                evidence=f"row_count = {row_count}, min_rows = {min_rows}",
                recommendation=(
                    "Use a longer historical period, or set min_rows to a "
                    "smaller value if you are intentionally testing with "
                    "short data."
                ),
            )
        )

    # ------------------------------------------------------------------
    # Determine overall pass/fail
    # A result passes when there are no error-severity findings.
    # Warnings alone do not cause a failure.
    # ------------------------------------------------------------------
    has_errors = any(f.severity == "error" for f in findings)
    return OHLCVValidationResult(
        passed=not has_errors,
        findings=findings,
        row_count=row_count,
        column_count=column_count,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _first_unsorted_position(index: pd.DatetimeIndex) -> int:
    """Return the position of the first timestamp that breaks ascending order."""
    for i in range(1, len(index)):
        if index[i] <= index[i - 1]:
            return i
    return -1  # should not happen if called only when not monotonic
