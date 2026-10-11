"""
inspection/ast_inspector.py
----------------------------
Inspects Python strategy source code for patterns that may indicate
look-ahead bias or other common backtesting mistakes.

How it works
------------
Python's built-in `ast` module parses source code into a tree structure
(Abstract Syntax Tree) without executing any of it. Each node in the tree
represents a syntactic element — a function call, an attribute access,
a constant value, etc.

We walk the tree and look for specific patterns. When we find one, we
produce an InspectionFinding with a code, certainty level, evidence, and
a recommendation.

Certainty levels
----------------
  "confirmed"  — the pattern is a definite violation of a stated rule.
                 Example: shift(-1) always accesses future data.
  "suspicious" — the pattern is often a mistake but might be intentional.
                 Example: .mean() on a full series leaks future data in most
                 contexts, but could be a one-time calculation done before
                 signal generation starts.

What this inspector can and cannot do
--------------------------------------
CAN detect:
  - Negative shifts (.shift(-N))
  - Centered rolling windows (.rolling(center=True))
  - Aggregate statistics on full series without windowing
    (.mean(), .std(), .max(), .min(), .sum(), .var(), .median())
  - Negative iloc indexing (.iloc[-N])
  - Expanding window aggregates (.expanding().mean(), etc.) — flagged
    suspicious because expanding windows use all past data up to T,
    which can be valid but deserves attention

CANNOT detect:
  - Look-ahead bias hidden inside custom functions passed to .apply()
  - Bias introduced through data preprocessing done before calling the strategy
  - Subtle indexing bugs that depend on the shape of the input DataFrame
  - Any runtime behaviour — only static structure is analysed

This inspector is a helper, not a guarantee. A strategy that passes all
checks may still have look-ahead bias.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

# Aggregate methods that, when applied to a full Series without a rolling
# or expanding window, use data from the entire dataset — including the future.
_GLOBAL_AGGREGATE_METHODS = frozenset(
    {"mean", "std", "var", "max", "min", "sum", "median", "corr", "cov", "kurt", "skew"}
)

# Methods that chain after .expanding() — flagged as suspicious because
# expanding windows grow with each row and include all past data up to T,
# which can be a valid design choice but is worth reviewing.
_EXPANDING_AGGREGATE_METHODS = frozenset({"mean", "std", "var", "max", "min", "sum"})

# Window methods whose results ARE safe (windowed lookback only).
_WINDOWED_METHODS = frozenset({"rolling", "ewm"})


@dataclass
class InspectionFinding:
    """
    A single finding from the AST strategy inspector.

    Attributes
    ----------
    code : str
        Short stable identifier for the type of finding.
    certainty : str
        'confirmed' — definite rule violation.
        'suspicious' — warrants review but may be intentional.
    description : str
        Plain-language explanation of what was found.
    evidence : str
        The specific code fragment and line number.
    line_number : int
        Source line where the pattern was detected (-1 if unknown).
    recommendation : str
        Practical suggestion for fixing or investigating the issue.
    """

    code: str
    certainty: str
    description: str
    evidence: str
    line_number: int
    recommendation: str

    def __str__(self) -> str:
        return (
            f"[{self.certainty.upper()}] {self.code} "
            f"(line {self.line_number}): {self.description}"
        )


@dataclass
class InspectionResult:
    """
    The complete output of inspect_strategy_source().

    Attributes
    ----------
    findings : list[InspectionFinding]
        All findings, in the order the patterns were encountered.
    source_line_count : int
        Number of lines in the inspected source code.
    parse_error : str | None
        Set to an error message if the source code could not be parsed.
        When set, findings will be empty.
    """

    findings: list[InspectionFinding] = field(default_factory=list)
    source_line_count: int = 0
    parse_error: str | None = None

    @property
    def confirmed(self) -> list[InspectionFinding]:
        """Findings with certainty == 'confirmed'."""
        return [f for f in self.findings if f.certainty == "confirmed"]

    @property
    def suspicious(self) -> list[InspectionFinding]:
        """Findings with certainty == 'suspicious'."""
        return [f for f in self.findings if f.certainty == "suspicious"]

    @property
    def has_confirmed_violations(self) -> bool:
        """True when at least one confirmed finding exists."""
        return len(self.confirmed) > 0

    @property
    def is_parseable(self) -> bool:
        """True when the source code was parsed without error."""
        return self.parse_error is None


def inspect_strategy_source(source_code: str) -> InspectionResult:
    """
    Parse and inspect Python source code for look-ahead bias patterns.

    Parameters
    ----------
    source_code : str
        The raw Python source code to inspect. Can be a full module,
        a class definition, or a standalone function.

    Returns
    -------
    InspectionResult
        Contains all findings, source line count, and any parse error.

    Examples
    --------
    >>> source = '''
    ... def generate_signals(df):
    ...     future = df['Close'].shift(-1)  # BUG: looks one day ahead
    ...     return (future > df['Close']).astype(int)
    ... '''
    >>> result = inspect_strategy_source(source)
    >>> result.has_confirmed_violations
    True
    >>> result.confirmed[0].code
    'NEGATIVE_SHIFT'
    """
    line_count = len(source_code.splitlines())

    try:
        tree = ast.parse(source_code)
    except SyntaxError as exc:
        return InspectionResult(
            parse_error=f"SyntaxError at line {exc.lineno}: {exc.msg}",
            source_line_count=line_count,
        )

    visitor = _StrategyVisitor()
    visitor.visit(tree)

    return InspectionResult(
        findings=visitor.findings,
        source_line_count=line_count,
    )


# ---------------------------------------------------------------------------
# Internal AST visitor
# ---------------------------------------------------------------------------

class _StrategyVisitor(ast.NodeVisitor):
    """
    Walks an AST and collects suspicious or confirmed problematic patterns.

    This class is an implementation detail. Callers should use
    inspect_strategy_source() rather than instantiating this directly.
    """

    def __init__(self) -> None:
        self.findings: list[InspectionFinding] = []

    def visit_Call(self, node: ast.Call) -> None:
        """Check every function/method call in the source."""
        self._check_negative_shift(node)
        self._check_centered_rolling(node)
        self._check_global_aggregate(node)
        self._check_expanding_aggregate(node)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        """Check for negative iloc indexing."""
        self._check_negative_iloc(node)
        self.generic_visit(node)

    # ------------------------------------------------------------------
    # Pattern checkers
    # ------------------------------------------------------------------

    def _check_negative_shift(self, node: ast.Call) -> None:
        """
        Detect .shift(-N) where N > 0.

        A negative shift moves the series forward in time, so the value
        at position T becomes the value that was at T+N — future data.
        """
        if not _is_method_call(node, "shift"):
            return

        if not node.args:
            return  # shift() with no arguments defaults to 1 — safe

        first_arg = node.args[0]
        if not _is_negative_number(first_arg):
            return

        self.findings.append(
            InspectionFinding(
                code="NEGATIVE_SHIFT",
                certainty="confirmed",
                description=(
                    ".shift() was called with a negative value. "
                    "A negative shift moves the series forward in time, "
                    "giving the current row access to a future row's value."
                ),
                evidence=f"Line {node.lineno}: .shift({ast.unparse(first_arg)})",
                line_number=node.lineno,
                recommendation=(
                    "Use .shift(1) or a positive value to lag data backward. "
                    "Positive shifts use past data, negative shifts use future data."
                ),
            )
        )

    def _check_centered_rolling(self, node: ast.Call) -> None:
        """
        Detect .rolling(..., center=True).

        A centered rolling window of size N uses (N-1)//2 rows from the
        future to compute the value at each position. This is valid for
        smoothing historical data offline, but is look-ahead bias when
        used inside signal generation.
        """
        if not _is_method_call(node, "rolling"):
            return

        for kw in node.keywords:
            if kw.arg == "center" and isinstance(kw.value, ast.Constant):
                if kw.value.value is True:
                    self.findings.append(
                        InspectionFinding(
                            code="CENTERED_ROLLING",
                            certainty="confirmed",
                            description=(
                                "rolling(center=True) uses future data to compute "
                                "the value at each position. A window of size N "
                                "looks (N-1)//2 rows ahead."
                            ),
                            evidence=(
                                f"Line {node.lineno}: .rolling(..., center=True)"
                            ),
                            line_number=node.lineno,
                            recommendation=(
                                "Remove center=True. The default (center=False) "
                                "only uses past and current data."
                            ),
                        )
                    )

    def _check_global_aggregate(self, node: ast.Call) -> None:
        """
        Detect aggregate methods called directly on a full Series or DataFrame
        without a preceding rolling() or ewm() call.

        For example: df['Close'].mean() computes the mean of the entire
        column, including all future values. Assigning this to a variable
        and comparing it to each row's price constitutes look-ahead bias.

        We allow the aggregate if it is chained after .rolling() or .ewm()
        because those methods only use data up to each row.
        """
        if not isinstance(node.func, ast.Attribute):
            return

        method = node.func.attr
        if method not in _GLOBAL_AGGREGATE_METHODS:
            return

        # The object this method is called on.
        receiver = node.func.value

        # If the receiver is itself a call to rolling() or ewm(), it is safe.
        if isinstance(receiver, ast.Call) and isinstance(receiver.func, ast.Attribute):
            if receiver.func.attr in _WINDOWED_METHODS:
                return

        self.findings.append(
            InspectionFinding(
                code="GLOBAL_STATISTIC",
                certainty="suspicious",
                description=(
                    f".{method}() appears to be called on a full Series or "
                    "DataFrame without a rolling or ewm window. If this "
                    "computes a statistic over the entire dataset, future "
                    "values influence every row — look-ahead bias."
                ),
                evidence=f"Line {node.lineno}: .{method}()",
                line_number=node.lineno,
                recommendation=(
                    f"Use .rolling(window=N).{method}() to compute a "
                    f"backward-looking windowed statistic instead of "
                    f".{method}() on the full column."
                ),
            )
        )

    def _check_expanding_aggregate(self, node: ast.Call) -> None:
        """
        Detect aggregate methods chained after .expanding().

        .expanding() grows the window from the first row to the current row,
        so it never uses future data. However, it means the strategy behaviour
        at early rows depends on very little history, which can produce
        unrealistic signals. We flag it as suspicious for review.
        """
        if not isinstance(node.func, ast.Attribute):
            return

        method = node.func.attr
        if method not in _EXPANDING_AGGREGATE_METHODS:
            return

        receiver = node.func.value
        if not (
            isinstance(receiver, ast.Call)
            and isinstance(receiver.func, ast.Attribute)
            and receiver.func.attr == "expanding"
        ):
            return

        self.findings.append(
            InspectionFinding(
                code="EXPANDING_WINDOW",
                certainty="suspicious",
                description=(
                    f".expanding().{method}() uses all available past data "
                    "up to each row. This does not look into the future, but "
                    "early signals may be based on very few observations, "
                    "making them unreliable."
                ),
                evidence=f"Line {node.lineno}: .expanding().{method}()",
                line_number=node.lineno,
                recommendation=(
                    "Consider using .rolling(window=N) with a fixed lookback "
                    "period for more stable and comparable signals across all rows."
                ),
            )
        )

    def _check_negative_iloc(self, node: ast.Subscript) -> None:
        """
        Detect .iloc[-N] — negative integer indexing from the end of the Series.

        In a strategy function that receives the full DataFrame at once,
        .iloc[-1] accesses the last row of the entire dataset — which is a
        future row for any bar except the final one.
        """
        if not (
            isinstance(node.value, ast.Attribute)
            and node.value.attr == "iloc"
        ):
            return

        if not _is_negative_number(node.slice):
            return

        line = getattr(node, "lineno", -1)
        self.findings.append(
            InspectionFinding(
                code="NEGATIVE_ILOC",
                certainty="suspicious",
                description=(
                    ".iloc[] was called with a negative index, which accesses "
                    "rows from the end of the DataFrame. If the DataFrame "
                    "contains the full dataset, .iloc[-1] is the last row — "
                    "a future row for every bar except the final one."
                ),
                evidence=f"Line {line}: .iloc[{ast.unparse(node.slice)}]",
                line_number=line,
                recommendation=(
                    "Verify that .iloc[-N] is intentional here. "
                    "If used inside signal generation, ensure it is "
                    "only reading already-elapsed history."
                ),
            )
        )


# ---------------------------------------------------------------------------
# Small helper functions
# ---------------------------------------------------------------------------

def _is_method_call(node: ast.Call, method_name: str) -> bool:
    """True when node is a call to an attribute named method_name."""
    return (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == method_name
    )


def _is_negative_number(node: ast.expr) -> bool:
    """
    True when the AST node represents a negative numeric literal.

    Handles two forms:
      -1      → UnaryOp(USub, Constant(1))
      -0.5    → UnaryOp(USub, Constant(0.5))

    Python 3.8+ sometimes folds -1 into a Constant(-1) depending on context,
    so we check for that too.
    """
    # Form 1: -N written as UnaryOp(USub, Constant(N))
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float))
        and node.operand.value > 0
    ):
        return True

    # Form 2: already-folded negative constant (rare in practice but safe to handle)
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and node.value < 0
    ):
        return True

    return False
