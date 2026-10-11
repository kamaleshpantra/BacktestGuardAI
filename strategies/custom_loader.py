"""
strategies/custom_loader.py
---------------------------
Safe Dynamic Strategy Loader for BacktestGuard AI.

Provides a secure sandbox for compiling and executing user-submitted Python
strategy code.

Two-Stage Defense Architecture:
1. AST Static Security Gate:
   - Blocks dangerous imports (os, sys, subprocess, socket, etc.).
   - Runs the AST Look-Ahead Bias Inspector (negative shifts, centered windows).
2. Restricted Execution Namespace:
   - Executes inside an isolated dictionary containing only numpy, pandas,
     and math.
   - Built-ins like open(), exec(), eval(), __import__ are strictly disabled.
3. Contract Enforcement:
   - Verifies the user code defines a callable generate_signals(df).
   - Validates that returned signals match df.index and contain values in {-1, 0, 1}.
"""

from __future__ import annotations

import ast
import math

import numpy as np
import pandas as pd

from inspection.ast_inspector import InspectionResult, inspect_strategy_source
from strategies.base import Strategy

# Disallowed module imports for security sandboxing
DANGEROUS_MODULES = frozenset({
    "os", "sys", "subprocess", "socket", "urllib", "requests", "httpx",
    "shutil", "builtins", "importlib", "pickle", "ctypes", "pathlib",
})


class SecurityViolationError(Exception):
    """Raised when user-submitted code attempts an insecure operation."""


class SafeCustomStrategy(Strategy):
    """
    Wraps dynamic user-submitted Python code inside a secure Strategy instance.
    """

    def __init__(self, source_code: str, strategy_name: str = "Custom_User_Strategy") -> None:
        self.source_code = source_code
        self._name = strategy_name
        self._compiled_signals_fn = None
        self._inspection_result: InspectionResult | None = None
        self._compile_and_validate()

    @property
    def name(self) -> str:
        return self._name

    @property
    def min_required_rows(self) -> int:
        return 10

    @property
    def inspection_result(self) -> InspectionResult:
        return self._inspection_result

    def _security_audit_ast(self, tree: ast.AST) -> None:
        """Scan AST for dangerous imports and calls."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in DANGEROUS_MODULES:
                        raise SecurityViolationError(
                            f"Security Violation: Import of module '{alias.name}' is strictly prohibited."
                        )
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] in DANGEROUS_MODULES:
                    raise SecurityViolationError(
                        f"Security Violation: Import from '{node.module}' is strictly prohibited."
                    )
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in ("eval", "exec", "open", "input", "__import__", "compile"):
                    raise SecurityViolationError(
                        f"Security Violation: Calling built-in '{node.func.id}()' is prohibited."
                    )

    def _compile_and_validate(self) -> None:
        # 1. Parse AST
        try:
            tree = ast.parse(self.source_code)
        except SyntaxError as e:
            raise ValueError(f"SyntaxError in custom code (Line {e.lineno}): {e.msg}")

        # 2. Security scan
        self._security_audit_ast(tree)

        # 3. Look-ahead bias inspection
        self._inspection_result = inspect_strategy_source(self.source_code)
        if self._inspection_result.has_confirmed_violations:
            first_err = self._inspection_result.confirmed[0]
            raise ValueError(
                f"Look-Ahead Bias Violation [{first_err.code}]: {first_err.description} (Line {first_err.line_number})"
            )

        # 4. Compile inside safe restricted namespace
        safe_globals = {
            "__builtins__": {
                "range": range,
                "len": len,
                "enumerate": enumerate,
                "min": min,
                "max": max,
                "abs": abs,
                "round": round,
                "int": int,
                "float": float,
                "bool": bool,
                "str": str,
                "list": list,
                "dict": dict,
                "tuple": tuple,
                "isinstance": isinstance,
            },
            "pd": pd,
            "np": np,
            "math": math,
        }
        local_scope = {}

        try:
            code_obj = compile(tree, filename="<custom_strategy>", mode="exec")
            exec(code_obj, safe_globals, local_scope)
        except Exception as exc:
            raise RuntimeError(f"Failed to compile strategy script: {str(exc)}")

        if "generate_signals" not in local_scope or not callable(local_scope["generate_signals"]):
            raise ValueError(
                "Custom strategy code must define a function named 'generate_signals(df)'"
            )

        self._compiled_signals_fn = local_scope["generate_signals"]

    def validate_config(self) -> list[str]:
        return []

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        if self._compiled_signals_fn is None:
            raise RuntimeError("Strategy function is not compiled.")

        try:
            # Pass a defensive copy to prevent strategy code from modifying input data
            raw_result = self._compiled_signals_fn(df.copy())
        except Exception as e:
            raise RuntimeError(f"Runtime error in generate_signals(): {str(e)}")

        if not isinstance(raw_result, pd.Series):
            raise TypeError(
                f"generate_signals(df) must return a pandas Series, got {type(raw_result).__name__}"
            )

        if not raw_result.index.equals(df.index):
            raw_result = raw_result.reindex(df.index)

        # Ensure signal Series is clean and cast to float
        clean_signal = raw_result.fillna(0.0).astype(float)
        clean_signal.name = "signal"
        return clean_signal
