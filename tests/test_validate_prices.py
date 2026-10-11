"""
test_validate_prices.py
-----------------------
Automated tests for the validate_prices() function.

Test categories:
  - Valid inputs  : should return an empty list (no errors)
  - Invalid inputs: should return a list containing specific error messages
  - Edge cases    : boundary values, mixed-problem inputs, unexpected types
"""

import pytest

from validate_prices import validate_prices

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def passes(prices):
    """Return True when validate_prices finds no errors."""
    return validate_prices(prices) == []


def fails_with(prices, fragment):
    """Return True when at least one error message contains `fragment`."""
    errors = validate_prices(prices)
    return any(fragment in e for e in errors)


# ---------------------------------------------------------------------------
# Valid inputs — expect no errors
# ---------------------------------------------------------------------------

class TestValidPrices:

    def test_typical_closing_prices(self):
        assert passes([100.0, 101.5, 99.75, 102.0])

    def test_single_price(self):
        assert passes([50.0])

    def test_integer_prices(self):
        # integers are acceptable numeric values
        assert passes([10, 20, 30])

    def test_mixed_int_and_float(self):
        assert passes([10, 20.5, 30])

    def test_very_small_positive_price(self):
        # penny stocks can have very low prices — still valid
        assert passes([0.001])

    def test_large_prices(self):
        assert passes([1_000_000.0, 999_999.99])


# ---------------------------------------------------------------------------
# Wrong container type
# ---------------------------------------------------------------------------

class TestWrongContainerType:

    @pytest.mark.parametrize("bad_input, expected_fragment", [
        (None,           "prices must be a list"),
        ("100,200",      "prices must be a list"),
        (100.0,          "prices must be a list"),
        ((100.0, 101.0), "prices must be a list"),  # tuple, not list
        ({"a": 1},       "prices must be a list"),
    ])
    def test_non_list_inputs(self, bad_input, expected_fragment):
        assert fails_with(bad_input, expected_fragment)

    def test_non_list_returns_immediately(self):
        # When the container type is wrong, we should get exactly one error
        # and not attempt to iterate over it.
        errors = validate_prices(42)
        assert len(errors) == 1


# ---------------------------------------------------------------------------
# Empty list
# ---------------------------------------------------------------------------

class TestEmptyList:

    def test_empty_list(self):
        assert fails_with([], "prices list is empty")

    def test_empty_list_returns_exactly_one_error(self):
        assert len(validate_prices([])) == 1


# ---------------------------------------------------------------------------
# Wrong element types
# ---------------------------------------------------------------------------

class TestWrongElementTypes:

    def test_string_element(self):
        assert fails_with(["100.0"], "expected a number, got str")

    def test_none_element(self):
        assert fails_with([100.0, None, 102.0], "expected a number, got NoneType")

    def test_list_element(self):
        assert fails_with([[100.0]], "expected a number, got list")

    def test_bool_is_treated_as_number(self):
        # In Python, bool is a subclass of int, so True == 1 and False == 0.
        # True (==1) is positive and finite — passes the numeric checks.
        # False (==0) fails the positivity check.
        # This is a documented edge case, not a bug we need to fix right now.
        assert fails_with([False], "price must be positive")

    @pytest.mark.parametrize("index, bad_value", [
        (0, "hello"),
        (1, None),
        (2, []),
    ])
    def test_error_reports_correct_index(self, index, bad_value):
        prices = [10.0, 20.0, 30.0]
        prices[index] = bad_value
        errors = validate_prices(prices)
        assert any(f"index {index}" in e for e in errors)


# ---------------------------------------------------------------------------
# Non-finite values
# ---------------------------------------------------------------------------

class TestNonFiniteValues:

    def test_positive_infinity(self):
        assert fails_with([float("inf")], "not finite")

    def test_negative_infinity(self):
        assert fails_with([float("-inf")], "not finite")

    def test_nan(self):
        # float('nan') is not finite — math.isfinite(nan) is False
        assert fails_with([float("nan")], "not finite")

    def test_finite_check_at_correct_index(self):
        errors = validate_prices([100.0, float("inf"), 102.0])
        assert any("index 1" in e for e in errors)


# ---------------------------------------------------------------------------
# Non-positive values
# ---------------------------------------------------------------------------

class TestNonPositiveValues:

    def test_zero_price(self):
        assert fails_with([0.0], "price must be positive")

    def test_negative_price(self):
        assert fails_with([-1.0], "price must be positive")

    def test_negative_integer(self):
        assert fails_with([-10], "price must be positive")

    def test_zero_integer(self):
        assert fails_with([0], "price must be positive")

    def test_negative_at_correct_index(self):
        errors = validate_prices([100.0, 101.0, -5.0])
        assert any("index 2" in e for e in errors)


# ---------------------------------------------------------------------------
# Multiple errors in a single list
# ---------------------------------------------------------------------------

class TestMultipleErrors:

    def test_collects_all_errors(self):
        # A list with three separate problems should produce three errors.
        prices = [100.0, -5.0, "bad", float("inf")]
        errors = validate_prices(prices)
        assert len(errors) == 3

    def test_error_order_matches_index_order(self):
        prices = [-1.0, float("nan"), 100.0]
        errors = validate_prices(prices)
        # First error should mention index 0, second should mention index 1
        assert "index 0" in errors[0]
        assert "index 1" in errors[1]

    def test_valid_elements_not_flagged(self):
        # Only the bad element should be flagged, not the good ones.
        errors = validate_prices([100.0, -5.0, 102.0])
        assert len(errors) == 1
        assert "index 1" in errors[0]


# ---------------------------------------------------------------------------
# Boundary values
# ---------------------------------------------------------------------------

class TestBoundaryValues:

    def test_smallest_positive_float(self):
        import sys
        assert passes([sys.float_info.min])

    def test_largest_finite_float(self):
        import sys
        assert passes([sys.float_info.max])

    def test_exactly_zero_fails(self):
        assert fails_with([0.0], "price must be positive")

    def test_epsilon_above_zero_passes(self):
        assert passes([1e-300])
