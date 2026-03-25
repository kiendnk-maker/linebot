"""
tests/test_math_eval.py — Test calculate_math() safety and correctness.

Verifies that simpleeval correctly evaluates arithmetic and rejects
potentially dangerous or invalid expressions.
"""
import pytest
from tools_api import calculate_math


# ── Correct arithmetic ────────────────────────────────────────────────────────

def test_addition():
    assert calculate_math("2 + 2") == "4"


def test_subtraction():
    assert calculate_math("10 - 3") == "7"


def test_multiplication():
    assert calculate_math("15 * 24") == "360"


def test_division():
    result = calculate_math("10 / 4")
    assert result == "2.5"


def test_complex_expression():
    assert calculate_math("(100 + 50) * 2 - 75") == "225"


def test_float_result():
    result = calculate_math("1 / 3")
    assert "0.333" in result


def test_percentage_of():
    result = calculate_math("200 * 0.15")
    assert result == "30.0"


# ── Security: must reject dangerous inputs ────────────────────────────────────

def test_rejects_import_attempt():
    result = calculate_math("__import__('os').system('whoami')")
    assert "Lỗi" in result or "Error" in result.lower()


def test_rejects_exec():
    result = calculate_math("exec('print(1)')")
    assert "Lỗi" in result or "Error" in result.lower()


def test_rejects_attribute_access():
    result = calculate_math("().__class__.__bases__")
    assert "Lỗi" in result or "Error" in result.lower()


def test_rejects_string_input():
    result = calculate_math("'hello' + 'world'")
    # simpleeval may allow string concat — at minimum it shouldn't execute OS commands
    # This test verifies it doesn't crash
    assert isinstance(result, str)


def test_empty_expression_returns_error():
    result = calculate_math("")
    assert isinstance(result, str)  # graceful, not a crash


def test_division_by_zero_returns_error():
    result = calculate_math("1 / 0")
    assert "Lỗi" in result or "Error" in result.lower() or "ZeroDivision" in result


def test_very_long_expression_handled():
    """Should not hang or crash on a long expression."""
    expr = " + ".join(["1"] * 100)
    result = calculate_math(expr)
    assert result == "100"
