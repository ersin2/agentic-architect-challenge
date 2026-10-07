"""Part 3 calculator: correct arithmetic, and no way to run code."""

from __future__ import annotations

import pytest

from agentkit.errors import ToolError
from part3_agent.calculator import calculate


@pytest.mark.parametrize("expression, expected", [
    ("4 * 220", 880),
    ("round(236 * 0.62, 2)", 146.32),
    ("60 * 0.75 * 2 + 60", 150),
    ("2 + 3 * 4", 14),
    ("(2 + 3) * 4", 20),
    ("-5 + 2", -3),
    ("0.1 + 0.2", 0.3),           # float noise hidden
    ("2 ^ 10", 1024),             # models often write ^ for "power"
    ("7 // 2", 3),
    ("sqrt(16)", 4),
    ("max(160, 220, 120)", 220),
    ("4 × 220", 880),        # multiplication sign
])
def test_arithmetic(expression, expected):
    assert calculate(expression) == expected


@pytest.mark.parametrize("expression", [
    "__import__('os').system('echo hacked')",
    "().__class__.__bases__",
    "open('secrets.txt').read()",
    "print(1)",
    "'a' * 3",
    "lambda: 1",
    "[1, 2, 3]",
    "x + 1",
    "True + 1",                   # booleans are not numbers here
    "1j * 2",
    "9 ** 9 ** 9",                # would hang the process
    "10 ** 20 * 10 ** 20",
    "1 / 0",
    "sqrt(-1)",
    "1 + " * 100 + "1",           # too long
    "",
    "2 +",
])
def test_anything_but_arithmetic_is_rejected(expression):
    with pytest.raises(ToolError):
        calculate(expression)
