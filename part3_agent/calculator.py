"""A safe calculator. It never uses eval().

The expression is parsed into a syntax tree with Python's `ast` module, and
only a whitelist of node types is evaluated: numbers, + - * / // % **, unary
plus/minus, parentheses, and a few functions. Anything else (names, attribute
access, strings, lambdas, imports) is rejected. Limits stop "denial of service"
inputs such as 9**9**9 or a 10,000-character expression.
"""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable

from agentkit.errors import ToolError

MAX_LENGTH = 200
MAX_EXPONENT = 100
MAX_MAGNITUDE = 1e15

_BINARY: dict[type, Callable] = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY: dict[type, Callable] = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS: dict[str, Callable] = {"round": round, "min": min, "max": max, "abs": abs, "sqrt": math.sqrt}


def calculate(expression: str) -> int | float:
    text = expression.strip().replace("×", "*").replace("÷", "/").replace("^", "**")
    if not text:
        raise ToolError("the expression is empty")
    if len(text) > MAX_LENGTH:
        raise ToolError(f"the expression is longer than {MAX_LENGTH} characters")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        raise ToolError(f"not a valid arithmetic expression: {expression!r}") from None
    value = _evaluate(tree.body)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ToolError("the result is not a finite number")
        value = round(value, 10)  # hide float noise such as 0.1 + 0.2 = 0.30000000000000004
        if value.is_integer():
            value = int(value)
    return value


def _check(value: int | float) -> int | float:
    if abs(value) > MAX_MAGNITUDE:
        raise ToolError("the result is too large")
    return value


def _evaluate(node: ast.AST) -> int | float:
    # bool is a subclass of int, so it is excluded explicitly; str and complex fail the isinstance check
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return _check(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ToolError(f"exponents larger than {MAX_EXPONENT} are not allowed")
        try:
            return _check(_BINARY[type(node.op)](left, right))
        except ZeroDivisionError:
            raise ToolError("division by zero") from None
        except OverflowError:
            raise ToolError("the result is too large") from None
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_evaluate(node.operand))
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS
            and not node.keywords and node.args):
        args = [_evaluate(arg) for arg in node.args]
        try:
            return _check(_FUNCTIONS[node.func.id](*args))
        except (TypeError, ValueError) as exc:
            raise ToolError(f"invalid arguments for {node.func.id}(): {exc}") from None
    raise ToolError(f"unsupported element in expression: {type(node).__name__}")
