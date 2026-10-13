"""Safe expression evaluator.

Restricted to arithmetic, comparisons, boolean ops, bounded generator
expressions, dictionary lookups, and whitelisted method/function calls used by
fusion rules and signal extraction.
"""
from __future__ import annotations

import ast
from typing import Any

_WINDOW_METHODS = {
    "latest", "mean", "max", "min", "count",
    "fraction_above", "fraction_below", "slope", "derivative",
}
_DICT_METHODS = {"get"}
_ALLOWED_METHODS = _WINDOW_METHODS | _DICT_METHODS
_SAFE_FUNCS = {
    "abs": abs, "min": min, "max": max, "len": len,
    "round": round, "any": any, "all": all,
    "int": int, "float": float, "bool": bool,
}
_FORBIDDEN_NAMES = {"__import__", "__builtins__", "eval", "exec", "open"}
_ALLOWED_NODES = (
    ast.Expression,
    ast.BoolOp,
    ast.BinOp,
    ast.UnaryOp,
    ast.Compare,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Constant,
    ast.Call,
    ast.Attribute,
    ast.Subscript,
    ast.Slice,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Set,
    ast.GeneratorExp,
    ast.comprehension,
    ast.keyword,
    ast.And,
    ast.Or,
    ast.Not,
    ast.UAdd,
    ast.USub,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Is,
    ast.IsNot,
    ast.In,
    ast.NotIn,
)


def safe_eval(expr: str, namespace: dict[str, Any]) -> Any:
    tree = ast.parse(expr, mode="eval")
    _validate(tree)
    return eval(compile(tree, "<expr>", "eval"),
                {"__builtins__": {}, **_SAFE_FUNCS}, namespace)


def _validate(node: ast.AST) -> None:
    for n in ast.walk(node):
        if not isinstance(n, _ALLOWED_NODES):
            raise ValueError(f"forbidden expression element: {type(n).__name__}")
        if isinstance(n, ast.Name) and (n.id in _FORBIDDEN_NAMES or n.id.startswith("_")):
            raise ValueError(f"forbidden name: {n.id}")
        if isinstance(n, ast.Attribute) and n.attr.startswith("_"):
            raise ValueError(f"forbidden attribute access: {n.attr}")
        if isinstance(n, ast.Call):
            _validate_call(n)


def _validate_call(node: ast.Call) -> None:
    if isinstance(node.func, ast.Name):
        if node.func.id not in _SAFE_FUNCS:
            raise ValueError(f"forbidden function call: {node.func.id}")
    elif isinstance(node.func, ast.Attribute):
        if node.func.attr not in _ALLOWED_METHODS:
            raise ValueError(f"forbidden method call: {node.func.attr}")
    else:
        raise ValueError("forbidden dynamic function call")
    for kw in node.keywords:
        if kw.arg is None:
            raise ValueError("forbidden expanded keyword arguments")
