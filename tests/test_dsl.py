"""Validate the safe expression DSL allows/forbids the right things."""
import pytest

from inferops.domain.fusion.dsl import safe_eval


def test_basic_arith():
    assert safe_eval("1 + 2 * 3", {}) == 7


def test_method_call_on_window_like():
    class W:
        def fraction_above(self, t):
            return 0.75
    assert safe_eval("phone.fraction_above(0.4) > 0.5", {"phone": W()}) is True


def test_generator_extract_expression():
    class R:
        def __init__(self):
            self.raw = {"by_class": {"phone": [{"score": 0.2}, {"score": 0.9}]}}

    expr = "max((d['score'] for r in results if r.raw for d in (r.raw.get('by_class', {}).get('phone', []))), default=0.0)"
    assert safe_eval(expr, {"results": [R()]}) == 0.9


def test_forbidden_import():
    with pytest.raises(ValueError):
        safe_eval("__import__('os').system('echo bad')", {})


def test_forbidden_dunder():
    with pytest.raises(ValueError):
        safe_eval("(1).__class__", {})


def test_forbidden_lambda():
    with pytest.raises(ValueError):
        safe_eval("(lambda: 1)()", {})


def test_forbidden_unlisted_method_call():
    with pytest.raises(ValueError):
        safe_eval("payload.clear()", {"payload": {"x": 1}})
