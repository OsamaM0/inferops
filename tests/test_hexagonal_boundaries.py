"""Executable dependency rules for the inward layers."""
from __future__ import annotations

import ast
from importlib.util import resolve_name
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
INFRASTRUCTURE_PACKAGES = {
    "sqlalchemy", "asyncpg", "boto3", "botocore", "redis", "aiokafka",
    "fastapi", "httpx", "prometheus_client", "opentelemetry", "structlog",
}


def _imports(path: Path):
    package = ".".join(path.relative_to(ROOT).with_suffix("").parts[:-1])
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                module = resolve_name("." * node.level + module, package)
            yield node.lineno, module
        elif isinstance(node, ast.Call) and node.args:
            name = (node.func.attr if isinstance(node.func, ast.Attribute)
                    else node.func.id if isinstance(node.func, ast.Name) else "")
            argument = node.args[0]
            if (name in {"import_module", "__import__"} and isinstance(argument, ast.Constant)
                    and isinstance(argument.value, str)):
                yield node.lineno, argument.value


@pytest.mark.parametrize("layer", ["domain", "application", "ports"])
def test_inward_layers_import_only_allowed_dependencies(layer):
    allowed_layers = {"domain": {"domain"}, "application": {"application", "domain", "ports"},
                      "ports": {"ports", "domain"}}[layer]
    errors = []
    for path in sorted((ROOT / "inferops" / layer).rglob("*.py")):
        for line, module in _imports(path):
            internal = module == "inferops" or module.startswith("inferops.")
            allowed = (any(module == f"inferops.{candidate}" or module.startswith(f"inferops.{candidate}.")
                              for candidate in allowed_layers))
            infrastructure = module.split(".", 1)[0] in INFRASTRUCTURE_PACKAGES
            if infrastructure or (internal and not allowed):
                errors.append(f"{path.relative_to(ROOT)}:{line}: {module}")
    assert not errors, "Hexagonal dependency boundary violations:\n" + "\n".join(errors)
