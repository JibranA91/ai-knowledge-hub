"""Architecture guard: LLM connections may only be created in app/providers/.

Everything else must go through `app.model`. This keeps the role → model → vendor
mapping in one place so a new provider is a single new file plus a registry entry.

The scan is AST-based, not textual, so docstrings and code samples embedded in
strings (e.g. the export bundle's README template) are not false positives.

If this test fails, don't add an exemption — move the client construction into a
provider and reach it via `app.model`.
"""
import ast
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parents[2] / "app"

# Only app/providers/ may construct vendor clients. app/services/bedrock.py is a
# deprecated re-export shim that constructs nothing; it is allowed until removed.
ALLOWED_PACKAGES = {"providers"}
ALLOWED_FILES = {"services/bedrock.py"}

# Vendor SDKs / LangChain integrations that imply a direct model connection.
FORBIDDEN_MODULES = {
    "langchain_aws", "langchain_openai", "langchain_anthropic", "langchain_google_genai",
    "openai", "anthropic", "mistralai", "cohere", "ollama", "google.generativeai",
}

# Constructors that open a connection to a model endpoint.
FORBIDDEN_CALLS = {
    "ChatBedrock", "ChatBedrockConverse", "BedrockLLM", "BedrockEmbeddings",
    "OpenAI", "AsyncOpenAI", "AzureOpenAI", "ChatOpenAI", "AzureChatOpenAI",
    "Anthropic", "AsyncAnthropic", "ChatAnthropic",
}

# boto3 service names that are model endpoints (S3/STS/etc. are fine anywhere).
FORBIDDEN_BOTO3_SERVICES = {"bedrock", "bedrock-runtime", "sagemaker-runtime"}


def _app_modules():
    """Yield (relative_path, parsed_ast) for every app module under the guard."""
    for path in sorted(APP_DIR.rglob("*.py")):
        rel = path.relative_to(APP_DIR).as_posix()
        if rel.split("/")[0] in ALLOWED_PACKAGES or rel in ALLOWED_FILES:
            continue
        yield rel, ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _func_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _dotted(node: ast.Call) -> str:
    """'boto3.client' for boto3.client(...) — empty string if not an attribute call."""
    func = node.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return f"{func.value.id}.{func.attr}"
    return ""


def test_no_llm_clients_outside_providers():
    violations = []
    for rel, tree in _app_modules():
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in FORBIDDEN_MODULES or alias.name in FORBIDDEN_MODULES:
                        violations.append(f"app/{rel}:{node.lineno} — imports {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
                if root in FORBIDDEN_MODULES or (node.module or "") in FORBIDDEN_MODULES:
                    violations.append(f"app/{rel}:{node.lineno} — imports from {node.module}")
            elif isinstance(node, ast.Call):
                name = _func_name(node)
                if name in FORBIDDEN_CALLS:
                    violations.append(f"app/{rel}:{node.lineno} — constructs {name}()")
                elif _dotted(node) == "boto3.client" and node.args:
                    service = node.args[0]
                    if isinstance(service, ast.Constant) and service.value in FORBIDDEN_BOTO3_SERVICES:
                        violations.append(f"app/{rel}:{node.lineno} — boto3.client({service.value!r})")

    assert not violations, (
        "LLM connections must be created inside app/providers/ and reached via "
        "app.model. Offending lines:\n  " + "\n  ".join(violations)
    )


def test_providers_are_not_imported_directly():
    """Application code imports app.model, never a concrete provider module.

    `app.providers.base` / `app.providers.usage` are vendor-neutral and exempt.
    """
    neutral = {"app.providers.base", "app.providers.usage", "app.providers"}
    offenders = []
    for rel, tree in _app_modules():
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("app.providers"):
                if node.module not in neutral:
                    offenders.append(f"app/{rel}:{node.lineno} — from {node.module} import …")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("app.providers.") and alias.name not in neutral:
                        offenders.append(f"app/{rel}:{node.lineno} — import {alias.name}")

    assert not offenders, (
        "Import app.model instead of a concrete provider:\n  " + "\n  ".join(offenders)
    )


def test_guard_detects_a_planted_violation(tmp_path):
    """The guard would actually catch a regression — not vacuously passing."""
    tree = ast.parse('import boto3\nc = boto3.client("bedrock-runtime")\n')
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert any(
        _dotted(n) == "boto3.client"
        and isinstance(n.args[0], ast.Constant)
        and n.args[0].value in FORBIDDEN_BOTO3_SERVICES
        for n in calls
    )


@pytest.mark.parametrize("role", ["ingest_plan", "ingest_write", "query", "recalibrate",
                                  "draft_agent", "edit", "embedding"])
def test_every_role_resolves_to_a_real_setting(role):
    """Each Role must map to a real Settings field, so none silently resolves to ''."""
    from app import model
    from app.config import Settings
    assert model.Role(role) in model._ROLE_SETTING
    assert model._ROLE_SETTING[model.Role(role)] in Settings.model_fields
