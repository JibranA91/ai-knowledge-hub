"""One catalogue of friendly names, provider IDs and locally checked capabilities."""
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

from app.providers import available
from app.providers.base import UnknownModelError

Capability = Literal["chat", "tools", "converse", "stream", "embedding"]


class Binding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = Field(min_length=1, pattern=r"^\S+$")
    capabilities: frozenset[Capability] = Field(min_length=1)
    inference_geos: frozenset[str] = frozenset()
    dimensions: frozenset[PositiveInt] = frozenset()
    temperature: bool = False
    unavailable_reason: str = ""

    @model_validator(mode="after")
    def check_capabilities(self):
        if "embedding" in self.capabilities:
            if self.capabilities != {"embedding"} or not self.dimensions or self.inference_geos or self.temperature:
                raise ValueError("Embedding bindings need dimensions and cannot have text capabilities or profiles")
        elif self.dimensions:
            raise ValueError("Only embedding bindings have dimensions")
        if "tools" in self.capabilities and "chat" not in self.capabilities:
            raise ValueError("Tools require chat support")
        if "stream" in self.capabilities and "converse" not in self.capabilities:
            raise ValueError("Streaming requires converse support")
        if any(not geo.isalpha() or not geo.islower() for geo in self.inference_geos):
            raise ValueError("Inference geographies must be lowercase letters")
        return self

    def require(self, capabilities: set[Capability], dimensions: int = 1536):
        if self.unavailable_reason:
            raise UnknownModelError(f"{self.model_id}: {self.unavailable_reason}")
        missing = capabilities - self.capabilities
        if missing:
            raise UnknownModelError(f"{self.model_id} does not support {', '.join(sorted(missing))}")
        if "embedding" in capabilities and dimensions not in self.dimensions:
            raise UnknownModelError(f"{self.model_id} cannot produce the {dimensions} dimensions required by the database")


class ModelEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    providers: dict[str, Binding] = Field(min_length=1)


class _UniqueKeysLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        keys = [self.construct_object(key, deep=deep) for key, _ in node.value]
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate catalogue key")
        return super().construct_mapping(node, deep=deep)


def normalize(name: str) -> str:
    return "".join(c for c in name.lower() if c.isalnum())


def parse_catalog(text: str) -> dict[str, ModelEntry]:
    """Validate the entire manifest, including unused entries, before resolving."""
    try:
        data = yaml.load(text, Loader=_UniqueKeysLoader)
        if not isinstance(data, dict) or not data:
            raise ValueError("Expected a non-empty mapping of model names")
        catalog = {}
        ids = set()
        for name, entry in data.items():
            if not isinstance(name, str) or not name or name != normalize(name):
                raise ValueError("Catalogue keys must be normalized alphanumeric friendly names")
            catalog[name] = ModelEntry.model_validate(entry)
            for provider, binding in catalog[name].providers.items():
                if provider not in available():
                    raise ValueError(f"{name}: provider {provider!r} is not registered")
                if binding.inference_geos and provider != "bedrock":
                    raise ValueError(f"{name}: inference_geos is only supported by bedrock")
                if (provider, binding.model_id) in ids:
                    raise ValueError(f"{name}: duplicate model ID for {provider}")
                ids.add((provider, binding.model_id))
        return catalog
    except (ValueError, TypeError, yaml.YAMLError) as exc:
        raise UnknownModelError(f"Invalid app/model_catalog.yaml: {exc}") from exc


@lru_cache(maxsize=1)
def load_catalog() -> dict[str, ModelEntry]:
    return parse_catalog(Path(__file__).with_suffix(".yaml").read_text(encoding="utf-8"))


def known_models(provider: str) -> list[str]:
    return sorted(name for name, entry in load_catalog().items() if provider in entry.providers)


def resolve(name: str, provider: str, *, geo: str = "", capabilities: set[Capability] | None = None,
            dimensions: int = 1536) -> str:
    entry = load_catalog().get(normalize(name))
    if entry is None or provider not in entry.providers:
        raise UnknownModelError(
            f"No mapping for model {name!r} on {provider!r}. Known names: {', '.join(known_models(provider))}. "
            "Use a friendly name; add new provider IDs in app/model_catalog.yaml, not environment settings.")
    binding = entry.providers[provider]
    binding.require(capabilities or set(), dimensions)
    geo = geo.strip().lower().rstrip(".")
    if binding.inference_geos and geo:
        if geo not in binding.inference_geos:
            raise UnknownModelError(f"{name}: no {geo!r} inference profile; set BEDROCK_INFERENCE_GEO to "
                                    f"{', '.join(sorted(binding.inference_geos))}, or blank for the foundation ID")
        return f"{geo}.{binding.model_id}"
    return binding.model_id


def binding_for_id(provider: str, model_id: str) -> Binding:
    """Adapter payload metadata for an already resolved ID; never a raw-ID escape hatch."""
    for entry in load_catalog().values():
        binding = entry.providers.get(provider)
        if binding and model_id == binding.model_id:
            return binding
    raise UnknownModelError(f"No catalogue metadata for {provider} model {model_id!r}")
