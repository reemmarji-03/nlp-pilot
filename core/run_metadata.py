from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from typing import Any

from core.settings_manager import settings


PACKAGES = [
    "bertopic",
    "customtkinter",
    "faiss-cpu",
    "langchain",
    "langchain-community",
    "langchain-ollama",
    "nltk",
    "numpy",
    "pandas",
    "scikit-learn",
    "sentence-transformers",
    "torch",
    "transformers",
]


def package_versions() -> dict[str, str]:
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def safe_settings_snapshot() -> dict[str, Any]:
    data = settings.data
    active = data.get("active_provider", "ollama")
    return {
        "active_provider": active,
        "random_seed": settings.get_seed(),
        "ollama": {
            "url": data.get("ollama", {}).get("url"),
            "model": data.get("ollama", {}).get("model"),
        },
        "openai": {
            "model": data.get("openai", {}).get("model"),
            "credential_configured": bool(data.get("openai", {}).get("api_key")),
        },
        "anthropic": {
            "model": data.get("anthropic", {}).get("model"),
            "credential_configured": bool(data.get("anthropic", {}).get("api_key")),
        },
    }


def run_metadata(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": package_versions(),
        "settings": safe_settings_snapshot(),
        "reproducibility_notes": [
            "Local random seeds are applied to Python, NumPy, Torch, train/test splits, CV, K-Means, and BERTopic UMAP.",
            "Torch deterministic algorithms are requested with warn_only=True.",
            "LLM outputs may still vary by provider/runtime even with temperature set to 0.",
            "GPU kernels may emit warnings or use nondeterministic fallbacks depending on installed drivers and libraries.",
        ],
    }
    if extra:
        payload["run"] = extra
    return payload
