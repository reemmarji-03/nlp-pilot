import copy
import json
from pathlib import Path
from typing import Optional


DEFAULTS: dict = {
    "active_provider": "ollama",
    "random_seed": 42,
    "ollama": {
        "url": "http://localhost:11434",
        "model": "qwen3:4b",
    },
    "openai": {
        "api_key": "",
        "model": "gpt-4o-mini",
    },
    "anthropic": {
        "api_key": "",
        "model": "claude-sonnet-4-6",
    },
}


class SettingsManager:
    def __init__(self, config_dir: Optional[Path] = None):
        self._config_dir = config_dir or (Path.home() / ".nlp_pilot")
        self._config_file = self._config_dir / "config.json"
        self.data: dict = {}
        self.load()

    def load(self) -> None:
        self._config_dir.mkdir(parents=True, exist_ok=True)
        if self._config_file.exists():
            try:
                with open(self._config_file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                self.data = copy.deepcopy(DEFAULTS)
                self.data.update({k: v for k, v in loaded.items() if k not in ("ollama", "openai", "anthropic")})
                for key in ("ollama", "openai", "anthropic"):
                    self.data[key] = {**DEFAULTS[key], **loaded.get(key, {})}
            except Exception as exc:
                import sys
                print(f"[SettingsManager] Config load failed: {exc}", file=sys.stderr)
                self.data = copy.deepcopy(DEFAULTS)
                self.save()
        else:
            self.data = copy.deepcopy(DEFAULTS)
            self.save()

    def save(self) -> None:
        self._config_dir.mkdir(parents=True, exist_ok=True)
        with open(self._config_file, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)

    def get_llm(self):
        provider = self.data.get("active_provider", "ollama")
        try:
            if provider == "ollama":
                from langchain_ollama import ChatOllama
                cfg = self.data["ollama"]
                if not cfg.get("url") or not cfg.get("model"):
                    return None
                return ChatOllama(base_url=cfg["url"], model=cfg["model"])
            elif provider == "openai":
                from langchain_openai import ChatOpenAI
                cfg = self.data["openai"]
                if not cfg.get("api_key"):
                    return None
                return ChatOpenAI(api_key=cfg["api_key"], model=cfg["model"], temperature=0.2)
            elif provider == "anthropic":
                from langchain_anthropic import ChatAnthropic
                cfg = self.data["anthropic"]
                if not cfg.get("api_key"):
                    return None
                return ChatAnthropic(api_key=cfg["api_key"], model=cfg["model"], temperature=0.2)
        except Exception:
            return None
        return None

    def get_seed(self) -> int:
        return int(self.data.get("random_seed", 42))


settings = SettingsManager()
