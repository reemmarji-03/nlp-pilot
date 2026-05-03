import json
import pytest
from core.settings_manager import SettingsManager, DEFAULTS


def test_creates_default_config_on_first_run(tmp_path):
    sm = SettingsManager(config_dir=tmp_path)
    assert (tmp_path / "config.json").exists()
    assert sm.data["active_provider"] == "ollama"
    assert sm.data["random_seed"] == 42


def test_save_and_reload_preserves_values(tmp_path):
    sm = SettingsManager(config_dir=tmp_path)
    sm.data["active_provider"] = "openai"
    sm.data["openai"]["api_key"] = "sk-test"
    sm.data["random_seed"] = 7
    sm.save()

    sm2 = SettingsManager(config_dir=tmp_path)
    assert sm2.data["active_provider"] == "openai"
    assert sm2.data["openai"]["api_key"] == "sk-test"
    assert sm2.data["random_seed"] == 7


def test_corrupted_config_resets_to_defaults(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text("not valid json", encoding="utf-8")

    sm = SettingsManager(config_dir=tmp_path)
    assert sm.data["active_provider"] == "ollama"
    with open(config_file) as f:
        saved = json.load(f)
    assert saved["active_provider"] == "ollama"


def test_get_seed_returns_int(tmp_path):
    sm = SettingsManager(config_dir=tmp_path)
    seed = sm.get_seed()
    assert isinstance(seed, int)
    assert seed == 42


def test_get_llm_returns_none_when_openai_key_missing(tmp_path):
    sm = SettingsManager(config_dir=tmp_path)
    sm.data["active_provider"] = "openai"
    sm.data["openai"]["api_key"] = ""
    assert sm.get_llm() is None


def test_get_llm_returns_none_when_anthropic_key_missing(tmp_path):
    sm = SettingsManager(config_dir=tmp_path)
    sm.data["active_provider"] = "anthropic"
    sm.data["anthropic"]["api_key"] = ""
    assert sm.get_llm() is None


def test_partial_config_merges_with_defaults(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"active_provider": "openai"}), encoding="utf-8")

    sm = SettingsManager(config_dir=tmp_path)
    assert sm.data["active_provider"] == "openai"
    assert "ollama" in sm.data
    assert sm.data["random_seed"] == 42
