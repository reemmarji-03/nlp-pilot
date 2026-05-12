from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal


TaskKind = Literal["classification", "regression"]
ModelFactory = Callable[..., object]

CONFIG_DIR = Path.home() / ".nlp_pilot"
USER_MODELS_FILE = "custom_models.json"

_REGISTRY: dict[TaskKind, dict[str, ModelFactory]] = {
    "classification": {},
    "regression": {},
}
_USER_SPECS: list["UserModelSpec"] = []


@dataclass(frozen=True)
class UserModelSpec:
    task_type: TaskKind
    name: str
    module_path: str
    object_name: str


def register_model(task_type: TaskKind, name: str, factory: ModelFactory) -> None:
    """
    Register a supervised model factory.

    The factory should return a scikit-learn compatible estimator. It may accept
    keyword arguments such as random_state and hyperparameters.
    """
    _validate_task(task_type)
    if not name.strip():
        raise ValueError("Model name cannot be empty")
    _REGISTRY[task_type][name] = factory


def get_registered_models(task_type: TaskKind) -> dict[str, ModelFactory]:
    _validate_task(task_type)
    return dict(_REGISTRY[task_type])


def registered_model_names(task_type: TaskKind) -> list[str]:
    return sorted(get_registered_models(task_type).keys())


def list_user_model_specs(config_dir: Path | None = None) -> list[UserModelSpec]:
    if config_dir is None:
        return list(_USER_SPECS)
    return _read_specs(config_dir)


def save_user_model_spec(spec: UserModelSpec, config_dir: Path | None = None) -> None:
    validate_user_model_spec(spec)
    specs = _read_specs(config_dir)
    specs = [
        s for s in specs
        if not (s.task_type == spec.task_type and s.name == spec.name)
    ]
    specs.append(spec)
    _write_specs(specs, config_dir)
    reload_user_model_specs(config_dir)


def remove_user_model_spec(
    task_type: TaskKind,
    name: str,
    config_dir: Path | None = None,
) -> None:
    _validate_task(task_type)
    specs = [
        s for s in _read_specs(config_dir)
        if not (s.task_type == task_type and s.name == name)
    ]
    _write_specs(specs, config_dir)
    reload_user_model_specs(config_dir)


def reload_user_model_specs(config_dir: Path | None = None) -> None:
    global _USER_SPECS
    specs = _read_specs(config_dir)

    # Remove previously loaded user specs without touching programmatic models.
    for spec in _USER_SPECS:
        _REGISTRY[spec.task_type].pop(spec.name, None)

    loaded: list[UserModelSpec] = []
    for spec in specs:
        try:
            validate_user_model_spec(spec)
            register_model(spec.task_type, spec.name, _factory_from_spec(spec))
            loaded.append(spec)
        except Exception as exc:
            print(f"[model_registry] Skipping {spec.name}: {exc}", file=sys.stderr)
    _USER_SPECS = loaded


def validate_user_model_spec(spec: UserModelSpec) -> None:
    _validate_task(spec.task_type)
    if not spec.name.strip():
        raise ValueError("Display name is required.")
    if not spec.object_name.strip():
        raise ValueError("Class or factory name is required.")
    path = Path(spec.module_path)
    if not path.exists() or not path.is_file():
        raise ValueError("Python module file does not exist.")
    if path.suffix.lower() != ".py":
        raise ValueError("Model module must be a .py file.")

    estimator = _factory_from_spec(spec)(random_state=42)
    _validate_estimator(estimator)


def _factory_from_spec(spec: UserModelSpec) -> ModelFactory:
    def factory(**kwargs):
        obj = _load_object(spec.module_path, spec.object_name)
        accepted = _filter_kwargs(obj, kwargs)
        estimator = obj(**accepted)
        _validate_estimator(estimator)
        return estimator

    return factory


def _load_object(module_path: str, object_name: str):
    path = Path(module_path)
    module_name = f"nlp_pilot_user_model_{abs(hash(path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    obj = getattr(module, object_name, None)
    if obj is None:
        raise ValueError(f"Object '{object_name}' was not found in {module_path}.")
    if not callable(obj):
        raise ValueError(f"Object '{object_name}' is not callable.")
    return obj


def _filter_kwargs(callable_obj, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return {}
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def _validate_estimator(estimator) -> None:
    required = ("fit", "predict", "get_params")
    missing = [name for name in required if not hasattr(estimator, name)]
    if missing:
        raise ValueError(
            "Model must be scikit-learn compatible; missing: "
            + ", ".join(missing)
        )


def _read_specs(config_dir: Path | None = None) -> list[UserModelSpec]:
    path = _models_path(config_dir)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    specs = []
    for item in raw if isinstance(raw, list) else []:
        try:
            specs.append(UserModelSpec(**item))
        except TypeError:
            continue
    return specs


def _write_specs(specs: list[UserModelSpec], config_dir: Path | None = None) -> None:
    path = _models_path(config_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(s) for s in specs], f, indent=2)


def _models_path(config_dir: Path | None = None) -> Path:
    return (config_dir or CONFIG_DIR) / USER_MODELS_FILE


def _validate_task(task_type: TaskKind) -> None:
    if task_type not in _REGISTRY:
        raise ValueError("task_type must be 'classification' or 'regression'")


reload_user_model_specs()
