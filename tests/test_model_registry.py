from sklearn.dummy import DummyClassifier

from core.model_registry import (
    UserModelSpec,
    list_user_model_specs,
    register_model,
    registered_model_names,
    reload_user_model_specs,
    remove_user_model_spec,
    save_user_model_spec,
)
from core.supervised import available_model_names


def test_register_model_exposes_name_to_supervised_model_list():
    register_model(
        "classification",
        "Dummy Majority",
        lambda random_state=None: DummyClassifier(strategy="most_frequent"),
    )

    assert "Dummy Majority" in registered_model_names("classification")
    assert "Dummy Majority" in available_model_names("classification")


def test_user_model_spec_persists_and_loads(tmp_path):
    module_path = tmp_path / "custom_model.py"
    module_path.write_text(
        "from sklearn.dummy import DummyClassifier\n\n"
        "class MyClassifier(DummyClassifier):\n"
        "    def __init__(self, strategy='most_frequent'):\n"
        "        super().__init__(strategy=strategy)\n",
        encoding="utf-8",
    )
    spec = UserModelSpec(
        task_type="classification",
        name="My UI Classifier",
        module_path=str(module_path),
        object_name="MyClassifier",
    )

    save_user_model_spec(spec, config_dir=tmp_path)
    reload_user_model_specs(config_dir=tmp_path)

    assert spec in list_user_model_specs(config_dir=tmp_path)
    assert "My UI Classifier" in registered_model_names("classification")

    remove_user_model_spec("classification", "My UI Classifier", config_dir=tmp_path)
    assert "My UI Classifier" not in registered_model_names("classification")
