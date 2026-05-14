import pandas as pd

from core.agent_graph import (
    _clean_direct_address,
    clustering_run_node,
    final_message_node,
    model_config_node,
    render_message_node,
    task_selection_node,
    vector_run_node,
)
from core.agent_schemas import ModelConfigDecision, TaskSelectionDecision
from core.english_state import EnglishState


class _FakeStructuredLLM:
    def __init__(self, response):
        self.response = response

    def invoke(self, _messages):
        return self.response


def test_task_selection_unknown_stays_in_task_selection(monkeypatch):
    decision = TaskSelectionDecision(
        task_type="unknown",
        explanation="Please choose a task.",
        needs_clarification=True,
        clarification_question="What would you like to do?",
    )
    monkeypatch.setattr(
        "core.agent_graph._get_structured_llm",
        lambda _schema: _FakeStructuredLLM(decision),
    )

    state = {
        "user_message": "hello",
        "english_state": EnglishState(),
        "phase": {"stage": "task_selection"},
        "decisions": {},
    }

    new_state = task_selection_node(state)

    assert new_state["phase"]["stage"] == "task_selection"
    assert new_state["task_type"] is None


def test_vector_run_routes_clustering_to_clustering_config():
    state = {
        "english_state": EnglishState(last_vector_method="TF-IDF (word)"),
        "phase": {"stage": "vector_run"},
        "task_type": "clustering",
        "decisions": {},
    }

    new_state = vector_run_node(state)

    assert new_state["phase"]["stage"] == "clustering_config"


def test_task_selection_renderer_handles_unrelated_input_without_greeting():
    state = {
        "english_state": EnglishState(),
        "phase": {"stage": "task_selection"},
        "decisions": {},
        "ui_intent": "task_selection_feedback",
        "ui_payload": {
            "task_type": "unknown",
            "needs_clarification": True,
            "clarification_question": "What do you want to do with hamburger?",
        },
    }

    new_state = render_message_node(state)
    message = new_state["assistant_message"].lower()

    assert not message.startswith("hi")
    assert "hamburger" not in message
    assert "classify" in message
    assert "predict" in message
    assert "cluster" in message


def test_vector_run_includes_candidate_columns_and_suggested_label():
    state_obj = EnglishState(
        df=pd.DataFrame(
            {
                "text": ["good", "bad", "great"],
                "label": ["positive", "negative", "positive"],
                "score": [5, 1, 4],
            }
        ),
        csv_text_column="text",
        last_vector_method="TF-IDF (word)",
    )
    state = {
        "english_state": state_obj,
        "phase": {"stage": "vector_run"},
        "task_type": "classification",
        "decisions": {},
    }

    new_state = vector_run_node(state)

    assert new_state["phase"]["stage"] == "model_config"
    assert new_state["ui_payload"]["candidate_columns"] == ["label", "score"]
    assert new_state["ui_payload"]["suggested_label"] == "label"


def test_completed_workflow_followup_has_non_llm_idle_message():
    state = {
        "english_state": EnglishState(),
        "phase": {"stage": "results_explained"},
        "decisions": {},
    }

    rendered = render_message_node(final_message_node(state))

    assert "review step" in rendered["assistant_message"].lower()


def test_direct_address_cleanup_removes_third_person_user_phrasing():
    message = _clean_direct_address(
        "User has confirmed 3 clusters. The user wants to proceed with K-Means."
    )

    assert "User has" not in message
    assert "The user" not in message
    assert message.startswith("You have confirmed")
    assert "You want to proceed" in message


def test_model_config_defaults_to_auto_for_incompatible_model(monkeypatch):
    decision = ModelConfigDecision(
        label_column="score",
        model_name="Random Forest",
        test_size=0.2,
        explanation="Use the numeric label.",
    )
    monkeypatch.setattr(
        "core.agent_graph._get_structured_llm",
        lambda _schema: _FakeStructuredLLM(decision),
    )
    state_obj = EnglishState(
        df=pd.DataFrame({"text": ["a", "b", "c"], "score": [1.0, 2.0, 3.0]}),
        csv_text_column="text",
    )
    state = {
        "user_message": "predict score",
        "english_state": state_obj,
        "phase": {"stage": "model_config"},
        "task_type": "regression",
        "decisions": {},
    }

    new_state = model_config_node(state)

    assert new_state["decisions"]["model_name"] == "Auto"


def test_clustering_run_stores_export_on_shared_state():
    state_obj = EnglishState(
        df=pd.DataFrame(
            {
                "text": [
                    "cat kitten feline",
                    "dog puppy canine",
                    "stock market shares",
                    "equity trading market",
                ]
            }
        ),
        csv_text_column="text",
        last_vector_method="TF-IDF (word)",
        last_vector_ngram=(1, 1),
        last_vector_max_features=100,
    )
    state = {
        "english_state": state_obj,
        "phase": {"stage": "clustering_run"},
        "decisions": {"clustering_config": {"n_clusters": 2}},
    }

    new_state = clustering_run_node(state)

    assert new_state["phase"]["stage"] == "results_explained"
    assert state_obj.last_cluster_export is not None
    assert list(state_obj.last_cluster_export.columns) == ["sample", "cluster"]
    assert len(state_obj.last_cluster_export) == 4
