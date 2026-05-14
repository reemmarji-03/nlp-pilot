from typing import TypedDict, Literal, Optional, Dict, Any, List
from dataclasses import dataclass

import pandas as pd
from langgraph.graph import StateGraph, END
from langchain.messages import HumanMessage

from core.english_state import EnglishState
from core import english_nlp as enlp
from core import supervised as sup
from core import unsupervised as unsup
from core.agent_schemas import (
    TaskSelectionDecision,
    PreprocessConfigDecision,
    VectorConfigDecision,
    ModelConfigDecision,
    ClusteringConfigDecision,
)
from core import prompts as agent_prompts
from core.settings_manager import settings

_NO_LLM_MSG = (
    "No LLM provider configured. Open Settings to select Ollama, OpenAI, or Anthropic."
)


def _clean_direct_address(text: str) -> str:
    """Remove third-person phrasing that can leak from LLM decision explanations."""
    replacements = {
        "The user wants to": "You want to",
        "the user wants to": "you want to",
        "User wants to": "You want to",
        "The user has": "You have",
        "the user has": "you have",
        "User has": "You have",
        "The user is": "You are",
        "the user is": "you are",
        "User is": "You are",
        "The user": "You",
        "the user": "you",
    }
    cleaned = text or ""
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    return cleaned


def _get_llm():
    return settings.get_llm()


def _get_structured_llm(schema):
    llm = _get_llm()
    if llm is None:
        return None
    try:
        return llm.with_structured_output(schema)
    except Exception:
        return None


def _candidate_label_context(eng: EnglishState, task_type: str | None = None) -> dict:
    if eng.df is None:
        return {"columns": [], "profiles": [], "suggested_label": None}

    columns = list(eng.df.columns)
    if eng.csv_text_column in columns:
        columns.remove(eng.csv_text_column)

    profiles = []
    for col in columns:
        series = eng.df[col]
        non_null = series.dropna()
        samples = [str(v)[:60] for v in non_null.head(5).tolist()]
        profiles.append(
            {
                "name": col,
                "dtype": str(series.dtype),
                "missing": int(series.isna().sum()),
                "unique": int(non_null.nunique(dropna=True)),
                "samples": samples,
            }
        )

    def name_score(name: str) -> int:
        lowered = name.lower()
        score = 0
        for token in ("label", "target", "class", "category", "sentiment", "outcome"):
            if token in lowered:
                score += 10
        for token in ("rating", "score", "value", "y"):
            if token == lowered or token in lowered:
                score += 6
        return score

    suggested = None
    if profiles:
        if task_type == "regression":
            numeric = [
                p for p in profiles
                if pd.api.types.is_numeric_dtype(eng.df[p["name"]])
            ]
            pool = numeric or profiles
        elif task_type == "classification":
            pool = [
                p for p in profiles
                if not pd.api.types.is_numeric_dtype(eng.df[p["name"]]) or p["unique"] <= 50
            ] or profiles
        else:
            pool = profiles
        suggested = max(pool, key=lambda p: (name_score(p["name"]), -p["unique"]))["name"]

    return {"columns": columns, "profiles": profiles, "suggested_label": suggested}


def _route_followup_from_review(state: "AgentState") -> str | None:
    msg = (state.get("user_message") or "").lower()
    if not msg:
        return None

    if any(k in msg for k in ("preprocess", "preprocessing", "stem", "stemming", "lemma", "lemmat", "stopword", "lowercase")):
        state["phase"]["stage"] = "preprocess_config"
        return "preprocess_config_node"
    if any(k in msg for k in ("vector", "vectorization", "tf-idf", "tfidf", "bag of words", "embedding", "embeddings", "ngram", "n-gram")):
        state["phase"]["stage"] = "vector_config"
        return "vector_config_node"
    if any(k in msg for k in ("label", "model", "train", "test size", "fold", "classifier", "regressor", "random forest", "svm", "logistic", "ridge")):
        state["phase"]["stage"] = "model_config"
        return "model_config_node"
    if any(k in msg for k in ("cluster", "clustering", "kmeans", "k-means", "group similar")):
        state["task_type"] = "clustering"
        state["phase"]["stage"] = "clustering_config"
        return "clustering_config_node"
    if any(k in msg for k in ("classify", "classification", "predict", "regression", "start over", "new task")):
        state["phase"]["stage"] = "task_selection"
        return "task_selection_node"
    return None
# ===================== Agent State ===================== #

class PipelinePhase(TypedDict, total=False):
    stage: Literal[
        "idle",
        "task_selection",
        "preprocess_config",
        "preprocess_run",
        "vector_config",
        "vector_run",
        "model_config",
        "model_run",
        "clustering_config",
        "clustering_run",
        "results_explained",
    ]


class AgentState(TypedDict, total=False):
    """
    State passed between LangGraph nodes.

    This is separate from EnglishState, which holds the actual
    document/CSV, preprocessing config, vectorization config, etc.
    """
    user_message: str
    assistant_message: str

    # Shared app state
    english_state: EnglishState

    # Conversation / pipeline state
    phase: PipelinePhase
    task_type: Optional[Literal["classification", "regression", "clustering"]]

    # decisions made so far (preprocessing, vectorization, model, etc.)
    decisions: Dict[str, Any]

    # UI rendering hints (logic vs. message separation)
    ui_intent: Optional[str]
    ui_payload: Dict[str, Any]


# A small helper: create default phase/decisions if missing
def _ensure_defaults(state: AgentState) -> AgentState:
    if "phase" not in state or "stage" not in state["phase"]:
        state["phase"] = {"stage": "task_selection"}
    if "decisions" not in state:
        state["decisions"] = {}
    if "ui_payload" not in state:
        state["ui_payload"] = {}
    if "ui_intent" not in state:
        state["ui_intent"] = None
    return state

# ===================== Explanation intent helper ===================== #

_EXPLANATION_KEYWORDS = [
    "explain",
    "what is",
    "what are",
    "what's",
    "why ",
    "difference between",
    "meaning of",
    "how does",
    "how do",
]


def _is_explanation_request(state: AgentState) -> bool:
    """
    Lightweight heuristic: decide if the user is asking for a conceptual explanation
    rather than changing config.

    This does NOT change any pipeline settings; it only affects routing to
    explanation_node. After explanation, we keep the same phase.stage.
    """
    msg = (state.get("user_message") or "").lower()
    if not msg:
        return False

    return any(kw in msg for kw in _EXPLANATION_KEYWORDS)

# ===================== Nodes ===================== #

def router_node(state: AgentState) -> AgentState:
    """
    Entry node: just ensure defaults and return the state.
    The actual routing decision is done by route_fn below.
    """
    state = _ensure_defaults(state)
    return state
def explanation_node(state: AgentState) -> AgentState:
    """
    Pure explanation mode: answer conceptual questions about the current step
    (preprocessing, vectorization, models, or results) without changing any config.

    We KEEP state['phase']['stage'] unchanged, so the user returns to the same step
    after getting an explanation.
    """
    state = _ensure_defaults(state)
    eng = state["english_state"]
    stage = state["phase"]["stage"]
    user_msg = state.get("user_message", "")

    # Rough context description based on current stage
    if stage.startswith("preprocess"):
        context = (
            "We are currently configuring the text preprocessing pipeline "
            "(operations like lowercasing, removing URLs, stopwords, etc.)."
        )
    elif stage.startswith("vector"):
        context = (
            "We are currently configuring how to turn text into numeric vectors "
            "(methods like TF-IDF, Bag-of-Words, or transformer embeddings)."
        )
    elif stage.startswith("model"):
        context = (
            "We are currently configuring or training a supervised model "
            "(e.g. logistic regression, SVM, random forest) on the vectorized text."
        )
    elif stage.startswith("results"):
        context = (
            "We have already trained a model and are looking at its evaluation results "
            "(e.g. accuracy, confusion matrix, or regression metrics)."
        )
    else:
        context = (
            "We are inside an NLP workbench that helps configure preprocessing, "
            "vectorization, and supervised models on text data."
        )

    # Optional: include a tiny bit of current config, just for richer explanations
    text_col = getattr(eng, "csv_text_column", None)
    has_df = eng.df is not None
    dataset_context = ""
    if has_df and text_col:
        dataset_context = (
            f"The user has loaded a table with a text column named '{text_col}'. "
        )
    elif eng.text:
        dataset_context = "The user has loaded free-form text (not a CSV). "

    prompt = (
        "You are an NLP tutor inside a graphical NLP workbench application.\n"
        f"{context}\n"
        f"{dataset_context}\n"
        "The user is asking for an explanation or clarification. "
        "You should **not** change any configuration; just explain the concept "
        "in clear, friendly language.\n\n"
        "User question:\n"
        f"{user_msg}\n\n"
        "Please answer in 1-3 short paragraphs, optionally with a simple example. "
        "Avoid code unless the user explicitly asked for code. "
        "Do not ask the user questions back; just explain.\n"
    )

    llm = _get_llm()
    if llm is None:
        state["assistant_message"] = _NO_LLM_MSG
        return state
    resp = llm.invoke([HumanMessage(content=prompt)])
    state["assistant_message"] = resp.content

    # IMPORTANT: do NOT change state['phase']['stage'] here.
    # We want to stay in the same step after explaining.
    return state


def task_selection_node(state: AgentState) -> AgentState:
    """
    Use the user's message to decide (via LLM) what they want:
    - classification
    - regression
    - clustering
    """
    state = _ensure_defaults(state)
    user_msg = state.get("user_message", "")

    llm = _get_structured_llm(TaskSelectionDecision)
    if llm is None:
        state["assistant_message"] = _NO_LLM_MSG
        return state
    decision: TaskSelectionDecision = llm.invoke(
        [HumanMessage(content=agent_prompts.build_task_selection_prompt(user_msg))]
    )

    if decision.task_type == "unknown":
        state["task_type"] = None
    else:
        state["task_type"] = decision.task_type

    # Store UI intent/payload for renderer
    state["ui_intent"] = "task_selection_feedback"
    state["ui_payload"] = {
        "user_message": user_msg,
        "task_type": decision.task_type,
        "explanation": decision.explanation,
        "needs_clarification": decision.needs_clarification,
        "clarification_question": decision.clarification_question,
    }

    if decision.task_type == "unknown" or decision.needs_clarification:
        state["phase"]["stage"] = "task_selection"
    else:
        state["phase"]["stage"] = "preprocess_config"
    return state


def preprocess_config_node(state: AgentState) -> AgentState:
    """
    Use the LLM (structured) to propose or UPDATE preprocessing steps.
    This node loops until the LLM marks the config as ready_to_apply=True.
    """
    state = _ensure_defaults(state)
    eng = state["english_state"]
    decisions = state["decisions"]

    user_msg = state.get("user_message", "")

    # Get a small sample text
    if enlp.is_csv_mode(eng) and eng.df is not None and eng.csv_text_column:
        sample_series = eng.df[eng.csv_text_column].dropna().astype(str)
        sample_rows = sample_series.head(3).tolist()
        sample_text = "\n---\n".join(sample_rows)
    else:
        sample_text = (eng.text or "")[:1000]

    # Check if we already have a pipeline proposal from a previous turn
    existing_cfg = decisions.get("preprocess_config")
    previous_steps = None
    if existing_cfg and "steps" in existing_cfg:
        previous_steps = existing_cfg["steps"]

    llm = _get_structured_llm(PreprocessConfigDecision)
    if llm is None:
        state["assistant_message"] = _NO_LLM_MSG
        return state
    decision: PreprocessConfigDecision = llm.invoke(
        [
            HumanMessage(
                content=agent_prompts.build_preprocess_config_prompt(
                    sample_text=sample_text,
                    previous_steps=previous_steps,
                    user_message=user_msg,
                )
            )
        ]
    )

    # Store the latest config
    decisions["preprocess_config"] = decision.model_dump()

    # UI intent: show explanation + possibly ask for further edits
    state["ui_intent"] = "preprocess_config_proposal"
    state["ui_payload"] = {
        "explanation": decision.explanation,
        "steps": [s.model_dump() for s in decision.steps],
        "needs_clarification": decision.needs_clarification,
        "clarification_question": decision.clarification_question,
        "ready_to_apply": decision.ready_to_apply,
    }

    # If ready_to_apply -> next stage is preprocess_run (apply pipeline)
    # else stay in preprocess_config (user can respond with more edits)
    if decision.ready_to_apply:
        state["phase"]["stage"] = "preprocess_run"
    else:
        state["phase"]["stage"] = "preprocess_config"

    return state



def preprocess_run_node(state: AgentState) -> AgentState:
    """
    Apply preprocessing config to a small preview, using the LLM-proposed
    pipeline if available.
    """
    state = _ensure_defaults(state)
    eng = state["english_state"]
    decisions = state["decisions"]

    # If we have an LLM-proposed pipeline, use it; otherwise fall back to a reasonable default
    preprocess_cfg = decisions.get("preprocess_config")

    if preprocess_cfg and "steps" in preprocess_cfg:
        steps_cfg = []
        for s in preprocess_cfg["steps"]:
            step = {
                "id": s["id"],
                "enabled": s.get("enabled", True),
            }
            if s["id"] == "short_tokens" and s.get("min_len") is not None:
                step["min_len"] = s["min_len"]
            steps_cfg.append(step)
        eng.pipeline_config = {"steps": steps_cfg}
    else:
        # Fallback default if something went wrong
        eng.pipeline_config = {
            "steps": [
                {"id": "lowercase", "enabled": True},
                {"id": "urls", "enabled": True},
                {"id": "contractions", "enabled": True},
                {"id": "stopwords", "enabled": True},
                {"id": "short_tokens", "enabled": True, "min_len": 3},
            ]
        }

    # Build BEFORE/AFTER preview
    preview_before: List[str] = []
    preview_after: List[str] = []

    if enlp.is_csv_mode(eng) and eng.df is not None and eng.csv_text_column:
        series = eng.df[eng.csv_text_column].dropna().astype(str).head(3)
        for s in series:
            preview_before.append(s)
            preview_after.append(
                enlp.apply_pipeline(s, cfg=eng.pipeline_config, stopword_set=eng.stopwords)
            )
    else:
        full = eng.text or ""
        chunks = [c for c in full.split("\n") if c.strip()][:3]
        for s in chunks:
            preview_before.append(s)
            preview_after.append(
                enlp.apply_pipeline(s, cfg=eng.pipeline_config, stopword_set=eng.stopwords)
            )

    state["ui_intent"] = "preprocess_preview"
    state["ui_payload"] = {
        "preview_before": preview_before,
        "preview_after": preview_after,
    }

    state["phase"]["stage"] = "vector_config"
    return state


def vector_config_node(state: AgentState) -> AgentState:
    """
    Ask the LLM (structured) to propose or UPDATE a vectorization method + params.
    This node loops until ready_to_apply=True.
    """
    state = _ensure_defaults(state)
    eng = state["english_state"]
    decisions = state["decisions"]

    user_msg = state.get("user_message", "")

    # Rough size estimate
    n_rows = 0
    if eng.df is not None and eng.csv_text_column:
        n_rows = eng.df[eng.csv_text_column].dropna().shape[0]
    elif eng.text:
        n_rows = len([l for l in eng.text.splitlines() if l.strip()])

    # Existing vector config, if any
    existing_vec_cfg = decisions.get("vector_config")

    llm = _get_structured_llm(VectorConfigDecision)
    if llm is None:
        state["assistant_message"] = _NO_LLM_MSG
        return state
    decision: VectorConfigDecision = llm.invoke(
        [
            HumanMessage(
                content=agent_prompts.build_vector_config_prompt(
                    n_rows=n_rows,
                    previous_config=existing_vec_cfg,
                    user_message=user_msg,
                )
            )
        ]
    )

    # Store in english_state
    eng.last_vector_method = decision.method
    eng.last_vector_ngram = decision.ngram_range
    eng.last_vector_max_features = decision.max_features

    # Also store full decision in decisions
    decisions["vector_config"] = decision.model_dump()

    # UI intent: explain suggestion + ask for confirmation/modification
    state["ui_intent"] = "vector_config_feedback"
    state["ui_payload"] = {
        "n_rows": n_rows,
        "method": decision.method,
        "ngram_range": list(decision.ngram_range),
        "max_features": decision.max_features,
        "explanation": decision.explanation,
        "needs_clarification": decision.needs_clarification,
        "clarification_question": decision.clarification_question,
        "ready_to_apply": decision.ready_to_apply,
    }

    # If ready_to_apply -> move on to vector_run, else stay here for more edits
    if decision.ready_to_apply:
        state["phase"]["stage"] = "vector_run"
    else:
        state["phase"]["stage"] = "vector_config"

    return state



def vector_run_node(state: AgentState) -> AgentState:
    """
    Acknowledge the chosen vectorization settings and move to model config.
    Message text is delegated to renderer.
    """
    state = _ensure_defaults(state)
    eng = state["english_state"]

    method = eng.last_vector_method or "TF-IDF (word)"
    ngram = eng.last_vector_ngram or (1, 2)
    max_feat = eng.last_vector_max_features
    label_context = _candidate_label_context(eng, state.get("task_type"))

    state["ui_intent"] = "vector_config_confirmed"
    state["ui_payload"] = {
        "method": method,
        "ngram_range": list(ngram),
        "max_features": max_feat,
        "task_type": state.get("task_type"),
        "candidate_columns": label_context["columns"],
        "column_profiles": label_context["profiles"],
        "suggested_label": label_context["suggested_label"],
    }

    if state.get("task_type") == "clustering":
        state["phase"]["stage"] = "clustering_config"
    else:
        state["phase"]["stage"] = "model_config"
    return state


def clustering_config_node(state: AgentState) -> AgentState:
    state = _ensure_defaults(state)
    eng = state["english_state"]
    decisions = state["decisions"]
    user_msg = state.get("user_message", "")

    n_rows = 0
    if eng.df is not None and eng.csv_text_column:
        n_rows = eng.df[eng.csv_text_column].dropna().shape[0]
    elif eng.text:
        n_rows = len([l for l in eng.text.splitlines() if l.strip()])

    llm = _get_structured_llm(ClusteringConfigDecision)
    if llm is None:
        state["assistant_message"] = _NO_LLM_MSG
        return state
    decision: ClusteringConfigDecision = llm.invoke(
        [
            HumanMessage(
                content=agent_prompts.build_clustering_config_prompt(
                    user_message=user_msg,
                    n_rows=n_rows,
                    previous_config=decisions.get("clustering_config"),
                )
            )
        ]
    )
    n_clusters = max(2, min(int(decision.n_clusters or 3), 20))
    decisions["clustering_config"] = {
        "n_clusters": n_clusters,
        "explanation": decision.explanation,
    }

    state["ui_intent"] = "clustering_config_feedback"
    state["ui_payload"] = {
        "n_clusters": n_clusters,
        "explanation": decision.explanation,
        "needs_clarification": decision.needs_clarification,
        "clarification_question": decision.clarification_question,
        "ready_to_apply": decision.ready_to_apply,
    }
    state["phase"]["stage"] = "clustering_run" if decision.ready_to_apply else "clustering_config"
    return state


def clustering_run_node(state: AgentState) -> AgentState:
    state = _ensure_defaults(state)
    eng = state["english_state"]
    decisions = state["decisions"]

    vector_method = eng.last_vector_method or "TF-IDF (word)"
    ngram = eng.last_vector_ngram or (1, 2)
    max_feat = eng.last_vector_max_features or 5000
    n_clusters = int(decisions.get("clustering_config", {}).get("n_clusters", 3))

    try:
        X, labels_for_display = unsup.build_X_from_state(
            eng,
            vector_method=vector_method,
            ngram_range=tuple(ngram),
            max_features=max_feat,
        )
        n_clusters = max(2, min(n_clusters, len(labels_for_display)))
        result = unsup.run_kmeans(X, n_clusters=n_clusters, random_state=settings.get_seed())
    except Exception as e:
        state["ui_intent"] = "training_failed"
        state["ui_payload"] = {"error": str(e)}
        state["phase"]["stage"] = "clustering_config"
        return state

    export_df = pd.DataFrame(
        {
            "sample": labels_for_display,
            "cluster": result.labels,
        }
    )
    eng.last_cluster_export = export_df

    cluster_sizes = {}
    examples = {}
    for cluster_id in sorted(set(int(x) for x in result.labels)):
        mask = result.labels == cluster_id
        cluster_sizes[str(cluster_id)] = int(mask.sum())
        examples[str(cluster_id)] = [
            labels_for_display[i]
            for i, belongs in enumerate(mask)
            if belongs
        ][:5]

    state["ui_intent"] = "clustering_summary"
    state["ui_payload"] = {
        "n_clusters": int(result.n_clusters),
        "cluster_sizes": cluster_sizes,
        "examples": examples,
        "vectorization": {
            "method": vector_method,
            "ngram_range": list(ngram),
            "max_features": max_feat,
        },
        "seed": settings.get_seed(),
    }
    state["phase"]["stage"] = "results_explained"
    return state


def model_config_node(state: AgentState) -> AgentState:
    """
    Use LLM (structured) to interpret user's preferences about label column
    and model choice, then store them in decisions.
    """
    state = _ensure_defaults(state)
    eng = state["english_state"]
    decisions = state["decisions"]

    user_msg = state.get("user_message", "").strip()

    label_context = _candidate_label_context(eng, state.get("task_type"))
    candidate_cols: List[str] = label_context["columns"]

    llm = _get_structured_llm(ModelConfigDecision)
    if llm is None:
        state["assistant_message"] = _NO_LLM_MSG
        return state
    decision: ModelConfigDecision = llm.invoke(
        [
            HumanMessage(
                content=agent_prompts.build_model_config_prompt(
                    user_message=user_msg,
                    candidate_columns=candidate_cols,
                    candidate_profiles=label_context["profiles"],
                    suggested_label=label_context["suggested_label"],
                    task_type=state.get("task_type"),
                )
            )
        ]
    )

    decisions["label_column"] = decision.label_column
    decisions["model_name"] = decision.model_name
    decisions["test_size"] = decision.test_size
    decisions["cv_folds"] = max(2, min(int(decision.cv_folds or 5), 20))
    decisions["auto_subset_size"] = max(0, int(decision.auto_subset_size or 0))
    decisions["rf_estimators"] = max(10, min(int(decision.rf_estimators or 200), 2000))

    if not decision.label_column:
        # Needs clarification about label column
        state["ui_intent"] = "model_config_needs_clarification"
        state["ui_payload"] = {
            "candidate_columns": candidate_cols,
            "suggested_label": label_context["suggested_label"],
            "explanation": decision.explanation,
            "needs_clarification": True,
            "clarification_question": decision.clarification_question
            or "Please specify exactly which column you'd like to predict.",
        }
        # Stay in model_config until user clarifies
        state["phase"]["stage"] = "model_config"
        return state

    if eng.df is not None and decision.label_column in eng.df.columns:
        inferred_task = (
            "regression"
            if pd.api.types.is_numeric_dtype(eng.df[decision.label_column])
            else "classification"
        )
        if decision.model_name not in sup.available_model_names(inferred_task):
            decisions["model_name"] = "Auto"

    # Label column decided: proceed to training next
    state["ui_intent"] = "model_config_confirmed"
    state["ui_payload"] = {
        "label_column": decision.label_column,
        "model_name": decisions["model_name"],
        "test_size": decision.test_size,
        "cv_folds": decisions["cv_folds"],
        "auto_subset_size": decisions["auto_subset_size"],
        "rf_estimators": decisions["rf_estimators"],
        "explanation": decision.explanation,
    }

    state["phase"]["stage"] = "model_run"
    return state


def model_run_node(state: AgentState) -> AgentState:
    """
    Call your supervised training pipeline and store a structured summary
    of the results for the renderer to explain.
    """
    state = _ensure_defaults(state)
    eng = state["english_state"]
    decisions = state["decisions"]

    label_col = decisions.get("label_column")
    if not label_col:
        # No label column: go back to model_config
        state["ui_intent"] = "training_failed"
        state["ui_payload"] = {"error": "No label column has been set."}
        state["phase"]["stage"] = "model_config"
        return state

    vector_method = eng.last_vector_method or "TF-IDF (word)"
    ngram = eng.last_vector_ngram or (1, 2)
    max_feat = eng.last_vector_max_features or 5000
    model_name = decisions.get("model_name", "Auto")
    test_size = float(decisions.get("test_size", 0.2))
    cv_folds = int(decisions.get("cv_folds", 5))
    auto_subset_size = int(decisions.get("auto_subset_size", 0)) or None
    rf_estimators = int(decisions.get("rf_estimators", 200))
    model_params = {"n_estimators": rf_estimators}

    try:
        result = sup.train_supervised_from_state(
            eng,
            label_column=label_col,
            vector_method=vector_method,
            ngram_range=tuple(ngram),
            max_features=max_feat,
            model_name=model_name,
            test_size=test_size,
            random_state=settings.get_seed(),
            cv_folds=cv_folds,
            auto_subset_size=auto_subset_size,
            model_params=model_params,
        )
    except Exception as e:
        state["ui_intent"] = "training_failed"
        state["ui_payload"] = {"error": str(e)}
        state["phase"]["stage"] = "model_config"
        return state

    eng.last_supervised_result = result
    eng.last_supervised_context = {
        "label_column": label_col,
        "vector_method": vector_method,
        "ngram_range": list(ngram),
        "max_features": max_feat,
        "test_size": test_size,
        "seed": settings.get_seed(),
        "cv_folds": cv_folds,
        "auto_subset_size": auto_subset_size,
        "model_params": model_params,
        "source": "Agent Lab",
    }

    # Prepare structured summary for renderer
    if result.task_type == "classification":
        acc = float(result.metrics.get("accuracy", 0.0))
        rep = result.metrics.get("report", {})
        cm = result.cm
        metrics_payload = {
            "task_type": result.task_type,
            "model_name": result.model_name,
            "accuracy": acc,
            "classification_report": rep,
            "confusion_matrix_shape": list(cm.shape),
            "confusion_matrix": cm.tolist(),
        }
    else:
        regression_metrics = {}
        for k, v in result.metrics.items():
            try:
                regression_metrics[k] = float(v)
            except Exception:
                regression_metrics[k] = v
        metrics_payload = {
            "task_type": result.task_type,
            "model_name": result.model_name,
            "metrics": regression_metrics,
        }

    state["ui_intent"] = "results_summary"
    state["ui_payload"] = metrics_payload
    state["phase"]["stage"] = "results_explained"
    return state


def render_message_node(state: AgentState) -> AgentState:
    """
    Turn ui_intent + ui_payload into a natural language message.
    This is the only node that writes assistant_message.
    """
    state = _ensure_defaults(state)
    intent = state.get("ui_intent")
    payload = state.get("ui_payload", {}) or {}
    msg = ""

    if intent == "task_selection_feedback":
        task_type = payload.get("task_type")

        if task_type in ("classification", "regression", "clustering"):
            nice = {
                "classification": "classify texts into categories (e.g. positive/negative, spam/not spam)",
                "regression": "predict a numeric value from text (e.g. rating, score)",
                "clustering": "group similar texts together without labels",
            }
            desc = nice.get(task_type, task_type)
            msg = (
                f"Great, I'll treat this as a **{task_type}** problem, so we'll {desc}.\n\n"
                "Next, I'll propose a preprocessing pipeline for your text. "
                "If this isn't what you meant, just tell me."
            )
        else:
            msg = (
                "I can help once you choose the NLP task for this dataset.\n\n"
                "Pick one of these:\n"
                "- classify texts into categories\n"
                "- predict a numeric value from text\n"
                "- cluster similar texts together\n"
            )

    elif intent == "preprocess_config_proposal":
        msg = _clean_direct_address(payload.get("explanation", ""))
        ready = payload.get("ready_to_apply", False)
        if not ready:
            if payload.get("needs_clarification") and payload.get("clarification_question"):
                msg += "\n\n" + payload["clarification_question"]
            else:
                msg += "\n\nIf this looks good, say so or tell me which steps you'd like to add/remove."
        else:
            msg += (
                "\n\nGot it. I'll use this preprocessing pipeline and apply it to a small "
                "sample next so you can see a before/after preview."
            )

    elif intent == "vector_config_feedback":
        msg = _clean_direct_address(payload.get("explanation", ""))
        ready = payload.get("ready_to_apply", False)
        if not ready:
            if payload.get("needs_clarification") and payload.get("clarification_question"):
                msg += "\n\n" + payload["clarification_question"]
            else:
                msg += (
                    "\n\nIf this vectorization setup looks OK, say 'OK' or 'go ahead'. "
                    "If you prefer another method (Bag-of-Words, TF-IDF character n-grams, "
                    "transformer embeddings), tell me."
                )
        else:
            msg += (
                "\n\nThese vectorization settings are locked in. Next I'll move on to "
                "configuring the prediction model."
            )

    elif intent == "clustering_config_feedback":
        msg = _clean_direct_address(payload.get("explanation", ""))
        if payload.get("ready_to_apply"):
            msg += (
                f"\n\nI'll run K-Means with {payload.get('n_clusters', 3)} clusters "
                "using the current vectorization settings."
            )
        elif payload.get("needs_clarification") and payload.get("clarification_question"):
            msg += "\n\n" + payload["clarification_question"]
        else:
            msg += "\n\nIf this looks good, say 'go ahead', or tell me how many clusters you want to use."

    elif intent in {
        "preprocess_preview",
        "vector_config_confirmed",
        "results_summary",
        "clustering_summary",
        "training_failed",
    }:
        llm = _get_llm()
        if llm is None:
            state["assistant_message"] = _NO_LLM_MSG
            return state
        prompt = agent_prompts.build_renderer_prompt(intent, payload)
        resp = llm.invoke([HumanMessage(content=prompt)])
        msg = resp.content

    elif intent == "model_config_needs_clarification":
        columns = payload.get("candidate_columns") or []
        suggested = payload.get("suggested_label")
        msg = _clean_direct_address(payload.get("explanation", "")).strip()
        if columns:
            choices = ", ".join(f"`{col}`" for col in columns)
            suffix = f"\n\nAvailable label columns: {choices}."
            if suggested:
                suffix += f"\nMy best guess is `{suggested}`. Confirm that or name another column."
            else:
                suffix += "\nWhich one should I predict?"
            msg = f"{msg}{suffix}" if msg else suffix.strip()
        elif payload.get("clarification_question"):
            msg += "\n\n" + payload["clarification_question"]
        else:
            msg += "\n\nPlease specify exactly which column you'd like to predict."

    elif intent == "model_config_confirmed":
        msg = (
            f"Great. I'll treat `{payload.get('label_column')}` as the label column.\n"
            f"Model choice: **{payload.get('model_name')}**. "
            f"Test size: {payload.get('test_size', 0.2):.2f}. "
            f"CV folds: {payload.get('cv_folds', 5)}. "
            f"Auto subset size: {payload.get('auto_subset_size', 0)}. "
            f"Random forest trees: {payload.get('rf_estimators', 200)}.\n\n"
            f"{_clean_direct_address(payload.get('explanation', ''))}\n\n"
            "I'll now train a model using your current preprocessing and vectorization settings "
            "and then explain the results."
        )

    elif intent == "workflow_idle":
        msg = (
            "We are at the review step. Tell me what you want to adjust next: "
            "preprocessing, vectorization, model or label settings, clustering, or start a new task."
        )

    else:
        llm = _get_llm()
        if llm is None:
            state["assistant_message"] = _NO_LLM_MSG
            return state
        prompt = agent_prompts.build_renderer_prompt(intent or "generic", payload)
        resp = llm.invoke([HumanMessage(content=prompt)])
        msg = resp.content

    state["assistant_message"] = msg
    state["ui_intent"] = None
    state["ui_payload"] = {}

    return state
def final_message_node(state: AgentState) -> AgentState:
    """
    Handle follow-up turns after a workflow has already produced results.
    """
    state = _ensure_defaults(state)
    state["ui_intent"] = "workflow_idle"
    state["ui_payload"] = {}
    return state

# ===================== Build graph ===================== #

def build_agent_graph():
    """
    Build and compile the LangGraph graph for the Agent tab.
    One user turn:
      router -> (logic node chosen by route_fn) -> render_message_node -> END
    """
    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("router", router_node)
    graph.add_node("task_selection_node", task_selection_node)
    graph.add_node("preprocess_config_node", preprocess_config_node)
    graph.add_node("preprocess_run_node", preprocess_run_node)
    graph.add_node("vector_config_node", vector_config_node)
    graph.add_node("vector_run_node", vector_run_node)
    graph.add_node("model_config_node", model_config_node)
    graph.add_node("model_run_node", model_run_node)
    graph.add_node("clustering_config_node", clustering_config_node)
    graph.add_node("clustering_run_node", clustering_run_node)
    graph.add_node("render_message_node", render_message_node)
    graph.add_node("explanation_node", explanation_node)
    graph.add_node("final_message_node", final_message_node)

    # Entry
    graph.set_entry_point("router")

    # router's conditional edges:
    def route_fn(state: AgentState) -> str:
        """
        Decide which node to go to next, based on state.phase.stage and
        whether the user is asking for a conceptual explanation.

        This MUST return a string node name, NOT the state.
        """
        state = _ensure_defaults(state)

        # 1) Explanation override
        if _is_explanation_request(state):
            return "explanation_node"

        # 2) Normal stage-based routing
        stage = state["phase"]["stage"]
        if stage == "results_explained":
            followup_route = _route_followup_from_review(state)
            if followup_route:
                return followup_route

        if stage == "task_selection":
            return "task_selection_node"
        elif stage == "preprocess_config":
            return "preprocess_config_node"
        elif stage == "preprocess_run":
            return "preprocess_run_node"
        elif stage == "vector_config":
            return "vector_config_node"
        elif stage == "vector_run":
            return "vector_run_node"
        elif stage == "model_config":
            return "model_config_node"
        elif stage == "model_run":
            return "model_run_node"
        elif stage == "clustering_config":
            return "clustering_config_node"
        elif stage == "clustering_run":
            return "clustering_run_node"
        elif stage == "results_explained":
            return "final_message_node"
        else:
            state["phase"]["stage"] = "task_selection"
            return "task_selection_node"

    graph.add_conditional_edges(
        "router",
        route_fn,
        {
            "task_selection_node": "task_selection_node",
            "preprocess_config_node": "preprocess_config_node",
            "preprocess_run_node": "preprocess_run_node",
            "vector_config_node": "vector_config_node",
            "vector_run_node": "vector_run_node",
            "model_config_node": "model_config_node",
            "model_run_node": "model_run_node",
            "clustering_config_node": "clustering_config_node",
            "clustering_run_node": "clustering_run_node",
            "final_message_node": "final_message_node",
            "explanation_node": "explanation_node",
        },
    )

    # After every logic node, go to renderer;
    # explanation_node already sets assistant_message, so it goes straight to END.
    for node in [
        "task_selection_node",
        "preprocess_config_node",
        "preprocess_run_node",
        "vector_config_node",
        "vector_run_node",
        "model_config_node",
        "model_run_node",
        "clustering_config_node",
        "clustering_run_node",
        "final_message_node",
    ]:
        graph.add_edge(node, "render_message_node")

    # Explanation node: direct reply, no extra rendering step
    graph.add_edge("explanation_node", END)

    # Renderer -> END
    graph.add_edge("render_message_node", END)

    return graph.compile()


