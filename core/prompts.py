from typing import Any, Dict, List, Optional
import json


# ============ Structured decision prompts ============ #
def build_task_selection_prompt(user_message: str) -> str:
    return (
        "You are a friendly assistant inside an NLP workbench.\n"
        "You are talking directly to the person using the app.\n"
        "Never refer to them as 'the user' or describe what they said; "
        "just respond to them naturally.\n\n"
        "Your goal is to help them choose what they want to do with their text data.\n"
        "Common options include:\n"
        "- text classification (e.g., positive vs negative, spam vs not spam)\n"
        "- regression (predicting a numeric value from text)\n"
        "- clustering (grouping similar texts)\n\n"
        "If their message is vague, random, or unrelated to an NLP task (for example "
        "'start', 'hi', 'hamburger', 'pizza'), set task_type='unknown', "
        "needs_clarification=true, and ask them to choose classification, regression, "
        "or clustering. Do not interpret a random word as a dataset topic.\n"
        "Do not greet them again; the app already has an opening message.\n\n"
        "Style rules:\n"
        "- Be concise and conversational.\n"
        "- Do not explain your reasoning out loud.\n"
        "- Do not mention that you're choosing a task type.\n"
        "- Do not talk about 'their message' or 'the user'.\n\n"
        f"User message:\n{user_message}"
    )


def build_preprocess_config_prompt(
    sample_text: str,
    previous_steps: Optional[List[Dict[str, Any]]] = None,
    user_message: str = "",
) -> str:
    """
    If previous_steps is None -> first proposal (never final).
    If previous_steps is provided -> update existing pipeline according to user_message,
    and mark it final (ready_to_apply=true) when the user clearly specifies or approves it.
    """
    if previous_steps is None:
        prev_part = (
            "There is currently no preprocessing pipeline. "
            "You must propose an initial pipeline, but always set ready_to_apply=false "
            "so that the user can review and modify it.\n"
        )
        confirm_rules = (
            "Because this is the first proposal, even if the user seems happy, "
            "you must still return ready_to_apply=false.\n"
        )
    else:
        prev_part = (
            "The user already has a proposed preprocessing pipeline, given in JSON below. "
            "You must update this existing pipeline according to the user's latest message, "
            "instead of ignoring it.\n\n"
            f"Existing pipeline JSON:\n"
            f"{json.dumps(previous_steps, indent=2, ensure_ascii=False)}\n\n"
        )
        confirm_rules = (
            "Because this is an update to an existing pipeline, use the following rule:\n"
            "- If the user's latest message clearly specifies the pipeline they want "
            "(e.g. 'just do lower casing', 'only lowercase', 'turn off stopwords') "
            "or explicitly approves the current pipeline (e.g. 'ok', 'looks good', "
            "'that is fine', 'go ahead'), then set ready_to_apply=true.\n"
            "- Only keep ready_to_apply=false if the user seems unsure or is asking open-ended "
            "questions and will likely want more changes.\n"
        )

    return (
        "You are an NLP preprocessing assistant working inside a GUI workbench.\n"
        "The user has loaded some text data. Here is a small sample:\n\n"
        f"{sample_text}\n\n"
        "You can choose from these preprocessing steps:\n"
        "- lowercase\n"
        "- urls (remove URLs)\n"
        "- contractions (expand/normalize contractions)\n"
        "- numbers (normalize numeric tokens)\n"
        "- stopwords (remove common stopwords)\n"
        "- short_tokens (remove tokens whose length < 3; if you use this, set min_len)\n"
        "- stem (apply word stemming, e.g. 'running' -> 'run')\n"
        "- lemma (apply lemmatization, e.g. 'better' -> 'good')\n\n"
        f"{prev_part}"
        "User's latest message (may be empty if this is the first proposal):\n"
        f"{user_message}\n\n"
        "Your output must be a JSON object with fields:\n"
        "- steps: list of step objects: {id: one of the step names, enabled: bool, min_len: optional int for short_tokens}\n"
        "- explanation: friendly explanation of the resulting pipeline, addressed directly to the person using 'you'. Never say 'the user'.\n"
        "- ready_to_apply: boolean flag as described in the rules below\n"
        "- needs_clarification: boolean\n"
        "- clarification_question: optional string\n\n"
        f"{confirm_rules}"
    )


def build_vector_config_prompt(
    n_rows: int,
    previous_config: Optional[Dict[str, Any]] = None,
    user_message: str = "",
) -> str:
    """
    If previous_config is None -> first suggestion.
    If previous_config is provided -> update existing vectorization config according to user_message.
    """
    if previous_config is None:
        prev_part = (
            "There is currently no vectorization configuration. "
            "You must propose an initial configuration based on the dataset size and the user's message.\n"
        )
        confirm_rules = (
            "Because this is the first proposal, you should normally set ready_to_apply=false "
            "so the user can adjust it, unless the user explicitly says they are happy and want to proceed.\n"
        )
    else:
        prev_part = (
            "There is an existing vectorization configuration in JSON form below. "
            "You must update this configuration according to the user's latest message, "
            "rather than ignoring it.\n\n"
            f"Existing vectorization config JSON:\n"
            f"{json.dumps(previous_config, indent=2, ensure_ascii=False)}\n\n"
        )
        confirm_rules = (
            "Because this is an update to an existing configuration, follow these rules:\n"
            "- If the user's latest message clearly asks for a specific method "
            "(e.g. 'use bag of words', 'use tf-idf', 'switch to character n-grams'), "
            "you must override any previous method choice and set method to exactly what they asked for.\n"
            "- If the user's message explicitly approves the current configuration "
            "(e.g. 'ok', 'looks good', 'go ahead'), set ready_to_apply=true.\n"
            "- If the user seems unsure or is asking open-ended questions, keep ready_to_apply=false "
            "so they can make further changes.\n"
        )

    return (
        "You are an NLP vectorization assistant inside a GUI workbench.\n"
        f"The dataset has approximately {n_rows} text rows.\n\n"
        f"{prev_part}"
        "User's latest message (may be empty if this is the first suggestion):\n"
        f"{user_message}\n\n"
        "Your job:\n"
        "- Decide which vectorization method is appropriate: "
        "'TF-IDF (word)', 'Bag-of-Words', 'TF-IDF (char)', or 'Transformer embeddings'.\n"
        "- Decide an ngram_range (e.g. [1, 2]) when relevant (for TF-IDF or Bag-of-Words).\n"
        "- Decide a suitable max_features (integer) or null if not applicable.\n"
        "- Explain your reasoning in 'explanation'.\n"
        "- The explanation must address the person directly using 'you'. Never say 'the user'.\n"
        "- Set ready_to_apply based on the rules below.\n"
        "- Optionally ask a clarification question if the user's request is ambiguous.\n\n"
        "Your JSON output must include keys:\n"
        "- method\n"
        "- ngram_range (list of two integers)\n"
        "- max_features (integer or null)\n"
        "- explanation\n"
        "- ready_to_apply (boolean)\n"
        "- needs_clarification (boolean)\n"
        "- clarification_question (string or null)\n\n"
        f"{confirm_rules}"
    )


def build_model_config_prompt(
    user_message: str,
    candidate_columns: List[str],
    task_type: str | None = None,
    candidate_profiles: Optional[List[Dict[str, Any]]] = None,
    suggested_label: Optional[str] = None,
) -> str:
    if task_type == "regression":
        model_options = "'Linear Regression', 'Ridge Regression', 'Random Forest Regressor', 'Auto'"
    else:
        model_options = "'Logistic Regression', 'Linear SVM', 'Random Forest', 'Auto'"

    return (
        "You are configuring a supervised NLP model inside a GUI workbench.\n"
        f"High-level task type (may be null): {task_type}\n\n"
        "These columns are available as potential label columns (text column excluded):\n"
        f"{candidate_columns}\n\n"
        "Column profiles from a small peek at the data:\n"
        f"{json.dumps(candidate_profiles or [], indent=2, ensure_ascii=False)}\n\n"
        f"Suggested label candidate, if any: {suggested_label}\n\n"
        "User message:\n"
        f"{user_message}\n\n"
        "Your job:\n"
        "- Decide which column (if any) should be the label_column.\n"
        "- If the user says something like 'the label column' or misspells 'label', "
        "prefer an actual column named 'label' if present, otherwise use the suggested "
        "label candidate when it is plausible.\n"
        "- If you are not confident, do not invent a column. Ask them to pick from the "
        "available columns exactly as named.\n"
        f"- Choose an appropriate model_name among: {model_options}.\n"
        "- Choose a test_size between 0.1 and 0.4. "
        "If the user mentions a split, honor it; otherwise pick a sensible default.\n"
        "- Choose cv_folds between 2 and 20 for Auto mode, defaulting to 5.\n"
        "- Choose auto_subset_size as 0 unless the user asks to use a subset for faster Auto selection.\n"
        "- Choose rf_estimators between 10 and 2000, defaulting to 200. This only affects random forest models.\n"
        "- Explain your choices directly to the person using 'you'. Never say 'the user'.\n"
        "- If you cannot confidently pick a label column, set needs_clarification=true "
        "and provide a clear question asking the user which label they want.\n"
    )


def build_clustering_config_prompt(
    user_message: str,
    n_rows: int,
    previous_config: Optional[Dict[str, Any]] = None,
) -> str:
    prev_part = ""
    if previous_config:
        prev_part = (
            "There is an existing clustering configuration:\n"
            f"{json.dumps(previous_config, indent=2, ensure_ascii=False)}\n\n"
        )
    return (
        "You are configuring K-Means clustering inside an NLP GUI workbench.\n"
        f"The dataset has approximately {n_rows} text rows.\n\n"
        f"{prev_part}"
        "User message:\n"
        f"{user_message}\n\n"
        "Choose n_clusters as an integer between 2 and 20. If the user does not specify, "
        "use 3 for small exploratory corpora. Set ready_to_apply=true when the user has "
        "specified or approved the number of clusters; otherwise ask a concise clarification question.\n"
        "Write the explanation directly to the person using 'you'. Never say 'the user'.\n"
    )


# ============ Renderer prompts (message generation) ============ #
def build_renderer_prompt(intent: str, payload: Dict[str, Any]) -> str:
    """
    Build a prompt for the render_message_node for intents that rely on the LLM
    to turn JSON payload into a user-friendly message.
    """
    payload_json = json.dumps(payload, indent=2, ensure_ascii=False)

    if intent == "preprocess_preview":
        return (
            "You are an assistant inside an NLP GUI workbench.\n"
            "You have just applied a preprocessing pipeline to a few sample texts.\n"
            "Continue the existing workflow naturally. Do not greet the user again.\n"
            "You are given a JSON object with 'preview_before' and 'preview_after' arrays.\n\n"
            "Write a concise explanation that:\n"
            "- Shows each BEFORE/AFTER pair in a clear way.\n"
            "- Briefly explains that this pipeline will be reused later.\n"
            "- Ends by saying that next you'll help choose a vectorization method.\n\n"
            f"JSON payload:\n{payload_json}"
        )

    if intent == "vector_config_confirmed":
        return (
            "You are an assistant inside an NLP GUI workbench.\n"
            "You have just finalized a vectorization configuration.\n"
            "Continue the existing workflow naturally. Do not greet the user again.\n"
            "You are given a JSON payload with:\n"
            "- method (string)\n"
            "- ngram_range ([min, max])\n"
            "- max_features (int or null)\n"
            "- task_type (may be null)\n"
            "- candidate_columns (possible label columns)\n"
            "- column_profiles (dtype, missing values, unique count, and sample values)\n"
            "- suggested_label (best guess, may be null)\n\n"
            "Write a short, friendly message that:\n"
            "- Summarizes the chosen vectorization settings.\n"
            "- Explains why this is a reasonable choice.\n"
            "- If this is classification or regression, lists the available label columns "
            "and asks the user to confirm the label column.\n"
            "- If suggested_label is present, suggest it as the likely label and explain briefly why.\n"
            "- Do not ask a generic label question without showing the exact column names.\n\n"
            f"JSON payload:\n{payload_json}"
        )

    if intent == "results_summary":
        return (
            "You are an assistant inside an NLP GUI workbench.\n"
            "A text model has just been trained. Continue naturally and do not greet the user.\n"
            "You are given a JSON payload describing the results. For classification, the payload contains:\n"
            "- task_type\n"
            "- model_name\n"
            "- accuracy\n"
            "- classification_report (a mapping from label -> metrics)\n"
            "- confusion_matrix (2D array)\n"
            "- confusion_matrix_shape\n"
            "For regression, the payload contains:\n"
            "- task_type\n"
            "- model_name\n"
            "- metrics (e.g. MAE, MSE, R2).\n\n"
            "Write a concise, user-friendly explanation of the results:\n"
            "- For classification, explain accuracy and the confusion matrix "
            "(rows = true labels, columns = predicted labels).\n"
            "- Comment briefly on whether performance is good/poor and any imbalance.\n"
            "- For regression, explain MAE/MSE/R2 in simple terms.\n\n"
            f"JSON payload:\n{payload_json}"
        )

    if intent == "clustering_summary":
        return (
            "You are an assistant inside an NLP GUI workbench.\n"
            "K-Means clustering has just completed. Continue naturally and do not greet the user.\n"
            "You are given a JSON payload with n_clusters, cluster_sizes, vectorization settings, "
            "and a few example rows per cluster.\n\n"
            "Write a concise, user-friendly summary that:\n"
            "- states how many clusters were found\n"
            "- summarizes the cluster sizes\n"
            "- mentions that the result can be exported as CSV from Prediction\n"
            "- does not overclaim semantic meaning from K-Means alone\n\n"
            f"JSON payload:\n{payload_json}"
        )

    if intent == "training_failed":
        return (
            "You are an assistant inside an NLP GUI workbench.\n"
            "Model training has failed. Continue naturally and do not greet the user.\n"
            "You are given a JSON payload with an 'error' string.\n"
            "Write a short, empathetic message that:\n"
            "- States that training failed.\n"
            "- Shows the raw error message in a code block.\n"
            "- Suggests a couple of likely causes and next steps (e.g. check label column, "
            "missing values, small dataset).\n\n"
            f"JSON payload:\n{payload_json}"
        )

    return (
        "You are an assistant inside an NLP GUI workbench.\n"
        "Continue naturally and do not greet the user. "
        "You are given a JSON payload describing the situation. "
        "Write a short, friendly message explaining what happened.\n\n"
        f"Intent: {intent}\n"
        f"Payload:\n{payload_json}"
    )
