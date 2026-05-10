from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from core import english_nlp as enlp
from core import vectorization as vec
from core.english_state import EnglishState


TaskType = Literal["classification", "regression"]


@dataclass
class SupervisedResult:
    task_type: TaskType
    model_name: str
    model: Any
    metrics: Dict[str, Any]
    cm: Optional[np.ndarray]
    y_true: np.ndarray
    y_pred: np.ndarray
    test_indices: Optional[np.ndarray] = None


def build_docs_y_from_state(
    state: EnglishState,
    label_column: str,
) -> Tuple[List[str], np.ndarray, TaskType, np.ndarray]:
    """Return preprocessed documents, labels, task type, and source row indices."""
    if not enlp.is_csv_mode(state):
        raise ValueError("Supervised learning currently only supports CSV mode.")

    if state.df is None or state.csv_text_column is None:
        raise ValueError("CSV and text column must be set in state.")

    df = state.df.dropna(subset=[state.csv_text_column, label_column]).copy()
    raw_docs = df[state.csv_text_column].astype(str).tolist()
    docs = [
        enlp.apply_pipeline(
            d or "",
            cfg=state.pipeline_config,
            stopword_set=state.stopwords,
        ).strip()
        for d in raw_docs
    ]

    keep_positions = [i for i, d in enumerate(docs) if d]
    docs = [docs[i] for i in keep_positions]
    df = df.iloc[keep_positions]

    if not docs:
        raise ValueError("No non-empty documents after preprocessing.")

    y = df[label_column].values
    task_type: TaskType = (
        "regression" if np.issubdtype(df[label_column].dtype, np.number) else "classification"
    )
    return docs, y, task_type, df.index.to_numpy()


def build_xy_from_state(
    state: EnglishState,
    label_column: str,
    vector_method: str,
    ngram_range: Tuple[int, int],
    max_features: int,
) -> Tuple[np.ndarray, np.ndarray, TaskType]:
    """Compatibility helper for callers that still want a prebuilt feature matrix."""
    docs, y, task_type, _ = build_docs_y_from_state(state, label_column)
    X = _vectorize_docs(docs, vector_method, ngram_range, max_features)
    return X, y, task_type


def _vectorize_docs(
    docs: List[str],
    vector_method: str,
    ngram_range: Tuple[int, int],
    max_features: int,
) -> np.ndarray:
    if vector_method == "Bag-of-Words":
        res, _ = vec.vectorize_bow(docs, ngram_range=ngram_range, max_features=max_features)
        return res.vector.toarray()
    if vector_method == "TF-IDF (word)":
        res, _ = vec.vectorize_tfidf(
            docs, ngram_range=ngram_range, max_features=max_features, analyzer="word"
        )
        return res.vector.toarray()
    if vector_method == "TF-IDF (char)":
        res, _ = vec.vectorize_tfidf(
            docs, ngram_range=ngram_range, max_features=max_features, analyzer="char"
        )
        return res.vector.toarray()
    if vector_method == "Transformer embeddings":
        return vec.transformer_sentence_embeddings(docs).vector
    raise ValueError(f"Unknown vectorization method: {vector_method}")


def _make_vectorizer(vector_method: str, ngram_range: Tuple[int, int], max_features: int):
    if vector_method == "Bag-of-Words":
        return CountVectorizer(ngram_range=ngram_range, max_features=max_features)
    if vector_method == "TF-IDF (word)":
        return TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            analyzer="word",
        )
    if vector_method == "TF-IDF (char)":
        return TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            analyzer="char",
        )
    raise ValueError(f"Unknown vectorization method: {vector_method}")


def _make_classification_model(model_name: str, random_state: int = 42):
    if model_name == "Logistic Regression":
        return LogisticRegression(max_iter=500, n_jobs=-1, random_state=random_state), model_name
    if model_name == "Linear SVM":
        return LinearSVC(random_state=random_state), model_name
    if model_name == "Random Forest":
        return (
            RandomForestClassifier(
                n_estimators=200,
                n_jobs=-1,
                random_state=random_state,
            ),
            model_name,
        )
    raise ValueError(f"Unknown classification model: {model_name}")


def _make_regression_model(model_name: str, random_state: int = 42):
    if model_name == "Linear Regression":
        return LinearRegression(), model_name
    if model_name == "Ridge Regression":
        return Ridge(alpha=1.0, random_state=random_state), model_name
    if model_name == "Random Forest Regressor":
        return (
            RandomForestRegressor(
                n_estimators=200,
                n_jobs=-1,
                random_state=random_state,
            ),
            model_name,
        )
    raise ValueError(f"Unknown regression model: {model_name}")


def _candidate_models(task_type: TaskType, random_state: int):
    if task_type == "classification":
        names = ["Logistic Regression", "Linear SVM", "Random Forest"]
        return [
            (name, model)
            for model, name in (
                _make_classification_model(n, random_state) for n in names
            )
        ]
    names = ["Linear Regression", "Ridge Regression", "Random Forest Regressor"]
    return [
        (name, model)
        for model, name in (
            _make_regression_model(n, random_state) for n in names
        )
    ]


def _single_model(task_type: TaskType, model_name: str, random_state: int):
    if task_type == "classification":
        return _make_classification_model(model_name, random_state)
    return _make_regression_model(model_name, random_state)


def _stratify_or_none(y: np.ndarray):
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) > 1 and counts.min() >= 2:
        return y
    return None


def _cv_for(task_type: TaskType, y_train: np.ndarray, random_state: int, max_folds: int):
    n_samples = len(y_train)
    if n_samples < 4:
        return None
    folds = min(max_folds, n_samples)
    if task_type == "classification":
        _, counts = np.unique(y_train, return_counts=True)
        folds = min(folds, int(counts.min()))
        if folds < 2:
            return None
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    if folds < 2:
        return None
    return KFold(n_splits=folds, shuffle=True, random_state=random_state)


def _score_name(task_type: TaskType) -> str:
    return "accuracy" if task_type == "classification" else "r2"


def _evaluate_predictions(
    estimator,
    X_test,
    y_test,
    task_type: TaskType,
    name: str,
    candidate_scores: Optional[Dict[str, float]] = None,
    selection_strategy: Optional[str] = None,
) -> SupervisedResult:
    y_pred = estimator.predict(X_test)
    if task_type == "classification":
        metrics: Dict[str, Any] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        }
        if candidate_scores:
            metrics["candidate_scores"] = candidate_scores
        if selection_strategy:
            metrics["selection_strategy"] = selection_strategy
        return SupervisedResult(
            task_type="classification",
            model_name=name,
            model=estimator,
            metrics=metrics,
            cm=confusion_matrix(y_test, y_pred),
            y_true=y_test,
            y_pred=y_pred,
        )

    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred),
    }
    if candidate_scores:
        metrics["candidate_scores"] = candidate_scores
    if selection_strategy:
        metrics["selection_strategy"] = selection_strategy
    return SupervisedResult(
        task_type="regression",
        model_name=name,
        model=estimator,
        metrics=metrics,
        cm=None,
        y_true=y_test,
        y_pred=y_pred,
    )


def _select_candidate(
    candidates,
    X_train,
    y_train,
    task_type: TaskType,
    random_state: int,
    cv_folds: int,
    progress_callback=None,
):
    cv = _cv_for(task_type, y_train, random_state, cv_folds)
    scoring = _score_name(task_type)
    results: List[Tuple[str, Any, float]] = []
    selection_strategy = "cross_validation" if cv is not None else "training_score_no_cv"

    for i, (name, estimator) in enumerate(candidates):
        if progress_callback is not None:
            progress_callback(i / max(len(candidates), 1), f"Trying {name}...")
        try:
            if cv is not None:
                scores = cross_val_score(estimator, X_train, y_train, cv=cv, scoring=scoring)
                score = float(np.nanmean(scores))
            else:
                fitted = clone(estimator).fit(X_train, y_train)
                score = float(fitted.score(X_train, y_train))
            results.append((name, estimator, score))
        except Exception:
            continue

    if not results:
        raise RuntimeError(f"All candidate {task_type} models failed during Auto mode.")
    best_name, best_estimator, _ = max(results, key=lambda item: item[2])
    return best_name, best_estimator, {name: score for name, _, score in results}, selection_strategy


def train_supervised_model(
    X: np.ndarray,
    y: np.ndarray,
    task_type: TaskType,
    model_name: str = "Auto",
    test_size: float = 0.2,
    random_state: int = 42,
    cv_folds: int = 5,
    cancel_event=None,
    progress_callback=None,
) -> SupervisedResult:
    """Train on a feature matrix. Prefer train_supervised_from_state for text data."""
    stratify = _stratify_or_none(y) if task_type == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    if cancel_event is not None and cancel_event.is_set():
        from core.task_runner import CancelledError

        raise CancelledError()

    if model_name == "Auto":
        name, estimator, scores, strategy = _select_candidate(
            _candidate_models(task_type, random_state),
            X_train,
            y_train,
            task_type,
            random_state,
            cv_folds,
            progress_callback,
        )
    else:
        estimator, name = _single_model(task_type, model_name, random_state)
        scores = None
        strategy = None

    estimator.fit(X_train, y_train)
    if progress_callback is not None:
        progress_callback(1.0, "Done.")
    return _evaluate_predictions(estimator, X_test, y_test, task_type, name, scores, strategy)


def train_supervised_from_state(
    state: EnglishState,
    label_column: str,
    vector_method: str,
    ngram_range: Tuple[int, int],
    max_features: int,
    model_name: str = "Auto",
    test_size: float = 0.2,
    random_state: int = 42,
    cv_folds: int = 5,
    cancel_event=None,
    progress_callback=None,
) -> SupervisedResult:
    """
    Split raw/preprocessed documents first, then fit vectorizers only on training data.

    Auto mode uses cross-validation on the training split only, then evaluates the
    selected model on the held-out test split.
    """
    docs, y, task_type, source_indices = build_docs_y_from_state(state, label_column)
    all_positions = np.arange(len(docs))
    stratify = _stratify_or_none(y) if task_type == "classification" else None

    train_idx, test_idx = train_test_split(
        all_positions,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    train_docs = [docs[i] for i in train_idx]
    test_docs = [docs[i] for i in test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    if cancel_event is not None and cancel_event.is_set():
        from core.task_runner import CancelledError

        raise CancelledError()

    if vector_method == "Transformer embeddings":
        if progress_callback is not None:
            progress_callback(0.1, "Building transformer embeddings...")
        X_train = vec.transformer_sentence_embeddings(train_docs).vector
        X_test = vec.transformer_sentence_embeddings(test_docs).vector

        if model_name == "Auto":
            name, estimator, scores, strategy = _select_candidate(
                _candidate_models(task_type, random_state),
                X_train,
                y_train,
                task_type,
                random_state,
                cv_folds,
                progress_callback,
            )
        else:
            estimator, name = _single_model(task_type, model_name, random_state)
            scores = None
            strategy = None
        estimator.fit(X_train, y_train)
        result = _evaluate_predictions(estimator, X_test, y_test, task_type, name, scores, strategy)
        result.test_indices = source_indices[test_idx]
        return result

    vectorizer = _make_vectorizer(vector_method, ngram_range, max_features)

    def as_pipeline(model):
        return Pipeline([("vectorizer", clone(vectorizer)), ("model", model)])

    if model_name == "Auto":
        pipe_candidates = [
            (name, as_pipeline(model))
            for name, model in _candidate_models(task_type, random_state)
        ]
        name, estimator, scores, strategy = _select_candidate(
            pipe_candidates,
            train_docs,
            y_train,
            task_type,
            random_state,
            cv_folds,
            progress_callback,
        )
    else:
        model, name = _single_model(task_type, model_name, random_state)
        estimator = as_pipeline(model)
        scores = None
        strategy = None

    estimator.fit(train_docs, y_train)
    if progress_callback is not None:
        progress_callback(1.0, "Done.")
    result = _evaluate_predictions(estimator, test_docs, y_test, task_type, name, scores, strategy)
    result.test_indices = source_indices[test_idx]
    return result
