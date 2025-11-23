# core/supervised.py

from dataclasses import dataclass
from typing import Literal, Any, Dict, Optional, Tuple, List

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    Ridge,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from core.english_state import EnglishState
from core import english_nlp as enlp
from core import vectorization as vec


TaskType = Literal["classification", "regression"]


@dataclass
class SupervisedResult:
    task_type: TaskType
    model_name: str
    model: Any
    metrics: Dict[str, Any]          # can include "candidate_scores" when Auto is used
    cm: Optional[np.ndarray]         # confusion matrix for classification
    y_true: np.ndarray
    y_pred: np.ndarray


# -------------------------------------------------------------------
# Build X, y from EnglishState
# -------------------------------------------------------------------

def build_xy_from_state(
    state: EnglishState,
    label_column: str,
    vector_method: str,
    ngram_range: Tuple[int, int],
    max_features: int,
) -> Tuple[np.ndarray, np.ndarray, TaskType]:
    """
    Build feature matrix X and targets y from the current CSV in state.

    - Uses state.csv_text_column as input text
    - Uses label_column as target
    - Applies the same preprocessing pipeline
    """
    if not enlp.is_csv_mode(state):
        raise ValueError("Supervised learning currently only supports CSV mode.")

    if state.df is None or state.csv_text_column is None:
        raise ValueError("CSV and text column must be set in state.")

    df = state.df

    # Drop rows where either text or label is missing
    df = df.dropna(subset=[state.csv_text_column, label_column]).copy()

    # Preprocess text
    raw_docs = df[state.csv_text_column].astype(str).tolist()
    docs = [
        enlp.apply_pipeline(
            d or "",
            cfg=state.pipeline_config,
            stopword_set=state.stopwords,
        ).strip()
        for d in raw_docs
    ]
    # Filter out completely empty docs (keep label in sync)
    mask = [bool(t) for t in docs]
    df = df.loc[mask]
    docs = [d for d in docs if d]

    if not docs:
        raise ValueError("No non-empty documents after preprocessing.")

    # Build y
    y = df[label_column].values

    # Decide task type: numeric → regression, else classification
    if np.issubdtype(df[label_column].dtype, np.number):
        task_type: TaskType = "regression"
    else:
        task_type = "classification"

    # Vectorize
    if vector_method == "Bag-of-Words":
        res, _ = vec.vectorize_bow(
            docs,
            ngram_range=ngram_range,
            max_features=max_features,
        )
        X = res.vector.toarray()
    elif vector_method == "TF-IDF (word)":
        res, _ = vec.vectorize_tfidf(
            docs,
            ngram_range=ngram_range,
            max_features=max_features,
            analyzer="word",
        )
        X = res.vector.toarray()
    elif vector_method == "TF-IDF (char)":
        res, _ = vec.vectorize_tfidf(
            docs,
            ngram_range=ngram_range,
            max_features=max_features,
            analyzer="char",
        )
        X = res.vector.toarray()
    elif vector_method == "Transformer embeddings":
        res = vec.transformer_sentence_embeddings(docs)
        X = res.vector
    else:
        raise ValueError(f"Unknown vectorization method: {vector_method}")

    return X, y, task_type


# -------------------------------------------------------------------
# Internal helpers to construct models
# -------------------------------------------------------------------

def _make_classification_model(model_name: str):
    if model_name == "Logistic Regression":
        return LogisticRegression(max_iter=200, n_jobs=-1), "Logistic Regression"
    if model_name == "Linear SVM":
        return LinearSVC(), "Linear SVM"
    if model_name == "Random Forest":
        return RandomForestClassifier(n_estimators=200, n_jobs=-1), "Random Forest"
    raise ValueError(f"Unknown classification model: {model_name}")


def _make_regression_model(model_name: str):
    if model_name == "Linear Regression":
        return LinearRegression(), "Linear Regression"
    if model_name == "Ridge Regression":
        return Ridge(alpha=1.0), "Ridge Regression"
    if model_name == "Random Forest Regressor":
        return RandomForestRegressor(n_estimators=200, n_jobs=-1), "Random Forest Regressor"
    raise ValueError(f"Unknown regression model: {model_name}")


# -------------------------------------------------------------------
# Evaluation helpers
# -------------------------------------------------------------------

def _evaluate_classifier(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    name: str,
) -> Dict[str, Any]:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    rep = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "name": name,
        "model": model,
        "y_pred": y_pred,
        "accuracy": acc,
        "report": rep,
        "cm": cm,
    }


def _evaluate_regressor(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    name: str,
) -> Dict[str, Any]:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        "name": name,
        "model": model,
        "y_pred": y_pred,
        "MAE": mae,
        "MSE": mse,
        "R2": r2,
    }


# -------------------------------------------------------------------
# Public training API
# -------------------------------------------------------------------

def train_supervised_model(
    X: np.ndarray,
    y: np.ndarray,
    task_type: TaskType,
    model_name: str = "Auto",
    test_size: float = 0.2,
    random_state: int = 42,
) -> SupervisedResult:
    """
    Train either a classifier or regressor and return metrics.

    If model_name == "Auto":
        - For classification: tries multiple models and picks the best by accuracy.
        - For regression: tries multiple models and picks the best by R^2.
        - Also returns 'candidate_scores' in metrics so the UI can show the sweep.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if task_type == "classification":
        if model_name == "Auto":
            # Try multiple classification models
            candidates = [
                ("Logistic Regression", LogisticRegression(max_iter=200, n_jobs=-1)),
                ("Linear SVM", LinearSVC()),
                ("Random Forest", RandomForestClassifier(n_estimators=200, n_jobs=-1)),
            ]

            results: List[Dict[str, Any]] = []
            for name, clf in candidates:
                try:
                    r = _evaluate_classifier(clf, X_train, X_test, y_train, y_test, name)
                    results.append(r)
                except Exception:
                    # If a model fails (e.g. convergence issues), skip it
                    continue

            if not results:
                raise RuntimeError("All candidate classification models failed during Auto mode.")

            # Choose best by accuracy
            best = max(results, key=lambda r: r["accuracy"])
            candidate_scores = {r["name"]: r["accuracy"] for r in results}

            metrics = {
                "accuracy": best["accuracy"],
                "report": best["report"],
                "candidate_scores": candidate_scores,
            }

            return SupervisedResult(
                task_type="classification",
                model_name=best["name"],
                model=best["model"],
                metrics=metrics,
                cm=best["cm"],
                y_true=y_test,
                y_pred=best["y_pred"],
            )

        # Non-Auto: single chosen model
        clf, used_name = _make_classification_model(model_name)
        r = _evaluate_classifier(clf, X_train, X_test, y_train, y_test, used_name)
        metrics = {
            "accuracy": r["accuracy"],
            "report": r["report"],
        }

        return SupervisedResult(
            task_type="classification",
            model_name=used_name,
            model=clf,
            metrics=metrics,
            cm=r["cm"],
            y_true=y_test,
            y_pred=r["y_pred"],
        )

    # ---------------- Regression ----------------
    else:
        if model_name == "Auto":
            candidates = [
                ("Linear Regression", LinearRegression()),
                ("Ridge Regression", Ridge(alpha=1.0)),
                ("Random Forest Regressor", RandomForestRegressor(n_estimators=200, n_jobs=-1)),
            ]

            results: List[Dict[str, Any]] = []
            for name, reg in candidates:
                try:
                    r = _evaluate_regressor(reg, X_train, X_test, y_train, y_test, name)
                    results.append(r)
                except Exception:
                    continue

            if not results:
                raise RuntimeError("All candidate regression models failed during Auto mode.")

            # Choose best by R^2
            best = max(results, key=lambda r: r["R2"])
            candidate_scores = {r["name"]: r["R2"] for r in results}

            metrics = {
                "MAE": best["MAE"],
                "MSE": best["MSE"],
                "R2": best["R2"],
                "candidate_scores": candidate_scores,
            }

            return SupervisedResult(
                task_type="regression",
                model_name=best["name"],
                model=best["model"],
                metrics=metrics,
                cm=None,
                y_true=y_test,
                y_pred=best["y_pred"],
            )

        # Non-Auto: single chosen model
        reg, used_name = _make_regression_model(model_name)
        r = _evaluate_regressor(reg, X_train, X_test, y_train, y_test, used_name)
        metrics = {
            "MAE": r["MAE"],
            "MSE": r["MSE"],
            "R2": r["R2"],
        }

        return SupervisedResult(
            task_type="regression",
            model_name=used_name,
            model=reg,
            metrics=metrics,
            cm=None,
            y_true=y_test,
            y_pred=r["y_pred"],
        )
