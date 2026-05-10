# core/unsupervised.py

from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
from sklearn.cluster import KMeans

from core.english_state import EnglishState
from core import english_nlp as enlp
from core import vectorization as vec


@dataclass
class ClusteringResult:
    labels: np.ndarray
    centroids: np.ndarray
    n_clusters: int


def build_X_from_state(
    state: EnglishState,
    vector_method: str,
    ngram_range: Tuple[int, int],
    max_features: int,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build feature matrix X from current CSV in state.

    CSV-only: one row = one sample (preprocessed text).
    Returns X and list of row labels for display ("Row 1", "Row 2", ...).
    """
    if not enlp.is_csv_mode(state):
        raise ValueError("Clustering currently only supports CSV mode.")

    if state.df is None or state.csv_text_column is None:
        raise ValueError("CSV and text column must be set in state.")

    df = state.df

    raw_docs = df[state.csv_text_column].astype(str).tolist()
    docs = [
        enlp.apply_pipeline(
            d or "",
            cfg=state.pipeline_config,
            stopword_set=state.stopwords,
        ).strip()
        for d in raw_docs
    ]
    mask = [bool(t) for t in docs]
    docs = [d for d in docs if d]

    if not docs:
        raise ValueError("No non-empty documents after preprocessing.")

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

    labels_for_display = [f"Row {i+1}" for i in range(len(docs))]
    return X, labels_for_display


def run_kmeans(
    X: np.ndarray,
    n_clusters: int = 3,
    random_state: int = 42,
    cancel_event=None,
    progress_callback=None,
) -> ClusteringResult:
    if cancel_event is not None and cancel_event.is_set():
        from core.task_runner import CancelledError
        raise CancelledError()
    if progress_callback is not None:
        progress_callback(0.0, "Running K-Means…")
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    km.fit(X)
    if progress_callback is not None:
        progress_callback(1.0, "Done.")
    return ClusteringResult(
        labels=km.labels_,
        centroids=km.cluster_centers_,
        n_clusters=n_clusters,
    )
