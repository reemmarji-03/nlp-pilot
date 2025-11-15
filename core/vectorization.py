from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import scipy.sparse as sp

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA
from numpy.linalg import norm

from transformers import AutoTokenizer, AutoModel
import torch


# ---------------- Models cached globally so they load once ----------------

_tf_tokenizer = None
_tf_model = None


def get_transformer_model():
    """
    Lazy-load a small-ish transformer for sentence embeddings.
    We use distilbert-base-uncased for now (available already in your project).
    """
    global _tf_tokenizer, _tf_model
    if _tf_tokenizer is None or _tf_model is None:
        model_name = "distilbert-base-uncased"
        _tf_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _tf_model = AutoModel.from_pretrained(model_name)
        _tf_model.eval()
    return _tf_tokenizer, _tf_model


# ---------------- Data structure for results ----------------

@dataclass
class VectorizationResult:
    name: str
    vector: np.ndarray | sp.spmatrix   # shape (n_samples, dim)
    dim: int
    is_sparse: bool
    nnz: Optional[int]
    density: float
    memory_kb: float
    vocab: Optional[List[str]] = None
    top_features: Optional[List[Tuple[str, float]]] = None


# ---------------- Utility functions ----------------

def _sparse_memory_kb(x: sp.spmatrix) -> float:
    return (x.data.nbytes + x.indptr.nbytes + x.indices.nbytes) / 1024.0


def _dense_memory_kb(x: np.ndarray) -> float:
    return x.nbytes / 1024.0


def _top_k_from_sparse_row(
    row: sp.spmatrix,
    feature_names: List[str],
    k: int = 20
) -> List[Tuple[str, float]]:
    """Top features from a single sparse row."""
    # convert to COO for convenience
    coo = row.tocoo()
    if coo.nnz == 0:
        return []

    values = coo.data
    idxs = coo.col

    order = np.argsort(values)[::-1][:k]
    top = []
    for i in order:
        feat = feature_names[idxs[i]]
        val = float(values[i])
        top.append((feat, val))
    return top


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Cosine similarity between two 1D vectors."""
    v1 = vec1.reshape(-1)
    v2 = vec2.reshape(-1)
    n1 = norm(v1)
    n2 = norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def pca_2d(X: np.ndarray) -> np.ndarray:
    """Project embeddings to 2D using PCA."""
    if X.shape[0] < 2:
        return np.zeros((X.shape[0], 2))
    pca = PCA(n_components=2)
    return pca.fit_transform(X)


# ---------------- Bag-of-Words / TF-IDF / Char n-grams ----------------

def vectorize_bow(
    texts: List[str],
    ngram_range: Tuple[int, int] = (1, 1),
    max_features: int = 5000
) -> Tuple[VectorizationResult, CountVectorizer]:
    vec = CountVectorizer(
        ngram_range=ngram_range,
        max_features=max_features
    )
    X = vec.fit_transform(texts)
    feature_names = vec.get_feature_names_out().tolist()

    n_samples, dim = X.shape
    nnz = int(X.nnz)
    density = nnz / (n_samples * dim) if dim > 0 else 0.0
    mem = _sparse_memory_kb(X)

    # top features for first doc
    top = _top_k_from_sparse_row(X[0], feature_names, k=20)

    res = VectorizationResult(
        name=f"Bag-of-Words {ngram_range}",
        vector=X,
        dim=dim,
        is_sparse=True,
        nnz=nnz,
        density=density,
        memory_kb=mem,
        vocab=feature_names,
        top_features=top
    )
    return res, vec


def vectorize_tfidf(
    texts: List[str],
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: int = 5000,
    analyzer: str = "word"    # "word" or "char"
) -> Tuple[VectorizationResult, TfidfVectorizer]:
    vec = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        analyzer=analyzer
    )
    X = vec.fit_transform(texts)
    feature_names = vec.get_feature_names_out().tolist()

    n_samples, dim = X.shape
    nnz = int(X.nnz)
    density = nnz / (n_samples * dim) if dim > 0 else 0.0
    mem = _sparse_memory_kb(X)

    top = _top_k_from_sparse_row(X[0], feature_names, k=20)

    res = VectorizationResult(
        name=f"TF-IDF ({analyzer}) {ngram_range}",
        vector=X,
        dim=dim,
        is_sparse=True,
        nnz=nnz,
        density=density,
        memory_kb=mem,
        vocab=feature_names,
        top_features=top
    )
    return res, vec


# ---------------- Transformer sentence embeddings ----------------

def transformer_sentence_embeddings(
    texts: List[str],
    pooling: str = "mean"
) -> VectorizationResult:
    """
    Use DistilBERT to build dense sentence embeddings.
    pooling: "mean" over tokens (simple but works fine).
    """
    tokenizer, model = get_transformer_model()
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**encoded)  # last_hidden_state: (B, T, H)

    token_embeddings = outputs.last_hidden_state  # (B, T, H)

    if pooling == "mean":
        # mean over non-padding tokens
        attention_mask = encoded["attention_mask"].unsqueeze(-1)  # (B, T, 1)
        masked = token_embeddings * attention_mask
        summed = masked.sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1)
        sent_emb = (summed / counts).cpu().numpy()  # (B, H)
    else:
        sent_emb = token_embeddings[:, 0, :].cpu().numpy()

    X = sent_emb
    n_samples, dim = X.shape
    mem = _dense_memory_kb(X)

    res = VectorizationResult(
        name="Transformer (DistilBERT) sentence embeddings",
        vector=X,
        dim=dim,
        is_sparse=False,
        nnz=None,
        density=1.0,
        memory_kb=mem,
        vocab=None,
        top_features=None
    )
    return res
