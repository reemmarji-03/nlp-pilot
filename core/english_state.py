# core/english_state.py

from dataclasses import dataclass, field
from typing import Optional, Set, Dict, Any, Tuple
from transformers import Pipeline
import pandas as pd
from core import english_nlp as enlp


def default_pipeline_config() -> Dict[str, Any]:
    # initial pipeline
    return {
        # "steps": [
        #     {"id": "lowercase", "enabled": True},
        #     {"id": "urls", "enabled": True},
        #     {"id": "contractions", "enabled": True},
        #     {"id": "stopwords", "enabled": True},
        #     {"id": "short_tokens", "enabled": True, "min_len": 3},
        # ]
    }


@dataclass
class EnglishState:
    # For TXT / PDF / DOCX
    text: str = ""
    file_name: str = ""

    # Stopwords are initialized once via NLTK
    stopwords: Set[str] = field(default_factory=enlp.ensure_nltk)

    # HuggingFace pipelines (lazy-loaded)
    sentiment_model: Optional[Pipeline] = None
    ner_model: Optional[Pipeline] = None

    # Preprocessing pipeline configuration
    pipeline_config: Dict[str, Any] = field(default_factory=default_pipeline_config)

    # For CSV mode
    df: Optional[pd.DataFrame] = None           # holds the loaded CSV (if any)
    csv_text_column: Optional[str] = None       # which column is the "text" column

    #  Last vectorization settings (for Prediction tab to reuse) ---
    last_vector_method: Optional[str] = None
    last_vector_ngram: Optional[Tuple[int, int]] = None
    last_vector_max_features: Optional[int] = None

    # Results created by guided/agent workflows, available to other tabs.
    last_supervised_result: Optional[Any] = None
    last_supervised_context: Dict[str, Any] = field(default_factory=dict)
    last_cluster_export: Optional[pd.DataFrame] = None
