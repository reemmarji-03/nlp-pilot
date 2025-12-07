# core/agent_schemas.py

from typing import Literal, Optional, List, Tuple, Dict, Any
from pydantic import BaseModel, Field


# Task selection

class TaskSelectionDecision(BaseModel):
    task_type: Literal["classification", "regression", "clustering", "unknown"]
    explanation: str
    needs_clarification: bool = False
    clarification_question: Optional[str] = None


# Preprocessing config

class PreprocessStep(BaseModel):
    id: Literal[
        "lowercase",
        "urls",
        "contractions",
        "numbers",
        "stopwords",
        "short_tokens",
        "stem",
        "lemma",
    ]
    enabled: bool
    min_len: Optional[int] = None





class PreprocessConfigDecision(BaseModel):
    steps: List[PreprocessStep]
    explanation: str
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
    ready_to_apply: bool = False


# Vectorization config

class VectorConfigDecision(BaseModel):
    method: Literal["TF-IDF (word)", "Bag-of-Words", "Char n-grams", "Transformer embeddings"]

    # Use a list instead of a tuple so OpenAI's structured output schema is valid.
    # We enforce length = 2 so it still behaves like (min_n, max_n).
    ngram_range: List[int] = Field(..., min_length=2, max_length=2)

    max_features: Optional[int]
    explanation: str

    ready_to_apply: bool = False
    needs_clarification: bool = False
    clarification_question: Optional[str] = None



# Model config

class ModelConfigDecision(BaseModel):
    label_column: Optional[str]
    model_name: Literal["Logistic Regression", "Linear SVM", "Random Forest", "Auto"]
    test_size: float
    explanation: str
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
