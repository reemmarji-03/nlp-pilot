import pandas as pd

from core.english_state import EnglishState
from core.supervised import train_supervised_from_state


def test_train_supervised_from_state_classification_auto_uses_seeded_holdout():
    df = pd.DataFrame(
        {
            "text": [
                "great useful tool",
                "excellent nice result",
                "happy positive case",
                "bad broken tool",
                "awful poor result",
                "sad negative case",
                "great happy result",
                "bad awful tool",
            ],
            "label": ["pos", "pos", "pos", "neg", "neg", "neg", "pos", "neg"],
        }
    )
    state = EnglishState(stopwords=set(), df=df, csv_text_column="text")

    result = train_supervised_from_state(
        state,
        label_column="label",
        vector_method="TF-IDF (word)",
        ngram_range=(1, 1),
        max_features=100,
        model_name="Auto",
        test_size=0.25,
        random_state=7,
    )

    assert result.task_type == "classification"
    assert result.test_indices is not None
    assert len(result.y_true) == len(result.y_pred) == len(result.test_indices)
    assert result.metrics["selection_strategy"] in {
        "cross_validation",
        "training_score_no_cv",
    }
    assert result.metrics["candidate_scores"]


def test_train_supervised_from_state_regression_named_model():
    df = pd.DataFrame(
        {
            "text": [
                "short note",
                "medium length note",
                "long detailed note",
                "brief item",
                "expanded detailed item",
                "compact item",
            ],
            "score": [1.0, 2.0, 3.0, 1.2, 3.1, 1.1],
        }
    )
    state = EnglishState(stopwords=set(), df=df, csv_text_column="text")

    result = train_supervised_from_state(
        state,
        label_column="score",
        vector_method="Bag-of-Words",
        ngram_range=(1, 1),
        max_features=100,
        model_name="Ridge Regression",
        test_size=0.33,
        random_state=3,
    )

    assert result.task_type == "regression"
    assert "MAE" in result.metrics
    assert len(result.y_true) == len(result.y_pred)
