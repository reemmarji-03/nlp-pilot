import pandas as pd
import pytest
from core.data_quality import txt_stats, csv_stats


def test_txt_stats_known_string():
    text = "The cat sat. The dog ran."
    result = txt_stats(text)
    assert result["char_count"] == len(text)
    assert result["word_count"] == 6
    assert result["sentence_count"] == 2
    assert result["avg_sentence_len"] == pytest.approx(3.0)
    assert result["unique_word_count"] == 5


def test_txt_stats_empty():
    result = txt_stats("")
    assert result["char_count"] == 0
    assert result["word_count"] == 0
    assert result["sentence_count"] == 0
    assert result["avg_sentence_len"] == 0.0
    assert result["unique_word_count"] == 0


def test_csv_stats_basic():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    result = csv_stats(df, None)
    assert result["row_count"] == 3
    assert result["col_count"] == 2
    assert result["duplicate_rows"] == 0
    assert result["missing"] == {}
    assert result["text_col_stats"] is None


def test_csv_stats_missing_values():
    df = pd.DataFrame({"a": [1, None, 3], "b": ["x", "y", "z"]})
    result = csv_stats(df, None)
    assert "a" in result["missing"]
    assert result["missing"]["a"]["count"] == 1
    assert result["missing"]["a"]["pct"] == pytest.approx(33.3, abs=0.1)
    assert "b" not in result["missing"]


def test_csv_stats_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "z"]})
    result = csv_stats(df, None)
    assert result["duplicate_rows"] == 1


def test_csv_stats_text_col_stats():
    df = pd.DataFrame({"text": ["one two three", "four five", "six"]})
    result = csv_stats(df, "text")
    assert result["text_col_stats"]["avg_words"] == pytest.approx(2.0, rel=1e-6)
    assert result["text_col_stats"]["min_words"] == 1
    assert result["text_col_stats"]["max_words"] == 3


def test_csv_stats_no_text_col():
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = csv_stats(df, None)
    assert result["text_col_stats"] is None


def test_txt_stats_unique_word_deduplication():
    result = txt_stats("Good good good.")
    assert result["unique_word_count"] == 1


def test_csv_stats_nonexistent_text_col():
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = csv_stats(df, "nonexistent")
    assert result["text_col_stats"] is None
