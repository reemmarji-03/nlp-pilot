import re
import pandas as pd


def txt_stats(text: str | None) -> dict:
    """Return character, word, sentence, and vocabulary stats for a text string."""
    if not text:
        return {
            "char_count": 0,
            "word_count": 0,
            "sentence_count": 0,
            "avg_sentence_len": 0.0,
            "unique_word_count": 0,
        }
    words = text.split()
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    word_count = len(words)
    sentence_count = len(sentences)
    avg_sentence_len = round(word_count / sentence_count, 1) if sentence_count > 0 else 0.0
    clean = [re.sub(r"[^\w'-]", "", w).lower() for w in words]
    return {
        "char_count": len(text),
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_len": avg_sentence_len,
        "unique_word_count": len({w for w in clean if w}),
    }


def csv_stats(df: pd.DataFrame, text_col: str | None) -> dict:
    """Return row count, missing value, duplicate, and text-column stats for a DataFrame."""
    missing = {}
    for col, n in df.isnull().sum().items():
        if n > 0:
            missing[col] = {"count": int(n), "pct": round(int(n) / len(df) * 100, 1)}

    text_col_stats = None
    if text_col and text_col in df.columns:
        lengths = df[text_col].dropna().str.split().str.len()
        if len(lengths) > 0:
            text_col_stats = {
                "avg_words": round(float(lengths.mean()), 1),
                "min_words": int(lengths.min()),
                "max_words": int(lengths.max()),
            }

    return {
        "row_count": len(df),
        "col_count": len(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing": missing,
        "text_col_stats": text_col_stats,
    }
