# Wave 2 Design: Data Quality Tab & Richer Visualisations

**Date:** 2026-05-10
**Scope:** Wave 2 of enhancement roadmap — Data Quality Panel + Richer Document Tools Visualisations
**Approach:** Thin tab + dedicated core module (Approach A)

---

## Overview

Two parallel enhancements:

1. **Data Quality Tab** — a new dedicated tab that auto-refreshes whenever a file is loaded in Document Tools, surfacing stats for both TXT (word/sentence/character counts) and CSV (shape, missing values, duplicate rows) files.
2. **Richer Document Tools Visualisations** — upgrades the word frequency chart to horizontal bars, adds a confidence bar to the sentiment result, and adds an entity-type frequency chart to the NER result.

---

## File Structure

### New files
```
core/data_quality.py               pure stat functions: txt_stats(), csv_stats()
tabs/english/data_quality_tab.py   DataQualityTab(ctk.CTkFrame) with refresh()
tests/test_data_quality.py         unit tests for core/data_quality.py
```

### Modified files
```
tabs/english/main_tab.py           add Data Quality tab; wire on_state_changed → refresh()
tabs/english/document_tools_tab.py improve word freq chart; add sentiment bar; add NER chart
```

---

## Section 1: Data Quality Core (`core/data_quality.py`)

Pure functions — no UI imports, no side effects.

### `txt_stats(text: str) -> dict`

Returns:
```python
{
    "char_count": int,
    "word_count": int,
    "sentence_count": int,
    "avg_sentence_len": float,   # words per sentence
    "unique_word_count": int,
}
```

Implementation notes:
- Words: `text.split()`
- Sentences: `re.split(r'[.!?]+', text)` filtered for non-empty strips
- `avg_sentence_len`: `word_count / sentence_count` if `sentence_count > 0` else `0.0`
- Unique words: `len(set(w.lower() for w in text.split()))`

### `csv_stats(df: pd.DataFrame, text_col: str | None) -> dict`

Returns:
```python
{
    "row_count": int,
    "col_count": int,
    "duplicate_rows": int,
    "missing": {
        "<col_name>": {"count": int, "pct": float},
        ...                                           # only columns with missing > 0
    },
    "text_col_stats": {                               # None if text_col is None
        "avg_words": float,
        "min_words": int,
        "max_words": int,
    },
}
```

Implementation notes:
- `duplicate_rows`: `df.duplicated().sum()`
- `missing`: `{col: {"count": int(n), "pct": round(n / len(df) * 100, 1)} for col, n in df.isnull().sum().items() if n > 0}`
- `text_col_stats`: word counts via `df[text_col].dropna().str.split().str.len()`; returns `None` if `text_col` is `None` or not in `df.columns`

---

## Section 2: Data Quality Tab (`tabs/english/data_quality_tab.py`)

### Class

```python
class DataQualityTab(ctk.CTkFrame):
    def __init__(self, parent, state: EnglishState)
    def refresh(self)
```

### Layout

```
┌─────────────────────────────────────────────┐
│  Data Quality                    [Refresh]  │
├─────────────────────────────────────────────┤
│  (scrollable content area)                  │
│                                             │
│  ── TXT mode ──                             │
│  File:            report.txt                │
│  Characters:      12,430                   │
│  Words:           2,100                    │
│  Sentences:       98                       │
│  Avg sentence:    21.4 words               │
│  Unique words:    874                      │
│                                             │
│  ── CSV mode ──                             │
│  File:            dataset.csv               │
│  Rows:            1,200                    │
│  Columns:         5                        │
│  Duplicate rows:  3                        │
│                                             │
│  Missing values:                            │
│  ┌──────────────┬───────┬───────┐          │
│  │ Column       │ Count │   %   │          │
│  │ label        │    12 │  1.0% │          │
│  │ description  │     4 │  0.3% │          │
│  └──────────────┴───────┴───────┘          │
│                                             │
│  Text column (text):                        │
│  Avg words: 18.3 · Min: 2 · Max: 94        │
└─────────────────────────────────────────────┘
```

### Behaviour

- `refresh()` is called:
  - On init (shows "No data loaded" if state is empty)
  - By `main_tab.on_state_changed()` whenever a file is uploaded
  - On manual Refresh button click
- Detects mode: `enlp.is_csv_mode(state)` → CSV branch; `state.text` non-empty → TXT branch; else → placeholder
- All stat rows rendered as `CTkLabel` pairs (label left, value right) inside a `CTkScrollableFrame`
- Missing values table rendered as a grid of `CTkLabel` cells; hidden entirely if no missing values
- Text-column stats row hidden if `state.csv_text_column` is `None`

---

## Section 3: Main Tab Changes (`tabs/english/main_tab.py`)

```python
# Add tab (between Document Tools and Preprocessing)
dq_frame = self.section_tabs.add("Data Quality")
self.data_quality_tab = DataQualityTab(dq_frame, self.state)

# Extend on_state_changed
def on_state_changed(self):
    self.prediction_tab.sync_with_state()
    self.data_quality_tab.refresh()
```

Tab order after change: Document Tools → **Data Quality** → Preprocessing → Vectorization → Prediction → RAG → Agent

---

## Section 4: Richer Document Tools Visualisations (`tabs/english/document_tools_tab.py`)

### Word Frequency chart (`generate_charts` method)

Changes:
- Switch from vertical to **horizontal bar chart** (`barh`)
- Show top 20 words (was: variable, now fixed at 20)
- Bar color: `#6ea8fe` (app accent blue)
- Add value labels at end of each bar via `ax.bar_label(bars, padding=3)`
- Y-axis: words (most frequent at top — use `ax.invert_yaxis()`)

### Sentiment result (`sentiment` method)

Replace the plain text widget output with a matplotlib figure rendered via `FigureCanvasTkAgg` into `output_container`. The figure has two parts:
- Top: text annotation showing label and score (e.g. "POSITIVE — 0.94") via `fig.text()`
- Bottom: single horizontal bar, width = confidence score (0.0–1.0)
  - Bar color: green (`#4CAF50`) if score ≥ 0.7, orange (`#FF9800`) if 0.4–0.7, red (`#F44336`) if < 0.4
  - X-axis range 0–1, ticks at 0, 0.5, 1.0
  - Title: `"Confidence"`
- Figure background matches app dark theme (`#1a1a1a`)

### NER result (`named_entities` method)

After the existing entity list, append an entity-type frequency chart:
- Horizontal bar chart of entity type counts (e.g. PERSON: 4, ORG: 2, GPE: 1)
- Bar color: `#6ea8fe`
- Title: `"Entity Types"`
- Only rendered if at least one entity was found; skipped silently otherwise
- Rendered into `output_container` alongside the text list

---

## Error Handling

| Scenario | Behaviour |
|----------|-----------|
| No file loaded when Data Quality tab opened | Shows "No data loaded — upload a file in Document Tools" |
| `txt_stats` called with empty string | Returns all zeros |
| `csv_stats` called with empty DataFrame | Returns zeros; missing dict empty; text_col_stats None |
| Sentiment confidence unavailable | Confidence bar not rendered; text-only fallback |
| NER returns no entities | Chart section skipped; entity list shows "No entities found" |

---

## Testing

### `tests/test_data_quality.py`

- `test_txt_stats_known_string` — fixed string, assert all five stat values exactly
- `test_txt_stats_empty` — empty string returns all zeros
- `test_csv_stats_basic` — small DataFrame, assert row/col counts correct
- `test_csv_stats_missing_values` — DataFrame with known nulls, assert missing dict keys and percentages
- `test_csv_stats_duplicates` — DataFrame with one duplicate row, assert `duplicate_rows == 1`
- `test_csv_stats_text_col_stats` — DataFrame with text column, assert avg/min/max word counts
- `test_csv_stats_no_text_col` — `text_col=None`, assert `text_col_stats` is `None`

---

## Out of Scope (Waves 3–4)

- Experiment presets / logging
- Export artifacts
- Session management
- Language expansion
- Class imbalance chart (label distribution) in Data Quality tab
- Interactive charts (hover, zoom)
- Top-N slider on word frequency chart
