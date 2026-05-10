# Wave 2: Data Quality Tab & Richer Visualisations — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Data Quality tab that auto-refreshes when files are loaded, and enrich the Document Tools visualisations with a richer word frequency chart, a sentiment confidence bar, and an NER entity-type frequency chart.

**Architecture:** Pure stat functions in `core/data_quality.py` (TDD), a thin `DataQualityTab` widget that calls them in `refresh()`, wired into `main_tab.on_state_changed()`. Document Tools visualisation changes stay inline in `document_tools_tab.py`; the shared `build_word_freq_figure` helper in `english_nlp.py` gains a `dark` parameter to preserve PDF-export compatibility.

**Tech Stack:** customtkinter, matplotlib (FigureCanvasTkAgg), pandas, pytest

---

## File Structure

```
core/data_quality.py                  NEW — txt_stats(), csv_stats() pure functions
tabs/english/data_quality_tab.py      NEW — DataQualityTab(ctk.CTkFrame) with refresh()
tests/test_data_quality.py            NEW — 7 unit tests for core/data_quality.py
tabs/english/main_tab.py              MODIFY — add Data Quality tab, wire on_state_changed
tabs/english/document_tools_tab.py   MODIFY — fix text-upload trigger, richer charts
core/english_nlp.py                   MODIFY — build_word_freq_figure gains dark= param
```

---

## Task 1: Data Quality Core Functions

**Files:**
- Create: `core/data_quality.py`
- Test: `tests/test_data_quality.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_data_quality.py`:

```python
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
    assert result["text_col_stats"]["avg_words"] == pytest.approx(2.0, abs=0.1)
    assert result["text_col_stats"]["min_words"] == 1
    assert result["text_col_stats"]["max_words"] == 3


def test_csv_stats_no_text_col():
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = csv_stats(df, None)
    assert result["text_col_stats"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_data_quality.py -v
```

Expected: `ImportError` — `core.data_quality` does not exist yet.

- [ ] **Step 3: Implement `core/data_quality.py`**

Create `core/data_quality.py`:

```python
import re
import pandas as pd


def txt_stats(text: str) -> dict:
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
    return {
        "char_count": len(text),
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_len": avg_sentence_len,
        "unique_word_count": len(set(w.lower() for w in words)),
    }


def csv_stats(df: pd.DataFrame, text_col: str | None) -> dict:
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
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_data_quality.py -v
```

Expected: 7 PASSED.

- [ ] **Step 5: Commit**

```
git add core/data_quality.py tests/test_data_quality.py
git commit -m "feat: add data_quality core functions with tests"
```

---

## Task 2: DataQualityTab Widget

**Files:**
- Create: `tabs/english/data_quality_tab.py`

No unit tests — UI rendering; verified by running the app after Task 3 wires it in.

- [ ] **Step 1: Create `tabs/english/data_quality_tab.py`**

```python
import customtkinter as ctk

from core.english_state import EnglishState
from core import english_nlp as enlp
from core.data_quality import txt_stats, csv_stats


class DataQualityTab(ctk.CTkFrame):
    def __init__(self, parent, state: EnglishState):
        super().__init__(parent, fg_color="#0f0f0f")
        self.state = state
        self._build_ui()
        self.pack(fill="both", expand=True)
        self.refresh()

    def _build_ui(self):
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        sidebar = ctk.CTkFrame(self, fg_color="#121212", corner_radius=10)
        sidebar.grid(row=0, column=0, sticky="ns", padx=(10, 5), pady=10)

        ctk.CTkLabel(
            sidebar,
            text="Data Quality",
            text_color="#6ea8fe",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack(pady=(15, 20))

        ctk.CTkButton(
            sidebar,
            text="🔄 Refresh",
            command=self.refresh,
            fg_color="#0078ff",
            hover_color="#005dc1",
            corner_radius=6,
            height=38,
            font=ctk.CTkFont(size=14),
        ).pack(fill="x", padx=10, pady=6)

        self._scroll = ctk.CTkScrollableFrame(
            self, fg_color="#1a1a1a", corner_radius=10
        )
        self._scroll.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)

    def refresh(self):
        for w in self._scroll.winfo_children():
            w.destroy()

        in_csv = enlp.is_csv_mode(self.state)
        has_text = bool(self.state.text and self.state.text.strip())

        if not in_csv and not has_text:
            ctk.CTkLabel(
                self._scroll,
                text="No data loaded — upload a file in Document Tools",
                text_color="#888888",
                font=ctk.CTkFont(size=13),
            ).pack(pady=20, padx=20)
            return

        if in_csv:
            self._render_csv_stats()
        else:
            self._render_txt_stats()

    def _render_txt_stats(self):
        stats = txt_stats(self.state.text)
        self._section_header(f"📄  {self.state.file_name or 'Text file'}")
        rows = [
            ("Characters", f"{stats['char_count']:,}"),
            ("Words", f"{stats['word_count']:,}"),
            ("Sentences", f"{stats['sentence_count']:,}"),
            ("Avg sentence length", f"{stats['avg_sentence_len']:.1f} words"),
            ("Unique words", f"{stats['unique_word_count']:,}"),
        ]
        for label, value in rows:
            self._stat_row(label, value)

    def _render_csv_stats(self):
        stats = csv_stats(self.state.df, self.state.csv_text_column)
        self._section_header(f"📊  {self.state.file_name or 'CSV file'}")
        self._stat_row("Rows", f"{stats['row_count']:,}")
        self._stat_row("Columns", str(stats["col_count"]))
        self._stat_row("Duplicate rows", str(stats["duplicate_rows"]))

        if stats["missing"]:
            self._section_header("Missing Values")
            header = ctk.CTkFrame(self._scroll, fg_color="transparent")
            header.pack(fill="x", padx=20, pady=(2, 0))
            ctk.CTkLabel(
                header, text="Column", font=ctk.CTkFont(weight="bold"),
                text_color="#aaaaaa", width=160, anchor="w",
            ).pack(side="left")
            ctk.CTkLabel(
                header, text="Count", font=ctk.CTkFont(weight="bold"),
                text_color="#aaaaaa", width=60, anchor="e",
            ).pack(side="left")
            ctk.CTkLabel(
                header, text="%", font=ctk.CTkFont(weight="bold"),
                text_color="#aaaaaa", width=60, anchor="e",
            ).pack(side="left")
            for col, info in stats["missing"].items():
                row = ctk.CTkFrame(self._scroll, fg_color="transparent")
                row.pack(fill="x", padx=20, pady=1)
                ctk.CTkLabel(row, text=col, width=160, anchor="w",
                             text_color="white").pack(side="left")
                ctk.CTkLabel(row, text=str(info["count"]), width=60, anchor="e",
                             text_color="white").pack(side="left")
                ctk.CTkLabel(
                    row, text=f"{info['pct']:.1f}%", width=60, anchor="e",
                    text_color="#ffaa44",
                ).pack(side="left")
        else:
            self._stat_row("Missing values", "None ✓")

        if stats["text_col_stats"]:
            tcs = stats["text_col_stats"]
            self._section_header(f"Text column  ({self.state.csv_text_column})")
            self._stat_row("Avg words per row", f"{tcs['avg_words']:.1f}")
            self._stat_row("Min words", str(tcs["min_words"]))
            self._stat_row("Max words", str(tcs["max_words"]))

    def _section_header(self, text: str):
        ctk.CTkLabel(
            self._scroll,
            text=text,
            text_color="#6ea8fe",
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w",
        ).pack(fill="x", padx=20, pady=(16, 4))

    def _stat_row(self, label: str, value: str):
        row = ctk.CTkFrame(self._scroll, fg_color="transparent")
        row.pack(fill="x", padx=20, pady=2)
        ctk.CTkLabel(
            row, text=label, text_color="#aaaaaa", anchor="w", width=200,
        ).pack(side="left")
        ctk.CTkLabel(row, text=value, text_color="white", anchor="w").pack(side="left")
```

- [ ] **Step 2: Commit**

```
git add tabs/english/data_quality_tab.py
git commit -m "feat: add DataQualityTab widget"
```

---

## Task 3: Wire DataQualityTab into the App

**Files:**
- Modify: `tabs/english/main_tab.py` (full file)
- Modify: `tabs/english/document_tools_tab.py:156-171` (`_handle_text_like_file`)

This task has two parts:
1. Add the Data Quality tab to `main_tab.py` and extend `on_state_changed`.
2. Fix `_handle_text_like_file` so it fires `on_state_changed` when a TXT/PDF file loads (currently only CSV upload does this).

- [ ] **Step 1: Update `tabs/english/main_tab.py`**

Replace the entire file with:

```python
import customtkinter as ctk

from core.english_state import EnglishState
from .agent_tab import AgentTab
from .data_quality_tab import DataQualityTab
from .document_tools_tab import DocumentToolsTab
from .preprocess_tab import PreprocessTab
from .vector_tab import VectorTab
from .prediction_tab import PredictionTab
from .rag_tab import RAGTab


class EnglishTab(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent, fg_color="#0f0f0f")
        self.state = EnglishState()
        self.build_ui()

    def build_ui(self):
        self.section_tabs = ctk.CTkTabview(self)
        self.section_tabs.pack(fill="both", expand=True, padx=10, pady=10)

        doc_frame = self.section_tabs.add("Document Tools")
        dq_frame = self.section_tabs.add("Data Quality")
        prep_frame = self.section_tabs.add("Preprocessing")
        vect_frame = self.section_tabs.add("Vectorization")
        prediction_frame = self.section_tabs.add("Prediction")
        rag_frame = self.section_tabs.add("RAG")
        agent_frame = self.section_tabs.add("Agent")

        self.doc_tab = DocumentToolsTab(
            doc_frame, self.state, on_state_changed=self.on_state_changed
        )
        self.data_quality_tab = DataQualityTab(dq_frame, self.state)
        self.prep_tab = PreprocessTab(prep_frame, self.state)
        self.vect_tab = VectorTab(vect_frame, self.state)
        self.prediction_tab = PredictionTab(prediction_frame, self.state)
        self.rag_tab = RAGTab(rag_frame, self.state)
        self.agent_tab = AgentTab(agent_frame, self.state)

    def on_state_changed(self):
        self.prediction_tab.sync_with_state()
        self.data_quality_tab.refresh()
```

- [ ] **Step 2: Fix `_handle_text_like_file` in `tabs/english/document_tools_tab.py`**

Find `_handle_text_like_file` (around line 156). Replace the method with:

```python
def _handle_text_like_file(self, path: str):
    self.state.df = None
    self.state.csv_text_column = None

    text = enlp.load_text_from_file(path)
    if not text.strip():
        messagebox.showwarning("Warning", "Could not extract text from file.")
        return

    self.state.text = text
    self.state.file_name = os.path.basename(path)
    self.on_state_changed()

    self.file_label.configure(text=f"📂  {self.state.file_name}")
    self.output.delete("1.0", "end")
    self.output.insert(
        "end",
        f"File loaded successfully.\n Total Number of characters: {len(text)}",
        "left",
    )
```

- [ ] **Step 3: Run all tests**

```
pytest tests/ -v
```

Expected: all existing tests still pass (15 tests from Wave 1 + 7 new = 22 PASSED).

- [ ] **Step 4: Commit**

```
git add tabs/english/main_tab.py tabs/english/document_tools_tab.py
git commit -m "feat: wire DataQualityTab into app; trigger refresh on text file upload"
```

---

## Task 4: Richer Word Frequency Chart

**Files:**
- Modify: `core/english_nlp.py:192-213` (`build_word_freq_figure`)
- Modify: `tabs/english/document_tools_tab.py:295-310` (`generate_charts`)

The existing chart is already horizontal (`barh`). This task adds: dark theme, accent colour, value labels, inverted y-axis (most frequent at top), and bumps top-N to 20.

`build_word_freq_figure` gains a `dark=False` parameter so the PDF export (`export_report`) continues to work unchanged with its white background.

- [ ] **Step 1: Update `build_word_freq_figure` in `core/english_nlp.py`**

Find `build_word_freq_figure` (line 192). Replace it with:

```python
def build_word_freq_figure(freq_pairs, title="Top Words", dark=False):
    """
    Build and return a matplotlib Figure for a horizontal bar chart
    of word frequencies.

    freq_pairs: list of (word, count)
    dark: apply dark theme (used for on-screen display)
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    bg = "#1a1a1a" if dark else "white"
    fg = "white" if dark else "black"
    muted = "#aaaaaa" if dark else "#444444"
    bar_color = "#6ea8fe" if dark else "#4472C4"
    spine_color = "#333333" if dark else "#cccccc"

    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    if not freq_pairs:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", color=fg)
        ax.set_axis_off()
        return fig

    labels, values = zip(*freq_pairs)
    bars = ax.barh(labels, values, color=bar_color)
    ax.bar_label(bars, padding=3, color=fg, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Count", color=muted)
    ax.set_title(title, color=fg, fontsize=13)
    ax.tick_params(colors=muted)
    for spine in ax.spines.values():
        spine.set_color(spine_color)
    fig.tight_layout()
    return fig
```

- [ ] **Step 2: Update `generate_charts` in `tabs/english/document_tools_tab.py`**

Find `generate_charts` (line 295). Replace it with:

```python
def generate_charts(self):
    if not self.require_text():
        return

    text = self.get_active_text()
    freq = enlp.word_frequency(
        text,
        stopword_set=self.state.stopwords,
        top_n=20,
        remove_stopwords=True,
    )
    fig = enlp.build_word_freq_figure(freq, title="Top 20 Words", dark=True)
    self.show_chart(fig)
```

- [ ] **Step 3: Run all tests**

```
pytest tests/ -v
```

Expected: 22 PASSED (no new tests for rendering changes).

- [ ] **Step 4: Commit**

```
git add core/english_nlp.py tabs/english/document_tools_tab.py
git commit -m "feat: richer word frequency chart — accent color, labels, inverted axis, top 20"
```

---

## Task 5: Sentiment Confidence Figure

**Files:**
- Modify: `tabs/english/document_tools_tab.py:231-253` (`sentiment`)

Replace the text-only sentiment output with a matplotlib figure: suptitle shows label + explanation, a single horizontal bar shows the confidence score colour-coded by strength.

- [ ] **Step 1: Replace `sentiment` in `tabs/english/document_tools_tab.py`**

Find `sentiment` (line 231). Replace the method with:

```python
def sentiment(self):
    if not self.require_text():
        return

    model = self.get_sentiment_model()
    text = self.get_active_text()
    result = model(text[:2000])[0]
    label = result["label"]
    score = result["score"]

    explanation = {
        "POSITIVE": "Optimistic or confident tone.",
        "NEGATIVE": "Critical or dissatisfied tone.",
        "NEUTRAL": "Balanced, factual tone.",
    }.get(label, "Unclear tone detected.")

    if score >= 0.7:
        bar_color = "#4CAF50"
    elif score >= 0.4:
        bar_color = "#FF9800"
    else:
        bar_color = "#F44336"

    fig, ax = plt.subplots(figsize=(6, 2.2))
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_facecolor("#1a1a1a")
    fig.suptitle(
        f"{label}  ({score:.3f})  —  {explanation}",
        color="white",
        fontsize=11,
        y=0.98,
    )
    ax.barh([""], [score], color=bar_color, height=0.4)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(["0", "0.5", "1.0"], color="#aaaaaa")
    ax.set_title("Confidence", color="#aaaaaa", fontsize=10)
    ax.tick_params(colors="#aaaaaa", left=False, labelleft=False)
    for spine in ax.spines.values():
        spine.set_color("#333333")
    plt.tight_layout()
    self.show_chart(fig)
```

- [ ] **Step 2: Run all tests**

```
pytest tests/ -v
```

Expected: 22 PASSED.

- [ ] **Step 3: Commit**

```
git add tabs/english/document_tools_tab.py
git commit -m "feat: replace text-only sentiment output with confidence figure"
```

---

## Task 6: NER Entity-Type Frequency Chart

**Files:**
- Modify: `tabs/english/document_tools_tab.py` — add `show_text_and_chart` helper and update `named_entities`

The NER result keeps its entity list as text in a textbox (top) and adds an entity-type frequency bar chart (bottom). A new `show_text_and_chart` helper manages this two-widget layout in `output_container`.

- [ ] **Step 1: Add `show_text_and_chart` to `tabs/english/document_tools_tab.py`**

Add this method after `show_chart` (after line 408):

```python
def show_text_and_chart(self, text_content: str, fig):
    """Show a text widget above a matplotlib chart in output_container."""
    for w in self.output_container.winfo_children():
        w.destroy()

    self.output = ctk.CTkTextbox(
        self.output_container,
        fg_color="#1a1a1a",
        text_color="white",
        font=("Consolas", 13),
        wrap="word",
        height=200,
    )
    self.output.pack(fill="x", expand=False)
    self.output.insert("end", text_content, "left")

    canvas = FigureCanvasTkAgg(fig, master=self.output_container)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)
```

- [ ] **Step 2: Replace `named_entities` in `tabs/english/document_tools_tab.py`**

Find `named_entities` (line 255). Replace the method with:

```python
def named_entities(self):
    if not self.require_text():
        return

    model = self.get_ner_model()
    text = self.get_active_text()
    ents_raw = model(text[:1000])
    cleaned = enlp.merge_ner_entities(ents_raw)
    label_counts = Counter(lbl for _, lbl, _ in cleaned)

    if not label_counts:
        self.show_textbox()
        self.output.delete("1.0", "end")
        self.output.insert("end", "\n🧬 Named Entities\n" + "-" * 40 + "\nNo entities found.\n", "left")
        return

    lines = ["\n🧬 Named Entities", "-" * 40, "Summary by label:"]
    for lbl, cnt in label_counts.items():
        lines.append(f"{lbl:<10}: {cnt}")
    lines.append("-" * 40)
    for word, lbl, score in cleaned[:50]:
        lines.append(f"{word:<25}{lbl:<10}{score:.2f}")
    text_content = "\n".join(lines)

    types = list(label_counts.keys())
    counts = [label_counts[t] for t in types]
    fig_h = max(2.0, len(types) * 0.5 + 1.0)
    fig, ax = plt.subplots(figsize=(6, fig_h))
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_facecolor("#1a1a1a")
    bars = ax.barh(types, counts, color="#6ea8fe")
    ax.bar_label(bars, padding=3, color="white", fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Count", color="#aaaaaa")
    ax.set_title("Entity Types", color="white", fontsize=12)
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_color("#333333")
    fig.tight_layout()

    self.show_text_and_chart(text_content, fig)
```

- [ ] **Step 3: Run all tests**

```
pytest tests/ -v
```

Expected: 22 PASSED.

- [ ] **Step 4: Commit**

```
git add tabs/english/document_tools_tab.py
git commit -m "feat: add NER entity-type frequency chart below entity list"
```

---

## Final Check

- [ ] **Run full test suite one last time**

```
pytest tests/ -v
```

Expected: 22 PASSED, 0 failed.

- [ ] **Smoke test the app**

```
python main.py
```

Verify:
1. "Data Quality" tab appears between Document Tools and Preprocessing.
2. Upload a `.txt` file → Data Quality tab auto-refreshes with TXT stats.
3. Upload a `.csv` file → Data Quality tab auto-refreshes with CSV stats, missing values table.
4. Click "Generate Charts" → horizontal top-20 bar chart with blue accent.
5. Click "Sentiment" → confidence bar figure with colour-coded bar.
6. Click "Named Entities" → entity list (text, top) + entity-type chart (bottom).
7. Manual "Refresh" button in Data Quality tab works.
