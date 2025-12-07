import os
import re
from collections import Counter
from typing import Iterable

import docx
import pdfplumber
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import textstat
import matplotlib.pyplot as plt
import nltk

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

_STEMMER = PorterStemmer()
_LEMMATIZER = WordNetLemmatizer()
def _simple_tokenize(text: str) -> Iterable[str]:
    # Keep it simple; you might already have a better tokenizer.
    return text.split()

def _simple_detokenize(tokens: Iterable[str]) -> str:
    return " ".join(tokens)
# NLTK setup
def ensure_nltk():
    """
    Ensure required NLTK resources are available and
    return a set of English stopwords.
    This should be called once and the result reused.
    """
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")

    return set(stopwords.words("english"))


# File loading
def load_text_from_file(path: str) -> str:
    """
    Load text from .txt, .docx or .pdf.
    Returns the raw text as a string.
    """
    if not path:
        return ""

    path = os.path.abspath(path)

    if path.lower().endswith(".txt"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    if path.lower().endswith(".docx"):
        doc = docx.Document(path)
        return "\n".join(p.text for p in doc.paragraphs)

    if path.lower().endswith(".pdf"):
        text = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text.append(page.extract_text() or "")
        return "\n".join(text)

    # Unsupported extension
    return ""


# Tokenization & word frequency
def cleaned_words(
    text: str,
    stopword_set=None,
    remove_stopwords: bool = True,
    alpha_only: bool = True
):
    if not text:
        return []

    tokens = word_tokenize(text.lower())

    words = []
    for w in tokens:
        if alpha_only and not w.isalpha():
            continue
        # strip non-letters from edges
        w_clean = re.sub(r'[^a-zA-Z]+', '', w)
        if not w_clean:
            continue
        if remove_stopwords and stopword_set and w_clean in stopword_set:
            continue
        words.append(w_clean)

    return words


def word_frequency(
    text: str,
    stopword_set=None,
    top_n: int = 20,
    remove_stopwords: bool = True
):
    """
    Return a list of (word, count) pairs for the top_n most
    frequent words.
    """
    words = cleaned_words(
        text,
        stopword_set=stopword_set,
        remove_stopwords=remove_stopwords
    )
    return Counter(words).most_common(top_n)


# Readability
def readability_scores(text: str):
    """
    Compute Flesch Reading Ease and Flesch-Kincaid Grade.
    Returns (ease, grade).
    """
    if not text:
        return 0.0, 0.0

    ease = textstat.flesch_reading_ease(text)
    grade = textstat.flesch_kincaid_grade(text)
    return ease, grade


def readability_level(grade: float) -> str:
    """
    Simple interpretation string for a grade level.
    """
    if grade <= 6:
        return "Easy to read (simple language)."
    elif grade <= 10:
        return "Moderately complex text."
    else:
        return "Advanced reading level (academic/technical)."


# NER post-processing
def merge_ner_entities(hf_ents):
    """
    Merge WordPiece / subword outputs from a HuggingFace NER pipeline.

    Input: list of dicts, each having keys:
        'word', 'entity', 'score', etc.

    Output: list of (word, label, avg_score)
    """
    cleaned = []
    buffer_word = ""
    buffer_label = ""
    scores = []

    for e in hf_ents:
        word = e["word"]
        label = e["entity"]
        score = e["score"]

        # HuggingFace pipelines may produce tokens like "##ing"
        if word.startswith("##"):
            buffer_word += word[2:]
            scores.append(score)
        else:
            # flush previous buffered entity
            if buffer_word:
                avg_score = sum(scores) / len(scores)
                cleaned.append((buffer_word, buffer_label, avg_score))
            buffer_word = word
            buffer_label = label
            scores = [score]

    # flush final
    if buffer_word:
        avg_score = sum(scores) / len(scores)
        cleaned.append((buffer_word, buffer_label, avg_score))

    return cleaned


# Charts
def build_word_freq_figure(freq_pairs, title="Top Words"):
    """
    Build and return a matplotlib Figure for a horizontal bar chart
    of word frequencies.

    freq_pairs: list of (word, count)
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    if not freq_pairs:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return fig

    labels, values = zip(*freq_pairs)
    ax.barh(labels, values)
    ax.set_xlabel("Count")
    ax.set_ylabel("Word")
    ax.set_title(title)
    fig.tight_layout()

    return fig


def apply_pipeline(text: str, cfg, stopword_set=None) -> str:
    """
    Apply the configured preprocessing pipeline to a single text string.
    """
    steps = cfg.get("steps", [])
    out = text

    for step in steps:
        if not step.get("enabled", True):
            continue
        sid = step["id"]

        if sid == "lowercase":
            out = out.lower()

        elif sid == "urls":
            out = remove_urls_hashtags_mentions(out)

        elif sid == "contractions":
            out = expand_contractions(out)

        elif sid == "numbers":
            out = normalize_numbers(out)

        elif sid == "stopwords":
            out = remove_stopwords_step(out, stopword_set)

        elif sid == "short_tokens":
            min_len = step.get("min_len", 3)
            out = remove_short_tokens(out, min_len, stopword_set)

        elif sid == "stem":
            toks = _simple_tokenize(out)
            toks = [_STEMMER.stem(t) for t in toks]
            out = _simple_detokenize(toks)

            # NEW: lemmatization
        elif sid == "lemma":
            toks = _simple_tokenize(out)
            toks = [_LEMMATIZER.lemmatize(t) for t in toks]
            out = _simple_detokenize(toks)

            # you can add more custom steps here later: emoji_to_text, profanity, etc.

            # OPTIONAL REGEX STEP – this is *not* a pipeline step name, just extra config
    print("PIPELINE CFG:", cfg)
    print("REGEX CFG:", cfg.get("regex"))

    regex_cfg = cfg.get("regex")
    if regex_cfg:
        pattern = regex_cfg.get("pattern") or ""
        replacement = regex_cfg.get("replacement") or ""
        if pattern.strip():
            try:
                out = re.sub(pattern, replacement, out)
            except re.error:
                # invalid regex – you might want to log or ignore
                pass

    return out


def remove_urls_hashtags_mentions(text: str) -> str:
    pattern = r"(https?://\S+|www\.\S+|@\w+|#\w+)"
    return re.sub(pattern, " ", text)

def expand_contractions(text: str) -> str:
    text = text.replace("’", "'").replace("‘", "'")
    contractions = {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "cannot",
        "can't've": "cannot have",
        "’cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he will have",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "I'd": "I would",
        "I'd've": "I would have",
        "I'll": "I will",
        "I'll've": "I will have",
        "I'm": "I am",
        "I've": "I have",
        "isn't": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so is",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who's": "who is",
        "who've": "who have",
        "why's": "why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you would",
        "you'd've": "you would have",
        "you'll": "you will",
        "you'll've": "you will have",
        "you're": "you are",
        "you've": "you have",
        # Common informal contractions
        "gonna": "going to",
        "wanna": "want to",
        "gotta": "got to",
        "lemme": "let me",
        "gimme": "give me",
        "kinda": "kind of",
        "sorta": "sort of",
        "outta": "out of",
        "lotta": "lot of",
        "’em": "them",
    }


    contractions_lc = {k.lower(): v for k, v in contractions.items()}

    def replace(match):
        original = match.group(0)        # as it appears in text
        c = original.lower()             # normalized
        full = contractions_lc.get(c, c)

        if original.isupper():
            # DON'T -> DO NOT
            full = full.upper()
        elif original[0].isupper():
            # Don't -> Do not
            full = full[0].upper() + full[1:]

        return full

    pattern = re.compile(
        r"\b(" + "|".join(re.escape(c) for c in contractions_lc.keys()) + r")\b",
        re.IGNORECASE
    )
    return pattern.sub(replace, text)

def normalize_numbers(text: str) -> str:
    return re.sub(r"\d+", " <NUM> ", text)


def remove_stopwords_step(text: str, stopword_set=None) -> str:
    words = cleaned_words(
        text,
        stopword_set=stopword_set,
        remove_stopwords=True,
        alpha_only=False
    )
    return " ".join(words)


def remove_short_tokens(text: str, min_len: int, stopword_set=None) -> str:
    words = text.split()
    kept = [w for w in words if len(w) >= min_len]
    return " ".join(kept)

def is_csv_mode(state) -> bool:
    """
    Return True if the state currently represents a loaded CSV file.
    """
    return getattr(state, "df", None) is not None and bool(
        getattr(state, "csv_text_column", None)
    )


def get_raw_documents_from_state(state) -> list[str]:
    """
    Return a list of *raw* documents based on the current state.

    - If CSV is loaded: one document per row in the chosen text column.
    - If a normal text file is loaded: a single document (state.text).
    """
    # CSV mode: one row per document
    if is_csv_mode(state):
        series = (
            state.df[state.csv_text_column]  # type: ignore[attr-defined]
            .fillna("")
            .astype(str)
        )
        docs = [s for s in series.tolist() if s.strip()]
        return docs

    # TXT/PDF mode: single document
    text = getattr(state, "text", "") or ""
    if text.strip():
        return [text]
    return []


def get_preprocessed_documents_from_state(state) -> list[str]:
    """
    Return a list of *preprocessed* documents based on the current state.

    This applies the current pipeline_config + stopwords
    to each raw document returned by get_raw_documents_from_state.
    """
    raw_docs = get_raw_documents_from_state(state)
    if not raw_docs:
        return []

    cfg = getattr(state, "pipeline_config", {}) or {}
    stopword_set = getattr(state, "stopwords", None)

    processed = [
        apply_pipeline(doc, cfg=cfg, stopword_set=stopword_set)
        for doc in raw_docs
    ]
    return [p for p in processed if p.strip()]
