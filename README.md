# NLP Pilot

NLP Pilot is a local-first desktop workbench for exploratory NLP on small and
medium corpora. It supports document ingestion, preprocessing, vectorization,
supervised prediction, clustering, topic modeling, RAG, and an agent-assisted
workflow.

## Requirements

- Python 3.11
- Windows, macOS, or Linux with a working Python/Tk installation
- Optional: Ollama for fully local LLM-backed RAG and agent workflows
- Optional: OpenAI or Anthropic API keys if you choose those providers

The application starts without any API key. LLM-backed features show a clear
message if no usable provider is configured.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

Some models are downloaded lazily on first use by Hugging Face, NLTK, or
BERTopic dependencies. For local LLM use, install Ollama and pull a model such
as:

```powershell
ollama pull qwen3:4b
```

Then open Settings in NLP Pilot and select Ollama.

## Typical Workflow

1. Open **Document Tools** and load a `.txt`, `.pdf`, `.docx`, or `.csv` file.
2. For CSV files, choose the text column when prompted.
3. Use **Data Quality** to inspect rows, missing values, duplicates, and text
   length statistics.
4. Use **Preprocessing** to build a cleaning pipeline and export a preprocessed
   CSV when needed.
5. Use **Vectorization** to compare Bag-of-Words, TF-IDF, character n-grams, and
   transformer embeddings.
6. Use **Prediction** for supervised classification/regression, clustering, and
   BERTopic topic modeling.
7. Use **RAG** or **Agent Lab** after configuring a local or cloud LLM provider
   in Settings.

## Reproducibility

The Settings window exposes a random seed. NLP Pilot uses it for train/test
splits, cross-validation shuffling, random forest estimators, K-Means,
BERTopic, NumPy, Python random sampling, and Torch where available. LLM outputs
and GPU kernels may still be nondeterministic depending on the provider and
runtime.

Supervised Auto mode evaluates candidate models with cross-validation on the
training split only, then fits the selected model on that training split and
reports metrics on the held-out test split.

## Exports

- PDF document analysis report
- JSON document analysis report
- CSV preprocessing output
- JSON supervised prediction results
- CSV supervised prediction results
- CSV/JSON BERTopic results

## Tests

```powershell
pytest
```
