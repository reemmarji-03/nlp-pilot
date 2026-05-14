# NLP Pilot

NLP Pilot is a local-first desktop workbench for exploratory NLP on small and
medium corpora. It brings document ingestion, data quality checks,
preprocessing, vectorization, supervised prediction, clustering, topic modeling,
retrieval-augmented generation, and an agent-guided workflow into one GUI.

The app is designed for teaching, prototyping, and research workflows where
users want to inspect each step rather than run a hidden end-to-end pipeline.

## Requirements

- Python 3.11
- Windows, macOS, or Linux with a working Python/Tk installation
- Optional: Ollama for fully local LLM-backed RAG and Agent Lab workflows
- Optional: OpenAI or Anthropic API keys if you choose those providers

The supported Python version is intentionally pinned to Python 3.11 in
`pyproject.toml`, `requirements.txt`, and `environment.yml`.

## Installation

For a plain Python environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

Conda users should prefer `environment.yml`. It installs compiled scientific
packages from `conda-forge` and then installs the smaller pip-only layer from
`requirements-conda.txt`:

```powershell
conda env create -f environment.yml
conda activate nlp-pilot
python main.py
```

The dependency files are checked in CI with both pip and micromamba. Locally,
you can verify an environment with:

```powershell
python -m pip check
pytest
```

Some models are downloaded lazily on first use by Hugging Face, NLTK, or
BERTopic dependencies.

## Settings And Local Operation

Open **Settings** to configure the global LLM provider, model, provider
credentials, Ollama URL, and random seed. NLP Pilot does not require an OpenAI
API key to launch. It can run in fully local mode with Ollama, and OpenAI or
Anthropic keys are only needed if you select those provider tabs.

For local LLM use, install Ollama and pull a model, for example:

```powershell
ollama pull qwen3:4b
```

Then open **Settings**, select the Ollama tab, refresh models if needed, and
choose the model. The same global model setting is used by RAG and Agent Lab.

## Main Workflow

1. Open **Document Tools** and load a `.txt`, `.pdf`, `.docx`, or `.csv` file.
2. For CSV files, choose the text column when prompted.
3. Use **Data Quality** from Document Tools to inspect rows, missing values,
   duplicates, and text length statistics.
4. Use **Preprocessing** to build a cleaning pipeline and export a preprocessed
   CSV when needed.
5. Use **Vectorization** to compare Bag-of-Words, TF-IDF, character n-grams, and
   transformer embeddings.
6. Use **Prediction** for supervised classification/regression, K-Means
   clustering, and BERTopic topic modeling.
7. Use **RAG** for question answering over text documents.
8. Use **Agent Lab** for a guided conversational workflow across task choice,
   preprocessing, vectorization, prediction, and clustering.

## Document Tools

Document Tools supports text, PDF, DOCX, and CSV loading. For CSV files, NLP
Pilot keeps the table structure and asks which column contains the text. Data
Quality is available directly from this tab.

Document analysis includes NER and sentiment output in a simpler text-first view.
The document report export uses one dialog where you choose the destination and
format:

- PDF
- JSON

## RAG

RAG is available for text documents (`.txt`, `.pdf`, `.docx`). CSV uploads are
intended for row-based prediction, clustering, and topic modeling, so the RAG tab
does not build an index from CSV rows.

When a text document is loaded, opening the RAG tab automatically builds the
index. The chat appears when the index and model are ready. User messages appear
on the right and model messages on the left. Response metadata from providers
such as Ollama is kept in the RAG log panel instead of being mixed into the chat
answer.

Changing the global model in **Settings** updates RAG automatically; users do not
need to rebuild manually just to switch models.

## Agent Lab

Agent Lab is a guided conversational workflow that uses the global Settings LLM.
It helps users:

- choose classification, regression, or clustering
- propose and revise preprocessing steps
- choose a vectorization strategy
- select or suggest a label column for supervised prediction
- configure supervised training settings
- run K-Means clustering
- review results and route follow-up requests naturally

Agent Lab now treats unusual or unrelated turns as clarification moments instead
of restarting the conversation. It also avoids repeated greetings mid-workflow
and addresses the user directly rather than referring to "the user".

For CSV prediction workflows, Agent Lab shows available candidate label columns
and suggests a likely label using column names, data types, unique counts,
missing values, and sample values.

## Prediction

The Prediction tab supports supervised classification and regression with:

- Logistic Regression
- Linear SVM
- Random Forest
- Linear/Ridge regression options for numeric targets
- Auto mode for model selection
- registered custom models

Common settings stay visible in the left panel, while detailed supervised
training settings are available from **Training Settings**. This keeps the tab
usable on smaller screens while still exposing important hyperparameters:

- test size
- Auto CV folds
- optional Auto subset size
- random forest tree count

Auto mode evaluates candidate models with cross-validation on the training split
only, then fits the selected model on that training split and reports metrics on
the held-out test split. Text vectorizers are fit only inside the training split
to avoid vocabulary leakage into the held-out test split.

Prediction exports use a single **Export Results** dialog:

- JSON supervised result summary
- CSV prediction rows

Clustering exports CSV cluster assignments. Topic modeling exports use one
**Export Topics** dialog with CSV or JSON output.

## Extending Models

Custom supervised models can be added from **Prediction > Manage Models**.
Choose a Python file, enter a class or factory function name, select
classification or regression, and NLP Pilot validates that the object returns a
scikit-learn compatible estimator. Saved model entries are persisted in
`~/.nlp_pilot/custom_models.json` and appear in the model dropdown.

This repository includes example custom estimators in
`examples/custom_supervised_models.py`. In **Manage Models**, select that file
and use one of:

- `NaiveBayesTextClassifier` for classification
- `sgd_logistic_classifier` for classification
- `elastic_net_regressor` for regression

Models can also be registered programmatically through
`core.model_registry.register_model`:

```python
from sklearn.naive_bayes import MultinomialNB
from core.model_registry import register_model

register_model("classification", "Naive Bayes", lambda random_state=None: MultinomialNB())
```

## Reproducibility

The Settings window exposes a random seed. NLP Pilot uses it for train/test
splits, cross-validation shuffling, random forest estimators, K-Means,
BERTopic, NumPy, Python random sampling, and Torch where available.

LLM outputs and GPU kernels may still be nondeterministic depending on the
provider, model, hardware, and runtime. The app exposes the model/provider
choice and seed so runs can be documented, but cloud LLM responses and some GPU
operations cannot be guaranteed bit-for-bit reproducible.

Create a reproducible capsule for the current environment with:

```powershell
python scripts/generate_reproducible_capsule.py
```

The capsule writes redacted settings, platform details, package versions, and
`pip freeze` output to `reproducible_capsule/`.

## Output Formats

Machine-readable outputs are available throughout the app:

- PDF or JSON document analysis report
- CSV preprocessed corpus
- JSON supervised prediction summary
- CSV supervised prediction rows
- CSV clustering assignments
- CSV or JSON topic modeling output
- reproducible capsule metadata

## Robustness

NLP Pilot includes validation and user-facing error handling for common issues:

- malformed or unsupported files
- missing CSV text columns
- missing values in supervised labels
- incompatible model/task selections
- CSV input in RAG
- model training failures
- provider/model configuration errors

Long transformer inputs are truncated safely for NER and sentiment pipelines
where model limits apply.

## Tests

Run the automated test suite with:

```powershell
pytest
```

The current suite covers settings, file loading, data quality, supervised
prediction, model registry behavior, reproducibility metadata, task runner
behavior, exports, and Agent Lab graph routing. Dependency install checks run in
GitHub Actions for both pip and micromamba/Conda-style environments.
