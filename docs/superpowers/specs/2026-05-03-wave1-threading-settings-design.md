# Wave 1 Design: Background Threading & Settings/Provider Panel

**Date:** 2026-05-03
**Scope:** Wave 1 of 9-enhancement roadmap — foundational layer all other waves depend on
**Approach:** Centralized `TaskRunner` + `SettingsManager` (Approach B)

---

## Overview

Two parallel foundations:

1. **Background Threading** — moves all heavy ML operations off the main thread, preventing UI freezes. Exposes a progress indicator and cancel button on every long-running operation.
2. **Settings & Provider Panel** — replaces the hardcoded OpenAI dependency with a multi-provider LLM selector (Ollama / OpenAI / Anthropic) persisted to a local config file. Accessible via a gear icon in the main window.

---

## File Structure

### New files
```
core/settings_manager.py     singleton config, get_llm(), persist to ~/.nlp_pilot/config.json
core/task_runner.py           TaskRunner: background thread + cancel event + progress callbacks
tabs/settings_window.py       SettingsWindow(CTkToplevel): gear-icon-triggered settings UI
tabs/progress_overlay.py      reusable ProgressOverlay widget (bar + label + Cancel button)
```

### Modified files
```
main.py                              add gear icon button in top-right header
core/agent_graph.py                  lazy LLM init via settings_manager.get_llm()
tabs/english/agent_tab.py            wrap graph.invoke() in TaskRunner
tabs/english/prediction_tab.py       wrap train/cluster/bertopic in TaskRunner
tabs/english/vector_tab.py           wrap vectorization in TaskRunner
core/supervised.py                   accept cancel_event + progress_callback params
core/unsupervised.py                 accept cancel_event + progress_callback params
```

---

## Section 1: SettingsManager (`core/settings_manager.py`)

### Config location
`~/.nlp_pilot/config.json` — created on first run via `pathlib.Path.home()`, cross-platform.

### Schema
```json
{
  "active_provider": "ollama",
  "random_seed": 42,
  "ollama": {
    "url": "http://localhost:11434",
    "model": "qwen3:4b"
  },
  "openai": {
    "api_key": "",
    "model": "gpt-4o-mini"
  },
  "anthropic": {
    "api_key": "",
    "model": "claude-sonnet-4-6"
  }
}
```

### Public API
| Method | Description |
|--------|-------------|
| `settings.load()` | Read config.json; create with defaults if missing. Called once at import. |
| `settings.save()` | Write current `settings.data` to config.json. |
| `settings.get_llm()` | Return a LangChain chat model for the active provider, or `None` if unconfigured. |
| `settings.get_seed()` | Return `random_seed` as `int`. |
| `settings.data` | Raw dict for direct field access/update from the Settings window. |

### `get_llm()` provider mapping
- `"ollama"` → `ChatOllama(base_url=..., model=...)`
- `"openai"` → `ChatOpenAI(api_key=..., model=...)`
- `"anthropic"` → `ChatAnthropic(api_key=..., model=...)`
- Returns `None` if provider is unconfigured; callers show a friendly message instead of crashing.

### Notes
- The `.env` file's `OPENAI_API_KEY` is ignored in favour of `settings.get_llm()`.
- `agent_graph.py` module-level LLM instantiation is removed entirely. Every node calls `settings.get_llm()` at invocation time, so missing keys never crash the app on startup.
- Module-level singleton: `settings = SettingsManager()` at bottom of file; imported as `from core.settings_manager import settings`.

---

## Section 2: TaskRunner (`core/task_runner.py`)

### Interface
```python
class TaskRunner:
    def __init__(self, root)           # root: any tkinter widget for .after() dispatch
    def run(self, fn, args=(), kwargs={},
            on_progress=None,          # callable(percent: float, message: str)
            on_done=None,              # callable(result)
            on_error=None,             # callable(exception)
            on_cancel=None)            # callable() — called when cancel() was used to stop the run
    def cancel(self)
    @property
    def is_running(self) -> bool
```

### Mechanics
- Spawns a `threading.Thread(daemon=True)`.
- Stores a `threading.Event` as `cancel_event`.
- Worker receives `cancel_event` and `progress_callback` as kwargs **only if** the function declares them — detected via `inspect.signature`. Existing functions with no such params run unchanged.
- `progress_callback(percent, message)` posts to UI thread via `root.after(0, ...)`.
- All three callbacks (`on_progress`, `on_done`, `on_error`) are dispatched to the UI thread.
- `cancel()` sets `cancel_event`; workers that check it return early, at which point `TaskRunner` calls `on_cancel()` on the UI thread. Workers that don't honour it finish naturally — `on_done` is called as normal.
- `run()` raises `RuntimeError` if already running. Callers disable their trigger button while `is_running`.

### Usage pattern
```python
self.runner = TaskRunner(root=self)

def _do_train():
    return sup.train_supervised_model(
        ..., cancel_event=cancel_event, progress_callback=progress_callback
    )

self.runner.run(
    _do_train,
    on_progress=lambda p, msg: self.progress.update(p, msg),
    on_done=lambda result: self._show_results(result),
    on_error=lambda e: messagebox.showerror("Error", str(e)),
    on_cancel=lambda: (self.progress.hide(), self.train_btn.configure(state="normal")),
)
```

---

## Section 3: ProgressOverlay (`tabs/progress_overlay.py`)

### Visual
```
┌─────────────────────────────┐
│ [████████░░░░░░░░░░] 60%    │
│ Training Random Forest...   │
│        [Cancel]             │
└─────────────────────────────┘
```

### Interface
```python
class ProgressOverlay(ctk.CTkFrame):
    def __init__(self, parent, task_runner: TaskRunner)
    def show(self, message: str = "Working...")  # pack self, start indeterminate if needed
    def update(self, percent: float, message: str)
    def hide(self)
```

### Behaviour
- Determinate mode (0.0–1.0) when `percent` is provided; indeterminate (animated) when progress is unknown (e.g. BERTopic).
- Cancel button calls `task_runner.cancel()`.
- Action button that triggered the operation is disabled while overlay is visible; re-enabled in `on_done` / `on_error`.
- One instance per tab sidebar, shown/hidden in-place — no new windows.

---

## Section 4: SettingsWindow (`tabs/settings_window.py`)

### Window
- `CTkToplevel`, ~500×550px, non-resizable.
- Only one instance open at a time (guarded by `main.py` reference check).

### Layout
```
┌─────────────────────────────────────────┐
│  NLP Pilot Settings                     │
│                                         │
│  Active provider:                       │
│  ◉ Ollama  ○ OpenAI  ○ Anthropic       │
│                                         │
│  ┌─[Ollama]──[OpenAI]──[Anthropic]────┐ │
│  │ URL:  [http://localhost:11434     ] │ │
│  │ Model:[qwen3:4b ▾] [Refresh]       │ │
│  │                      [Test ●]      │ │
│  └────────────────────────────────────┘ │
│                                         │
│  Random seed: [42    ]                  │
│                                         │
│           [Cancel]  [Save]              │
└─────────────────────────────────────────┘
```

### Per-provider tab contents
| Provider | Fields | Test method |
|----------|--------|-------------|
| **Ollama** | URL entry · Model `CTkComboBox` populated by `GET /api/tags` on Refresh | `GET /api/version` |
| **OpenAI** | API key entry (show/hide toggle) · Model dropdown (`gpt-4o-mini`, `gpt-4o`, `gpt-3.5-turbo`) | Minimal `ChatOpenAI` call |
| **Anthropic** | API key entry (show/hide toggle) · Model dropdown (`claude-sonnet-4-6`, `claude-opus-4-7`, `claude-haiku-4-5`) | Minimal `ChatAnthropic` call |

### Test button
- Runs in a background thread (UI stays responsive).
- Shows `● green` on success or `● red` on failure next to the button.
- Short error message displayed below the button on failure.

### Save / Cancel
- **Save**: writes all fields to `settings.data`, calls `settings.save()`, closes window. Changes take effect immediately — next `get_llm()` call picks up the new config.
- **Cancel**: discards in-memory changes, closes window.

---

## Section 5: Main Window Changes (`main.py`)

```
┌──────────────────────────────────────────────┐
│  NLP Pilot                              [⚙]  │
├──────────────────────────────────────────────┤
│  [English NLP]                               │
│  ┌───────────────────────────────────────┐   │
│  │  Document Tools │ Preprocessing │ ... │   │
│  └───────────────────────────────────────┘   │
└──────────────────────────────────────────────┘
```

- Add a `CTkFrame` header row (`pack` before the notebook, `fill="x"`).
- Left: app title label. Right: `CTkButton(text="⚙", width=32, height=32)`.
- On click: opens `SettingsWindow(self)` if not already open (checked via `self._settings_win` reference).

---

## Section 6: Per-tab Integration

### `core/supervised.py` and `core/unsupervised.py`
- `train_supervised_model()` gains `cancel_event=None` and `progress_callback=None` params.
- In Auto mode: `progress_callback` called between candidates (33%, 66%, 100%); `cancel_event` checked before each — raises `CancelledError` if set.
- `run_kmeans()` and `run_bertopic()`: same two params added. BERTopic reports indeterminate progress and checks cancel only before starting (no iteration hooks available).

### Each tab with heavy operations
1. `self.runner = TaskRunner(root=self)` in `__init__`.
2. `self.progress = ProgressOverlay(sidebar, task_runner=self.runner)` in sidebar.
3. Every "Run"/"Train" button callback: disable button → `overlay.show()` → `runner.run(...)` → `on_done` hides overlay and re-enables button.

### `core/agent_graph.py`
- Remove all module-level LLM instantiation (`base_llm`, `task_selection_llm`, etc.).
- Replace with `_get_llm()` helper calling `settings.get_llm()` at call time.
- If `_get_llm()` returns `None`, nodes set `assistant_message` to: *"No LLM provider configured — open Settings (⚙) to set one up."*
- `build_agent_graph()` structure unchanged; only LLM references inside nodes change.

### `tabs/english/agent_tab.py`
- `graph.invoke(...)` wrapped in `TaskRunner`.
- Send button replaced with spinner while waiting; re-enabled in `on_done`.

---

## Error Handling

| Scenario | Behaviour |
|----------|-----------|
| No provider configured | `get_llm()` returns `None`; caller shows "Open Settings (⚙) to configure a provider" |
| Provider unreachable | Test button shows red; LLM call raises exception caught by `TaskRunner.on_error` |
| Operation cancelled | Worker returns early; `TaskRunner` calls `on_cancel()` — callers use this to hide overlay and re-enable button, no error dialog shown |
| config.json corrupted | `load()` catches `JSONDecodeError`, resets to defaults, saves clean file |

---

## Out of Scope (Waves 2–4)

- Data Quality Panel
- Richer visualisations
- Experiment presets / logging
- Export artifacts
- Session management
- Language expansion
