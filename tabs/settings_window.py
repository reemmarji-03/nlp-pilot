import copy
import json
import threading
import urllib.request

import customtkinter as ctk
from tkinter import messagebox

from core.settings_manager import settings
from core.reproducibility import set_global_seed


class SettingsWindow(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Settings")
        self.geometry("520x600")
        self.resizable(False, False)
        self.grab_set()

        # Work on a deep copy so Cancel truly discards changes
        self._data = copy.deepcopy(settings.data)
        self._build_ui()

    # ── Build ──────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        ctk.CTkLabel(
            self,
            text="Settings",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#6ea8fe",
        ).pack(pady=(16, 8))

        self._tabs = ctk.CTkTabview(self, height=290)
        self._tabs.pack(fill="x", padx=20, pady=(0, 8))
        self._build_ollama_tab(self._tabs.add("Ollama"))
        self._build_openai_tab(self._tabs.add("OpenAI"))
        self._build_anthropic_tab(self._tabs.add("Anthropic"))
        active_tab = {
            "ollama": "Ollama",
            "openai": "OpenAI",
            "anthropic": "Anthropic",
        }.get(self._data.get("active_provider", "ollama"), "Ollama")
        self._tabs.set(active_tab)
        self._refresh_ollama_models(silent=True)

        seed_row = ctk.CTkFrame(self, fg_color="transparent")
        seed_row.pack(fill="x", padx=20, pady=(0, 16))
        ctk.CTkLabel(seed_row, text="Random seed:", text_color="#cccccc").pack(
            side="left"
        )
        self._seed_entry = ctk.CTkEntry(seed_row, width=70)
        self._seed_entry.insert(0, str(self._data.get("random_seed", 42)))
        self._seed_entry.pack(side="left", padx=(8, 0))

        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(fill="x", padx=20, pady=(0, 16))
        ctk.CTkButton(
            btn_row, text="Cancel", fg_color="#555555", hover_color="#444444",
            command=self.destroy, width=100,
        ).pack(side="right", padx=(8, 0))
        ctk.CTkButton(
            btn_row, text="Save", fg_color="#0078ff", hover_color="#005dc1",
            command=self._save, width=100,
        ).pack(side="right")

    def _build_ollama_tab(self, parent) -> None:
        ctk.CTkLabel(parent, text="URL:", text_color="#cccccc").pack(
            anchor="w", padx=8, pady=(8, 0)
        )
        self._ollama_url = ctk.CTkEntry(parent, width=300)
        self._ollama_url.insert(0, self._data["ollama"].get("url", "http://localhost:11434"))
        self._ollama_url.pack(anchor="w", padx=8, pady=(2, 8))

        ctk.CTkLabel(parent, text="Model:", text_color="#cccccc").pack(anchor="w", padx=8)
        model_row = ctk.CTkFrame(parent, fg_color="transparent")
        model_row.pack(fill="x", padx=8, pady=(2, 8))

        self._ollama_model = ctk.CTkComboBox(
            model_row,
            values=[self._data["ollama"].get("model", "qwen3:4b")],
            width=200,
        )
        self._ollama_model.set(self._data["ollama"].get("model", "qwen3:4b"))
        self._ollama_model.pack(side="left")

        ctk.CTkButton(
            model_row, text="Refresh", width=80,
            fg_color="#444444", hover_color="#333333",
            command=self._refresh_ollama_models,
        ).pack(side="left", padx=(8, 0))

        test_row = ctk.CTkFrame(parent, fg_color="transparent")
        test_row.pack(fill="x", padx=8)
        ctk.CTkButton(
            test_row, text="Test", width=80,
            fg_color="#444444", hover_color="#333333",
            command=self._test_ollama,
        ).pack(side="left")
        self._ollama_status = ctk.CTkLabel(test_row, text="", width=20)
        self._ollama_status.pack(side="left", padx=(8, 0))
        self._ollama_msg = ctk.CTkLabel(
            test_row, text="", text_color="#ff6666",
            font=ctk.CTkFont(size=10), wraplength=200,
        )
        self._ollama_msg.pack(side="left", padx=(4, 0))

    def _build_openai_tab(self, parent) -> None:
        ctk.CTkLabel(parent, text="API Key:", text_color="#cccccc").pack(
            anchor="w", padx=8, pady=(8, 0)
        )
        key_row = ctk.CTkFrame(parent, fg_color="transparent")
        key_row.pack(fill="x", padx=8, pady=(2, 8))

        self._openai_key = ctk.CTkEntry(key_row, width=240, show="*")
        self._openai_key.insert(0, self._data["openai"].get("api_key", ""))
        self._openai_key.pack(side="left")
        ctk.CTkButton(
            key_row, text="Show", width=60,
            fg_color="#444444", hover_color="#333333",
            command=lambda: self._toggle_show(self._openai_key),
        ).pack(side="left", padx=(8, 0))

        ctk.CTkLabel(parent, text="Model:", text_color="#cccccc").pack(anchor="w", padx=8)
        self._openai_model = ctk.CTkComboBox(
            parent,
            values=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            width=200,
        )
        self._openai_model.set(self._data["openai"].get("model", "gpt-4o-mini"))
        self._openai_model.pack(anchor="w", padx=8, pady=(2, 8))

        test_row = ctk.CTkFrame(parent, fg_color="transparent")
        test_row.pack(fill="x", padx=8)
        ctk.CTkButton(
            test_row, text="Test", width=80,
            fg_color="#444444", hover_color="#333333",
            command=self._test_openai,
        ).pack(side="left")
        self._openai_status = ctk.CTkLabel(test_row, text="", width=20)
        self._openai_status.pack(side="left", padx=(8, 0))
        self._openai_msg = ctk.CTkLabel(
            test_row, text="", text_color="#ff6666",
            font=ctk.CTkFont(size=10), wraplength=200,
        )
        self._openai_msg.pack(side="left", padx=(4, 0))

    def _build_anthropic_tab(self, parent) -> None:
        ctk.CTkLabel(parent, text="API Key:", text_color="#cccccc").pack(
            anchor="w", padx=8, pady=(8, 0)
        )
        key_row = ctk.CTkFrame(parent, fg_color="transparent")
        key_row.pack(fill="x", padx=8, pady=(2, 8))

        self._anthropic_key = ctk.CTkEntry(key_row, width=240, show="*")
        self._anthropic_key.insert(0, self._data["anthropic"].get("api_key", ""))
        self._anthropic_key.pack(side="left")
        ctk.CTkButton(
            key_row, text="Show", width=60,
            fg_color="#444444", hover_color="#333333",
            command=lambda: self._toggle_show(self._anthropic_key),
        ).pack(side="left", padx=(8, 0))

        ctk.CTkLabel(parent, text="Model:", text_color="#cccccc").pack(anchor="w", padx=8)
        self._anthropic_model = ctk.CTkComboBox(
            parent,
            values=["claude-sonnet-4-6", "claude-opus-4-7", "claude-haiku-4-5"],
            width=220,
        )
        self._anthropic_model.set(self._data["anthropic"].get("model", "claude-sonnet-4-6"))
        self._anthropic_model.pack(anchor="w", padx=8, pady=(2, 8))

        test_row = ctk.CTkFrame(parent, fg_color="transparent")
        test_row.pack(fill="x", padx=8)
        ctk.CTkButton(
            test_row, text="Test", width=80,
            fg_color="#444444", hover_color="#333333",
            command=self._test_anthropic,
        ).pack(side="left")
        self._anthropic_status = ctk.CTkLabel(test_row, text="", width=20)
        self._anthropic_status.pack(side="left", padx=(8, 0))
        self._anthropic_msg = ctk.CTkLabel(
            test_row, text="", text_color="#ff6666",
            font=ctk.CTkFont(size=10), wraplength=200,
        )
        self._anthropic_msg.pack(side="left", padx=(4, 0))

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _toggle_show(entry: ctk.CTkEntry) -> None:
        entry.configure(show="" if entry.cget("show") == "*" else "*")

    def _refresh_ollama_models(self, silent: bool = False) -> None:
        url = self._ollama_url.get().rstrip("/")
        current = self._ollama_model.get()

        def _worker():
            try:
                with urllib.request.urlopen(f"{url}/api/tags", timeout=5) as resp:
                    data = json.loads(resp.read())
                models = [m["name"] for m in data.get("models", [])]
                if models:
                    def _update():
                        if self.winfo_exists():
                            self._ollama_model.configure(values=models)
                            self._ollama_model.set(current if current in models else models[0])
                    self.after(0, _update)
            except Exception as exc:
                def _error(e=exc):
                    if self.winfo_exists() and not silent:
                        messagebox.showerror("Ollama Error", f"Could not fetch models:\n{e}", parent=self)
                self.after(0, _error)

        threading.Thread(target=_worker, daemon=True).start()

    def _test_ollama(self) -> None:
        url = self._ollama_url.get().rstrip("/")
        self._ollama_status.configure(text="…", text_color="#aaaaaa")
        self._ollama_msg.configure(text="")

        def _worker():
            try:
                with urllib.request.urlopen(f"{url}/api/version", timeout=5):
                    pass
                def _ok():
                    if self.winfo_exists():
                        self._ollama_status.configure(text="●", text_color="#44ff44")
                self.after(0, _ok)
            except Exception as exc:
                err = str(exc)
                def _fail(e=err):
                    if self.winfo_exists():
                        self._ollama_status.configure(text="●", text_color="#ff4444")
                        self._ollama_msg.configure(text=e[:80])
                self.after(0, _fail)

        threading.Thread(target=_worker, daemon=True).start()

    def _test_openai(self) -> None:
        key = self._openai_key.get()
        model = self._openai_model.get()
        self._openai_status.configure(text="…", text_color="#aaaaaa")
        self._openai_msg.configure(text="")

        def _worker():
            try:
                from langchain_openai import ChatOpenAI
                from langchain.messages import HumanMessage
                llm = ChatOpenAI(api_key=key, model=model, temperature=0, max_tokens=1)
                llm.invoke([HumanMessage(content="hi")])
                def _ok():
                    if self.winfo_exists():
                        self._openai_status.configure(text="●", text_color="#44ff44")
                self.after(0, _ok)
            except Exception as exc:
                err = str(exc)[:80]
                def _fail(e=err):
                    if self.winfo_exists():
                        self._openai_status.configure(text="●", text_color="#ff4444")
                        self._openai_msg.configure(text=e)
                self.after(0, _fail)

        threading.Thread(target=_worker, daemon=True).start()

    def _test_anthropic(self) -> None:
        key = self._anthropic_key.get()
        model = self._anthropic_model.get()
        self._anthropic_status.configure(text="…", text_color="#aaaaaa")
        self._anthropic_msg.configure(text="")

        def _worker():
            try:
                from langchain_anthropic import ChatAnthropic
                from langchain.messages import HumanMessage
                llm = ChatAnthropic(api_key=key, model=model, temperature=0, max_tokens=1)
                llm.invoke([HumanMessage(content="hi")])
                def _ok():
                    if self.winfo_exists():
                        self._anthropic_status.configure(text="●", text_color="#44ff44")
                self.after(0, _ok)
            except Exception as exc:
                err = str(exc)[:80]
                def _fail(e=err):
                    if self.winfo_exists():
                        self._anthropic_status.configure(text="●", text_color="#ff4444")
                        self._anthropic_msg.configure(text=e)
                self.after(0, _fail)

        threading.Thread(target=_worker, daemon=True).start()

    # ── Save ───────────────────────────────────────────────────────

    def _save(self) -> None:
        settings.data["active_provider"] = {
            "Ollama": "ollama",
            "OpenAI": "openai",
            "Anthropic": "anthropic",
        }.get(self._tabs.get(), "ollama")
        settings.data["ollama"]["url"] = self._ollama_url.get()
        settings.data["ollama"]["model"] = self._ollama_model.get()
        settings.data["openai"]["api_key"] = self._openai_key.get()
        settings.data["openai"]["model"] = self._openai_model.get()
        settings.data["anthropic"]["api_key"] = self._anthropic_key.get()
        settings.data["anthropic"]["model"] = self._anthropic_model.get()
        try:
            settings.data["random_seed"] = int(self._seed_entry.get())
        except ValueError:
            settings.data["random_seed"] = 42
        settings.save()
        set_global_seed(settings.get_seed())
        if hasattr(self.master, "on_settings_changed"):
            self.master.on_settings_changed()
        self.destroy()
