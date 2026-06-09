import customtkinter as ctk
from tkinter import ttk
from tabs.english import EnglishTab
from tabs.settings_window import SettingsWindow
from core.reproducibility import set_global_seed
from core.settings_manager import settings
from core import theme_manager as theme

APP_TITLE = "NLP Pilot"

class NLPApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1150x700")
        ctk.set_default_color_theme("dark-blue")
        theme.set_mode(settings.data.get("theme_mode", "light"))

        self._settings_win = None

        # Header bar
        header = ctk.CTkFrame(self, fg_color=("#e0e0e0", "#1a1a1a"), corner_radius=0)
        header.pack(fill="x", padx=0, pady=0)

        ctk.CTkLabel(
            header,
            text=APP_TITLE,
            text_color=("#0062cc", "#6ea8fe"),
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(side="left", padx=12, pady=6)

        ctk.CTkButton(
            header,
            text="⚙",
            width=32,
            height=28,
            fg_color=("#d0d0d0", "#2a2a2a"),
            hover_color=("#c0c0c0", "#3a3a3a"),
            command=self._open_settings,
        ).pack(side="right", padx=8, pady=4)

        # notebook-based tabs
        self._style = ttk.Style()
        self._style.theme_use('clam')
        self._apply_notebook_style()

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        # Tabs
        self.english_tab = EnglishTab(self.notebook)
        self.notebook.add(self.english_tab, text="English NLP")

    def _apply_notebook_style(self):
        s = theme.scalars()
        self._style.configure("TNotebook", background=s["nb_bg"], borderwidth=0)
        self._style.configure("TNotebook.Tab", background=s["nb_tab_bg"], foreground=s["nb_tab_fg"], padding=[15, 6])
        self._style.map("TNotebook.Tab", background=[("selected", s["nb_tab_sel"])])

    def _open_settings(self):
        if self._settings_win is None or not self._settings_win.winfo_exists():
            self._settings_win = SettingsWindow(self)
        else:
            self._settings_win.lift()
            self._settings_win.focus()

    def refresh_theme(self):
        self._apply_notebook_style()
        if hasattr(self, "english_tab"):
            self.english_tab.apply_theme()

    def on_settings_changed(self):
        self.refresh_theme()
        if hasattr(self, "english_tab"):
            self.english_tab.on_settings_changed()


if __name__ == "__main__":
    set_global_seed(settings.get_seed())
    print("NLPApp started successfully")
    app = NLPApp()
    app.mainloop()
