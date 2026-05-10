import customtkinter as ctk
from tkinter import ttk
from tabs.english import EnglishTab
from tabs.settings_window import SettingsWindow
from core.reproducibility import set_global_seed
from core.settings_manager import settings
import lookups

class NLPApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title(lookups.title)
        self.geometry("1150x700")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self._settings_win = None

        # Header bar
        header = ctk.CTkFrame(self, fg_color="#1a1a1a", corner_radius=0)
        header.pack(fill="x", padx=0, pady=0)

        title_label = ctk.CTkLabel(
            header,
            text=lookups.title,
            text_color="#6ea8fe",
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        title_label.pack(side="left", padx=12, pady=6)

        gear_btn = ctk.CTkButton(
            header,
            text="⚙",
            width=32,
            height=28,
            fg_color="#2a2a2a",
            hover_color="#3a3a3a",
            command=self._open_settings,
        )
        gear_btn.pack(side="right", padx=8, pady=4)

        # notebook-based tabs
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TNotebook", background="#0f0f0f", borderwidth=0)
        style.configure("TNotebook.Tab", background="#1a1a1a", foreground="white", padding=[15, 6])
        style.map("TNotebook.Tab", background=[("selected", "#0078ff")])

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        # Tabs
        self.english_tab = EnglishTab(self.notebook)
        self.notebook.add(self.english_tab, text="English NLP")

    def _open_settings(self):
        if self._settings_win is None or not self._settings_win.winfo_exists():
            self._settings_win = SettingsWindow(self)
        else:
            self._settings_win.lift()
            self._settings_win.focus()


if __name__ == "__main__":
    set_global_seed(settings.get_seed())
    print("NLPApp started successfully")
    app = NLPApp()
    app.mainloop()
