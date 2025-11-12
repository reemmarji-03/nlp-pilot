import customtkinter as ctk
from tkinter import ttk
from tabs.english_tab import EnglishTab

class NLPApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("NLP Pilot")
        self.geometry("1150x700")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        # --- notebook-based tabs ---
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

        self.arabic_tab = ctk.CTkFrame(self.notebook, fg_color="#0f0f0f")
        self.notebook.add(self.arabic_tab, text="Arabic NLP")

        self.youtube_tab = ctk.CTkFrame(self.notebook, fg_color="#0f0f0f")
        self.notebook.add(self.youtube_tab, text="YouTube Analysis")

if __name__ == "__main__":
    print("✅ NLPApp started successfully")
    app = NLPApp()
    app.mainloop()
