import ttkbootstrap as tb
from ttkbootstrap.constants import *

class ArabicTab(tb.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.create_layout()

    def create_layout(self):
        sidebar = tb.Frame(self, bootstyle=SECONDARY)
        sidebar.pack(side="left", fill="y", padx=10, pady=10)

        tb.Label(sidebar, text="Arabic NLP (SinaTools)", 
                 font=("Segoe UI", 14, "bold")).pack(pady=15)

        tb.Button(sidebar, text="Upload Document", bootstyle=INFO).pack(fill="x", pady=5)
        tb.Button(sidebar, text="Normalize Text", bootstyle=PRIMARY).pack(fill="x", pady=5)
        tb.Button(sidebar, text="Remove Diacritics", bootstyle=PRIMARY).pack(fill="x", pady=5)
        tb.Button(sidebar, text="Tokenize Arabic", bootstyle=PRIMARY).pack(fill="x", pady=5)
        tb.Button(sidebar, text="POS Tagging", bootstyle=PRIMARY).pack(fill="x", pady=5)
        tb.Button(sidebar, text="Lemmatization", bootstyle=PRIMARY).pack(fill="x", pady=5)
        tb.Button(sidebar, text="Sentiment Analysis", bootstyle=PRIMARY).pack(fill="x", pady=5)

        self.output = tb.Text(self, font=("Consolas", 12))
        self.output.pack(side="right", fill="both", expand=True, padx=10, pady=10)
