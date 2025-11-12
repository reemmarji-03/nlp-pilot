import customtkinter as ctk
from tkinter import filedialog, messagebox
import pdfplumber, docx, re, nltk, textstat
from collections import Counter
from nltk.tokenize import word_tokenize
from transformers import pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

nltk.download("punkt", quiet=True)
nltk.download('punkt_tab')


class EnglishTab(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent, fg_color="#0f0f0f")
        self.text = ""
        self.file_name = ""
        self.sentiment_model = pipeline("sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english")
        self.ner_model = pipeline("ner", model="dslim/bert-base-NER")
        self.build_ui()

    # =======================================================
    # BUILD INTERFACE
    # =======================================================
    def build_ui(self):
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # Sidebar
        sidebar = ctk.CTkFrame(self, fg_color="#121212", corner_radius=10)
        sidebar.grid(row=0, column=0, sticky="ns", padx=(10,5), pady=10)

        ctk.CTkLabel(
            sidebar,
            text="English NLP Tools",
            text_color="#6ea8fe",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=(15,20))

        self.add_button(sidebar, "📂 Upload File", self.upload_file)
        self.add_button(sidebar, "🔠 Word Frequency", self.word_freq)
        self.add_button(sidebar, "💬 Sentiment", self.sentiment)
        self.add_button(sidebar, "🧬 Named Entities", self.named_entities)
        self.add_button(sidebar, "📚 Readability", self.readability)

        # === Output area with embedded file label ===
        output_frame = ctk.CTkFrame(self, fg_color="#1a1a1a", corner_radius=10)
        output_frame.grid(row=0, column=1, sticky="nsew", padx=(5,10), pady=10)

        # small file label inside frame
        self.file_label = ctk.CTkLabel(
            output_frame,
            text="No file loaded",
            anchor="w",
            text_color="#6ea8fe",
            fg_color="#1a1a1a",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.file_label.pack(anchor="w", padx=10, pady=(6,2))

        # main output textbox
        self.output = ctk.CTkTextbox(
            output_frame,
            fg_color="#1a1a1a",
            text_color="white",
            font=("Consolas", 13),
            wrap="word"
        )
        self.output.pack(fill="both", expand=True, padx=10, pady=(0,10))
        self.output._textbox.tag_configure("left", justify="left")

    def add_button(self, parent, text, command):
        ctk.CTkButton(parent, text=text, command=command,
                      fg_color="#0078ff", hover_color="#005dc1",
                      corner_radius=6, height=38,
                      font=ctk.CTkFont(size=14)
        ).pack(fill="x", padx=10, pady=6)

    # =======================================================
    # FUNCTIONS
    # =======================================================
    def upload_file(self):
        path = filedialog.askopenfilename(filetypes=[("Documents", "*.pdf *.docx *.txt")])
        if not path:
            return
        if path.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                self.text = f.read()
        elif path.endswith(".docx"):
            doc = docx.Document(path)
            self.text = "\n".join(p.text for p in doc.paragraphs)
        elif path.endswith(".pdf"):
            text = ""
            with pdfplumber.open(path) as pdf:
                for p in pdf.pages:
                    text += (p.extract_text() or "") + "\n"
            self.text = text
        self.file_name = path.split("/")[-1]
        self.file_label.configure(text=f"📂  {self.file_name}")

    # ------------------------- Word Frequency -------------------------
    def word_freq(self):
        if not self.text:
            return messagebox.showwarning("Warning", "Please upload a file first.")
        words = word_tokenize(self.text.lower())
        words = [re.sub(r'[^a-zA-Z]+', '', w) for w in words if w.isalpha()]
        freq = Counter(words).most_common(20)
        self.output.delete("1.0", "end")
        self.output.insert("end", "\n🔠 Top 20 Words\n", "left")
        self.output.insert("end", "-"*40 + "\n", "left")
        for w, c in freq:
            self.output.insert("end", f"{w:<15}{c:>5}\n", "left")

    # ------------------------- Sentiment -------------------------
    def sentiment(self):
        if not self.text:
            return messagebox.showwarning("Warning", "Please upload a file first.")
        result = self.sentiment_model(self.text[:500])[0]
        label = result["label"]
        score = result["score"]
        explanation = {
            "POSITIVE": "Optimistic or confident tone.",
            "NEGATIVE": "Critical or dissatisfied tone.",
            "NEUTRAL": "Balanced, factual tone."
        }.get(label, "Unclear tone detected.")
        self.output.delete("1.0", "end")
        self.output.insert("end", f"\n💬 Sentiment Analysis\n", "left")
        self.output.insert("end", "-"*40 + "\n", "left")
        self.output.insert("end", f"Label: {label}\nScore: {score:.3f}\n", "left")
        self.output.insert("end", f"Interpretation: {explanation}\n", "left")

    # ------------------------- Named Entities -------------------------
    def named_entities(self):
        if not self.text:
            return messagebox.showwarning("Warning", "Please upload a file first.")
        ents = self.ner_model(self.text[:1000])
        self.output.delete("1.0", "end")
        self.output.insert("end", "\n🧬 Named Entities\n", "left")
        self.output.insert("end", "-"*40 + "\n", "left")
        for e in ents:
            self.output.insert("end", f"{e['word']:<25}{e['entity']:<15}{e['score']:.2f}\n", "left")

    # ------------------------- Readability -------------------------
    def readability(self):
        if not self.text:
            return messagebox.showwarning("Warning", "Please upload a file first.")
        ease = textstat.flesch_reading_ease(self.text)
        grade = textstat.flesch_kincaid_grade(self.text)
        self.output.delete("1.0", "end")
        self.output.insert("end", "\n📚 Readability Analysis\n", "left")
        self.output.insert("end", "-"*40 + "\n", "left")
        self.output.insert("end", f"Reading Ease: {ease:.2f}\nGrade Level: {grade:.2f}\n", "left")
        if grade <= 6:
            level = "Easy to read (simple language)."
        elif grade <= 10:
            level = "Moderately complex text."
        else:
            level = "Advanced reading level (academic/technical)."
        self.output.insert("end", f"Interpretation: {level}\n", "left")
