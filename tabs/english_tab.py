import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
import pdfplumber
import docx
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
from transformers import pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
nltk.download("punkt")


class EnglishTab(ctk.CTkFrame):

    def __init__(self, parent):
        super().__init__(parent)
        self.pack(fill="both", expand=True)

        # UI theme cleanup (remove grey)
        self.configure(fg_color="transparent")

        self.sidebar_expanded = True
        self.text = ""  # stores loaded document
        self.file_name = "No file yet"

        # Preload sentiment model
        self.sentiment_model = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )

        self.create_layout()

    # ==================================================
    # MAIN LAYOUT
    # ==================================================
    def create_layout(self):
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # Left Sidebar
        self.sidebar = ctk.CTkFrame(
            self, width=240,
            corner_radius=15,
            fg_color="#1f1b29", border_width=0
        )
        self.sidebar.grid(row=0, column=0, sticky="nsw", padx=10, pady=10)

        # Main content area
        self.main_area = ctk.CTkFrame(
            self, corner_radius=15,
            fg_color="#121214", border_width=0
        )
        self.main_area.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        self.build_sidebar()
        self.build_dashboard()

    # ==================================================
    # SIDEBAR WITH ICON BUTTONS
    # ==================================================
    def build_sidebar(self):
        for child in self.sidebar.winfo_children():
            child.destroy()

        ctk.CTkLabel(
            self.sidebar, text="English NLP Analysis",
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(pady=20)

        ctk.CTkButton(
            self.sidebar,
            text="⮜ Collapse",
            command=self.toggle_sidebar,
            fg_color="#8b39ef"
        ).pack(pady=10, fill="x")

        # Actual NLP Buttons
        self.add_icon_button("Upload Document", "upload.png", self.upload_file)
        self.add_icon_button("Word Frequency", "freq.png", self.do_word_frequency)
        self.add_icon_button("Topic Modeling", "lda.png", self.do_topic_modeling)
        self.add_icon_button("Sentiment Analysis", "sentiment.png", self.do_sentiment)
        self.add_icon_button("Summarize Document", "summary.png", self.fake_action)

    def add_icon_button(self, text, icon_name, callback):
        try:
            icon = ctk.CTkImage(
                light_image=Image.open(f"assets/icons/{icon_name}"),
                dark_image=Image.open(f"assets/icons/{icon_name}"),
                size=(20, 20)
            )
        except:
            icon = None

        ctk.CTkButton(
            self.sidebar,
            text=text,
            image=icon,
            compound="left",
            height=40,
            fg_color="#8b39ef",
            hover_color="#d3b2ff",
            command=callback
        ).pack(pady=8, fill="x")

    # ==================================================
    # COLLAPSIBLE SIDEBAR
    # ==================================================
    def toggle_sidebar(self):
        if self.sidebar_expanded:
            self.sidebar.configure(width=70)
            for child in self.sidebar.winfo_children():
                child.pack_forget()
            self.sidebar_expanded = False
        else:
            self.sidebar.configure(width=240)
            self.build_sidebar()
            self.sidebar_expanded = True

    # ==================================================
    # DASHBOARD
    # ==================================================
    def build_dashboard(self):
        top_frame = ctk.CTkFrame(
            self.main_area,
            corner_radius=15,
            fg_color="#1f1b29"
        )
        top_frame.pack(fill="x", pady=10)

        self.card_file = self.create_card(top_frame, "Uploaded File", self.file_name)
        self.card_wordcount = self.create_card(top_frame, "Word Count", "0")
        self.card_sentiment = self.create_card(top_frame, "Sentiment", "–")

        self.output = ctk.CTkTextbox(
            self.main_area, height=400,
            corner_radius=15, fg_color="#1f1b29",
            text_color="white", border_width=0
        )
        self.output.pack(fill="both", expand=True, pady=10)

    def create_card(self, parent, title, value):
        card = ctk.CTkFrame(parent, corner_radius=15, fg_color="#1f1b29")
        card.pack(side="left", padx=20, pady=10, fill="x", expand=True)

        label_title = ctk.CTkLabel(
            card, text=title, font=ctk.CTkFont(size=14, weight="bold")
        )
        label_title.pack(pady=(10, 0))

        label_value = ctk.CTkLabel(
            card, text=value, font=ctk.CTkFont(size=24, weight="bold")
        )
        label_value.pack(pady=10)

        return label_value  # return value label for updates

    # ==================================================
    # NLP ACTIONS
    # ==================================================
    def upload_file(self):
        self.file_path = filedialog.askopenfilename(
            filetypes=[("Documents", "*.pdf *.docx *.txt")]
        )
        if not self.file_path:
            return

        self.text = self.extract_text_from_file(self.file_path)
        self.file_name = self.file_path.split("/")[-1]

        self.card_file.configure(text=self.file_name)
        self.card_wordcount.configure(text=str(len(self.text.split())))
        self.output.insert("end", f"[✓] Loaded file: {self.file_name}\n")

    def extract_text_from_file(self, path):
        if path.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()

        elif path.endswith(".docx"):
            doc = docx.Document(path)
            return "\n".join([p.text for p in doc.paragraphs])

        elif path.endswith(".pdf"):
            text = ""
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text

    # -------------------------
    # Word Frequency
    # -------------------------
    def do_word_frequency(self):
        if not self.text:
            return self.output.insert("end", "No document loaded.\n")

        freq = self.compute_word_frequency(self.text)
        self.output.insert("end", "\n[Word Frequency]\n")
        for word, count in freq:
            self.output.insert("end", f"{word}: {count}\n")

    def compute_word_frequency(self, text):
        words = word_tokenize(text.lower())
        freq = Counter(words)
        return freq.most_common(20)

    # -------------------------
    # Sentiment
    # -------------------------
    def do_sentiment(self):
        if not self.text:
            return self.output.insert("end", "No document loaded.\n")

        result = self.analyze_sentiment(self.text)
        label = result["label"]
        score = round(result["score"], 3)

        self.card_sentiment.configure(text=label)
        self.output.insert("end", f"\n[Sentiment] → {label} ({score})\n")

    def analyze_sentiment(self, text):
        return self.sentiment_model(text[:500])[0]

    # -------------------------
    # Topic Modeling (LDA)
    # -------------------------
    def do_topic_modeling(self):
        if not self.text:
            return self.output.insert("end", "No document loaded.\n")

        topics = self.run_lda(self.text)
        self.output.insert("end", "\n[Topics]\n")
        for idx, words in topics:
            self.output.insert("end", f"Topic {idx}: {', '.join(words)}\n")

    def run_lda(self, text, n_topics=3):
        vectorizer = CountVectorizer(stop_words="english")
        X = vectorizer.fit_transform([text])

        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(X)

        words = vectorizer.get_feature_names_out()

        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words = [words[i] for i in topic.argsort()[-8:]]
            topics.append((topic_idx, top_words))
        return topics

    # -------------------------
    # Placeholder
    # -------------------------
    def fake_action(self):
        self.output.insert("end", "[TODO] Feature under construction.\n")

