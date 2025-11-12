import customtkinter as ctk
from tkinter import filedialog, messagebox
import pdfplumber, docx, re, nltk, textstat
from collections import Counter
from nltk.tokenize import word_tokenize
from transformers import pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
import tempfile, os

nltk.download("punkt", quiet=True)
nltk.download('punkt_tab')
from nltk.corpus import stopwords
nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))


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
        self.add_button(sidebar, "📈 Generate Charts", self.generate_charts)
        self.add_button(sidebar, "🧾 Export PDF Report", self.export_report)

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
        words = [
            re.sub(r'[^a-zA-Z]+', '', w)
            for w in words
            if w.isalpha() and w not in STOPWORDS
        ]
        freq = Counter(words).most_common(20)

        self.output.delete("1.0", "end")
        self.output.insert("end", "\n🔠 Top 20 Words (excluding stopwords)\n", "left")
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
        cleaned = []
        buffer_word, buffer_label, scores = "", "", []

        for e in ents:
            word = e["word"]
            label = e["entity"]
            score = e["score"]

            # Merge subwords (##...)
            if word.startswith("##"):
                buffer_word += word[2:]
                scores.append(score)
            else:
                # flush previous entity
                if buffer_word:
                    cleaned.append((buffer_word, buffer_label, sum(scores)/len(scores)))
                buffer_word = word
                buffer_label = label
                scores = [score]

        if buffer_word:
            cleaned.append((buffer_word, buffer_label, sum(scores)/len(scores)))

        # Show cleaned results
        self.output.delete("1.0", "end")
        self.output.insert("end", "\n🧬 Named Entities\n", "left")
        self.output.insert("end", "-"*40 + "\n", "left")

        for word, label, score in cleaned:
            self.output.insert("end", f"{word:<25}{label:<10}{score:.2f}\n", "left")

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

    # ------------------------- Charts -------------------------
    def generate_charts(self):
        if not self.text:
            return messagebox.showwarning("Warning", "Please upload a file first.")
        
        # ---- Word Frequency ----
        words = word_tokenize(self.text.lower())
        words = [re.sub(r'[^a-zA-Z]+', '', w) for w in words if w.isalpha()]
        freq = Counter(words).most_common(10)
        labels, values = zip(*freq)

        fig, ax = plt.subplots(figsize=(6,4))
        ax.barh(labels, values, color="#0078ff")
        ax.set_xlabel("Count")
        ax.set_ylabel("Word")
        ax.set_title("Top 10 Words")

        plt.tight_layout()

        # Embed chart into Tkinter
        chart_window = ctk.CTkToplevel(self)
        chart_window.title("Word Frequency Chart")
        chart_window.geometry("650x450")
        chart_window.configure(fg_color="#0f0f0f")

        canvas = FigureCanvasTkAgg(fig, master=chart_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)


    # ------------------------- Export Report -------------------------
    def export_report(self):
        if not self.text:
            return messagebox.showwarning("Warning", "Please upload a file first.")

        # ====== Compute analyses ======
        sentiment = self.sentiment_model(self.text[:500])[0]
        readability = textstat.flesch_kincaid_grade(self.text)
        words = word_tokenize(self.text.lower())
        words = [re.sub(r'[^a-zA-Z]+', '', w) for w in words if w.isalpha()]
        freq = Counter(words).most_common(10)
        labels, values = zip(*freq)

        # ====== Create chart images ======
        fig, ax = plt.subplots(figsize=(5,3))
        ax.barh(labels, values, color="#0078ff")
        ax.set_xlabel("Count")
        ax.set_title("Top 10 Words")
        chart_path = self.save_chart_image(fig, "freq_chart.png")
        plt.close(fig)

        # ====== Create PDF ======
        temp_pdf = os.path.join(tempfile.gettempdir(), "NLP_Report.pdf")
        doc = SimpleDocTemplate(temp_pdf, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        title_style = styles["Title"]
        title_style.textColor = "#0078ff"
        story.append(Paragraph("🧠 NLP Document Analysis Report", title_style))
        story.append(Spacer(1, 12))

        # File info
        story.append(Paragraph(f"<b>📂 File:</b> {self.file_name}", styles["Normal"]))
        story.append(Spacer(1, 6))

        # Sentiment
        story.append(Paragraph("<b>💬 Sentiment Analysis</b>", styles["Heading2"]))
        story.append(Paragraph(f"Result: {sentiment['label']} ({sentiment['score']:.3f})", styles["Normal"]))
        story.append(Spacer(1, 12))

        # Readability
        story.append(Paragraph("<b>📚 Readability</b>", styles["Heading2"]))
        story.append(Paragraph(f"Flesch–Kincaid Grade: {readability:.2f}", styles["Normal"]))
        story.append(Spacer(1, 12))

        # Word frequency chart
        story.append(Paragraph("<b>🔠 Word Frequency</b>", styles["Heading2"]))
        story.append(Spacer(1, 6))
        story.append(RLImage(chart_path, width=400, height=250))
        story.append(Spacer(1, 12))

        # Footer
        story.append(Paragraph("<font color='#888888'>Generated automatically by NLP Pilot</font>", styles["Normal"]))

        doc.build(story)
        messagebox.showinfo("Report Generated", f"✅ Report saved to:\n{temp_pdf}")
        os.startfile(temp_pdf)

    # ------------------------- Helper Functions -------------------------

    def save_chart_image(self, fig, name):
        temp_dir = tempfile.gettempdir()
        path = os.path.join(temp_dir, name)
        fig.savefig(path, bbox_inches="tight", facecolor="#ffffff")
        return path

