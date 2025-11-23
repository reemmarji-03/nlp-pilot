import os
import tempfile
from collections import Counter

import customtkinter as ctk
from tkinter import filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from transformers import pipeline

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet

from core import english_nlp as enlp
from core.english_state import EnglishState

class DocumentToolsTab(ctk.CTkFrame):
    def __init__(self, parent, state: EnglishState, on_state_changed):
        super().__init__(parent, fg_color="#0f0f0f")
        self.state = state

        self.file_label = None
        self.output = None

        self.build_ui()
        self.pack(fill="both", expand=True)

        self.on_state_changed = on_state_changed

    # UI
    def build_ui(self):
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # Sidebar
        sidebar = ctk.CTkFrame(self, fg_color="#121212", corner_radius=10)
        sidebar.grid(row=0, column=0, sticky="ns", padx=(10, 5), pady=10)

        ctk.CTkLabel(
            sidebar,
            text="English NLP Tools",
            text_color="#6ea8fe",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=(15, 20))

        self.add_button(sidebar, "📂 Upload File", self.upload_file)
        self.add_button(sidebar, "🔠 Word Frequency", self.word_freq)
        self.add_button(sidebar, "💬 Sentiment", self.sentiment)
        self.add_button(sidebar, "🧬 Named Entities", self.named_entities)
        # self.add_button(sidebar, "📚 Readability", self.readability)
        self.add_button(sidebar, "📈 Generate Charts", self.generate_charts)
        self.add_button(sidebar, "🧾 Export PDF Report", self.export_report)

        # Output area
        output_frame = ctk.CTkFrame(self, fg_color="#1a1a1a", corner_radius=10)
        output_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)

        self.file_label = ctk.CTkLabel(
            output_frame,
            text="No file loaded",
            anchor="w",
            text_color="#6ea8fe",
            fg_color="#1a1a1a",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.file_label.pack(anchor="w", padx=10, pady=(6, 2))

        self.output_container = ctk.CTkFrame(output_frame, fg_color="#1a1a1a")
        self.output_container.pack(fill="both", expand=True, padx=10, pady=(0,10))

        self.output = ctk.CTkTextbox(
            self.output_container,
            fg_color="#1a1a1a",
            text_color="white",
            font=("Consolas", 13),
            wrap="word"
        )
        self.output.pack(fill="both", expand=True)

        self.output._textbox.tag_configure("left", justify="left")

    def add_button(self, parent, text, command):
        ctk.CTkButton(
            parent,
            text=text,
            command=command,
            fg_color="#0078ff",
            hover_color="#005dc1",
            corner_radius=6,
            height=38,
            font=ctk.CTkFont(size=14)
        ).pack(fill="x", padx=10, pady=6)

    # ---------------- Helpers ----------------
    def require_text(self):
        """
        Ensure either a raw text file or a CSV has been loaded.
        """
        has_text = bool(self.state.text and self.state.text.strip())
        in_csv_mode = enlp.is_csv_mode(self.state)

        if not (has_text or in_csv_mode):
            messagebox.showwarning("Warning", "Please upload a file first.")
            return False
        return True

    def get_sentiment_model(self):
        if self.state.sentiment_model is None:
            self.output.delete("1.0", "end")
            self.output.insert("end", "⏳ Loading sentiment model...\n")
            self.update_idletasks()
            self.state.sentiment_model = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
        return self.state.sentiment_model

    def get_ner_model(self):
        if self.state.ner_model is None:
            self.output.delete("1.0", "end")
            self.output.insert("end", "⏳ Loading NER model...\n")
            self.update_idletasks()
            self.state.ner_model = pipeline("ner", model="dslim/bert-base-NER")
        return self.state.ner_model

    def get_active_text(self) -> str:
        """
        Return a single string representing the active text after preprocessing.

        - TXT/PDF: preprocessed whole document
        - CSV: all preprocessed rows in the chosen text column, joined together
        """
        docs = enlp.get_preprocessed_documents_from_state(self.state)
        return "\n".join(docs)

    # ---------------- Logic ----------------
    def upload_file(self):
        path = filedialog.askopenfilename(
            filetypes=[
                ("Documents", "*.pdf *.docx *.txt *.csv"),
                ("All files", "*.*"),
            ]
        )
        if not path:
            return

        if path.lower().endswith(".csv"):
            self._handle_csv_file(path)

        else:
            self._handle_text_like_file(path)

    def _handle_text_like_file(self, path: str):
        # Leave CSV mode
        self.state.df = None
        self.state.csv_text_column = None

        text = enlp.load_text_from_file(path)
        if not text.strip():
            messagebox.showwarning("Warning", "Could not extract text from file.")
            return

        self.state.text = text
        self.state.file_name = os.path.basename(path)

        self.file_label.configure(text=f"📂  {self.state.file_name}")
        self.output.delete("1.0", "end")
        self.output.insert("end", "File loaded successfully.\n", "left")

    def _handle_csv_file(self, path: str):
        try:
            df = pd.read_csv(path)
        except Exception as e:
            messagebox.showerror("Error", f"Could not read CSV file:\n{e}")
            return

        cols = df.columns.tolist()
        if not cols:
            messagebox.showwarning("Warning", "CSV file has no columns.")
            return

        dialog = ColumnChoiceDialog(self, columns=cols)
        self.wait_window(dialog)

        col = dialog.chosen
        if not col:
            # user cancelled
            return

        # enter CSV mode
        self.state.df = df
        self.state.csv_text_column = col

        # update state for prediction

        self.on_state_changed()
        # Clear text to make it obvious we're not in plain-text mode
        self.state.text = ""
        self.state.file_name = os.path.basename(path)

        self.file_label.configure(text=f"📂  {self.state.file_name} (CSV, col: {col})")
        self.output.delete("1.0", "end")
        self.output.insert(
            "end",
            f"CSV loaded successfully.\nUsing column '{col}' as text column.\n",
            "left"
        )

    def word_freq(self):
        if not self.require_text():
            return

        text = self.get_active_text()
        freq = enlp.word_frequency(
            text,
            stopword_set=self.state.stopwords,
            top_n=20,
            remove_stopwords=False
        )

        self.show_textbox()
        self.output.delete("1.0", "end")
        self.output.insert("end", "\n🔠 Top 20 Words", "left")
        self.output.insert("end", "-" * 40 + "\n", "left")
        for w, c in freq:
            self.output.insert("end", f"{w:<15}{c:>5}\n", "left")

    def sentiment(self):
        if not self.require_text():
            return

        model = self.get_sentiment_model()
        text = self.get_active_text()
        chunk = text[:2000]
        result = model(chunk)[0]
        label = result["label"]
        score = result["score"]

        explanation = {
            "POSITIVE": "Optimistic or confident tone.",
            "NEGATIVE": "Critical or dissatisfied tone.",
            "NEUTRAL": "Balanced, factual tone."
        }.get(label, "Unclear tone detected.")

        self.show_textbox()
        self.output.delete("1.0", "end")
        self.output.insert("end", f"\n💬 Sentiment Analysis\n", "left")
        self.output.insert("end", "-" * 40 + "\n", "left")
        self.output.insert("end", f"Label: {label}\nScore: {score:.3f}\n", "left")
        self.output.insert("end", f"Interpretation: {explanation}\n", "left")

    def named_entities(self):
        if not self.require_text():
            return

        model = self.get_ner_model()
        text = self.get_active_text()
        ents_raw = model(text[:1000])
        cleaned = enlp.merge_ner_entities(ents_raw)

        label_counts = Counter(lbl for _, lbl, _ in cleaned)

        self.show_textbox()
        self.output.delete("1.0", "end")
        self.output.insert("end", "\n🧬 Named Entities\n", "left")
        self.output.insert("end", "-" * 40 + "\n", "left")

        self.output.insert("end", "Summary by label:\n", "left")
        for lbl, cnt in label_counts.items():
            self.output.insert("end", f"{lbl:<10}: {cnt}\n", "left")
        self.output.insert("end", "-" * 40 + "\n", "left")

        for word, label, score in cleaned[:50]:
            self.output.insert("end", f"{word:<25}{label:<10}{score:.2f}\n", "left")

    def readability(self):
        if not self.require_text():
            return

        text = self.get_active_text()
        ease, grade = enlp.readability_scores(text)
        level = enlp.readability_level(grade)

        self.show_textbox()
        self.output.delete("1.0", "end")
        self.output.insert("end", "\n📚 Readability Analysis\n", "left")
        self.output.insert("end", "-" * 40 + "\n", "left")
        self.output.insert("end", f"Reading Ease: {ease:.2f}\n", "left")
        self.output.insert("end", f"Grade Level: {grade:.2f}\n", "left")
        self.output.insert("end", f"Interpretation: {level}\n", "left")

    def generate_charts(self):
        if not self.require_text():
            return

        text = self.get_active_text()
        freq = enlp.word_frequency(
            text,
            stopword_set=self.state.stopwords,
            top_n=10,
            remove_stopwords=True
        )
        fig = enlp.build_word_freq_figure(freq, title="Top 10 Words")

        # Show inside output area
        self.show_chart(fig)


    def export_report(self):
        if not self.require_text():
            return

        model = self.get_sentiment_model()
        text = self.get_active_text()

        sentiment = model(text[:2000])[0]
        _, grade = enlp.readability_scores(text)

        freq = enlp.word_frequency(
            text,
            stopword_set=self.state.stopwords,
            top_n=10,
            remove_stopwords=True
        )

        fig = enlp.build_word_freq_figure(freq, title="Top 10 Words")
        chart_path = self.save_chart_image(fig, "freq_chart.png")
        plt.close(fig)

        temp_pdf = os.path.join(tempfile.gettempdir(), "NLP_Report.pdf")
        doc = SimpleDocTemplate(temp_pdf, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        title_style = styles["Title"]
        story.append(Paragraph("NLP Document Analysis Report", title_style))
        story.append(Spacer(1, 12))

        story.append(Paragraph(f"<b>File:</b> {self.state.file_name}", styles["Normal"]))
        story.append(Spacer(1, 6))

        story.append(Paragraph("<b>Sentiment Analysis</b>", styles["Heading2"]))
        story.append(Paragraph(
            f"Result: {sentiment['label']} ({sentiment['score']:.3f})",
            styles["Normal"]
        ))
        story.append(Spacer(1, 12))

        story.append(Paragraph("<b>Readability</b>", styles["Heading2"]))
        story.append(Paragraph(
            f"Flesch–Kincaid Grade: {grade:.2f}",
            styles["Normal"]
        ))
        story.append(Spacer(1, 12))

        story.append(Paragraph("<b>Word Frequency</b>", styles["Heading2"]))
        story.append(Spacer(1, 6))
        if freq:
            story.append(RLImage(chart_path, width=400, height=250))
        else:
            story.append(Paragraph("Not enough words for a frequency chart.", styles["Normal"]))
        story.append(Spacer(1, 12))

        story.append(Paragraph(
            "<font color='#888888'>Generated automatically by NLP Pilot</font>",
            styles["Normal"]
        ))

        doc.build(story)
        messagebox.showinfo("Report Generated", f"✅ Report saved to:\n{temp_pdf}")
        try:
            os.startfile(temp_pdf)
        except Exception:
            pass

    def save_chart_image(self, fig, name):
        temp_dir = tempfile.gettempdir()
        path = os.path.join(temp_dir, name)
        fig.savefig(path, bbox_inches="tight", facecolor="#ffffff")
        return path
    
    def show_textbox(self):
        """Restore the text output and remove any chart canvas."""
        for w in self.output_container.winfo_children():
            w.destroy()

        # Recreate textbox
        self.output = ctk.CTkTextbox(
            self.output_container,
            fg_color="#1a1a1a",
            text_color="white",
            font=("Consolas", 13),
            wrap="word"
        )
        self.output.pack(fill="both", expand=True)

    def show_chart(self, fig):
        """Replace the textbox with a Matplotlib chart."""
        # Clear existing widgets (textbox or previous chart)
        for w in self.output_container.winfo_children():
            w.destroy()

        canvas = FigureCanvasTkAgg(fig, master=self.output_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)



class ColumnChoiceDialog(ctk.CTkToplevel):
    def __init__(self, master, columns: list[str]):
        super().__init__(master)
        self.title("Select text column")
        self.geometry("350x150")
        self.resizable(False, False)
        self.chosen = None

        ctk.CTkLabel(
            self,
            text="Select the column that contains the text:",
            wraplength=320
        ).pack(padx=10, pady=(10, 5))

        self.combo = ctk.CTkComboBox(
            self,
            values=columns,
            state="readonly",
            width=250
        )
        self.combo.pack(pady=(0, 10))
        if columns:
            self.combo.set(columns[0])

        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(pady=(0, 10))

        ctk.CTkButton(
            btn_frame, text="OK", width=80,
            command=self._on_ok
        ).pack(side="left", padx=5)
        ctk.CTkButton(
            btn_frame, text="Cancel", width=80,
            command=self._on_cancel
        ).pack(side="left", padx=5)

        self.grab_set()
        self.focus_force()

    def _on_ok(self):
        self.chosen = self.combo.get()
        self.destroy()

    def _on_cancel(self):
        self.chosen = None
        self.destroy()


