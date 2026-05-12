import os
import tempfile
import json
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
from core.data_quality import csv_stats, txt_stats
from core.run_metadata import run_metadata

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
        self.add_button(sidebar, "Data Quality", self.data_quality)
        self.add_button(sidebar, "🔠 Word Frequency", self.word_freq)
        self.add_button(sidebar, "💬 Sentiment", self.sentiment)
        self.add_button(sidebar, "🧬 Named Entities", self.named_entities)
        # self.add_button(sidebar, "📚 Readability", self.readability)
        self.add_button(sidebar, "📈 Generate Charts", self.generate_charts)
        self.add_button(sidebar, "Export Report", self.open_export_report_dialog)

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

    def run_sentiment_model(self, text: str) -> dict:
        model = self.get_sentiment_model()
        return model(self.truncate_for_pipeline(model, text))[0]

    def truncate_for_pipeline(self, model, text: str, max_length: int = 512) -> str:
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None:
            return text[:2000]

        tokenizer_limit = getattr(tokenizer, "model_max_length", max_length)
        if isinstance(tokenizer_limit, int) and tokenizer_limit > 0:
            max_length = min(max_length, tokenizer_limit)

        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
        input_ids = encoded.get("input_ids", [])
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        return tokenizer.decode(input_ids, skip_special_tokens=True)

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
        self.state.df = None
        self.state.csv_text_column = None

        text = enlp.load_text_from_file(path)
        if not text.strip():
            messagebox.showwarning("Warning", "Could not extract text from file.")
            return

        self.state.text = text
        self.state.file_name = os.path.basename(path)
        self.on_state_changed()

        self.file_label.configure(text=f"📂  {self.state.file_name}")
        self.output.delete("1.0", "end")
        self.output.insert(
            "end",
            f"File loaded successfully.\n Total Number of characters: {len(text)}",
            "left",
        )

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
        self.state.text = ""
        self.state.file_name = os.path.basename(path)
        self.on_state_changed()

        self.file_label.configure(text=f"📂  {self.state.file_name} (CSV, col: {col})")
        self.output.delete("1.0", "end")
        self.output.insert(
            "end",
            f"CSV loaded successfully.\nUsing column '{col}' as text column.\n Dataframe shape: {df.shape}",
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

    def data_quality(self):
        if not self.require_text():
            return

        self.show_textbox()
        self.output.delete("1.0", "end")
        self.output.insert("end", "Data Quality\n", "left")
        self.output.insert("end", "-" * 40 + "\n", "left")

        if enlp.is_csv_mode(self.state):
            stats = csv_stats(self.state.df, self.state.csv_text_column)
            self.output.insert("end", f"File: {self.state.file_name}\n", "left")
            self.output.insert("end", f"Rows: {stats['row_count']:,}\n", "left")
            self.output.insert("end", f"Columns: {stats['col_count']}\n", "left")
            self.output.insert("end", f"Duplicate rows: {stats['duplicate_rows']}\n", "left")

            if stats["missing"]:
                self.output.insert("end", "\nMissing values:\n", "left")
                for col, info in stats["missing"].items():
                    self.output.insert(
                        "end",
                        f"  {col}: {info['count']} ({info['pct']:.1f}%)\n",
                        "left",
                    )
            else:
                self.output.insert("end", "Missing values: none\n", "left")

            if stats["text_col_stats"]:
                tcs = stats["text_col_stats"]
                self.output.insert(
                    "end",
                    f"\nText column: {self.state.csv_text_column}\n"
                    f"Avg words per row: {tcs['avg_words']:.1f}\n"
                    f"Min words: {tcs['min_words']}\n"
                    f"Max words: {tcs['max_words']}\n",
                    "left",
                )
            return

        stats = txt_stats(self.state.text)
        self.output.insert("end", f"File: {self.state.file_name}\n", "left")
        self.output.insert("end", f"Characters: {stats['char_count']:,}\n", "left")
        self.output.insert("end", f"Words: {stats['word_count']:,}\n", "left")
        self.output.insert("end", f"Sentences: {stats['sentence_count']:,}\n", "left")
        self.output.insert(
            "end",
            f"Avg sentence length: {stats['avg_sentence_len']:.1f} words\n",
            "left",
        )
        self.output.insert("end", f"Unique words: {stats['unique_word_count']:,}\n", "left")

    def sentiment(self):
        if not self.require_text():
            return

        text = self.get_active_text()
        try:
            result = self.run_sentiment_model(text)
        except Exception as exc:
            messagebox.showerror("Sentiment error", f"Could not analyze sentiment:\n{exc}")
            return
        label = result["label"]
        score = result["score"]

        explanation = {
            "POSITIVE": "Optimistic or confident tone.",
            "NEGATIVE": "Critical or dissatisfied tone.",
            "NEUTRAL": "Balanced, factual tone.",
        }.get(label, "Unclear tone detected.")

        self.show_textbox()
        self.output.delete("1.0", "end")
        self.output.insert("end", "\nSentiment Analysis\n", "left")
        self.output.insert("end", "-" * 40 + "\n", "left")
        self.output.insert("end", f"Label: {label}\n", "left")
        self.output.insert("end", f"Confidence: {score:.3f}\n", "left")
        self.output.insert("end", f"Interpretation: {explanation}\n", "left")


    def named_entities(self):
        if not self.require_text():
            return

        model = self.get_ner_model()
        text = self.get_active_text()
        try:
            ents_raw = model(self.truncate_for_pipeline(model, text))
        except Exception as exc:
            messagebox.showerror("NER error", f"Could not extract named entities:\n{exc}")
            return
        cleaned = enlp.merge_ner_entities(ents_raw)
        label_counts = Counter(lbl for _, lbl, _ in cleaned)

        if not label_counts:
            self.show_textbox()
            self.output.delete("1.0", "end")
            self.output.insert("end", "\n🧬 Named Entities\n" + "-" * 40 + "\nNo entities found.\n", "left")
            return

        lines = ["\n🧬 Named Entities", "-" * 40, "Summary by label:"]
        for lbl, cnt in label_counts.items():
            lines.append(f"{lbl:<10}: {cnt}")
        lines.append("-" * 40)
        for word, lbl, score in cleaned[:50]:
            lines.append(f"{word:<25}{lbl:<10}{score:.2f}")
        text_content = "\n".join(lines)

        self.show_textbox()
        self.output.delete("1.0", "end")
        self.output.insert("end", text_content, "left")

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
            top_n=20,
            remove_stopwords=True,
        )
        fig = enlp.build_word_freq_figure(freq, title="Top 20 Words", dark=True)
        self.show_chart(fig)

    def open_export_report_dialog(self):
        if not self.require_text():
            return
        ReportExportDialog(self)

    def export_report(self, path: str | None = None):
        if not self.require_text():
            return
        if path is None:
            path = filedialog.asksaveasfilename(
                defaultextension=".pdf",
                filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
                initialfile="nlp_pilot_report.pdf",
                title="Save PDF report",
            )
        if not path:
            return

        text = self.get_active_text()

        sentiment = self.run_sentiment_model(text)
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

        doc = SimpleDocTemplate(path, pagesize=A4)
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
        messagebox.showinfo("Report Generated", f"Report saved to:\n{path}")

    def export_report_json(self, path: str | None = None):
        if not self.require_text():
            return

        text = self.get_active_text()
        ease, grade = enlp.readability_scores(text)
        freq = enlp.word_frequency(
            text,
            stopword_set=self.state.stopwords,
            top_n=50,
            remove_stopwords=True,
        )
        payload = {
            "metadata": run_metadata(
                {
                    "file_name": self.state.file_name,
                    "mode": "csv" if enlp.is_csv_mode(self.state) else "document",
                    "csv_text_column": self.state.csv_text_column,
                }
            ),
            "file_name": self.state.file_name,
            "mode": "csv" if enlp.is_csv_mode(self.state) else "document",
            "csv_text_column": self.state.csv_text_column,
            "character_count": len(text),
            "readability": {
                "flesch_reading_ease": ease,
                "flesch_kincaid_grade": grade,
            },
            "word_frequency": [
                {"term": term, "count": int(count)}
                for term, count in freq
            ],
        }

        if path is None:
            path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                initialfile="nlp_pilot_report.json",
                title="Save JSON report",
            )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            messagebox.showinfo("Report Generated", f"Report saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save JSON:\n{e}")

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
        for w in self.output_container.winfo_children():
            w.destroy()

        canvas = FigureCanvasTkAgg(fig, master=self.output_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig)

    def show_text_and_chart(self, text_content: str, fig):
        """Show a text widget above a matplotlib chart in output_container."""
        for w in self.output_container.winfo_children():
            w.destroy()

        self.output = ctk.CTkTextbox(
            self.output_container,
            fg_color="#1a1a1a",
            text_color="white",
            font=("Consolas", 13),
            wrap="word",
            height=200,
        )
        self.output.pack(fill="x", expand=False)
        self.output.insert("end", text_content, "left")

        canvas = FigureCanvasTkAgg(fig, master=self.output_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig)



class ReportExportDialog(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.title("Export Report")
        self.geometry("430x230")
        self.resizable(False, False)
        self.grab_set()

        ctk.CTkLabel(
            self,
            text="Export document report",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#6ea8fe",
        ).pack(anchor="w", padx=16, pady=(16, 8))

        ctk.CTkLabel(self, text="Format:").pack(anchor="w", padx=16, pady=(4, 2))
        self.format_combo = ctk.CTkComboBox(
            self,
            values=["PDF", "JSON"],
            state="readonly",
            command=lambda _: self._sync_default_path(),
        )
        self.format_combo.set("PDF")
        self.format_combo.pack(fill="x", padx=16, pady=(0, 8))

        ctk.CTkLabel(self, text="Save to:").pack(anchor="w", padx=16, pady=(4, 2))
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=16, pady=(0, 14))
        row.columnconfigure(0, weight=1)
        self.path_entry = ctk.CTkEntry(row)
        self.path_entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        ctk.CTkButton(
            row,
            text="Browse",
            width=86,
            fg_color="#444444",
            hover_color="#333333",
            command=self._browse,
        ).grid(row=0, column=1)

        buttons = ctk.CTkFrame(self, fg_color="transparent")
        buttons.pack(fill="x", padx=16, pady=(2, 16))
        ctk.CTkButton(
            buttons,
            text="Cancel",
            fg_color="#555555",
            hover_color="#444444",
            command=self.destroy,
        ).pack(side="right", padx=(8, 0))
        ctk.CTkButton(
            buttons,
            text="Export",
            fg_color="#0078ff",
            hover_color="#005dc1",
            command=self._export,
        ).pack(side="right")

        self._sync_default_path()

    def _format(self) -> str:
        return self.format_combo.get().lower()

    def _sync_default_path(self):
        current = self.path_entry.get().strip() if hasattr(self, "path_entry") else ""
        ext = ".pdf" if self._format() == "pdf" else ".json"
        if current and os.path.basename(current).split(".")[0] != "nlp_pilot_report":
            return
        self.path_entry.delete(0, "end")
        self.path_entry.insert(0, os.path.join(os.getcwd(), f"nlp_pilot_report{ext}"))

    def _browse(self):
        fmt = self._format()
        ext = ".pdf" if fmt == "pdf" else ".json"
        filetypes = [("PDF files", "*.pdf")] if fmt == "pdf" else [("JSON files", "*.json")]
        path = filedialog.asksaveasfilename(
            parent=self,
            defaultextension=ext,
            filetypes=filetypes + [("All files", "*.*")],
            initialfile=f"nlp_pilot_report{ext}",
            title="Choose export location",
        )
        if path:
            self.path_entry.delete(0, "end")
            self.path_entry.insert(0, path)

    def _export(self):
        path = self.path_entry.get().strip()
        if not path:
            messagebox.showwarning("Missing location", "Choose where to save the report.", parent=self)
            return
        fmt = self._format()
        try:
            if fmt == "pdf":
                self.parent.export_report(path)
            else:
                self.parent.export_report_json(path)
        finally:
            self.destroy()


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


