import customtkinter as ctk

from core.english_state import EnglishState
from core import english_nlp as enlp
from core.data_quality import txt_stats, csv_stats


class DataQualityTab(ctk.CTkFrame):
    def __init__(self, parent, state: EnglishState):
        super().__init__(parent, fg_color="#0f0f0f")
        self.state = state
        self._build_ui()
        self.pack(fill="both", expand=True)
        self.refresh()

    def _build_ui(self):
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        sidebar = ctk.CTkFrame(self, fg_color="#121212", corner_radius=10)
        sidebar.grid(row=0, column=0, sticky="ns", padx=(10, 5), pady=10)

        ctk.CTkLabel(
            sidebar,
            text="Data Quality",
            text_color="#6ea8fe",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack(pady=(15, 20))

        ctk.CTkButton(
            sidebar,
            text="🔄 Refresh",
            command=self.refresh,
            fg_color="#0078ff",
            hover_color="#005dc1",
            corner_radius=6,
            height=38,
            font=ctk.CTkFont(size=14),
        ).pack(fill="x", padx=10, pady=6)

        self._scroll = ctk.CTkScrollableFrame(
            self, fg_color="#1a1a1a", corner_radius=10
        )
        self._scroll.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)

    def refresh(self):
        for w in self._scroll.winfo_children():
            w.destroy()

        in_csv = enlp.is_csv_mode(self.state)
        has_text = bool(self.state.text and self.state.text.strip())

        if not in_csv and not has_text:
            ctk.CTkLabel(
                self._scroll,
                text="No data loaded — upload a file in Document Tools",
                text_color="#888888",
                font=ctk.CTkFont(size=13),
            ).pack(pady=20, padx=20)
            return

        if in_csv:
            self._render_csv_stats()
        else:
            self._render_txt_stats()

    def _render_txt_stats(self):
        stats = txt_stats(self.state.text)
        self._section_header(f"📄  {self.state.file_name or 'Text file'}")
        rows = [
            ("Characters", f"{stats['char_count']:,}"),
            ("Words", f"{stats['word_count']:,}"),
            ("Sentences", f"{stats['sentence_count']:,}"),
            ("Avg sentence length", f"{stats['avg_sentence_len']:.1f} words"),
            ("Unique words", f"{stats['unique_word_count']:,}"),
        ]
        for label, value in rows:
            self._stat_row(label, value)

    def _render_csv_stats(self):
        stats = csv_stats(self.state.df, self.state.csv_text_column)
        self._section_header(f"📊  {self.state.file_name or 'CSV file'}")
        self._stat_row("Rows", f"{stats['row_count']:,}")
        self._stat_row("Columns", str(stats["col_count"]))
        self._stat_row("Duplicate rows", str(stats["duplicate_rows"]))

        if stats["missing"]:
            self._section_header("Missing Values")
            header = ctk.CTkFrame(self._scroll, fg_color="transparent")
            header.pack(fill="x", padx=20, pady=(2, 0))
            ctk.CTkLabel(
                header, text="Column", font=ctk.CTkFont(weight="bold"),
                text_color="#aaaaaa", width=160, anchor="w",
            ).pack(side="left")
            ctk.CTkLabel(
                header, text="Count", font=ctk.CTkFont(weight="bold"),
                text_color="#aaaaaa", width=60, anchor="e",
            ).pack(side="left")
            ctk.CTkLabel(
                header, text="%", font=ctk.CTkFont(weight="bold"),
                text_color="#aaaaaa", width=60, anchor="e",
            ).pack(side="left")
            for col, info in stats["missing"].items():
                row = ctk.CTkFrame(self._scroll, fg_color="transparent")
                row.pack(fill="x", padx=20, pady=1)
                ctk.CTkLabel(row, text=col, width=160, anchor="w",
                             text_color="white").pack(side="left")
                ctk.CTkLabel(row, text=str(info["count"]), width=60, anchor="e",
                             text_color="white").pack(side="left")
                ctk.CTkLabel(
                    row, text=f"{info['pct']:.1f}%", width=60, anchor="e",
                    text_color="#ffaa44",
                ).pack(side="left")
        else:
            self._stat_row("Missing values", "None ✓")

        if stats["text_col_stats"]:
            tcs = stats["text_col_stats"]
            self._section_header(f"Text column  ({self.state.csv_text_column})")
            self._stat_row("Avg words per row", f"{tcs['avg_words']:.1f}")
            self._stat_row("Min words", str(tcs["min_words"]))
            self._stat_row("Max words", str(tcs["max_words"]))

    def _section_header(self, text: str):
        ctk.CTkLabel(
            self._scroll,
            text=text,
            text_color="#6ea8fe",
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w",
        ).pack(fill="x", padx=20, pady=(16, 4))

    def _stat_row(self, label: str, value: str):
        row = ctk.CTkFrame(self._scroll, fg_color="transparent")
        row.pack(fill="x", padx=20, pady=2)
        ctk.CTkLabel(
            row, text=label, text_color="#aaaaaa", anchor="w", width=200,
        ).pack(side="left")
        ctk.CTkLabel(row, text=value, text_color="white", anchor="w").pack(side="left")
