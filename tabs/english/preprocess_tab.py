import os

import customtkinter as ctk
from tkinter import messagebox, Listbox, END, filedialog

from core.english_state import EnglishState
from core import english_nlp as enlp
from core import theme_manager as theme


class PreprocessTab(ctk.CTkFrame):
    """
    Preprocessing pipeline builder:
    - left: available steps
    - middle: current pipeline (ordered, draggable)
    - right: parameters + preview + (optional) CSV export
    """

    def __init__(self, parent, state: EnglishState):
        super().__init__(parent, fg_color=("#f0f0f0", "#0f0f0f"))
        self.state = state
        self.regex_pattern_entry = None
        self.regex_replace_entry = None

        # widgets
        self.available_list = None
        self.pipeline_list = None
        self.preview_box = None
        self.short_min_len_entry = None

        # for drag-reorder
        self._drag_start_index = None

        # static metadata describing the steps
        self.available_steps_meta = [
            {"id": "lowercase", "label": "Lowercase / normalize"},
            {"id": "urls", "label": "Remove URLs / #hashtags / @mentions"},
            {"id": "contractions", "label": "Expand contractions (don't → do not)"},
            {"id": "numbers", "label": "Normalize numbers (123 → <NUM>)"},
            {"id": "stopwords", "label": "Remove stopwords"},
            {"id": "short_tokens", "label": "Remove short tokens"},
            {"id": "stem", "label": "Stemming (PorterStemmer)"},
            {"id": "lemma", "label": "Lemmatization (WordNet)"},
            # later: {"id": "emoji_text", "label": "Convert emoji to text"}, ...
        ]

        self.build_ui()
        self.load_from_state()
        self.pack(fill="both", expand=True)


    # UI
    def build_ui(self):
        self.columnconfigure((0, 1, 2), weight=1)
        self.rowconfigure(0, weight=1)

        # ========== LEFT: Available steps ==========
        left = ctk.CTkFrame(self, fg_color=("#e8e8e8", "#121212"), corner_radius=10)
        left.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)

        ctk.CTkLabel(
            left,
            text="Available Steps",
            text_color=("#0062cc", "#6ea8fe"),
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(10, 5))

        # container for padding & nicer look
        available_container = ctk.CTkFrame(left, fg_color=("#e8e8e8", "#121212"), corner_radius=8)
        available_container.pack(fill="both", expand=True, padx=10, pady=10)

        s = theme.scalars()
        self.available_list = Listbox(
            available_container,
            height=14,
            exportselection=False,
            font=("Segoe UI", 14),
            justify="center",
            activestyle="none",
            bg=s["listbox_bg"],
            fg=s["listbox_fg"],
            highlightthickness=1,
            highlightbackground=s["listbox_hl"],
            selectbackground="#0078ff",
            selectforeground="white",
            borderwidth=0,
            relief="flat"
        )
        self.available_list.pack(fill="both", expand=True, padx=4, pady=4)

        for step in self.available_steps_meta:
            self.available_list.insert(END, step["label"])

        # double-click to add
        self.available_list.bind("<Double-Button-1>", self.on_available_double_click)

        ctk.CTkButton(
            left,
            text="Add →",
            fg_color="#0078ff",
            hover_color="#005dc1",
            text_color="white",
            command=self.add_selected_step
        ).pack(padx=10, pady=(0, 10), anchor="e")

        # ========== MIDDLE: Current pipeline ==========
        mid = ctk.CTkFrame(self, fg_color=("#e8e8e8", "#121212"), corner_radius=10)
        mid.grid(row=0, column=1, sticky="nsew", padx=5, pady=10)

        ctk.CTkLabel(
            mid,
            text="Current Pipeline (top → bottom)",
            text_color=("#0062cc", "#6ea8fe"),
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(10, 5))

        pipeline_container = ctk.CTkFrame(mid, fg_color=("#e8e8e8", "#121212"), corner_radius=8)
        pipeline_container.pack(fill="both", expand=True, padx=10, pady=10)

        self.pipeline_list = Listbox(
            pipeline_container,
            height=16,
            exportselection=False,
            font=("Segoe UI", 14, "bold"),
            justify="center",
            activestyle="none",
            bg=s["listbox_bg"],
            fg=s["listbox_fg"],
            highlightthickness=1,
            highlightbackground=s["listbox_hl"],
            selectbackground="#0078ff",
            selectforeground="white",
            borderwidth=0,
            relief="flat"
        )
        self.pipeline_list.pack(fill="both", expand=True, padx=4, pady=4)

        # double-click to remove
        self.pipeline_list.bind("<Double-Button-1>", self.on_pipeline_double_click)

        # drag-to-reorder inside pipeline
        self.pipeline_list.bind("<Button-1>", self.on_pipeline_press)
        self.pipeline_list.bind("<ButtonRelease-1>", self.on_pipeline_release)

        btn_frame = ctk.CTkFrame(mid, fg_color=("#e8e8e8", "#121212"))
        btn_frame.pack(fill="x", padx=10, pady=(0, 10))

        ctk.CTkButton(
            btn_frame,
            text="← Remove",
            width=80,
            fg_color=("#a0a0a0", "#444444"),
            hover_color=("#888888", "#333333"),
            command=self.remove_selected_step
        ).pack(side="left", padx=2)

        ctk.CTkButton(
            btn_frame,
            text="Move Up",
            width=80,
            fg_color=("#a0a0a0", "#444444"),
            hover_color=("#888888", "#333333"),
            command=self.move_step_up
        ).pack(side="left", padx=2)

        ctk.CTkButton(
            btn_frame,
            text="Move Down",
            width=80,
            fg_color=("#a0a0a0", "#444444"),
            hover_color=("#888888", "#333333"),
            command=self.move_step_down
        ).pack(side="left", padx=2)

        # ========== RIGHT: Parameters + Preview + Export ==========
        right = ctk.CTkFrame(self, fg_color=("#ffffff", "#1a1a1a"), corner_radius=10)
        right.grid(row=0, column=2, sticky="nsew", padx=(5, 10), pady=10)

        ctk.CTkLabel(
            right,
            text="Parameters & Preview",
            text_color=("#0062cc", "#6ea8fe"),
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(10, 5), anchor="w", padx=10)

        # short token parameter (global for that step for now)
        param_frame = ctk.CTkFrame(right, fg_color=("#ffffff", "#1a1a1a"))
        param_frame.pack(fill="x", padx=10, pady=(0, 5))

        ctk.CTkLabel(
            param_frame,
            text="Min token length (for 'Remove short tokens'):",
            text_color=("#444444", "#cccccc"),
            font=ctk.CTkFont(size=13)
        ).pack(anchor="w")

        self.short_min_len_entry = ctk.CTkEntry(
            param_frame,
            width=80,
            placeholder_text="3"
        )
        self.short_min_len_entry.pack(anchor="w", pady=(2, 5))
        # --- NEW: Regex replace (GUI-only, not part of LangGraph steps) ---
        regex_frame = ctk.CTkFrame(right, fg_color=("#ffffff", "#1a1a1a"))
        regex_frame.pack(fill="x", padx=10, pady=(0, 5))

        ctk.CTkLabel(
            regex_frame,
            text="regex Pattern",
            text_color=("#444444", "#cccccc"),
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(anchor="w", pady=(0, 2))

        ctk.CTkLabel(
            regex_frame,
            text="Pattern (Python regex):",
            text_color=("#666666", "#aaaaaa"),
            font=ctk.CTkFont(size=12),
        ).pack(anchor="w")

        self.regex_pattern_entry = ctk.CTkEntry(
            regex_frame,
            width=260,
            placeholder_text=r"\d+"  # example
        )
        self.regex_pattern_entry.pack(anchor="w", pady=(0, 4))

        ctk.CTkLabel(
            regex_frame,
            text="Replacement string",
            text_color=("#666666", "#aaaaaa"),
            font=ctk.CTkFont(size=12),
        ).pack(anchor="w")

        self.regex_replace_entry = ctk.CTkEntry(
            regex_frame,
            width=260,
            placeholder_text=""  # default = remove matches
        )
        self.regex_replace_entry.pack(anchor="w", pady=(0, 4))

        # preview box
        self.preview_box = ctk.CTkTextbox(
            right,
            fg_color=("#ffffff", "#1a1a1a"),
            text_color=("#111111", "white"),
            font=("Consolas", 12),
            wrap="word"
        )
        self.preview_box.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # ---- Text tags for styled CSV preview ----
        tw = self.preview_box._textbox  # underlying tk.Text

        accent = "#0062cc" if not theme.is_dark() else "#6ea8fe"
        even_bg = "#f0f0f0" if not theme.is_dark() else "#181818"
        odd_bg = "#e8e8e8" if not theme.is_dark() else "#141414"
        tw.tag_configure("row_header", foreground=accent, font=("Consolas", 12, "bold"))
        tw.tag_configure("row_bg_even", background=even_bg)
        tw.tag_configure("row_bg_odd", background=odd_bg)

        # Buttons under preview
        btn_right_frame = ctk.CTkFrame(right, fg_color=("#ffffff", "#1a1a1a"))
        btn_right_frame.pack(fill="x", padx=10, pady=(0, 10))

        ctk.CTkButton(
            btn_right_frame,
            text="Preview on current document",
            fg_color="#0078ff",
            hover_color="#005dc1",
            command=self.preview_pipeline
        ).pack(side="left", padx=(0, 5))

        ctk.CTkButton(
            btn_right_frame,
            text="Apply to CSV & Download",
            fg_color=("#a0a0a0", "#444444"),
            hover_color=("#888888", "#333333"),
            command=self.export_preprocessed_csv
        ).pack(side="left", padx=(5, 0))


    # Helpers: mapping between IDs and labels

    def get_label_for_id(self, step_id: str) -> str:
        for s in self.available_steps_meta:
            if s["id"] == step_id:
                return s["label"]
        return step_id

    def get_id_for_label(self, label: str) -> str:
        for s in self.available_steps_meta:
            if s["label"] == label:
                return s["id"]
        # fallback: label itself if not found
        return label


    # State → GUI
    def load_from_state(self):
        """Populate pipeline_list and parameter widgets from state.pipeline_config."""
        self.pipeline_list.delete(0, END)

        steps = self.state.pipeline_config.get("steps", [])
        for step in steps:
            if not step.get("enabled", True):
                continue
            label = self.get_label_for_id(step["id"])
            self.pipeline_list.insert(END, label)

        # load short token min_len if present
        short_step = next(
            (s for s in steps if s["id"] == "short_tokens"),
            None
        )
        if short_step and "min_len" in short_step:
            self.short_min_len_entry.delete(0, END)
            self.short_min_len_entry.insert(0, str(short_step["min_len"]))
        else:
            # default
            self.short_min_len_entry.delete(0, END)
            self.short_min_len_entry.insert(0, "3")

        # NEW: load regex pattern/replacement if present
        regex_cfg = self.state.pipeline_config.get("regex", {})
        pattern = regex_cfg.get("pattern", "")
        repl = regex_cfg.get("replacement", "")

        if self.regex_pattern_entry is not None:
            self.regex_pattern_entry.delete(0, END)
            self.regex_pattern_entry.insert(0, pattern)

        if self.regex_replace_entry is not None:
            self.regex_replace_entry.delete(0, END)
            self.regex_replace_entry.insert(0, repl)

    # GUI → State
    def save_to_state(self):
        """Read pipeline_list and parameter widgets, update state.pipeline_config."""
        items = self.pipeline_list.get(0, END)
        steps = []

        # parse min_len
        min_len = 3
        txt_val = self.short_min_len_entry.get().strip()
        if txt_val.isdigit():
            min_len = int(txt_val)

        for lbl in items:
            sid = self.get_id_for_label(lbl)
            step = {"id": sid, "enabled": True}

            if sid == "short_tokens":
                step["min_len"] = min_len

            steps.append(step)

        # Make sure pipeline_config exists
        if not hasattr(self.state, "pipeline_config") or self.state.pipeline_config is None:
            self.state.pipeline_config = {}

        self.state.pipeline_config["steps"] = steps

        # NEW: save regex config
        if self.regex_pattern_entry is not None:
            pattern = self.regex_pattern_entry.get().strip()
        else:
            pattern = ""

        if self.regex_replace_entry is not None:
            replacement = self.regex_replace_entry.get().strip()
        else:
            replacement = ""

        self.state.pipeline_config["regex"] = {
            "pattern": pattern,
            "replacement": replacement,
        }

    # Button actions

    def add_selected_step(self):
        sel = self.available_list.curselection()
        if not sel:
            return
        label = self.available_list.get(sel[0])
        self.pipeline_list.insert(END, label)
        self.save_to_state()

    def remove_selected_step(self):
        sel = self.pipeline_list.curselection()
        if not sel:
            return
        self.pipeline_list.delete(sel[0])
        self.save_to_state()

    def move_step_up(self):
        sel = self.pipeline_list.curselection()
        if not sel:
            return
        idx = sel[0]
        if idx == 0:
            return
        label = self.pipeline_list.get(idx)
        self.pipeline_list.delete(idx)
        self.pipeline_list.insert(idx - 1, label)
        self.pipeline_list.selection_clear(0, "end")
        self.pipeline_list.selection_set(idx - 1)
        self.save_to_state()

    def move_step_down(self):
        sel = self.pipeline_list.curselection()
        if not sel:
            return
        idx = sel[0]
        if idx == self.pipeline_list.size() - 1:
            return
        label = self.pipeline_list.get(idx)
        self.pipeline_list.delete(idx)
        self.pipeline_list.insert(idx + 1, label)
        self.pipeline_list.selection_clear(0, "end")
        self.pipeline_list.selection_set(idx + 1)
        self.save_to_state()


    # Double-click handlers

    def on_available_double_click(self, event):
        self.add_selected_step()

    def on_pipeline_double_click(self, event):
        self.remove_selected_step()


    # Drag-to-reorder (pipeline list)

    def on_pipeline_press(self, event):
        # remember which index we started on
        idx = self.pipeline_list.nearest(event.y)
        self._drag_start_index = idx

    def on_pipeline_release(self, event):
        if self._drag_start_index is None:
            return

        start = self._drag_start_index
        end = self.pipeline_list.nearest(event.y)
        self._drag_start_index = None

        if start == end:
            return  # no movement

        label = self.pipeline_list.get(start)
        self.pipeline_list.delete(start)

        # if we removed an earlier item, the index after delete shifts
        if end > start:
            end -= 1

        self.pipeline_list.insert(end, label)
        self.pipeline_list.selection_clear(0, "end")
        self.pipeline_list.selection_set(end)
        self.save_to_state()


    # Preview
    def preview_pipeline(self):
        """
        Preview the current pipeline on the active document:

        - If CSV is loaded: preview on the first row of the chosen text column.
        - If TXT/PDF: preview on the full text (state.text).
        """
        # Is there any active document at all?
        has_text = bool(self.state.text and self.state.text.strip())
        in_csv_mode = enlp.is_csv_mode(self.state)

        if not (has_text or in_csv_mode):
            messagebox.showwarning(
                "Warning",
                "Please upload a file first in the Document Tools tab."
            )
            return

        # sync GUI → state first
        self.save_to_state()

        # Decide which raw sample text to preview on
        sample_text = ""

        if enlp.is_csv_mode(self.state):
            raw_docs = enlp.get_raw_documents_from_state(self.state)

            if not raw_docs:
                messagebox.showwarning(
                    "Warning",
                    "No text found in the selected CSV column to preview."
                )
                return

            # Use up to first 5 non-empty rows
            preview_rows = raw_docs[:5]

            # Apply preprocessing to each row
            processed_rows = [
                enlp.apply_pipeline(
                    row,
                    cfg=self.state.pipeline_config,
                    stopword_set=self.state.stopwords
                )
                for row in preview_rows
            ]

            # Format nicely
            preview_text = ""
            for idx, (raw, proc) in enumerate(zip(preview_rows, processed_rows), start=1):
                preview_text += f" Row {idx}: "
                preview_text += proc.strip() + "\n\n"

            # Write to UI
            self.preview_box.delete("1.0", "end")
            self.preview_box.insert(
                "end",
                f"(Preview on first {len(preview_rows)} rows of column '{self.state.csv_text_column}')\n\n"
            )
            self.preview_box.insert("end", preview_text)
            return

        else:
            # TXT/PDF: use state.text directly
            sample_text = self.state.text or ""
            if not sample_text.strip():
                messagebox.showwarning(
                    "Warning",
                    "Current document text is empty."
                )
                return

        processed = enlp.apply_pipeline(
            sample_text,
            cfg=self.state.pipeline_config,
            stopword_set=self.state.stopwords
        )

        self.preview_box.delete("1.0", "end")
        if enlp.is_csv_mode(self.state):
            self.preview_box.insert(
                "end",
                f"(Preview on first row of column '{self.state.csv_text_column}')\n\n"
            )
        else:
            self.preview_box.insert("end", "(Preview on full document)\n\n")
        self.preview_box.insert("end", processed)

    def apply_theme(self):
        s = theme.scalars()
        for lb in (self.available_list, self.pipeline_list):
            lb.configure(
                bg=s["listbox_bg"],
                fg=s["listbox_fg"],
                highlightbackground=s["listbox_hl"],
            )
        if self.preview_box:
            tw = self.preview_box._textbox
            accent = "#0062cc" if not theme.is_dark() else "#6ea8fe"
            even_bg = "#f0f0f0" if not theme.is_dark() else "#181818"
            odd_bg = "#e8e8e8" if not theme.is_dark() else "#141414"
            tw.tag_configure("row_header", foreground=accent)
            tw.tag_configure("row_bg_even", background=even_bg)
            tw.tag_configure("row_bg_odd", background=odd_bg)

    # CSV export: apply pipeline to all rows and download

    def export_preprocessed_csv(self):
        """
        Apply the current preprocessing pipeline to every row
        in the selected CSV text column, create a new '<col>_clean'
        column, and let the user download the resulting CSV.
        """
        if not enlp.is_csv_mode(self.state):
            messagebox.showwarning(
                "Warning",
                "This action is only available when a CSV file is loaded in the Document Tools tab."
            )
            return

        if self.state.df is None or not self.state.csv_text_column:
            messagebox.showwarning(
                "Warning",
                "No valid CSV and text column found."
            )
            return

        # sync GUI → state
        self.save_to_state()

        df = self.state.df.copy()
        col = self.state.csv_text_column

        if col not in df.columns:
            messagebox.showerror(
                "Error",
                f"The selected column '{col}' no longer exists in the CSV."
            )
            return

        series = df[col].fillna("").astype(str)

        cfg = self.state.pipeline_config
        stopwords = self.state.stopwords

        new_col = f"{col}_clean"

        # Apply pipeline row-wise
        processed_values = [
            enlp.apply_pipeline(text, cfg=cfg, stopword_set=stopwords)
            for text in series
        ]
        df[new_col] = processed_values

        # Ask user where to save
        base_name, _ = os.path.splitext(self.state.file_name or "preprocessed.csv")
        default_name = f"{base_name}_preprocessed.csv"

        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=default_name,
            title="Save preprocessed CSV"
        )

        if not path:
            return  # user cancelled

        try:
            df.to_csv(path, index=False, encoding="utf-8")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save CSV:\n{e}")
            return

        # update state.df to the new dataframe with the clean column
        self.state.df = df

        messagebox.showinfo(
            "Success",
            f"Preprocessed CSV saved successfully.\n\nNew column: '{new_col}'\nPath: {path}"
        )
