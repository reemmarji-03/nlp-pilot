import customtkinter as ctk
from tkinter import messagebox

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from core.english_state import EnglishState
from core import english_nlp as enlp
from core import vectorization as vec
from core.task_runner import TaskRunner
from tabs.progress_overlay import ProgressOverlay


class VectorTab(ctk.CTkFrame):
    """
    Vectorization Lab:
    - Compare BoW / TF-IDF / Char n-grams / Transformer embeddings
    - Inspect vocab & top features
    - Cosine similarity between two sentences
    - PCA 2D projection for sentences / CSV rows
    """

    def __init__(self, parent, state: EnglishState):
        super().__init__(parent, fg_color="#0f0f0f")
        self.state = state

        # widgets
        self.method_combo = None
        self.ng_min_entry = None
        self.ng_max_entry = None
        self.max_features_entry = None

        self.summary_box = None
        self.features_box = None
        self.plot_canvas = None
        self.plot_figure = None

        self.sent1_box = None
        self.sent2_box = None
        self.similarity_label = None

        self.vec_runner = TaskRunner(root=self)
        self.vec_progress = None
        self._vec_btn = None
        self._pca_btn = None

        self.build_ui()
        self.pack(fill="both", expand=True)

    # ---------------- UI ----------------
    def build_ui(self):
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # Sidebar
        sidebar = ctk.CTkFrame(self, fg_color="#121212", corner_radius=10)
        sidebar.grid(row=0, column=0, sticky="ns", padx=(10, 5), pady=10)

        ctk.CTkLabel(
            sidebar,
            text="Vectorization Lab",
            text_color="#6ea8fe",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=(10, 5))

        # --- Method selection ---
        ctk.CTkLabel(
            sidebar,
            text="Method:",
            text_color="#cccccc",
            font=ctk.CTkFont(size=13)
        ).pack(anchor="w", padx=10, pady=(10, 0))

        self.method_combo = ctk.CTkComboBox(
            sidebar,
            values=[
                "Bag-of-Words",
                "TF-IDF (word)",
                "TF-IDF (char)",
                "Transformer embeddings",
            ],
            state="readonly",
            width=200
        )
        self.method_combo.set("TF-IDF (word)")
        self.method_combo.pack(padx=10, pady=(2, 10))

        # --- n-gram range ---
        ngram_frame = ctk.CTkFrame(sidebar, fg_color="#121212")
        ngram_frame.pack(fill="x", padx=10, pady=(0, 5))

        ctk.CTkLabel(
            ngram_frame,
            text="n-gram range (min, max):",
            text_color="#cccccc",
            font=ctk.CTkFont(size=13)
        ).pack(anchor="w")

        row_ng = ctk.CTkFrame(ngram_frame, fg_color="#121212")
        row_ng.pack(fill="x", pady=(2, 0))

        self.ng_min_entry = ctk.CTkEntry(row_ng, width=40)
        self.ng_min_entry.insert(0, "1")
        self.ng_min_entry.pack(side="left", padx=(0, 5))

        self.ng_max_entry = ctk.CTkEntry(row_ng, width=40)
        self.ng_max_entry.insert(0, "2")
        self.ng_max_entry.pack(side="left")

        # --- max features ---
        ctk.CTkLabel(
            sidebar,
            text="Max features:",
            text_color="#cccccc",
            font=ctk.CTkFont(size=13)
        ).pack(anchor="w", padx=10, pady=(10, 0))

        self.max_features_entry = ctk.CTkEntry(sidebar, width=80)
        self.max_features_entry.insert(0, "5000")
        self.max_features_entry.pack(anchor="w", padx=10, pady=(2, 10))

        # --- Buttons ---
        self._vec_btn = ctk.CTkButton(
            sidebar,
            text="Vectorize current document",
            fg_color="#0078ff",
            hover_color="#005dc1",
            command=self.vectorize_document
        )
        self._vec_btn.pack(fill="x", padx=10, pady=(10, 5))

        self._pca_btn = ctk.CTkButton(
            sidebar,
            text="PCA 2D projection",
            fg_color="#444444",
            hover_color="#333333",
            command=self.plot_sentence_projection
        )
        self._pca_btn.pack(fill="x", padx=10, pady=(0, 15))

        self.vec_progress = ProgressOverlay(sidebar, task_runner=self.vec_runner)

        # --- Similarity sandbox ---
        ctk.CTkLabel(
            sidebar,
            text="Similarity sandbox:",
            text_color="#cccccc",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=10, pady=(5, 0))

        self.sent1_box = ctk.CTkTextbox(
            sidebar,
            height=40,
            fg_color="#1a1a1a",
            text_color="white",
            font=("Consolas", 11),
            wrap="word"
        )
        self.sent1_box.pack(fill="x", padx=10, pady=(4, 2))
        self.sent1_box.insert("end", "The movie was fantastic and inspiring.")

        self.sent2_box = ctk.CTkTextbox(
            sidebar,
            height=40,
            fg_color="#1a1a1a",
            text_color="white",
            font=("Consolas", 11),
            wrap="word"
        )
        self.sent2_box.pack(fill="x", padx=10, pady=(2, 4))
        self.sent2_box.insert("end", "I really enjoyed the film, it was great.")

        ctk.CTkButton(
            sidebar,
            text="Compute similarity",
            fg_color="#0078ff",
            hover_color="#005dc1",
            command=self.compute_similarity
        ).pack(fill="x", padx=10, pady=(0, 4))

        self.similarity_label = ctk.CTkLabel(
            sidebar,
            text="Similarity: –",
            text_color="#cccccc",
            font=ctk.CTkFont(size=13)
        )
        self.similarity_label.pack(anchor="w", padx=10, pady=(0, 10))

        # ---------------- Main output: Tabview ----------------
        right = ctk.CTkFrame(self, fg_color="#1a1a1a", corner_radius=10)
        right.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        tabview = ctk.CTkTabview(right, fg_color="#1a1a1a")
        tabview.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        summary_tab = tabview.add("Summary")
        features_tab = tabview.add("Top features")
        plot_tab = tabview.add("Plot")

        # Summary box
        self.summary_box = ctk.CTkTextbox(
            summary_tab,
            fg_color="#1a1a1a",
            text_color="white",
            font=("Consolas", 12),
            wrap="word"
        )
        self.summary_box.pack(fill="both", expand=True, padx=10, pady=10)

        # Features area ✅ (container for cards)
        self.features_container = ctk.CTkFrame(
            features_tab,
            fg_color="#1a1a1a",
            corner_radius=10,
        )
        self.features_container.pack(fill="both", expand=True, padx=10, pady=10)

        # we'll put up to 3 cards in columns 0,1,2
        for col in range(3):
            self.features_container.grid_columnconfigure(col, weight=1)

        # Plot area
        self.plot_container = ctk.CTkFrame(plot_tab, fg_color="#1a1a1a")
        self.plot_container.pack(fill="both", expand=True, padx=10, pady=10)

    # ---------------- Helpers ----------------

    def _has_any_document(self) -> bool:
        """
        True if either:
        - a plain text / pdf / docx has been loaded (state.text), or
        - we're in CSV mode (df + chosen text column).
        """
        in_csv_mode = enlp.is_csv_mode(self.state)
        has_text = bool(self.state.text and self.state.text.strip())
        return in_csv_mode or has_text

    def require_document(self) -> bool:
        if not self._has_any_document():
            messagebox.showwarning(
                "Warning",
                "Please upload a file first in the Document Tools tab."
            )
            return False
        return True

    def get_active_text(self) -> str:
        """
        Use the current preprocessing pipeline to get cleaned text
        for the single-document case (txt/pdf/docx).
        """
        return enlp.apply_pipeline(
            self.state.text,
            cfg=self.state.pipeline_config,
            stopword_set=self.state.stopwords
        )

    def _parse_ngrams(self) -> tuple[int, int]:
        try:
            n_min = int(self.ng_min_entry.get() or "1")
            n_max = int(self.ng_max_entry.get() or "1")
        except ValueError:
            n_min, n_max = 1, 1
        if n_min < 1:
            n_min = 1
        if n_max < n_min:
            n_max = n_min
        return n_min, n_max

    def _parse_max_features(self) -> int:
        try:
            mf = int(self.max_features_entry.get() or "5000")
        except ValueError:
            mf = 5000
        return max(100, mf)

    def _clear_plot(self):
        if self.plot_canvas is not None:
            self.plot_canvas.get_tk_widget().destroy()
            self.plot_canvas = None
        if self.plot_figure is not None:
            plt.close(self.plot_figure)
            self.plot_figure = None

    # ---------------- Main actions ----------------

    def _build_docs_for_vectorization(self) -> list[str]:
        """
        Returns a list of strings to vectorize.

        - CSV mode: one doc per row of the chosen text column
                    (after preprocessing pipeline).
        - Normal mode: a single doc = full preprocessed text.
        """
        if enlp.is_csv_mode(self.state):
            raw_docs = enlp.get_raw_documents_from_state(self.state)
            docs = [
                enlp.apply_pipeline(
                    d or "",
                    cfg=self.state.pipeline_config,
                    stopword_set=self.state.stopwords
                ).strip()
                for d in raw_docs
            ]
            # drop empty rows
            docs = [d for d in docs if d]
            return docs
        else:
            text = self.get_active_text().strip()
            return [text] if text else []

    def vectorize_document(self):
        if not self.require_document():
            return

        method = self.method_combo.get()
        ngram_range = self._parse_ngrams()
        max_features = self._parse_max_features()
        docs = self._build_docs_for_vectorization()
        if not docs:
            messagebox.showwarning("Warning", "No non-empty text found to vectorize.")
            return

        truncated = False
        if enlp.is_csv_mode(self.state) and method == "Transformer embeddings":
            if len(docs) > 256:
                docs = docs[:256]
                truncated = True

        self._vec_btn.configure(state="disabled")
        self._pca_btn.configure(state="disabled")
        self.vec_progress.show(f"Vectorizing with {method}…")

        captured_docs = docs
        captured_truncated = truncated

        def _work():
            if method == "Bag-of-Words":
                res, _ = vec.vectorize_bow(captured_docs, ngram_range=ngram_range, max_features=max_features)
            elif method == "TF-IDF (word)":
                res, _ = vec.vectorize_tfidf(captured_docs, ngram_range=ngram_range, max_features=max_features, analyzer="word")
            elif method == "TF-IDF (char)":
                res, _ = vec.vectorize_tfidf(captured_docs, ngram_range=ngram_range, max_features=max_features, analyzer="char")
            elif method == "Transformer embeddings":
                res = vec.transformer_sentence_embeddings(captured_docs)
            else:
                raise ValueError(f"Unknown method: {method}")
            return res

        def _on_done(res):
            self.vec_progress.hide()
            self._vec_btn.configure(state="normal")
            self._pca_btn.configure(state="normal")
            self.state.last_vector_method = method
            self.state.last_vector_ngram = ngram_range
            self.state.last_vector_max_features = max_features
            self._display_vectorization_results(res, method, captured_truncated, captured_docs)

        def _on_error(exc):
            self.vec_progress.hide()
            self._vec_btn.configure(state="normal")
            self._pca_btn.configure(state="normal")
            messagebox.showerror("Vectorization error", str(exc))

        def _on_cancel():
            self.vec_progress.hide()
            self._vec_btn.configure(state="normal")
            self._pca_btn.configure(state="normal")

        self.vec_runner.run(_work, on_done=_on_done, on_error=_on_error, on_cancel=_on_cancel)

    def _display_vectorization_results(self, res, method, truncated: bool, docs: list) -> None:
        n_samples = res.vector.shape[0]

        # --- Summary ---
        self.summary_box.delete("1.0", "end")
        mode_str = "CSV rows" if enlp.is_csv_mode(self.state) else "Single document"
        self.summary_box.insert("end", f"Mode: {mode_str}\n")
        self.summary_box.insert("end", f"Method: {res.name}\n")
        self.summary_box.insert("end", f"Samples (documents): {n_samples}\n")
        self.summary_box.insert("end", f"Dimensionality: {res.dim}\n")
        self.summary_box.insert("end", f"Sparse: {res.is_sparse}\n")
        if res.is_sparse:
            self.summary_box.insert("end", f"Non-zero entries: {res.nnz}\n")
            self.summary_box.insert("end", f"Density: {res.density:.6f}\n")
        self.summary_box.insert("end", f"Memory footprint: {res.memory_kb:.2f} KB\n")
        if truncated:
            self.summary_box.insert(
                "end",
                "\nNote: Transformer embeddings computed on first 256 rows for speed.\n"
            )

        # --- Top features as side-by-side cards ---
        # Make sure you have: self._clear_features() and self.features_container defined.
        self._clear_features()

        # Dense transformer embeddings don't have feature weights
        if not res.is_sparse:
            card = ctk.CTkFrame(
                self.features_container,
                fg_color="#222222",
                corner_radius=10
            )
            card.grid(row=0, column=0, sticky="nsew", padx=5, pady=5, columnspan=3)

            ctk.CTkLabel(
                card,
                text="Top features not available for dense embeddings.",
                text_color="#dddddd",
                font=ctk.CTkFont(size=13)
            ).pack(padx=10, pady=10)
            return

        # Sparse case: we can inspect per-feature weights
        feature_names = res.vocab or []
        X = res.vector  # sparse matrix

        import random

        if enlp.is_csv_mode(self.state):
            # Sample up to 3 random rows
            n_samples_mat = X.shape[0]
            num_samples = min(3, n_samples_mat)
            if num_samples == 0:
                card = ctk.CTkFrame(
                    self.features_container,
                    fg_color="#222222",
                    corner_radius=10
                )
                card.grid(row=0, column=0, sticky="nsew", padx=5, pady=5, columnspan=3)
                ctk.CTkLabel(
                    card,
                    text="No non-empty rows to show top features.",
                    text_color="#dddddd",
                    font=ctk.CTkFont(size=13)
                ).pack(padx=10, pady=10)
                return

            indices = random.sample(range(n_samples_mat), num_samples)

            # Header label
            header = ctk.CTkLabel(
                self.features_container,
                text=(
                    f"Sampled top features from {num_samples} random rows "
                    f"of column '{self.state.csv_text_column}':"
                ),
                text_color="#ffffff",
                font=ctk.CTkFont(size=14, weight="bold")
            )
            header.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8), padx=5)

            for col_idx, idx in enumerate(indices):
                card = ctk.CTkFrame(
                    self.features_container,
                    fg_color="#222222",
                    corner_radius=10
                )
                card.grid(row=1, column=col_idx, sticky="nsew", padx=5, pady=5)

                # Row title
                ctk.CTkLabel(
                    card,
                    text=f"Row {idx + 1}",
                    text_color="#6ea8fe",
                    font=ctk.CTkFont(size=13, weight="bold")
                ).pack(anchor="w", padx=8, pady=(6, 2))

                # Snippet of original (preprocessed) text
                row_text = docs[idx]
                snippet = row_text.replace("\n", " ")
                if len(snippet) > 200:
                    snippet = snippet[:200] + "..."

                ctk.CTkLabel(
                    card,
                    text=snippet,
                    text_color="#dddddd",
                    font=ctk.CTkFont(size=11),
                    wraplength=220,
                    justify="left"
                ).pack(anchor="w", padx=8, pady=(0, 6))

                # Top features for that row
                top_row_feats = vec._top_k_from_sparse_row(X[idx], feature_names, k=8)

                ctk.CTkLabel(
                    card,
                    text="Top features:",
                    text_color="#ffffff",
                    font=ctk.CTkFont(size=12, weight="bold")
                ).pack(anchor="w", padx=8, pady=(2, 0))

                features_text = "\n".join(
                    f"{feat:<15} {val:.3f}" for feat, val in top_row_feats
                ) or "(no non-zero features)"

                features_box = ctk.CTkTextbox(
                    card,
                    height=120,
                    fg_color="#1a1a1a",
                    text_color="white",
                    font=("Consolas", 11),
                    wrap="word"
                )
                features_box.pack(fill="both", expand=True, padx=8, pady=(2, 8))
                features_box.insert("end", features_text)
                features_box.configure(state="disabled")

        else:
            # Single-document case: one wide card
            card = ctk.CTkFrame(
                self.features_container,
                fg_color="#222222",
                corner_radius=10
            )
            card.grid(row=0, column=0, sticky="nsew", padx=5, pady=5, columnspan=3)

            ctk.CTkLabel(
                card,
                text="Top features (full document):",
                text_color="#6ea8fe",
                font=ctk.CTkFont(size=14, weight="bold")
            ).pack(anchor="w", padx=8, pady=(8, 4))

            top = vec._top_k_from_sparse_row(X[0], feature_names, k=20)
            features_text = "\n".join(
                f"{feat:<20} {val:.4f}" for feat, val in top
            ) or "(no non-zero features)"

            features_box = ctk.CTkTextbox(
                card,
                fg_color="#1a1a1a",
                text_color="white",
                font=("Consolas", 12),
                wrap="word",
                height=240
            )
            features_box.pack(fill="both", expand=True, padx=8, pady=(0, 8))
            features_box.insert("end", features_text)
            features_box.configure(state="disabled")

    def compute_similarity(self):
        """
        Similarity sandbox: encode two short sentences using current method,
        then compute cosine similarity.
        (Independent from uploaded document / CSV.)
        """
        method = self.method_combo.get()
        s1 = self.sent1_box.get("1.0", "end").strip()
        s2 = self.sent2_box.get("1.0", "end").strip()
        if not s1 or not s2:
            messagebox.showwarning("Warning", "Please enter both sentences.")
            return

        texts = [s1, s2]
        ngram_range = self._parse_ngrams()
        max_features = self._parse_max_features()

        try:
            if method == "Bag-of-Words":
                res, _ = vec.vectorize_bow(
                    texts,
                    ngram_range=ngram_range,
                    max_features=max_features
                )
                X = res.vector.toarray()
            elif method == "TF-IDF (word)":
                res, _ = vec.vectorize_tfidf(
                    texts,
                    ngram_range=ngram_range,
                    max_features=max_features,
                    analyzer="word"
                )
                X = res.vector.toarray()
            elif method == "TF-IDF (char)":
                res, _ = vec.vectorize_tfidf(
                    texts,
                    ngram_range=ngram_range,
                    max_features=max_features,
                    analyzer="char"
                )
                X = res.vector.toarray()
            elif method == "Transformer embeddings":
                res = vec.transformer_sentence_embeddings(texts)
                X = res.vector
            else:
                messagebox.showerror("Error", f"Unknown method: {method}")
                return
        except Exception as e:
            messagebox.showerror("Vectorization error", str(e))
            return

        sim = vec.cosine_similarity(X[0], X[1])
        self.similarity_label.configure(text=f"Similarity: {sim:.3f}")

    def plot_sentence_projection(self):
        """
        Visual 2D projection using PCA.

        - Normal text mode: split the preprocessed document into sentences.
        - CSV mode: each (preprocessed) row is one point.
        """
        if not self.require_document():
            return

        method = self.method_combo.get()
        ngram_range = self._parse_ngrams()
        max_features = self._parse_max_features()

        # ---- Build docs + labels depending on mode ----
        if enlp.is_csv_mode(self.state):
            raw_docs = enlp.get_raw_documents_from_state(self.state)
            docs = [
                enlp.apply_pipeline(
                    d or "",
                    cfg=self.state.pipeline_config,
                    stopword_set=self.state.stopwords
                ).strip()
                for d in raw_docs
            ]
            docs = [d for d in docs if d]

            # limit for plotting to avoid overclutter
            max_points = 200
            if len(docs) > max_points:
                docs = docs[:max_points]
            labels = [f"Row {i}" for i in range(1, len(docs) + 1)]
        else:
            # naive sentence split on preprocessed full text
            import re
            text = self.get_active_text().strip()
            sentences = [s.strip() for s in re.split(r"[.!?]\s+", text) if s.strip()]
            docs = sentences
            labels = [f"S{i}" for i in range(1, len(docs) + 1)]

        if len(docs) < 2:
            messagebox.showwarning(
                "Warning",
                "Not enough units (sentences/rows) for projection."
            )
            return

        self._vec_btn.configure(state="disabled")
        self._pca_btn.configure(state="disabled")
        self.vec_progress.show("Computing PCA projection…")

        captured_docs = docs
        captured_labels = labels

        def _work():
            if method == "Bag-of-Words":
                res, _ = vec.vectorize_bow(
                    captured_docs,
                    ngram_range=ngram_range,
                    max_features=max_features
                )
                X = res.vector.toarray()
            elif method == "TF-IDF (word)":
                res, _ = vec.vectorize_tfidf(
                    captured_docs,
                    ngram_range=ngram_range,
                    max_features=max_features,
                    analyzer="word"
                )
                X = res.vector.toarray()
            elif method == "TF-IDF (char)":
                res, _ = vec.vectorize_tfidf(
                    captured_docs,
                    ngram_range=ngram_range,
                    max_features=max_features,
                    analyzer="char"
                )
                X = res.vector.toarray()
            elif method == "Transformer embeddings":
                res = vec.transformer_sentence_embeddings(captured_docs)
                X = res.vector
            else:
                raise ValueError(f"Unknown method: {method}")

            coords = vec.pca_2d(X)
            return coords

        def _on_done(coords):
            self.vec_progress.hide()
            self._vec_btn.configure(state="normal")
            self._pca_btn.configure(state="normal")
            self.state.last_vector_method = method
            self.state.last_vector_ngram = ngram_range
            self.state.last_vector_max_features = max_features

            # Plot
            self._clear_plot()
            self.plot_figure = plt.figure(figsize=(5, 4))
            ax = self.plot_figure.add_subplot(111)
            ax.scatter(coords[:, 0], coords[:, 1])

            title = "Sentence embeddings (PCA 2D)"
            if enlp.is_csv_mode(self.state):
                title = f"CSV row embeddings (PCA 2D) – '{self.state.csv_text_column}'"
            ax.set_title(title)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")

            # label at most first 40 points to avoid clutter
            for i, label in enumerate(captured_labels[:40]):
                ax.text(coords[i, 0], coords[i, 1], label, fontsize=7)

            self.plot_canvas = FigureCanvasTkAgg(self.plot_figure, master=self.plot_container)
            self.plot_canvas.draw()
            self.plot_canvas.get_tk_widget().pack(fill="both", expand=True)

        def _on_error(exc):
            self.vec_progress.hide()
            self._vec_btn.configure(state="normal")
            self._pca_btn.configure(state="normal")
            messagebox.showerror("PCA error", str(exc))

        def _on_cancel():
            self.vec_progress.hide()
            self._vec_btn.configure(state="normal")
            self._pca_btn.configure(state="normal")

        self.vec_runner.run(_work, on_done=_on_done, on_error=_on_error, on_cancel=_on_cancel)

    def _clear_features(self):
        """Remove all existing feature cards."""
        for child in self.features_container.winfo_children():
            child.destroy()
