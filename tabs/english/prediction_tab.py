# tabs/prediction_tab.py
from sklearn.cluster import KMeans
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import customtkinter as ctk
from tkinter import messagebox

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from core.english_state import EnglishState
from core import english_nlp as enlp
from core import supervised as sup
from core import unsupervised as unsup
from core import vectorization as vec
from core.task_runner import TaskRunner
from tabs.progress_overlay import ProgressOverlay


# K-Means helper functions for animated mode

def euclid_dist(p, q):
    return float(np.sqrt((q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2))


def pick_centers(points, k):
    import random
    return random.sample(points, k)


def clusterize(points, centers):
    clusters = {i: [] for i in range(len(centers))}
    for point in points:
        dists = [euclid_dist(point, c) for c in centers]
        idx = int(np.argmin(dists))
        clusters[idx].append(point)
    return clusters


def update_centroids(clusters):
    new_centers = []
    for cluster_points in clusters.values():
        if not cluster_points:
            new_centers.append([0.0, 0.0])
            continue
        arr = np.array(cluster_points, dtype=float)
        center = arr.mean(axis=0).tolist()
        new_centers.append(center)
    return new_centers


def centroids_converged(old, new, tol=1e-4):
    for o, n in zip(old, new):
        if euclid_dist(o, n) > tol:
            return False
    return True


# PREDICTION TAB

class PredictionTab(ctk.CTkFrame):
    """
    Prediction Tab:
    - Supervised: classification/regression on CSV labels
      * auto-detect task type
      * user chooses model (LR/SVM/RF etc.)
      * uses last vectorization settings
    - Unsupervised: clustering
      * K-Means (animated) using PCA 2D doc embeddings
      * K-Means (static sklearn) + PCA scatter
    """

    def __init__(self, parent, state: EnglishState):
        super().__init__(parent, fg_color="#0f0f0f")
        self.state = state

        # TaskRunner instances for background work
        self.sup_runner = TaskRunner(root=self)
        self.clust_runner = TaskRunner(root=self)
        self.topic_runner = TaskRunner(root=self)

        # ProgressOverlay instances (created during build_ui)
        self.sup_progress = None
        self.clust_progress = None
        self.topic_progress = None

        # Button references (created during build_ui)
        self._sup_train_btn = None
        self._clust_run_btn = None
        self._clust_elbow_btn = None
        self._topic_run_btn = None

        # supervised-related
        self.supervised_label_combo = None
        self.supervised_model_combo = None
        self.supervised_vector_label = None
        self.test_size_entry = None
        self.supervised_metrics_box = None
        self.supervised_plot_figure = None
        self.supervised_plot_canvas = None
        self.supervised_plot_container = None
        self.current_task_type = None  # "classification" or "regression"

        # clustering-related
        self.clustering_vector_label = None
        self.cluster_k_entry = None
        self.cluster_mode_combo = None
        self.cluster_summary_box = None
        self.cluster_plot_container = None
        self.cluster_plot_figure = None
        self.cluster_plot_canvas = None
        self.topic_summary_box = None
        self.topic_plot_container = None
        self.topic_plot_figure = None
        self.topic_plot_canvas = None
        # animated K-Means state
        self.cluster_canvas = None
        self.points = []          # list [[x,y], ...] PCA coords mapped to canvas
        self.point_ids = []
        self.k = None
        self.centroids = []
        self.centroid_ids = []
        self.clusters = {}
        self.iteration = 0
        self.max_iterations = 30
        self.tolerance = 1e-3
        self.animating = False
        self.old_centroids = []
        self.new_centroids = []
        self.interpolate_steps = 15
        self.current_interp_step = 0
        self.cluster_colors = ["red", "lime", "cyan", "magenta", "yellow", "orange", "deepskyblue"]

        self.build_ui()
        self.pack(fill="both", expand=True)

        # Populate combos programmatically when created
        self.refresh_supervised_columns(silent=True)
        self._update_vector_labels()



    # --------------------------------------------------------
    # Public helper: call this after loading a new CSV
    # --------------------------------------------------------
    def sync_with_state(self):
        """
        Call this from the main app whenever:
        - You load a new CSV into state.df
        - You change state.csv_text_column

        It will:
        - Refresh label column combobox
        - Refresh the vectorization settings text
        """
        self.refresh_supervised_columns(silent=True)
        self._update_vector_labels()

    # --------------------------------------------------------
    # UI construction
    # --------------------------------------------------------
    def build_ui(self):
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        tabview = ctk.CTkTabview(self, fg_color="#1a1a1a")
        tabview.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        supervised_tab = tabview.add("Supervised")
        clustering_tab = tabview.add("Unsupervised (Clustering)")
        topic_tab = tabview.add("Topic Modeling (BERTopic)")

        self._build_supervised_tab(supervised_tab)
        self._build_clustering_tab(clustering_tab)
        self._build_topic_tab(topic_tab)
        # ============================================================
    #    SUPERVISED SUBTAB
    # ============================================================
    def _build_supervised_tab(self, parent):
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)

        # Sidebar
        sidebar = ctk.CTkFrame(parent, fg_color="#121212", corner_radius=10)
        sidebar.grid(row=0, column=0, sticky="ns", padx=(10, 5), pady=10)

        ctk.CTkLabel(
            sidebar,
            text="Supervised Prediction",
            text_color="#6ea8fe",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack(pady=(10, 5))

        ctk.CTkLabel(
            sidebar,
            text=(
                "Requires a CSV:\n"
                "- Text column already chosen\n"
                "- Label column for prediction\n\n"
                "Uses the last vectorization settings\n"
                "from the Vectorization tab."
            ),
            text_color="#bbbbbb",
            font=ctk.CTkFont(size=11),
            justify="left",
        ).pack(padx=10, pady=(0, 10))

        # Show current vectorization settings (read-only)
        self.supervised_vector_label = ctk.CTkLabel(
            sidebar,
            text=self._format_vector_settings(),
            text_color="#dddddd",
            font=ctk.CTkFont(size=11),
            justify="left",
        )
        self.supervised_vector_label.pack(padx=10, pady=(0, 10))

        # Label column selection
        ctk.CTkLabel(
            sidebar,
            text="Label column:",
            text_color="#cccccc",
            font=ctk.CTkFont(size=13),
        ).pack(anchor="w", padx=10, pady=(5, 0))

        self.supervised_label_combo = ctk.CTkComboBox(
            sidebar,
            values=[],
            state="readonly",
            width=200,
            command=self.on_supervised_label_changed,
        )
        self.supervised_label_combo.pack(padx=10, pady=(2, 10))

        # Model selection
        ctk.CTkLabel(
            sidebar,
            text="Model:",
            text_color="#cccccc",
            font=ctk.CTkFont(size=13),
        ).pack(anchor="w", padx=10, pady=(5, 0))

        self.supervised_model_combo = ctk.CTkComboBox(
            sidebar,
            values=["Auto"],
            state="readonly",
            width=200,
        )
        self.supervised_model_combo.set("Auto")
        self.supervised_model_combo.pack(padx=10, pady=(2, 10))

        # Test size
        ctk.CTkLabel(
            sidebar,
            text="Test size (0–0.9):",
            text_color="#cccccc",
            font=ctk.CTkFont(size=13),
        ).pack(anchor="w", padx=10, pady=(5, 0))

        self.test_size_entry = ctk.CTkEntry(sidebar, width=80)
        self.test_size_entry.insert(0, "0.2")
        self.test_size_entry.pack(anchor="w", padx=10, pady=(2, 10))

        # Train button
        self._sup_train_btn = ctk.CTkButton(
            sidebar,
            text="Train model",
            fg_color="#0078ff",
            hover_color="#005dc1",
            command=self.train_supervised_model,
        )
        self._sup_train_btn.pack(fill="x", padx=10, pady=(0, 10))
        self.sup_progress = ProgressOverlay(sidebar, task_runner=self.sup_runner)

        # Right side: metrics + plot
        right = ctk.CTkFrame(parent, fg_color="#1a1a1a", corner_radius=10)
        right.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        subtab = ctk.CTkTabview(right, fg_color="#1a1a1a")
        subtab.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        metrics_tab = subtab.add("Metrics")
        plot_tab = subtab.add("Plot")

        self.supervised_metrics_box = ctk.CTkTextbox(
            metrics_tab,
            fg_color="#1a1a1a",
            text_color="white",
            font=("Consolas", 11),
            wrap="word",
        )
        self.supervised_metrics_box.pack(fill="both", expand=True, padx=10, pady=10)

        self.supervised_plot_container = ctk.CTkFrame(plot_tab, fg_color="#1a1a1a")
        self.supervised_plot_container.pack(fill="both", expand=True, padx=10, pady=10)

    def refresh_supervised_columns(self, silent: bool = False):
        """
        Populate the label combobox with non-text columns from the CSV.

        Call this AFTER:
        - state.df is set (CSV loaded)
        - state.csv_text_column is set (user chose text column)
        """
        df = self.state.df

        if df is None:
            if not silent:
                messagebox.showwarning(
                    "Warning", "Please load a CSV file first in Document Tools."
                )
            return

        cols = list(df.columns)

        # Exclude the text column if we know it
        if self.state.csv_text_column and self.state.csv_text_column in cols:
            cols.remove(self.state.csv_text_column)

        if not cols:
            if not silent:
                messagebox.showwarning(
                    "Warning",
                    "No other columns found besides the text column.\n"
                    "You need at least one label/target column.",
                )
            return

        # Update combobox
        self.supervised_label_combo.configure(values=cols)
        # If current selection not valid, pick the first
        current = self.supervised_label_combo.get()
        if current not in cols:
            self.supervised_label_combo.set(cols[0])
            chosen = cols[0]
        else:
            chosen = current

        # Update model options based on this column's dtype
        self.on_supervised_label_changed(chosen)

    def _clear_supervised_plot(self):
        if self.supervised_plot_canvas is not None:
            self.supervised_plot_canvas.get_tk_widget().destroy()
            self.supervised_plot_canvas = None
        if self.supervised_plot_figure is not None:
            plt.close(self.supervised_plot_figure)
            self.supervised_plot_figure = None

    def _format_vector_settings(self) -> str:
        method = self.state.last_vector_method or "(default) TF-IDF (word)"
        ngram = self.state.last_vector_ngram or (1, 2)
        max_feat = self.state.last_vector_max_features or 5000
        return (
            "Vectorization settings:\n"
            f"- Method: {method}\n"
            f"- n-gram: {ngram[0]}–{ngram[1]}\n"
            f"- Max features: {max_feat}"
        )

    def _update_vector_labels(self):
        text = self._format_vector_settings()
        if self.supervised_vector_label is not None:
            self.supervised_vector_label.configure(text=text)
        if self.clustering_vector_label is not None:
            self.clustering_vector_label.configure(text=text)

    def on_supervised_label_changed(self, value: str):
        """When user picks a label column, decide task type and model options."""
        if self.state.df is None:
            return
        col = value
        series = self.state.df[col]
        # Decide task type: numeric -> regression, else classification
        if np.issubdtype(series.dtype, np.number):
            self.current_task_type = "regression"
            models = [
                "Auto",
                "Linear Regression",
                "Ridge Regression",
                "Random Forest Regressor",
            ]
        else:
            self.current_task_type = "classification"
            models = [
                "Auto",
                "Logistic Regression",
                "Linear SVM",
                "Random Forest",
            ]
        self.supervised_model_combo.configure(values=models)
        self.supervised_model_combo.set("Auto")

    def train_supervised_model(self):
        if not enlp.is_csv_mode(self.state) or self.state.df is None:
            messagebox.showwarning(
                "Warning", "Please load a CSV file first in Document Tools."
            )
            return

        label_col = self.supervised_label_combo.get()
        if not label_col:
            messagebox.showwarning(
                "Warning",
                "Please choose a label column (a CSV with labels is required).",
            )
            return

        # Capture params before entering the thread
        vector_method = self.state.last_vector_method or "TF-IDF (word)"
        ngram = self.state.last_vector_ngram or (1, 2)
        max_feat = self.state.last_vector_max_features or 5000

        try:
            test_size = float(self.test_size_entry.get() or "0.2")
        except ValueError:
            test_size = 0.2
        if test_size <= 0 or test_size >= 0.9:
            test_size = 0.2

        chosen_model = self.supervised_model_combo.get() or "Auto"

        # Disable UI and show progress
        self._sup_train_btn.configure(state="disabled")
        self.sup_progress.show("Building feature matrix…")

        def _work(progress_callback=None, cancel_event=None):
            if progress_callback:
                progress_callback(0.05, "Building feature matrix…")
            X, y, task_type = sup.build_xy_from_state(
                self.state,
                label_column=label_col,
                vector_method=vector_method,
                ngram_range=ngram,
                max_features=max_feat,
            )
            if cancel_event is not None and cancel_event.is_set():
                from core.task_runner import CancelledError
                raise CancelledError()
            if progress_callback:
                progress_callback(0.2, "Training…")
            result = sup.train_supervised_model(
                X,
                y,
                task_type=task_type,
                model_name=chosen_model,
                test_size=test_size,
                cancel_event=cancel_event,
                progress_callback=(
                    (lambda p, m: progress_callback(0.2 + p * 0.8, m))
                    if progress_callback else None
                ),
            )
            return result, task_type, vector_method, ngram, max_feat

        def _on_done(payload):
            result, task_type, vm, ng, mf = payload
            self.sup_progress.hide()
            self._sup_train_btn.configure(state="normal")
            self._display_supervised_results(result, vm, ng, mf)

        def _on_error(exc):
            self.sup_progress.hide()
            self._sup_train_btn.configure(state="normal")
            messagebox.showerror("Training error", str(exc))

        def _on_cancel():
            self.sup_progress.hide()
            self._sup_train_btn.configure(state="normal")

        self.sup_runner.run(
            _work,
            on_done=_on_done,
            on_error=_on_error,
            on_cancel=_on_cancel,
        )

    def _display_supervised_results(self, result, vector_method, ngram, max_feat):
        """Display metrics and confusion matrix after training completes."""
        self.supervised_metrics_box.delete("1.0", "end")
        self.supervised_metrics_box.insert("end", f"Task type: {result.task_type}\n")
        self.supervised_metrics_box.insert("end", f"Model: {result.model_name}\n\n")
        self.supervised_metrics_box.insert(
            "end",
            f"Vectorization:\n  method={vector_method}, "
            f"ngram={ngram}, max_features={max_feat}\n\n",
        )

        if result.task_type == "classification":
            acc = result.metrics["accuracy"]
            self.supervised_metrics_box.insert("end", f"Accuracy: {acc:.4f}\n\n")

            # If Auto was used, show candidate scores
            candidate_scores = result.metrics.get("candidate_scores")
            if candidate_scores:
                self.supervised_metrics_box.insert("end", "Auto model sweep:\n")
                # Sort descending by score
                for name, score in sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True):
                    if result.task_type == "classification":
                        self.supervised_metrics_box.insert(
                            "end", f"  {name:<25} accuracy = {score:.4f}\n"
                        )
                    else:
                        self.supervised_metrics_box.insert(
                            "end", f"  {name:<25} R2 = {score:.4f}\n"
                        )
                self.supervised_metrics_box.insert("end", "\n")

            rep = result.metrics["report"]
            self.supervised_metrics_box.insert("end", "Classification report:\n")
            for label, stats in rep.items():
                if label in {"accuracy", "macro avg", "weighted avg"}:
                    continue
                if isinstance(stats, dict):
                    self.supervised_metrics_box.insert(
                        "end",
                        f"  {label}: "
                        f"precision={stats['precision']:.3f}, "
                        f"recall={stats['recall']:.3f}, "
                        f"f1={stats['f1-score']:.3f}\n",
                    )

            self._plot_confusion_matrix(result.cm)
        else:
            for k, v in result.metrics.items():
                self.supervised_metrics_box.insert("end", f"{k}: {v:.4f}\n")

            self._clear_supervised_plot()

    def _plot_confusion_matrix(self, cm: np.ndarray):
        self._clear_supervised_plot()
        if cm is None:
            return

        self.supervised_plot_figure = plt.figure(figsize=(4, 3))
        ax = self.supervised_plot_figure.add_subplot(111)

        im = ax.imshow(cm, interpolation="nearest")
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)

        self.supervised_plot_canvas = FigureCanvasTkAgg(
            self.supervised_plot_figure, master=self.supervised_plot_container
        )
        self.supervised_plot_canvas.draw()
        self.supervised_plot_canvas.get_tk_widget().pack(
            fill="both", expand=True
        )

    # ============================================================
    #    CLUSTERING SUBTAB
    # ============================================================
    def _build_clustering_tab(self, parent):
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)

        # Sidebar
        sidebar = ctk.CTkFrame(parent, fg_color="#121212", corner_radius=10)
        sidebar.grid(row=0, column=0, sticky="ns", padx=(10, 5), pady=10)

        ctk.CTkLabel(
            sidebar,
            text="Unsupervised Clustering",
            text_color="#6ea8fe",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack(pady=(10, 5))

        ctk.CTkLabel(
            sidebar,
            text=(
                "Requires a CSV with a text column.\n"
                "Uses last vectorization settings\n"
                "to embed rows.\n\n"
                "Modes:\n"
                "- K-Means (animated)\n"
                "- K-Means (static sklearn)"
            ),
            text_color="#bbbbbb",
            font=ctk.CTkFont(size=11),
            justify="left",
        ).pack(padx=10, pady=(0, 10))

        self.clustering_vector_label = ctk.CTkLabel(
            sidebar,
            text=self._format_vector_settings(),
            text_color="#dddddd",
            font=ctk.CTkFont(size=11),
            justify="left",
        )
        self.clustering_vector_label.pack(padx=10, pady=(0, 10))

        # Mode selection
        ctk.CTkLabel(
            sidebar,
            text="Clustering mode:",
            text_color="#cccccc",
            font=ctk.CTkFont(size=13),
        ).pack(anchor="w", padx=10, pady=(5, 0))

        self.cluster_mode_combo = ctk.CTkComboBox(
            sidebar,
            values=["K-Means (animated)", "K-Means (static sklearn)"],
            state="readonly",
            width=220,
        )
        self.cluster_mode_combo.set("K-Means (animated)")
        self.cluster_mode_combo.pack(padx=10, pady=(2, 10))

        ctk.CTkLabel(
            sidebar,
            text="Number of clusters (k):",
            text_color="#cccccc",
            font=ctk.CTkFont(size=13),
        ).pack(anchor="w", padx=10, pady=(5, 0))

        self.cluster_k_entry = ctk.CTkEntry(sidebar, width=80)
        self.cluster_k_entry.insert(0, "3")
        self.cluster_k_entry.pack(anchor="w", padx=10, pady=(2, 10))

        self._clust_run_btn = ctk.CTkButton(
            sidebar,
            text="Run clustering",
            fg_color="#0078ff",
            hover_color="#005dc1",
            command=self.run_clustering,
        )
        self._clust_run_btn.pack(fill="x", padx=10, pady=(5, 10))
        self._clust_elbow_btn = ctk.CTkButton(
            sidebar,
            text="Elbow method (suggest k)",
            fg_color="#444444",
            hover_color="#333333",
            command=self.run_elbow_method,
        )
        self._clust_elbow_btn.pack(fill="x", padx=10, pady=(0, 10))
        self.clust_progress = ProgressOverlay(sidebar, task_runner=self.clust_runner)

        # Right side: summary + plot/animation
        right = ctk.CTkFrame(parent, fg_color="#1a1a1a", corner_radius=10)
        right.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        # Summary textbox (top)
        self.cluster_summary_box = ctk.CTkTextbox(
            right,
            fg_color="#1a1a1a",
            text_color="white",
            font=("Consolas", 11),
            wrap="word",
            height=120,
        )
        self.cluster_summary_box.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))

        # Plot/Canvas area (bottom)
        self.cluster_plot_container = ctk.CTkFrame(right, fg_color="#1a1a1a")
        self.cluster_plot_container.grid(row=1, column=0, sticky="nsew", padx=10, pady=(5, 10))

    def _clear_cluster_plot(self):
        if self.cluster_plot_canvas is not None:
            self.cluster_plot_canvas.get_tk_widget().destroy()
            self.cluster_plot_canvas = None
        if self.cluster_plot_figure is not None:
            plt.close(self.cluster_plot_figure)
            self.cluster_plot_figure = None
        if self.cluster_canvas is not None:
            self.cluster_canvas.destroy()
            self.cluster_canvas = None

    def run_clustering(self):
        if not enlp.is_csv_mode(self.state) or self.state.df is None:
            messagebox.showwarning(
                "Warning", "Please load a CSV file first in Document Tools."
            )
            return

        # Capture params before entering the thread
        vector_method = self.state.last_vector_method or "TF-IDF (word)"
        ngram = self.state.last_vector_ngram or (1, 2)
        max_feat = self.state.last_vector_max_features or 5000

        try:
            k = int(self.cluster_k_entry.get() or "3")
        except ValueError:
            k = 3
        if k < 2:
            k = 2

        mode = self.cluster_mode_combo.get()

        # Disable UI and show progress
        self._clust_run_btn.configure(state="disabled")
        self._clust_elbow_btn.configure(state="disabled")
        self.clust_progress.show("Building feature matrix…")

        def _work(progress_callback=None, cancel_event=None):
            if progress_callback:
                progress_callback(0.1, "Building feature matrix…")
            X, labels_for_display = unsup.build_X_from_state(
                self.state,
                vector_method=vector_method,
                ngram_range=ngram,
                max_features=max_feat,
            )
            if cancel_event is not None and cancel_event.is_set():
                from core.task_runner import CancelledError
                raise CancelledError()
            return X, labels_for_display, vector_method, ngram, max_feat, k, mode

        def _on_done(payload):
            X, labels_for_display, vm, ng, mf, _k, _mode = payload
            self.clust_progress.hide()
            self._clust_run_btn.configure(state="normal")
            self._clust_elbow_btn.configure(state="normal")
            # Summary header
            self.cluster_summary_box.delete("1.0", "end")
            self.cluster_summary_box.insert(
                "end",
                f"Vectorization method: {vm}, "
                f"ngram={ng}, max_features={mf}\n"
                f"Samples: {X.shape[0]}, Dim: {X.shape[1]}\n\n",
            )
            if _mode == "K-Means (static sklearn)":
                self._run_static_kmeans(X, labels_for_display, _k)
            else:
                self._run_animated_kmeans(X, _k)

        def _on_error(exc):
            self.clust_progress.hide()
            self._clust_run_btn.configure(state="normal")
            self._clust_elbow_btn.configure(state="normal")
            messagebox.showerror("Error building X", str(exc))

        def _on_cancel():
            self.clust_progress.hide()
            self._clust_run_btn.configure(state="normal")
            self._clust_elbow_btn.configure(state="normal")

        self.clust_runner.run(
            _work,
            on_done=_on_done,
            on_error=_on_error,
            on_cancel=_on_cancel,
        )
    def run_elbow_method(self):
        """
        Run K-Means for a range of k values and plot the elbow curve (k vs inertia).
        Uses the current vectorization settings and suggests a reasonable k by
        looking for where the relative improvement drops.
        """
        if not enlp.is_csv_mode(self.state) or self.state.df is None:
            messagebox.showwarning(
                "Warning", "Please load a CSV file first in Document Tools."
            )
            return

        # Capture params before entering the thread
        vector_method = self.state.last_vector_method or "TF-IDF (word)"
        ngram = self.state.last_vector_ngram or (1, 2)
        max_feat = self.state.last_vector_max_features or 5000

        # Disable UI and show progress
        self._clust_run_btn.configure(state="disabled")
        self._clust_elbow_btn.configure(state="disabled")
        self.clust_progress.show("Building feature matrix…")

        def _work(progress_callback=None, cancel_event=None):
            if progress_callback:
                progress_callback(0.05, "Building feature matrix…")
            X, _labels_for_display = unsup.build_X_from_state(
                self.state,
                vector_method=vector_method,
                ngram_range=ngram,
                max_features=max_feat,
            )
            if cancel_event is not None and cancel_event.is_set():
                from core.task_runner import CancelledError
                raise CancelledError()

            n_samples = X.shape[0]

            # Choose a reasonable k range: 2 .. min(10, n_samples-1)
            max_k = int(min(10, max(2, n_samples - 1)))
            k_values = list(range(2, max_k + 1))

            inertias = []
            total_k = len(k_values)
            for ki, k in enumerate(k_values):
                if cancel_event is not None and cancel_event.is_set():
                    from core.task_runner import CancelledError
                    raise CancelledError()
                if progress_callback:
                    progress_callback(0.1 + 0.9 * ki / max(total_k, 1), f"Fitting k={k}…")
                km = KMeans(
                    n_clusters=k,
                    random_state=42,
                    n_init="auto" if hasattr(KMeans, "n_init") else 10,
                )
                km.fit(X)
                inertias.append(float(km.inertia_))

            return X, n_samples, k_values, inertias, vector_method, ngram, max_feat

        def _on_done(payload):
            X, n_samples, k_values, inertias, vm, ng, mf = payload
            self.clust_progress.hide()
            self._clust_run_btn.configure(state="normal")
            self._clust_elbow_btn.configure(state="normal")
            if n_samples < 5:
                messagebox.showwarning(
                    "Warning",
                    "Not enough samples for a meaningful elbow curve (need at least ~5 rows).",
                )
                return
            self._display_elbow_results(n_samples, k_values, inertias, vm, ng, mf)

        def _on_error(exc):
            self.clust_progress.hide()
            self._clust_run_btn.configure(state="normal")
            self._clust_elbow_btn.configure(state="normal")
            messagebox.showerror("Elbow method error", str(exc))

        def _on_cancel():
            self.clust_progress.hide()
            self._clust_run_btn.configure(state="normal")
            self._clust_elbow_btn.configure(state="normal")

        self.clust_runner.run(
            _work,
            on_done=_on_done,
            on_error=_on_error,
            on_cancel=_on_cancel,
        )

    def _display_elbow_results(self, n_samples, k_values, inertias, vector_method, ngram, max_feat):
        """Display the elbow curve plot and summary after the k-loop completes."""
        # Clear old plot and summary, then show elbow info
        self._clear_cluster_plot()
        self.cluster_summary_box.delete("1.0", "end")
        self.cluster_summary_box.insert(
            "end",
            "Elbow method (K-Means inertia vs k)\n\n"
            f"Vectorization: {vector_method}, ngram={ngram}, max_features={max_feat}\n"
            f"Samples: {n_samples}\n\n"
        )

        # Plot k vs inertia
        self.cluster_plot_figure = plt.figure(figsize=(5, 4))
        ax = self.cluster_plot_figure.add_subplot(111)
        ax.plot(k_values, inertias, marker="o")
        ax.set_xlabel("Number of clusters k")
        ax.set_ylabel("Inertia (within-cluster SSE)")
        ax.set_title("Elbow curve")

        self.cluster_plot_canvas = FigureCanvasTkAgg(
            self.cluster_plot_figure, master=self.cluster_plot_container
        )
        self.cluster_plot_canvas.draw()
        self.cluster_plot_canvas.get_tk_widget().pack(fill="both", expand=True)

        # --- Simple heuristic to suggest k (where improvement starts to flatten) ---
        suggested_k = k_values[-1]  # default to max_k
        if len(inertias) >= 2:
            improvements = []
            for i in range(1, len(inertias)):
                prev = inertias[i - 1]
                curr = inertias[i]
                # relative improvement when going from k_{i-1} -> k_i
                rel = (prev - curr) / max(prev, 1e-9)
                improvements.append(rel)

            # Find first k where improvement drops below a small threshold (e.g. 10%)
            threshold = 0.10
            for idx, rel_imp in enumerate(improvements):
                if rel_imp < threshold:
                    suggested_k = k_values[idx]  # k at previous step
                    break

        self.cluster_summary_box.insert(
            "end",
            "Inertia values:\n" +
            "".join(f"  k={k}: {inertias[i]:.2f}\n" for i, k in enumerate(k_values))
        )
        self.cluster_summary_box.insert(
            "end",
            f"\nSuggested k (approx. elbow): {suggested_k}\n"
            "You can adjust this suggestion by eye from the curve above, then "
            "update the 'Number of clusters (k)' field and click 'Run clustering'.\n",
        )

        # Pre-fill k entry with suggested value
        self.cluster_k_entry.delete(0, "end")
        self.cluster_k_entry.insert(0, str(suggested_k))

    # ---------- Static K-Means using sklearn + PCA plot ---------- #
    def _run_static_kmeans(self, X: np.ndarray, labels_for_display, k: int):
        try:
            result = unsup.run_kmeans(X, n_clusters=k)
        except Exception as e:
            messagebox.showerror("Clustering error", str(e))
            return

        unique, counts = np.unique(result.labels, return_counts=True)
        self.cluster_summary_box.insert("end", f"Static K-Means with k={k}\n\n")
        self.cluster_summary_box.insert("end", "Cluster sizes:\n")
        for u, c in zip(unique, counts):
            self.cluster_summary_box.insert("end", f"  Cluster {u}: {c} samples\n")

        # Show first few rows per cluster
        max_examples_per_cluster = 3
        for u in unique:
            idxs = np.where(result.labels == u)[0]
            self.cluster_summary_box.insert("end", f"\nCluster {u} examples:\n")
            for idx in idxs[:max_examples_per_cluster]:
                self.cluster_summary_box.insert("end", f"  - {labels_for_display[idx]}\n")

        # Scatter via PCA
        self._plot_cluster_pca(X, result.labels)

    def _plot_cluster_pca(self, X: np.ndarray, cluster_labels: np.ndarray):
        self._clear_cluster_plot()
        if X.shape[0] < 2:
            return

        coords = vec.pca_2d(X)

        self.cluster_plot_figure = plt.figure(figsize=(5, 4))
        ax = self.cluster_plot_figure.add_subplot(111)

        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=cluster_labels)
        ax.set_title("K-Means Clusters (PCA 2D)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

        self.cluster_plot_canvas = FigureCanvasTkAgg(
            self.cluster_plot_figure, master=self.cluster_plot_container
        )
        self.cluster_plot_canvas.draw()
        self.cluster_plot_canvas.get_tk_widget().pack(fill="both", expand=True)

    # ---------- Animated K-Means on PCA coords using Canvas ---------- #
    def _run_animated_kmeans(self, X: np.ndarray, k: int):
        # Prepare PCA 2D
        if X.shape[0] < 2:
            messagebox.showwarning("Warning", "Not enough samples for PCA/animation.")
            return

        coords = vec.pca_2d(X)  # shape (n_samples, 2)
        self.cluster_summary_box.insert("end", f"Animated K-Means with k={k}\n\n")

        # Normalize coords to canvas space
        self._clear_cluster_plot()
        self.cluster_canvas = ctk.CTkCanvas(
            self.cluster_plot_container,
            width=600,
            height=450,
            bg="white",
            highlightthickness=1,
            highlightbackground="#444444",
        )
        self.cluster_canvas.pack(fill="both", expand=True, padx=10, pady=10)

        self.points = self._normalize_coords_to_canvas(coords, width=600, height=450)
        self.point_ids.clear()
        self.centroids.clear()
        self.centroid_ids.clear()
        self.clusters.clear()
        self.iteration = 0
        self.animating = True
        self.k = k

        # Draw points
        for x, y in self.points:
            pid = self.cluster_canvas.create_oval(
                x - 3, y - 3, x + 3, y + 3, fill="gray", outline=""
            )
            self.point_ids.append(pid)

        # Initialize centroids
        self.centroids = pick_centers(self.points, self.k)

        # Draw centroids
        for cx, cy in self.centroids:
            cid = self.cluster_canvas.create_oval(
                cx - 7, cy - 7, cx + 7, cy + 7,
                outline="black", width=2, fill=""
            )
            self.centroid_ids.append(cid)

        # Kick off first iteration
        self.after(100, self._kmeans_iteration_step)

    def _normalize_coords_to_canvas(self, coords: np.ndarray, width: int, height: int):
        xs = coords[:, 0]
        ys = coords[:, 1]
        min_x, max_x = float(xs.min()), float(xs.max())
        min_y, max_y = float(ys.min()), float(ys.max())
        pad = 20.0

        # Avoid division by zero
        range_x = max_x - min_x if max_x != min_x else 1.0
        range_y = max_y - min_y if max_y != min_y else 1.0

        scaled = []
        for x, y in coords:
            sx = pad + (x - min_x) / range_x * (width - 2 * pad)
            sy = pad + (y - min_y) / range_y * (height - 2 * pad)
            scaled.append([sx, sy])
        return scaled

    def _kmeans_iteration_step(self):
        if not self.animating:
            return

        if self.iteration >= self.max_iterations:
            self.cluster_summary_box.insert(
                "end",
                f"Reached max iterations ({self.max_iterations}). Stopping animation.\n",
            )
            self.animating = False
            return

        # Cluster assignment
        self.clusters = clusterize(self.points, self.centroids)

        # Compute new centroids
        new_centroids = update_centroids(self.clusters)
        self.old_centroids = self.centroids
        self.new_centroids = new_centroids

        # Recolor points by cluster
        self._color_points_by_cluster()

        # Animate centroid movement over several frames
        self.current_interp_step = 0
        self._animate_centroids()

    def _animate_centroids(self):
        if not self.animating:
            return

        steps = self.interpolate_steps
        t = self.current_interp_step / steps  # 0 → 1

        for i, cid in enumerate(self.centroid_ids):
            ox, oy = self.old_centroids[i]
            nx, ny = self.new_centroids[i]
            ix = ox + (nx - ox) * t
            iy = oy + (ny - oy) * t
            self.cluster_canvas.coords(cid, ix - 7, iy - 7, ix + 7, iy + 7)

        if self.current_interp_step < steps:
            self.current_interp_step += 1
            self.after(40, self._animate_centroids)
        else:
            # Finish iteration
            self.centroids = self.new_centroids
            self.iteration += 1

            if centroids_converged(self.old_centroids, self.new_centroids, self.tolerance):
                self.cluster_summary_box.insert(
                    "end", f"Converged after {self.iteration} iterations.\n"
                )
                self.animating = False
            else:
                self.cluster_summary_box.insert(
                    "end", f"Iteration {self.iteration} completed. Continuing...\n"
                )
                self.after(200, self._kmeans_iteration_step)

    def _color_points_by_cluster(self):
        point_to_cluster = {}
        for cluster_idx, pts in self.clusters.items():
            for p in pts:
                try:
                    idx = self.points.index(p)
                    point_to_cluster[idx] = cluster_idx
                except ValueError:
                    pass

        for idx, pid in enumerate(self.point_ids):
            c_idx = point_to_cluster.get(idx, None)
            if c_idx is None:
                color = "gray"
            else:
                color = self.cluster_colors[c_idx % len(self.cluster_colors)]
            self.cluster_canvas.itemconfig(pid, fill=color)

    #    TOPIC MODELING (BERTopic) SUBTAB
    # ============================================================
    #    TOPIC MODELING (BERTopic) SUBTAB
    # ============================================================

    def _build_topic_tab(self, parent):
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)

        # Sidebar
        sidebar = ctk.CTkFrame(parent, fg_color="#121212", corner_radius=10)
        sidebar.grid(row=0, column=0, sticky="ns", padx=(10, 5), pady=10)

        ctk.CTkLabel(
            sidebar,
            text="Topic Modeling (BERTopic)",
            text_color="#6ea8fe",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack(pady=(10, 5))

        ctk.CTkLabel(
            sidebar,
            text=(
                "Uses BERTopic on the current CSV text column.\n\n"
                "Steps:\n"
                "1) Load CSV in Document Tools\n"
                "2) Choose text column there\n"
                "3) Click 'Run BERTopic' below."
            ),
            text_color="#bbbbbb",
            font=ctk.CTkFont(size=11),
            justify="left",
        ).pack(padx=10, pady=(0, 10))

        self._topic_run_btn = ctk.CTkButton(
            sidebar,
            text="Run BERTopic",
            fg_color="#0078ff",
            hover_color="#005dc1",
            command=self.run_bertopic,
        )
        self._topic_run_btn.pack(fill="x", padx=10, pady=(5, 10))
        self.topic_progress = ProgressOverlay(sidebar, task_runner=self.topic_runner)

        # Right side: Summary + Plot
        right = ctk.CTkFrame(parent, fg_color="#1a1a1a", corner_radius=10)
        right.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        # TOPIC SUMMARY BOX
        self.topic_summary_box = ctk.CTkTextbox(
            right,
            fg_color="#1a1a1a",
            text_color="white",
            font=("Consolas", 11),
            wrap="word",
            height=160,
        )
        self.topic_summary_box.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))

        # Bind click
        self.topic_summary_box.bind("<Double-1>", self._show_topic_examples)

        # Plot area
        self.topic_plot_container = ctk.CTkFrame(right, fg_color="#1a1a1a")
        self.topic_plot_container.grid(row=1, column=0, sticky="nsew", padx=10, pady=(5, 10))

        self.topic_plot_canvas = None
        self.topic_plot_figure = None

    def _clear_topic_plot(self):
        if self.topic_plot_canvas is not None:
            self.topic_plot_canvas.get_tk_widget().destroy()
            self.topic_plot_canvas = None
        if self.topic_plot_figure is not None:
            plt.close(self.topic_plot_figure)
            self.topic_plot_figure = None

    def run_bertopic(self):
        if not enlp.is_csv_mode(self.state) or self.state.df is None:
            messagebox.showwarning("Warning", "Please load a CSV first in Document Tools.")
            return

        text_col = self.state.csv_text_column
        docs = self.state.df[text_col].dropna().astype(str).tolist()

        if len(docs) < 5:
            messagebox.showwarning("Warning", "Need at least 5 rows.")
            return

        # Disable UI and show progress
        self._topic_run_btn.configure(state="disabled")
        self.topic_progress.show("Running BERTopic…")

        def _work(progress_callback=None, cancel_event=None):
            vectorizer_model = CountVectorizer(stop_words="english")
            topic_model = BERTopic(vectorizer_model=vectorizer_model, verbose=False)
            topics, probs = topic_model.fit_transform(docs)
            topic_info = topic_model.get_topic_info()
            doc_info = topic_model.get_document_info(docs)
            return topic_model, docs, topic_info, doc_info

        def _on_done(payload):
            topic_model, _docs, topic_info, doc_info = payload
            self.topic_progress.hide()
            self._topic_run_btn.configure(state="normal")
            # Save data for later
            self.topic_model = topic_model
            self.docs = _docs
            self.topic_info = topic_info
            self.doc_info = doc_info
            self._display_bertopic_results(topic_model, topic_info)

        def _on_error(exc):
            self.topic_progress.hide()
            self._topic_run_btn.configure(state="normal")
            messagebox.showerror("BERTopic error", str(exc))

        def _on_cancel():
            self.topic_progress.hide()
            self._topic_run_btn.configure(state="normal")

        self.topic_runner.run(
            _work,
            on_done=_on_done,
            on_error=_on_error,
            on_cancel=_on_cancel,
        )

    def _display_bertopic_results(self, topic_model, topic_info):
        """Display topic summary and bar chart after BERTopic completes."""
        # TOPIC SUMMARY
        self.topic_summary_box.delete("1.0", "end")
        self.topic_summary_box.insert("end", "Top Topics (double-click to inspect):\n\n")

        top_info = topic_info[topic_info["Topic"] != -1]

        for _, row in top_info.iterrows():
            tid = row["Topic"]
            words = topic_model.get_topic(tid) or []
            top_words = ", ".join(w for w, _ in words[:5])
            self.topic_summary_box.insert("end", f"[Topic {tid}] ({row['Count']} docs): {top_words}\n")

        # Plot topic sizes
        self._clear_topic_plot()
        self.topic_plot_figure = plt.figure(figsize=(6, 4))
        ax = self.topic_plot_figure.add_subplot(111)

        ax.bar(top_info["Topic"].astype(str), top_info["Count"])
        ax.set_xlabel("Topic ID")
        ax.set_ylabel("Document count")
        ax.set_title("BERTopic: Topic Sizes")

        self.topic_plot_canvas = FigureCanvasTkAgg(self.topic_plot_figure, master=self.topic_plot_container)
        self.topic_plot_canvas.draw()
        self.topic_plot_canvas.get_tk_widget().pack(fill="both", expand=True)

    # -----------------------------------------------------------
    # NEW: Clicking a topic shows example documents
    # -----------------------------------------------------------
    def _show_topic_examples(self, event):
        """Triggered when user double-clicks the topic summary box."""
        try:
            index = self.topic_summary_box.index("@%s,%s" % (event.x, event.y))
            line = self.topic_summary_box.get(index + " linestart", index + " lineend")
        except:
            return

        if "Topic" not in line:
            return

        # Extract topic id
        try:
            tid = int(line.split("]")[0].split(" ")[1])
        except:
            return

        # Get top docs
        rows = self.doc_info[self.doc_info["Topic"] == tid].head(5)

        # Popup window
        win = ctk.CTkToplevel(self)
        win.title(f"Topic {tid} – Example Documents")
        win.geometry("650x500")

        txt = ctk.CTkTextbox(win, fg_color="#101010", text_color="white", wrap="word")
        txt.pack(fill="both", expand=True, padx=10, pady=10)

        # Add terms
        txt.insert("end", f"Top words for Topic {tid}:\n")
        for w, _ in self.topic_model.get_topic(tid)[:10]:
            txt.insert("end", f"- {w}\n")

        txt.insert("end", "\nExample documents:\n\n")
        for _, row in rows.iterrows():
            txt.insert("end", f"• {row['Document'][:300]}...\n\n")

        # OPTIONAL: show term bar chart
        self._plot_topic_terms(tid)

    # -----------------------------------------------------------
    # NEW: Show bar chart for topic terms
    # -----------------------------------------------------------
    def _plot_topic_terms(self, topic_id):
        """Plot top 10 terms for a topic."""
        words = self.topic_model.get_topic(topic_id)
        if not words:
            return

        terms = [w for w, _ in words[:10]]
        scores = [s for _, s in words[:10]]

        self._clear_topic_plot()

        self.topic_plot_figure = plt.figure(figsize=(6, 4))
        ax = self.topic_plot_figure.add_subplot(111)

        ax.barh(terms, scores)
        ax.set_title(f"Top Terms for Topic {topic_id}")
        ax.invert_yaxis()

        self.topic_plot_canvas = FigureCanvasTkAgg(self.topic_plot_figure, master=self.topic_plot_container)
        self.topic_plot_canvas.draw()
        self.topic_plot_canvas.get_tk_widget().pack(fill="both", expand=True)
