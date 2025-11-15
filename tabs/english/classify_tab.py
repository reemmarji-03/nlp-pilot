import customtkinter as ctk
from core.english_state import EnglishState


class ClassifyTab(ctk.CTkFrame):
    def __init__(self, parent, state: EnglishState):
        super().__init__(parent, fg_color="#0f0f0f")
        self.state = state

        self.build_ui()
        self.pack(fill="both", expand=True)

    def build_ui(self):
        frame = ctk.CTkFrame(self, fg_color="#121212", corner_radius=10)
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(
            frame,
            text="Text Classification Playground (coming soon)",
            text_color="#6ea8fe",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=(15, 10))

        ctk.CTkLabel(
            frame,
            text=(
                "Here you'll be able to load labeled datasets,\n"
                "train classifiers (Naive Bayes, Logistic\n"
                "Regression, SVM, etc.), and see metrics like\n"
                "accuracy, F1-score, and confusion matrix."
            ),
            justify="left",
            text_color="#cccccc"
        ).pack(padx=10, pady=10, anchor="nw")
