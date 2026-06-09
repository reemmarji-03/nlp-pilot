import customtkinter as ctk

from core.english_state import EnglishState
from .agent_tab import AgentTab
from .document_tools_tab import DocumentToolsTab
from .preprocess_tab import PreprocessTab
from .vector_tab import VectorTab
from .prediction_tab import PredictionTab
from .rag_tab import RAGTab


class EnglishTab(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent, fg_color=("#f0f0f0", "#0f0f0f"))
        self.state = EnglishState()
        self.build_ui()

    def build_ui(self):
        self.section_tabs = ctk.CTkTabview(self, command=self.on_tab_selected)
        self.section_tabs.pack(fill="both", expand=True, padx=10, pady=10)

        doc_frame = self.section_tabs.add("Document Tools")
        prep_frame = self.section_tabs.add("Preprocessing")
        vect_frame = self.section_tabs.add("Vectorization")
        prediction_frame = self.section_tabs.add("Prediction")
        rag_frame = self.section_tabs.add("RAG")
        agent_frame = self.section_tabs.add("Agent Lab")

        self.doc_tab = DocumentToolsTab(
            doc_frame, self.state, on_state_changed=self.on_state_changed
        )
        self.prep_tab = PreprocessTab(prep_frame, self.state)
        self.vect_tab = VectorTab(vect_frame, self.state)
        self.prediction_tab = PredictionTab(prediction_frame, self.state)
        self.rag_tab = RAGTab(rag_frame, self.state)
        self.agent_tab = AgentTab(agent_frame, self.state)

    def on_tab_selected(self):
        if self.section_tabs.get() == "RAG":
            self.rag_tab.on_show()

    def on_state_changed(self):
        self.prediction_tab.sync_with_state()
        if hasattr(self, "rag_tab"):
            self.rag_tab.invalidate_index()

    def apply_theme(self):
        if hasattr(self, "prep_tab"):
            self.prep_tab.apply_theme()

    def on_settings_changed(self):
        if hasattr(self, "rag_tab"):
            self.rag_tab.refresh_provider()
