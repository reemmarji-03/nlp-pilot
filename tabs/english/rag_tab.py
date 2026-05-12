from sys import prefix
from threading import Thread

import customtkinter as ctk
from tkinter import messagebox

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core import english_nlp as enlp
from core.settings_manager import settings


class RAGTab(ctk.CTkFrame):
    def __init__(self, parent, state):
        super().__init__(parent, fg_color="#0f0f0f")
        self.state = state

        self.db = None
        self.rag_chain = None
        self._indexing = False
        self._index_signature = None
        self._chain_signature = None

        self.pack(fill="both", expand=True)
        self._build_ui()

    def _build_ui(self):
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)

        sidebar = ctk.CTkFrame(self, fg_color="#121212", corner_radius=10)
        sidebar.grid(row=0, column=0, sticky="ns", padx=10, pady=10)

        ctk.CTkLabel(
            sidebar,
            text="RAG Assistant",
            text_color="#6ea8fe",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack(pady=(15, 20))

        self.rebuild_btn = ctk.CTkButton(
            sidebar,
            text="Rebuild Index",
            command=lambda: self._build_index(force=True),
            fg_color="#0078ff",
            hover_color="#005dc1",
            corner_radius=6,
            height=38,
            font=ctk.CTkFont(size=14),
        )
        self.rebuild_btn.pack(fill="x", padx=10, pady=6)

        self.status_label = ctk.CTkLabel(
            sidebar,
            text="Open RAG after loading a document.",
            text_color="#aaaaaa",
            wraplength=180,
            justify="left",
        )
        self.status_label.pack(fill="x", padx=10, pady=(8, 0))

        self.index_log = ctk.CTkTextbox(
            self, fg_color="#1a1a1a", text_color="white", wrap="word"
        )
        self.index_log.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        self.chat_frame = ctk.CTkScrollableFrame(
            self,
            fg_color="#101010",
        )
        self.chat_frame.grid(row=0, column=2, sticky="nsew", padx=10, pady=10)
        self.add_message("ai", "RAG chat will be ready after the index is built.")

        self.entry = ctk.CTkEntry(self, placeholder_text="Ask something...")
        self.entry.grid(row=1, column=2, sticky="ew", padx=10, pady=(0, 10))
        self.entry.bind("<Return>", lambda e: self._ask())
        self.entry.configure(state="disabled")

    def on_show(self):
        if enlp.is_csv_mode(self.state):
            self.db = None
            self.rag_chain = None
            self._index_signature = None
            self._chain_signature = None
            self._set_status("RAG works with text documents only.")
            self._set_chat_ready(False)
            self.clear_chat("RAG is available for text documents only. Load a TXT, PDF, or DOCX file in Document Tools.")
            self.index_log.delete("1.0", "end")
            self.index_log.insert("end", "CSV input detected. RAG indexing is disabled for CSV rows.\n")
            return
        self._build_index(force=False)

    def invalidate_index(self):
        self.db = None
        self.rag_chain = None
        self._index_signature = None
        self._chain_signature = None
        self.entry.configure(state="disabled")
        self.clear_chat("RAG chat will be ready after the index is built.")

    def refresh_provider(self):
        self.rag_chain = None
        self._chain_signature = None
        if self.db is not None:
            self._ensure_chain_current()

    def _active_docs(self) -> list[str]:
        if enlp.is_csv_mode(self.state):
            return []
        return enlp.get_preprocessed_documents_from_state(self.state)

    def _document_signature(self):
        docs = self._active_docs()
        if not docs:
            return None
        return (
            self.state.file_name,
            self.state.csv_text_column,
            len(docs),
            sum(len(d) for d in docs),
            hash("\n---\n".join(docs[:20])),
        )

    def _settings_signature(self):
        provider = settings.data.get("active_provider", "ollama")
        cfg = settings.data.get(provider, {})
        return provider, cfg.get("url"), cfg.get("model"), bool(cfg.get("api_key"))

    def _log(self, message: str):
        def write():
            if self.winfo_exists():
                self.index_log.insert("end", message)
                self.index_log.see("end")

        self.after(0, write)

    def _set_status(self, text: str):
        self.after(0, lambda: self.status_label.configure(text=text))

    def _set_chat_ready(self, ready: bool):
        def update():
            if self.winfo_exists():
                self.entry.configure(state="normal" if ready else "disabled")

        self.after(0, update)

    def add_message(self, role: str, text: str):
        container = ctk.CTkFrame(self.chat_frame, fg_color="transparent")
        container.pack(fill="x", pady=4, padx=4)

        if role == "user":
            bubble_color = "#175ea8"
            justify = "right"
            container.columnconfigure(0, weight=1)
            column = 1
            sticky = "e"
            padx = (40, 0)
        else:
            bubble_color = "#242424"
            justify = "left"
            container.columnconfigure(1, weight=1)
            column = 0
            sticky = "w"
            padx = (0, 40)

        bubble = ctk.CTkFrame(container, fg_color=bubble_color, corner_radius=10)
        bubble.grid(row=0, column=column, sticky=sticky, padx=padx)

        label = ctk.CTkLabel(
            bubble,
            text=f"{text}",
            text_color="white",
            font=("Consolas", 11),
            justify=justify,
            wraplength=600,
        )
        label.pack(padx=8, pady=6)
        self.after(50, self.scroll_to_bottom)
        return label

    def clear_chat(self, message: str | None = None):
        for child in self.chat_frame.winfo_children():
            child.destroy()
        if message:
            self.add_message("ai", message)

    def update_message(self, label, role: str, text: str):
        label.configure(text=f"{text}")
        self.after(50, self.scroll_to_bottom)

    def scroll_to_bottom(self):
        try:
            self.chat_frame._parent_canvas.yview_moveto(1.0)
        except Exception:
            pass

    def _build_index(self, force: bool = False):
        if enlp.is_csv_mode(self.state):
            if force:
                messagebox.showwarning(
                    "RAG unavailable",
                    "RAG only works with text documents. Load a TXT, PDF, or DOCX file first.",
                )
            self.db = None
            self.rag_chain = None
            self._index_signature = None
            self._chain_signature = None
            self._set_status("RAG works with text documents only.")
            self._set_chat_ready(False)
            self.clear_chat("RAG is available for text documents only. Load a TXT, PDF, or DOCX file in Document Tools.")
            return

        docs = self._active_docs()
        signature = self._document_signature()
        if not docs or signature is None:
            if force:
                messagebox.showwarning("Warning", "Upload a document first.")
            self._set_status("No document loaded.")
            self._set_chat_ready(False)
            return

        if self._indexing:
            return

        if not force and self.db is not None and self._index_signature == signature:
            self._ensure_chain_current()
            return

        self._indexing = True
        self.rebuild_btn.configure(state="disabled")
        self._set_chat_ready(False)
        self.index_log.delete("1.0", "end")
        self.index_log.insert("end", "Starting RAG index build...\n")
        self._set_status("Building index...")

        def worker():
            try:
                self._log("Splitting documents...\n")
                splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
                chunks = splitter.create_documents(docs)
                self._log(f"{len(chunks)} chunks created.\n")

                self._log("Loading embeddings...\n")
                embed = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )

                self._log("Building FAISS index...\n")
                self.db = FAISS.from_documents(chunks, embed)
                self._index_signature = signature
                self.rag_chain = None
                self._chain_signature = None

                self._log("Index built successfully.\n")
                self._ensure_chain_current()
            except Exception as exc:
                self._log(f"ERROR: {exc}\n")
                self._set_status("Index build failed.")
                self._set_chat_ready(False)
            finally:
                self._indexing = False
                self.after(0, lambda: self.rebuild_btn.configure(state="normal"))

        Thread(target=worker, daemon=True).start()

    def _ensure_chain_current(self) -> bool:
        if self.db is None:
            self._set_status("Build the index first.")
            self._set_chat_ready(False)
            return False

        signature = self._settings_signature()
        if self.rag_chain is not None and self._chain_signature == signature:
            self._set_status("Ready.")
            self._set_chat_ready(True)
            return True

        llm = settings.get_llm()
        if llm is None:
            self._log(
                "No LLM provider configured. Open Settings to select Ollama, OpenAI, or Anthropic.\n"
            )
            self._set_status("Configure an LLM provider in Settings.")
            self._set_chat_ready(False)
            return False

        self._log(f"Using provider/model: {signature[0]} / {signature[2]}\n")
        prompt = ChatPromptTemplate.from_template(
            """You are a helpful assistant. Use ONLY the context below:

{context}

Question: {question}
Answer:"""
        )
        retriever = self.db.as_retriever(search_kwargs={"k": 3})
        self.rag_chain = (
            {
                "context": retriever
                | (lambda docs: "\n\n".join([d.page_content for d in docs])),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        self._chain_signature = signature
        self._set_status("Ready.")
        self._set_chat_ready(True)
        self._log("RAG chat ready.\n")
        return True

    def _extract_answer_and_metadata(self, response):
        content = getattr(response, "content", response)
        if not isinstance(content, str):
            content = str(content)

        metadata = {
            "response_metadata": getattr(response, "response_metadata", None),
            "usage_metadata": getattr(response, "usage_metadata", None),
            "id": getattr(response, "id", None),
        }
        metadata = {k: v for k, v in metadata.items() if v}
        return content, metadata

    def _ask(self):
        query = self.entry.get().strip()
        if not query:
            return
        if not self._ensure_chain_current():
            self.add_message("ai", "RAG is not ready yet.")
            return

        self.entry.delete(0, "end")
        self.add_message("user", query)
        self.entry.configure(state="disabled")
        thinking_label = self.add_message("ai", "Thinking...")

        def worker():
            try:
                response = self.rag_chain.invoke(query)
                answer, metadata = self._extract_answer_and_metadata(response)

                def show_answer():
                    self.update_message(thinking_label, "ai", answer)

                self.after(0, show_answer)
                if metadata:
                    self._log(f"Last response metadata: {metadata}\n")
            except Exception as exc:
                self.after(
                    0,
                    lambda e=exc: self.update_message(thinking_label, "ai", f"ERROR: {e}"),
                )
            finally:
                self.after(0, lambda: self.entry.configure(state="normal"))

        Thread(target=worker, daemon=True).start()
