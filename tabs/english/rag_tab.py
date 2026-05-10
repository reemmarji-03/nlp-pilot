import customtkinter as ctk
from tkinter import messagebox
from threading import Thread

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from core.settings_manager import settings
import time


class RAGTab(ctk.CTkFrame):
    def __init__(self, parent, state):
        super().__init__(parent, fg_color="#0f0f0f")
        self.state = state

        self.db = None
        self.rag_chain = None

        self.pack(fill="both", expand=True)
        self._build_ui()


    # -------------------------------------------------------------
    # UI BUILDING
    # -------------------------------------------------------------
    def _build_ui(self):
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)

        # ============ SIDEBAR =============
        sidebar = ctk.CTkFrame(self, fg_color="#121212", corner_radius=10)
        sidebar.grid(row=0, column=0, sticky="ns", padx=10, pady=10)

        ctk.CTkLabel(
            sidebar,
            text="RAG Assistant",
            text_color="#6ea8fe",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=(15, 20))

        self._add_button(sidebar, "📄 Build Index", self._build_index)
        self._add_button(sidebar, "💬 Ask Question", self._toggle_chat)


        # ============ INDEX LOG BOX ============
        self.index_log = ctk.CTkTextbox(
            self, fg_color="#1a1a1a", text_color="white", wrap="word"
        )
        self.index_log.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)


        # ============ CHAT BOX (initially empty) ============
        self.chat_box = ctk.CTkTextbox(
            self, fg_color="#101010", text_color="white", wrap="word"
        )
        self.chat_box.grid(row=0, column=2, sticky="nsew", padx=10, pady=10)

        # ============ INPUT FIELD (HIDDEN INITIALLY) ============
        self.entry = ctk.CTkEntry(self, placeholder_text="Ask something...")
        self.entry.grid(row=1, column=2, sticky="ew", padx=10, pady=(0, 10))
        self.entry.bind("<Return>", lambda e: self._ask())
        self.entry.grid_remove()  

         # ============ LOADING LABEL (HIDDEN INITIALLY) ============
        self.loading_label = ctk.CTkLabel(
            self,
            text="",
            text_color="#6ea8fe",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.loading_label.grid(row=2, column=2, sticky="w", padx=10)
        self.loading_label.grid_remove()   



    def _add_button(self, parent, text, command):
        ctk.CTkButton(
            parent,
            text=text,
            command=command,
            fg_color="#0078ff",
            hover_color="#005dc1",
            corner_radius=6,
            height=38,
            font=ctk.CTkFont(size=14),
        ).pack(fill="x", padx=10, pady=6)


    # -------------------------------------------------------------
    # BUILD INDEX
    # -------------------------------------------------------------
    def _build_index(self):
        text = self.state.text
        if not text:
            return messagebox.showwarning("Warning", "Upload a document first.")

        self.index_log.delete("1.0", "end")
        self.index_log.insert("end", "⚙️ Starting index build...\n")

        def worker():
            try:
                # Split
                self.index_log.insert("end", "🔪 Splitting document...\n")
                splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
                docs = splitter.create_documents([text])
                self.index_log.insert("end", f"📄 {len(docs)} chunks created.\n")

                # Embeddings
                self.index_log.insert("end", "🧠 Loading embeddings...\n")
                embed = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )

                # Vector DB
                self.index_log.insert("end", "🔍 Building FAISS index...\n")
                self.db = FAISS.from_documents(docs, embed)

                # LLM
                self.index_log.insert("end", "Loading configured LLM provider...\n")
                llm = settings.get_llm()
                if llm is None:
                    self.index_log.insert(
                        "end",
                        "No LLM provider configured. Open Settings to select Ollama, OpenAI, or Anthropic.\n",
                    )
                    return

                # Build prompt
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
                        "question": RunnablePassthrough()
                    }
                    | prompt
                    | llm
                )

                self.index_log.insert("end", "🎉 Index built successfully!\n")

            except Exception as e:
                self.index_log.insert("end", f"❌ ERROR: {str(e)}\n")

        Thread(target=worker, daemon=True).start()


    # -------------------------------------------------------------
    # ENTER ASK MODE
    # -------------------------------------------------------------
    def _toggle_chat(self):
        self.chat_box.delete("1.0", "end")
        self.chat_box.insert("end", "💬 Ask your questions now.\n")
        self.chat_box.see("end")

        # Show input field
        self.entry.grid()


    def _animate_loading(self):
        dots = ["", ".", "..", "..."]
        i = 0
        while self._loading:
            self.loading_label.configure(text=f"Thinking{dots[i % 4]}")
            i += 1
            self.loading_label.update_idletasks()
            time.sleep(0.4)
        self.loading_label.configure(text="")


    # -------------------------------------------------------------
    # ASK QUESTION
    # -------------------------------------------------------------
    def _ask(self):
        query = self.entry.get().strip()
        if not query:
            return
        if not self.rag_chain:
            return self.chat_box.insert("end", "⚠ Please build the index first.\n")

        self.chat_box.insert("end", f"\n🧠 Q: {query}\n")
        self._loading = True
        self.loading_label.grid() 
        Thread(target=self._animate_loading, daemon=True).start()


        def worker():
            try:
                answer = self.rag_chain.invoke(query)
                self.chat_box.insert("end", f"→ {answer}\n")
                self.entry.delete(0, "end")
            except Exception as e:
                self.chat_box.insert("end", f"❌ ERROR: {str(e)}\n")

            finally:
                self._loading = False
                self.loading_label.grid_remove()
                self.entry.delete(0, "end")

        Thread(target=worker, daemon=True).start()
