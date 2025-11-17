import customtkinter as ctk
from tkinter import messagebox
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from threading import Thread


class RAGTab(ctk.CTkFrame):
    def __init__(self, parent, state):
        super().__init__(parent, fg_color="#0f0f0f")
        self.state = state
        self.db = None
        self.qa_chain = None
        self._build_ui()

    # ======================================
    # UI
    # ======================================
    def _build_ui(self):
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        sidebar = ctk.CTkFrame(self, fg_color="#121212", corner_radius=10)
        sidebar.grid(row=0, column=0, sticky="ns", padx=(10,5), pady=10)

        ctk.CTkLabel(sidebar, text="RAG Assistant",
                     text_color="#6ea8fe",
                     font=ctk.CTkFont(size=18, weight="bold")).pack(pady=(15,20))

        self._add_button(sidebar, "📄 Build Index", self._build_index)
        self._add_button(sidebar, "💬 Ask", self._toggle_chat)

        self.chat_box = ctk.CTkTextbox(self, fg_color="#1a1a1a", text_color="white", wrap="word")
        self.chat_box.grid(row=0, column=1, sticky="nsew", padx=(5,10), pady=10)

        self.entry = ctk.CTkEntry(self, placeholder_text="Ask something about your document...")
        self.entry.grid(row=1, column=1, sticky="ew", padx=(5,10), pady=(0,10))
        self.entry.bind("<Return>", lambda e: self._ask())

    def _add_button(self, parent, text, command):
        ctk.CTkButton(parent, text=text, command=command,
                      fg_color="#0078ff", hover_color="#005dc1",
                      corner_radius=6, height=38,
                      font=ctk.CTkFont(size=14)).pack(fill="x", padx=10, pady=6)

    # ======================================
    # INDEX BUILDING
    # ======================================
    def _build_index(self):
        text = self.state.text
        if not text:
            return messagebox.showwarning("Warning", "Upload a document first in Document Tools.")
        self.chat_box.insert("end", "⚙️ Building index...\n")
        self.update()

        def worker():
            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
            docs = splitter.create_documents([text])
            embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            self.db = FAISS.from_documents(docs, embedder)
            self.chat_box.insert("end", "✅ Index built successfully.\n")

        Thread(target=worker, daemon=True).start()

    # ======================================
    # CHAT
    # ======================================
    def _toggle_chat(self):
        self.chat_box.insert("end", "💬 You can now ask questions below.\n")

    def _ask(self):
        query = self.entry.get().strip()
        if not query:
            return
        if not self.db:
            return self.chat_box.insert("end", "⚠️ Please build the index first.\n")

        self.chat_box.insert("end", f"\n🧠 Q: {query}\n", "left")
        self.update()

        def worker():
            llm = OpenAI(temperature=0)
            retriever = self.db.as_retriever(search_kwargs={"k": 3})
            qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
            answer = qa.run(query)
            self.chat_box.insert("end", f"→ {answer}\n\n")
            self.entry.delete(0, "end")

        Thread(target=worker, daemon=True).start()
