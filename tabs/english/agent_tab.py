import customtkinter as ctk

from core.english_state import EnglishState
from core.agent_graph import build_agent_graph, AgentState
from core.task_runner import TaskRunner


class AgentTab(ctk.CTkFrame):
    """
    A conversational tab that uses LangGraph to orchestrate actions
    (preprocessing → vectorization → supervised classification)
    based on user instructions.
    """

    def __init__(self, parent, state: EnglishState):
        super().__init__(parent, fg_color="#0f0f0f")
        self.state = state

        # LangGraph app
        self.graph = build_agent_graph()

        # We keep phase/task_type/decisions across turns in this object
        self.current_phase = {"stage": "task_selection"}
        self.current_task_type = None
        self.decisions = {}

        self.runner = TaskRunner(root=self)
        self._send_btn = None   # set in _build_ui

        self._build_ui()
        self.pack(fill="both", expand=True)

        # Initial greeting from the agent
        self.agent_greeting()

    # ================== UI BUILDING ================== #

    def _build_ui(self):
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)

        # Top bar: title + current stage
        top_frame = ctk.CTkFrame(self, fg_color="transparent")
        top_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 0))
        top_frame.columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            top_frame,
            text="Agent Lab (LangGraph)",
            text_color="#6ea8fe",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        title.grid(row=0, column=0, sticky="w")

        self.phase_label = ctk.CTkLabel(
            top_frame,
            text="Stage: task_selection",
            text_color="#aaaaaa",
            font=ctk.CTkFont(size=12),
        )
        self.phase_label.grid(row=0, column=1, sticky="e", padx=(10, 0))

        # Chat area: scrollable frame with bubbles
        self.chat_frame = ctk.CTkScrollableFrame(
            self,
            fg_color="#101010",
        )
        self.chat_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(5, 5))

        # Input + send + reset
        input_frame = ctk.CTkFrame(self, fg_color="#121212")
        input_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))
        input_frame.columnconfigure(0, weight=1)

        self.input_box = ctk.CTkTextbox(
            input_frame,
            height=50,
            fg_color="#1a1a1a",
            text_color="white",
            font=("Consolas", 11),
            wrap="word",
        )
        self.input_box.grid(row=0, column=0, sticky="ew", padx=(5, 5), pady=5)

        btn_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
        btn_frame.grid(row=0, column=1, sticky="ns", padx=(0, 5), pady=5)

        self._send_btn = ctk.CTkButton(
            btn_frame,
            text="Send",
            fg_color="#0078ff",
            hover_color="#005dc1",
            command=self.on_send,
            width=70,
        )
        self._send_btn.pack(side="top", fill="x", pady=(0, 5))

        reset_btn = ctk.CTkButton(
            btn_frame,
            text="Reset",
            fg_color="#333333",
            hover_color="#444444",
            command=self.reset_agent,
            width=70,
        )
        reset_btn.pack(side="top", fill="x")

    # ================== MESSAGE HELPERS ================== #

    def add_message(self, role: str, text: str):
        """
        Add a chat bubble to the scrollable chat frame.
        role: "user" or "agent"
        """
        # Container frame to control alignment
        container = ctk.CTkFrame(self.chat_frame, fg_color="transparent")
        container.pack(fill="x", pady=4, padx=4)

        # Bubble styling
        if role == "user":
            bubble_color = "#175ea8"  # blue-ish
            anchor = "e"
            justify = "right"
            text_prefix = "You"
        else:
            bubble_color = "#242424"  # dark gray
            anchor = "w"
            justify = "left"
            text_prefix = "Agent"

        bubble = ctk.CTkFrame(
            container,
            fg_color=bubble_color,
            corner_radius=10,
        )

        # Use grid to align bubble right/left inside container
        if role == "user":
            container.columnconfigure(0, weight=1)
            bubble.grid(row=0, column=1, sticky="e", padx=(40, 0))
        else:
            container.columnconfigure(1, weight=1)
            bubble.grid(row=0, column=0, sticky="w", padx=(0, 40))

        label = ctk.CTkLabel(
            bubble,
            text=f"{text_prefix}:\n{text}",
            text_color="white",
            font=("Consolas", 11),
            justify=justify,
            wraplength=600,
        )
        label.pack(padx=8, pady=6)

        # Autoscroll to bottom after a short delay
        self.after(50, self.scroll_to_bottom)

    def scroll_to_bottom(self):
        try:
            # CTkScrollableFrame uses an internal canvas
            self.chat_frame._parent_canvas.yview_moveto(1.0)
        except Exception:
            pass

    def update_phase_label(self):
        stage = self.current_phase.get("stage", "task_selection")
        self.phase_label.configure(text=f"Stage: {stage}")

    # ================== AGENT CONTROL ================== #

    def agent_greeting(self):
        """
        Run a first turn where the agent greets the user and explains what it can do.
        We fake a simple 'start' message from the user to trigger task_selection_node.
        """
        agent_state: AgentState = {
            "user_message": "Hey!",  # simple placeholder
            "assistant_message": "",
            "english_state": self.state,
            "phase": self.current_phase,
            "task_type": self.current_task_type,
            "decisions": self.decisions,
        }

        def _work():
            return self.graph.invoke(agent_state)

        def _on_done(new_state):
            self.state = new_state["english_state"]
            self.current_phase = new_state.get("phase", self.current_phase)
            self.current_task_type = new_state.get("task_type", self.current_task_type)
            self.decisions = new_state.get("decisions", self.decisions)
            self.update_phase_label()
            msg = new_state.get("assistant_message", "").strip()
            if msg:
                self.add_message("agent", msg)

        def _on_error(exc):
            self.add_message("agent", f"Could not start agent:\n{exc}")

        self.runner.run(_work, on_done=_on_done, on_error=_on_error)

    def reset_agent(self):
        """
        Reset the agent's conversation state, but keep the loaded EnglishState.
        Optionally also clear some configs in EnglishState.
        """
        if self.runner.is_running:
            return
        # Reset conversation-level state
        self.current_phase = {"stage": "task_selection"}
        self.current_task_type = None
        self.decisions = {}

        # Optionally clear some config in EnglishState (defensive checks)
        if hasattr(self.state, "pipeline_config"):
            self.state.pipeline_config = {}
        if hasattr(self.state, "last_vector_method"):
            self.state.last_vector_method = None
        if hasattr(self.state, "last_vector_ngram"):
            self.state.last_vector_ngram = None
        if hasattr(self.state, "last_vector_max_features"):
            self.state.last_vector_max_features = None

        # Clear chat
        for child in self.chat_frame.winfo_children():
            child.destroy()

        self.update_phase_label()

        # Re-greet the user
        self.agent_greeting()

    # ================== EVENT HANDLERS ================== #

    def on_send(self):
        user_msg = self.input_box.get("1.0", "end").strip()
        if not user_msg:
            return
        if self.runner.is_running:
            return

        self.input_box.delete("1.0", "end")
        self.add_message("user", user_msg)
        self._send_btn.configure(state="disabled", text="…")

        agent_state: AgentState = {
            "user_message": user_msg,
            "assistant_message": "",
            "english_state": self.state,
            "phase": self.current_phase,
            "task_type": self.current_task_type,
            "decisions": self.decisions,
        }

        def _work():
            return self.graph.invoke(agent_state)

        def _on_done(new_state):
            self._send_btn.configure(state="normal", text="Send")
            self.state = new_state["english_state"]
            self.current_phase = new_state.get("phase", self.current_phase)
            self.current_task_type = new_state.get("task_type", self.current_task_type)
            self.decisions = new_state.get("decisions", self.decisions)
            self.update_phase_label()
            msg = new_state.get("assistant_message", "").strip()
            if msg:
                self.add_message("agent", msg)
            else:
                self.add_message("agent", "(no response)")

        def _on_error(exc):
            self._send_btn.configure(state="normal", text="Send")
            self.add_message("agent", f"Oops, the agent crashed:\n`{exc}`")

        self.runner.run(_work, on_done=_on_done, on_error=_on_error)
