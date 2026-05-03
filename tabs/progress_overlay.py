from __future__ import annotations
from typing import TYPE_CHECKING

import customtkinter as ctk

if TYPE_CHECKING:
    from core.task_runner import TaskRunner


class ProgressOverlay(ctk.CTkFrame):
    """
    Inline progress widget. Call show() to pack it, hide() to unpack.
    Starts hidden — do not pack it manually.
    """

    def __init__(self, parent, task_runner: "TaskRunner"):
        super().__init__(parent, fg_color="#1e1e1e", corner_radius=8)
        self._runner = task_runner
        self._build()

    def _build(self) -> None:
        self._bar = ctk.CTkProgressBar(self, width=200)
        self._bar.pack(padx=12, pady=(10, 4))
        self._bar.set(0)

        self._label = ctk.CTkLabel(
            self,
            text="Working...",
            text_color="#aaaaaa",
            font=ctk.CTkFont(size=11),
        )
        self._label.pack(padx=12, pady=(0, 4))

        ctk.CTkButton(
            self,
            text="Cancel",
            width=80,
            height=26,
            fg_color="#555555",
            hover_color="#444444",
            command=self._runner.cancel,
        ).pack(pady=(0, 10))

    def show(self, message: str = "Working...") -> None:
        self._label.configure(text=message)
        self._bar.configure(mode="indeterminate")
        self._bar.start()
        self.pack(fill="x", padx=10, pady=(0, 8))

    def update(self, percent: float, message: str) -> None:
        self._label.configure(text=message)
        if self._bar.cget("mode") == "indeterminate":
            self._bar.stop()
            self._bar.configure(mode="determinate")
        self._bar.set(max(0.0, min(1.0, percent)))

    def hide(self) -> None:
        self._bar.stop()
        self.pack_forget()
