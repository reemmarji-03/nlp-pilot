import customtkinter as ctk

_mode: str = "light"

# Scalar colors for native Tk widgets (Listbox, ttk.Style) and matplotlib
_LIGHT = {
    "listbox_bg":  "#f8f8f8",
    "listbox_fg":  "#111111",
    "listbox_hl":  "#cccccc",
    "nb_bg":       "#f0f0f0",
    "nb_tab_bg":   "#e8e8e8",
    "nb_tab_fg":   "#111111",
    "nb_tab_sel":  "#0078ff",
}

_DARK = {
    "listbox_bg":  "#1e1e1e",
    "listbox_fg":  "white",
    "listbox_hl":  "#333333",
    "nb_bg":       "#0f0f0f",
    "nb_tab_bg":   "#1a1a1a",
    "nb_tab_fg":   "white",
    "nb_tab_sel":  "#0078ff",
}


def mode() -> str:
    return _mode


def scalars() -> dict:
    return _LIGHT if _mode == "light" else _DARK


def is_dark() -> bool:
    return _mode == "dark"


def set_mode(m: str) -> None:
    global _mode
    _mode = m
    ctk.set_appearance_mode(m)
