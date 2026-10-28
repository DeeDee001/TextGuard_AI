"""
TextGuard AI - History Panel
Displays past analysis records, search filtering, detailed inspection,
deletion operations, and user privacy toggles.
"""

from typing import Callable, Optional
import customtkinter as ctk
from tkinter import messagebox

from gui.theme import THEME, FONTS
from data.history_manager import HistoryManager


class HistoryPanel(ctk.CTkFrame):
    """
    Dedicated view for inspecting, searching, and managing past text analysis records.
    """

    def __init__(
        self,
        master,
        history_manager: HistoryManager,
        on_load_record_callback: Callable[[dict], None],
        current_theme: str = "dark",
        **kwargs
    ):
        super().__init__(master, **kwargs)
        self.history_manager = history_manager
        self.on_load_record_callback = on_load_record_callback
        self.current_theme = current_theme
        self.colors = THEME[self.current_theme]

        self._init_ui()

    def _init_ui(self):
        """Construct the search bar, record list, and controls."""
        self.configure(fg_color=self.colors["bg_card"], corner_radius=12)

        # Header with Title and Privacy Controls
        top_bar = ctk.CTkFrame(self, fg_color="transparent")
        top_bar.pack(fill="x", padx=16, pady=(14, 8))

        lbl_title = ctk.CTkLabel(
            top_bar, text="Analysis History & Audit Log", font=FONTS["title_medium"], text_color=self.colors["text_primary"]
        )
        lbl_title.pack(side="left")

        # Privacy Toggles
        self.switch_save = ctk.CTkSwitch(
            top_bar,
            text="Save History",
            font=FONTS["body_small"],
            text_color=self.colors["text_secondary"],
            progress_color=self.colors["accent_primary"],
            command=self._on_privacy_change,
        )
        if self.history_manager.settings.get("save_history_enabled", True):
            self.switch_save.select()
        else:
            self.switch_save.deselect()
        self.switch_save.pack(side="right", padx=6)

        # Search & Action Toolbar
        toolbar = ctk.CTkFrame(self, fg_color="transparent")
        toolbar.pack(fill="x", padx=16, pady=(0, 10))

        self.search_entry = ctk.CTkEntry(
            toolbar,
            placeholder_text="Search past records by keyword, ID, or verdict...",
            font=FONTS["body_normal"],
            fg_color=self.colors["bg_secondary"],
            border_color=self.colors["border"],
            height=32,
        )
        self.search_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.search_entry.bind("<KeyRelease>", self._on_search)

        btn_clear_all = ctk.CTkButton(
            toolbar,
            text="Clear All",
            font=FONTS["body_small"],
            width=85,
            height=32,
            fg_color="#991B1B",
            hover_color="#DC2626",
            command=self._clear_all,
        )
        btn_clear_all.pack(side="right")

        btn_refresh = ctk.CTkButton(
            toolbar,
            text="Refresh",
            font=FONTS["body_small"],
            width=75,
            height=32,
            fg_color=self.colors["bg_secondary"],
            hover_color=self.colors["border"],
            text_color=self.colors["text_primary"],
            command=self.refresh_records,
        )
        btn_refresh.pack(side="right", padx=4)

        # Scrollable Record Container
        self.scroll_frame = ctk.CTkScrollableFrame(
            self,
            fg_color=self.colors["bg_primary"],
            corner_radius=8,
            border_color=self.colors["border"],
            border_width=1,
        )
        self.scroll_frame.pack(fill="both", expand=True, padx=16, pady=(0, 14))

        self.refresh_records()

    def refresh_records(self, query: str = ""):
        """Render the list of analysis records."""
        # Clear existing children
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()

        records = self.history_manager.search_records(query) if query else self.history_manager.get_all_records()

        if not records:
            lbl_empty = ctk.CTkLabel(
                self.scroll_frame,
                text="No analysis records found.",
                font=FONTS["body_normal"],
                text_color=self.colors["text_muted"],
            )
            lbl_empty.pack(pady=40)
            return

        for r in records:
            self._render_record_row(r)

    def _render_record_row(self, r: dict):
        """Render a single interactive record card."""
        card = ctk.CTkFrame(
            self.scroll_frame,
            fg_color=self.colors["bg_secondary"],
            border_color=self.colors["border"],
            border_width=1,
            corner_radius=8,
        )
        card.pack(fill="x", padx=6, pady=4)

        # Top row: ID, Timestamp, Verdict Badge
        top_row = ctk.CTkFrame(card, fg_color="transparent")
        top_row.pack(fill="x", padx=10, pady=(8, 2))

        lbl_id = ctk.CTkLabel(
            top_row, text=r.get("id", "N/A"), font=FONTS["body_bold"], text_color=self.colors["text_primary"]
        )
        lbl_id.pack(side="left")

        lbl_time = ctk.CTkLabel(
            top_row, text=f"•  {r.get('timestamp', '')}", font=FONTS["caption"], text_color=self.colors["text_muted"]
        )
        lbl_time.pack(side="left", padx=8)

        # Verdict Badge with color
        verdict = r.get("verdict", "Unknown")
        if "Human" in verdict:
            v_color = self.colors["human_emerald"]
        elif "AI" in verdict:
            v_color = self.colors["ai_rose"]
        else:
            v_color = self.colors["borderline_amber"]

        badge = ctk.CTkLabel(top_row, text=verdict, font=FONTS["body_bold"], text_color=v_color)
        badge.pack(side="right")

        # Middle row: Text preview
        preview = r.get("text_preview", "")
        lbl_preview = ctk.CTkLabel(
            card,
            text=f'"{preview}"',
            font=FONTS["body_small"],
            text_color=self.colors["text_secondary"],
            anchor="w",
            wraplength=600,
            justify="left",
        )
        lbl_preview.pack(fill="x", padx=10, pady=(2, 6))

        # Bottom row: Key stats & Action buttons
        bottom_row = ctk.CTkFrame(card, fg_color="transparent")
        bottom_row.pack(fill="x", padx=10, pady=(0, 8))

        stats_text = (
            f"Words: {r.get('word_count', 0):,}  |  "
            f"AI Likelihood: {r.get('ai_probability', 0.0):.1f}%  |  "
            f"Human Likelihood: {r.get('human_probability', 0.0):.1f}%  |  "
            f"Diversity: {r.get('lexical_diversity', 0.0):.2f}"
        )
        lbl_stats = ctk.CTkLabel(
            bottom_row, text=stats_text, font=FONTS["caption"], text_color=self.colors["text_muted"]
        )
        lbl_stats.pack(side="left")

        # Buttons
        btn_delete = ctk.CTkButton(
            bottom_row,
            text="Delete",
            font=FONTS["body_small"],
            width=65,
            height=24,
            fg_color="#7F1D1D",
            hover_color="#991B1B",
            command=lambda rid=r.get("id"): self._delete_record(rid),
        )
        btn_delete.pack(side="right", padx=(4, 0))

        btn_load = ctk.CTkButton(
            bottom_row,
            text="Inspect Text",
            font=FONTS["body_small"],
            width=90,
            height=24,
            fg_color=self.colors["accent_primary"],
            hover_color=self.colors["accent_hover"],
            command=lambda rec=r: self.on_load_record_callback(rec),
        )
        btn_load.pack(side="right")

    def _on_search(self, event=None):
        """Filter records on search keystroke."""
        q = self.search_entry.get().strip()
        self.refresh_records(q)

    def _delete_record(self, record_id: str):
        """Confirm and delete single record."""
        confirm = messagebox.askyesno("Confirm Delete", f"Delete record '{record_id}' permanently?")
        if confirm:
            self.history_manager.delete_record(record_id)
            self.refresh_records(self.search_entry.get().strip())

    def _clear_all(self):
        """Confirm and clear all records."""
        confirm = messagebox.askyesno("Clear All History", "Are you sure you want to delete ALL saved analysis records?")
        if confirm:
            self.history_manager.clear_all_history()
            self.refresh_records()

    def _on_privacy_change(self):
        """Update privacy settings."""
        enabled = self.switch_save.get() == 1
        self.history_manager.update_privacy_settings(save_enabled=enabled, full_text_enabled=True)
