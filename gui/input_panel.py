"""
TextGuard AI - Text Input Panel
Provides a large responsive editor, real-time counters, file loading,
clipboard pasting, and sample text loaders.
"""

import os
import re
from typing import Callable, Optional
import customtkinter as ctk
from tkinter import filedialog, messagebox

from gui.theme import THEME, FONTS


class InputPanel(ctk.CTkFrame):
    """
    Panel housing the primary text editor, input metrics, and action buttons.
    """

    def __init__(
        self,
        master,
        on_analyze_callback: Callable[[str], None],
        current_theme: str = "dark",
        **kwargs
    ):
        super().__init__(master, **kwargs)
        self.on_analyze_callback = on_analyze_callback
        self.current_theme = current_theme
        self.colors = THEME[self.current_theme]

        self._init_ui()

    def _init_ui(self):
        """Construct the widgets and layout."""
        self.configure(fg_color=self.colors["bg_card"], corner_radius=12)

        # Header Label with Instruction
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.pack(fill="x", padx=16, pady=(14, 6))

        title_lbl = ctk.CTkLabel(
            header_frame,
            text="Text Input & Analysis Editor",
            font=FONTS["title_small"],
            text_color=self.colors["text_primary"],
        )
        title_lbl.pack(side="left")

        # Load samples quick-buttons
        samples_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        samples_frame.pack(side="right")

        btn_sample_human = ctk.CTkButton(
            samples_frame,
            text="Load Human Sample",
            font=FONTS["body_small"],
            width=135,
            height=26,
            fg_color="#065F46",
            hover_color="#047857",
            command=self._load_sample_human,
        )
        btn_sample_human.pack(side="left", padx=4)

        btn_sample_ai = ctk.CTkButton(
            samples_frame,
            text="Load AI Sample",
            font=FONTS["body_small"],
            width=120,
            height=26,
            fg_color="#9F1239",
            hover_color="#BE123C",
            command=self._load_sample_ai,
        )
        btn_sample_ai.pack(side="left", padx=4)

        # Text Editor
        self.textbox = ctk.CTkTextbox(
            self,
            font=FONTS["body_normal"],
            wrap="word",
            fg_color=self.colors["bg_primary"],
            text_color=self.colors["text_primary"],
            border_color=self.colors["border"],
            border_width=1,
            corner_radius=8,
        )
        self.textbox.pack(fill="both", expand=True, padx=16, pady=6)
        self.textbox.bind("<KeyRelease>", self._on_text_change)

        # Short text warning label (hidden by default)
        self.warning_label = ctk.CTkLabel(
            self,
            text="",
            font=FONTS["body_small"],
            text_color=self.colors["borderline_amber"],
            anchor="w",
        )
        self.warning_label.pack(fill="x", padx=18, pady=(0, 4))

        # Bottom Bar: Counters + Action Buttons
        bottom_frame = ctk.CTkFrame(self, fg_color="transparent")
        bottom_frame.pack(fill="x", padx=16, pady=(4, 14))

        # Real-time Counters
        self.counter_label = ctk.CTkLabel(
            bottom_frame,
            text="Words: 0   |   Characters: 0   |   Sentences: 0",
            font=FONTS["body_small"],
            text_color=self.colors["text_secondary"],
        )
        self.counter_label.pack(side="left")

        # Action Buttons
        btn_container = ctk.CTkFrame(bottom_frame, fg_color="transparent")
        btn_container.pack(side="right")

        self.btn_clear = ctk.CTkButton(
            btn_container,
            text="Clear",
            font=FONTS["body_small"],
            width=70,
            height=32,
            fg_color=self.colors["bg_secondary"],
            hover_color=self.colors["border"],
            text_color=self.colors["text_primary"],
            command=self.clear_text,
        )
        self.btn_clear.pack(side="left", padx=4)

        self.btn_paste = ctk.CTkButton(
            btn_container,
            text="Paste",
            font=FONTS["body_small"],
            width=75,
            height=32,
            fg_color=self.colors["bg_secondary"],
            hover_color=self.colors["border"],
            text_color=self.colors["text_primary"],
            command=self._paste_clipboard,
        )
        self.btn_paste.pack(side="left", padx=4)

        self.btn_load_file = ctk.CTkButton(
            btn_container,
            text="Open .TXT",
            font=FONTS["body_small"],
            width=85,
            height=32,
            fg_color=self.colors["bg_secondary"],
            hover_color=self.colors["border"],
            text_color=self.colors["text_primary"],
            command=self._load_from_file,
        )
        self.btn_load_file.pack(side="left", padx=4)

        self.btn_analyze = ctk.CTkButton(
            btn_container,
            text="Analyze Text",
            font=FONTS["body_bold"],
            width=130,
            height=32,
            fg_color=self.colors["accent_primary"],
            hover_color=self.colors["accent_hover"],
            text_color="#FFFFFF",
            command=self._trigger_analysis,
        )
        self.btn_analyze.pack(side="left", padx=(8, 0))

        # Initial text update
        self._update_counters()

    def get_text(self) -> str:
        """Retrieve current text content from textbox."""
        return self.textbox.get("1.0", "end-1c").strip()

    def set_text(self, text: str):
        """Set textbox content and refresh counters."""
        self.textbox.delete("1.0", "end")
        self.textbox.insert("1.0", text)
        self._update_counters()

    def clear_text(self):
        """Clear editor content and reset warnings."""
        self.textbox.delete("1.0", "end")
        self.warning_label.configure(text="")
        self._update_counters()

    def _paste_clipboard(self):
        """Paste system clipboard text into editor."""
        try:
            clipboard_text = self.clipboard_get()
            if clipboard_text:
                self.textbox.insert("insert", clipboard_text)
                self._update_counters()
        except Exception:
            pass

    def _load_from_file(self):
        """Open a file dialog to load text from a .txt or .md file."""
        filepath = filedialog.askopenfilename(
            title="Select Text File to Analyze",
            filetypes=[("Text Documents", "*.txt;*.md"), ("All Files", "*.*")]
        )
        if filepath:
            try:
                with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                self.set_text(content)
            except Exception as e:
                messagebox.showerror("File Error", f"Unable to read file: {e}")

    def _load_sample_human(self):
        """Load the pre-packaged human sample text."""
        sample_path = os.path.join("data", "sample_human_text.txt")
        if os.path.exists(sample_path):
            with open(sample_path, "r", encoding="utf-8") as f:
                self.set_text(f.read())
        else:
            self.set_text(
                "When I first started programming, I thought debugging meant rewriting everything from scratch. "
                "Over the years, I learned that patience and careful reading of traceback logs solve almost every issue. "
                "It's always a wonderful feeling when you finally get the green checkmark on all your unit tests!"
            )

    def _load_sample_ai(self):
        """Load the pre-packaged AI sample text."""
        sample_path = os.path.join("data", "sample_ai_text.txt")
        if os.path.exists(sample_path):
            with open(sample_path, "r", encoding="utf-8") as f:
                self.set_text(f.read())
        else:
            self.set_text(
                "Artificial intelligence and computational systems represent a critical domain in contemporary discourse. "
                "With increasing attention on operational efficacy, industry stakeholders are systematically reevaluating "
                "traditional methodologies. Furthermore, machine learning pipelines enhance decision-making structures. "
                "In conclusion, collaborative policy frameworks remain paramount for sustainable technical advancement."
            )

    def _on_text_change(self, event=None):
        """Event triggered on typing or deleting text."""
        self._update_counters()

    def _update_counters(self):
        """Calculate and display live word, character, and sentence counts."""
        text = self.get_text()
        char_count = len(text)
        words = re.findall(r"\b[\w'-]+\b", text)
        word_count = len(words)

        if text:
            sentence_splits = re.split(r"[.!?]+", text)
            sentence_count = len([s for s in sentence_splits if s.strip()])
        else:
            sentence_count = 0

        self.counter_label.configure(
            text=f"Words: {word_count:,}   |   Characters: {char_count:,}   |   Sentences: {sentence_count:,}"
        )

        # Warning logic for short input
        if 0 < word_count < 25:
            self.warning_label.configure(
                text=f"Notice: Text contains only {word_count} words. Recommendations: provide 30+ words for reliable stylometry."
            )
        else:
            self.warning_label.configure(text="")

    def _trigger_analysis(self):
        """Validate input and invoke the analysis callback."""
        text = self.get_text()
        if not text:
            messagebox.showwarning("Empty Text", "Please enter or paste text to perform analysis.")
            return
        self.on_analyze_callback(text)

    def set_loading(self, is_loading: bool):
        """Toggle button state during asynchronous analysis."""
        if is_loading:
            self.btn_analyze.configure(state="disabled", text="Analyzing...")
        else:
            self.btn_analyze.configure(state="normal", text="Analyze Text")
