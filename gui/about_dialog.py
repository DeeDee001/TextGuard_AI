"""
TextGuard AI - About & Ethics Dialog
Presents system architecture, course project details, machine learning methodology,
limitations, and ethical guidelines for text authorship classification.
"""

import customtkinter as ctk
from gui.theme import THEME, FONTS


class AboutDialog(ctk.CTkToplevel):
    """
    Modal window explaining project objectives, academic attribution, and ethical constraints.
    """

    def __init__(self, master, current_theme: str = "dark", **kwargs):
        super().__init__(master, **kwargs)
        self.title("About TextGuard AI — System Architecture & Ethics")
        self.geometry("640x520")
        self.resizable(False, False)
        
        self.current_theme = current_theme
        self.colors = THEME[self.current_theme]
        self.configure(fg_color=self.colors["bg_primary"])

        # Center on parent window
        self.transient(master)
        self.grab_set()

        self._init_ui()

    def _init_ui(self):
        """Construct dialog content."""
        scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=20, pady=20)

        # Title Header
        title_lbl = ctk.CTkLabel(
            scroll,
            text="TextGuard AI",
            font=FONTS["title_large"],
            text_color=self.colors["accent_primary"],
        )
        title_lbl.pack(anchor="w")

        subtitle_lbl = ctk.CTkLabel(
            scroll,
            text="AI Text Detection System Using Python & Machine Learning",
            font=FONTS["title_small"],
            text_color=self.colors["text_primary"],
        )
        subtitle_lbl.pack(anchor="w", pady=(0, 12))

        # Academic Attribution Box
        acad_box = ctk.CTkFrame(scroll, fg_color=self.colors["bg_card"], corner_radius=8, border_color=self.colors["border"], border_width=1)
        acad_box.pack(fill="x", pady=6)

        acad_text = (
            "American International University-Bangladesh (AIUB)\n"
            "Faculty of Science & Technology (FST) | Department of Computer Science\n"
            "Course: CSC 4162 — Programming in Python | Semester: Fall 26-27"
        )
        lbl_acad = ctk.CTkLabel(
            acad_box, text=acad_text, font=FONTS["body_small"], text_color=self.colors["text_secondary"], justify="left"
        )
        lbl_acad.pack(padx=14, pady=10, anchor="w")

        # Sections Helper
        def add_section(heading: str, body: str):
            lbl_h = ctk.CTkLabel(scroll, text=heading, font=FONTS["body_bold"], text_color=self.colors["text_primary"])
            lbl_h.pack(anchor="w", pady=(12, 4))
            lbl_b = ctk.CTkLabel(
                scroll, text=body, font=FONTS["body_normal"], text_color=self.colors["text_secondary"],
                wraplength=580, justify="left"
            )
            lbl_b.pack(anchor="w", pady=(0, 6))

        add_section(
            "1. Purpose & Motivation",
            "TextGuard AI is developed to analyze user-submitted passages and estimate their likelihood "
            "of being synthetic (AI-generated) or human-authored. As generative models proliferate across "
            "academic and creative domains, understanding the stylometric, vocabulary, and syntactic patterns "
            "that distinguish artificial from human text is an essential educational objective."
        )

        add_section(
            "2. Machine Learning Architecture",
            "The detection engine uses a Scikit-learn Pipeline combining Sublinear TF-IDF N-gram Vectorization "
            "(word n-grams 1-2, 20,000 maximum vocabulary) and a Calibrated Logistic Regression classifier. "
            "Trained on the 50,000-sample AI vs Human Text Classification Benchmark covering models such as "
            "ChatGPT-4o, Claude 3.5 Sonnet, Llama 3 70B, Gemini 1.5 Pro, and Mistral Large."
        )

        add_section(
            "3. Stylometric & Linguistic Analysis",
            "In addition to ML classification, TextGuard AI computes descriptive features including "
            "Type-Token Ratio (vocabulary diversity), Sentence Length Variance (burstiness rhythm), "
            "Punctuation Density, Transition Word Counts, and Flesch-Kincaid Readability metrics."
        )

        add_section(
            "4. Limitations & Ethical Notice",
            "IMPORTANT: TextGuard AI is an educational classification system and NOT a forensic proof tool. "
            "Short texts (<30 words), technical documents, code snippets, and translated passages can yield "
            "borderline or misleading scores. AI text detectors should never be used as sole evidence for "
            "academic discipline or punitive judgments."
        )

        # Close button
        btn_close = ctk.CTkButton(
            scroll,
            text="Close",
            font=FONTS["body_bold"],
            width=100,
            fg_color=self.colors["accent_primary"],
            hover_color=self.colors["accent_hover"],
            command=self.destroy,
        )
        btn_close.pack(anchor="center", pady=(16, 8))
