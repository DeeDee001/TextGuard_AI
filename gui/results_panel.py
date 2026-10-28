"""
TextGuard AI - Results & Diagnostics Panel
Displays the classification verdict, calibrated probability meters,
stylometric metric cards, reasoning explanations, and report export triggers.
"""

from typing import Dict, Any, Callable, Optional
import customtkinter as ctk
from tkinter import filedialog, messagebox

from gui.theme import THEME, FONTS
from analysis.report_generator import ReportGenerator


class ResultsPanel(ctk.CTkFrame):
    """
    Results display dashboard presenting model prediction, scores, and linguistic metrics.
    """

    def __init__(
        self,
        master,
        current_theme: str = "dark",
        **kwargs
    ):
        super().__init__(master, **kwargs)
        self.current_theme = current_theme
        self.colors = THEME[self.current_theme]
        self.current_report_data: Optional[Dict[str, Any]] = None

        self._init_ui()

    def _init_ui(self):
        """Construct results presentation widgets."""
        self.configure(fg_color=self.colors["bg_card"], corner_radius=12)

        # Header Title
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.pack(fill="x", padx=16, pady=(14, 6))

        title_lbl = ctk.CTkLabel(
            header_frame,
            text="Analysis Verdict & Classification Metrics",
            font=FONTS["title_small"],
            text_color=self.colors["text_primary"],
        )
        title_lbl.pack(side="left")

        # Copy summary button
        self.btn_copy = ctk.CTkButton(
            header_frame,
            text="Copy Summary",
            font=FONTS["body_small"],
            width=100,
            height=26,
            fg_color=self.colors["bg_secondary"],
            hover_color=self.colors["border"],
            text_color=self.colors["text_primary"],
            command=self._copy_summary,
        )
        self.btn_copy.pack(side="right")

        # -------------------------------------------------------------
        # Verdict Card (Hero Badge)
        # -------------------------------------------------------------
        self.verdict_frame = ctk.CTkFrame(
            self,
            fg_color=self.colors["bg_secondary"],
            border_color=self.colors["border"],
            border_width=1,
            corner_radius=10,
        )
        self.verdict_frame.pack(fill="x", padx=16, pady=8)

        self.verdict_badge = ctk.CTkLabel(
            self.verdict_frame,
            text="Ready for Analysis",
            font=FONTS["title_medium"],
            text_color=self.colors["text_muted"],
        )
        self.verdict_badge.pack(anchor="w", padx=16, pady=(10, 2))

        self.confidence_lbl = ctk.CTkLabel(
            self.verdict_frame,
            text="Submit text to view machine learning predictions and stylometric measurements.",
            font=FONTS["body_small"],
            text_color=self.colors["text_secondary"],
            wraplength=520,
            justify="left",
        )
        self.confidence_lbl.pack(anchor="w", padx=16, pady=(0, 10))

        # -------------------------------------------------------------
        # Probability Meters Frame
        # -------------------------------------------------------------
        self.scores_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.scores_frame.pack(fill="x", padx=16, pady=4)
        self.scores_frame.grid_columnconfigure(0, weight=1)
        self.scores_frame.grid_columnconfigure(1, weight=1)

        # Human Score Container
        human_box = ctk.CTkFrame(self.scores_frame, fg_color=self.colors["bg_secondary"], corner_radius=8)
        human_box.grid(row=0, column=0, sticky="ew", padx=(0, 6), pady=4)

        self.lbl_human_title = ctk.CTkLabel(
            human_box, text="Human-Written Score", font=FONTS["caption"], text_color=self.colors["text_secondary"]
        )
        self.lbl_human_title.pack(anchor="w", padx=10, pady=(6, 0))

        self.lbl_human_score = ctk.CTkLabel(
            human_box, text="-- %", font=FONTS["title_medium"], text_color=self.colors["human_emerald"]
        )
        self.lbl_human_score.pack(anchor="w", padx=10, pady=(0, 2))

        self.progress_human = ctk.CTkProgressBar(human_box, progress_color=self.colors["human_emerald"], height=8)
        self.progress_human.set(0)
        self.progress_human.pack(fill="x", padx=10, pady=(0, 8))

        # AI Score Container
        ai_box = ctk.CTkFrame(self.scores_frame, fg_color=self.colors["bg_secondary"], corner_radius=8)
        ai_box.grid(row=0, column=1, sticky="ew", padx=(6, 0), pady=4)

        self.lbl_ai_title = ctk.CTkLabel(
            ai_box, text="AI-Associated Score", font=FONTS["caption"], text_color=self.colors["text_secondary"]
        )
        self.lbl_ai_title.pack(anchor="w", padx=10, pady=(6, 0))

        self.lbl_ai_score = ctk.CTkLabel(
            ai_box, text="-- %", font=FONTS["title_medium"], text_color=self.colors["ai_rose"]
        )
        self.lbl_ai_score.pack(anchor="w", padx=10, pady=(0, 2))

        self.progress_ai = ctk.CTkProgressBar(ai_box, progress_color=self.colors["ai_rose"], height=8)
        self.progress_ai.set(0)
        self.progress_ai.pack(fill="x", padx=10, pady=(0, 8))

        # -------------------------------------------------------------
        # Stylometric Metric Cards Grid (3x2)
        # -------------------------------------------------------------
        self.metrics_grid = ctk.CTkFrame(self, fg_color="transparent")
        self.metrics_grid.pack(fill="x", padx=16, pady=6)
        self.metrics_grid.grid_columnconfigure((0, 1, 2), weight=1)

        self.card_lexical = self._create_metric_card(self.metrics_grid, 0, 0, "Lexical Diversity (TTR)", "--")
        self.card_burstiness = self._create_metric_card(self.metrics_grid, 0, 1, "Sentence Burstiness", "--")
        self.card_sentence_len = self._create_metric_card(self.metrics_grid, 0, 2, "Avg Sentence Length", "--")

        self.card_words = self._create_metric_card(self.metrics_grid, 1, 0, "Total Word Count", "--")
        self.card_punctuation = self._create_metric_card(self.metrics_grid, 1, 1, "Punctuation Density", "--")
        self.card_transitions = self._create_metric_card(self.metrics_grid, 1, 2, "Transition Markers", "--")

        # -------------------------------------------------------------
        # Export Actions Bar
        # -------------------------------------------------------------
        export_frame = ctk.CTkFrame(self, fg_color="transparent")
        export_frame.pack(fill="x", padx=16, pady=(8, 4))

        lbl_export = ctk.CTkLabel(
            export_frame, text="Export Detailed Report:", font=FONTS["body_bold"], text_color=self.colors["text_secondary"]
        )
        lbl_export.pack(side="left")

        self.btn_export_txt = ctk.CTkButton(
            export_frame,
            text="TXT Report",
            font=FONTS["body_small"],
            width=90,
            height=28,
            fg_color=self.colors["bg_secondary"],
            hover_color=self.colors["border"],
            command=self._export_txt,
        )
        self.btn_export_txt.pack(side="left", padx=4)

        self.btn_export_json = ctk.CTkButton(
            export_frame,
            text="JSON Data",
            font=FONTS["body_small"],
            width=90,
            height=28,
            fg_color=self.colors["bg_secondary"],
            hover_color=self.colors["border"],
            command=self._export_json,
        )
        self.btn_export_json.pack(side="left", padx=4)

        self.btn_export_pdf = ctk.CTkButton(
            export_frame,
            text="PDF Report",
            font=FONTS["body_small"],
            width=95,
            height=28,
            fg_color=self.colors["accent_primary"],
            hover_color=self.colors["accent_hover"],
            command=self._export_pdf,
        )
        self.btn_export_pdf.pack(side="left", padx=4)

        # -------------------------------------------------------------
        # Limitations & Disclaimer Footnote
        # -------------------------------------------------------------
        self.disclaimer_lbl = ctk.CTkLabel(
            self,
            text="Disclaimer: Machine learning predictions are statistical estimations based on linguistic n-grams and do not provide absolute proof of authorship.",
            font=FONTS["caption"],
            text_color=self.colors["text_muted"],
            wraplength=540,
            justify="center",
        )
        self.disclaimer_lbl.pack(fill="x", padx=16, pady=(6, 12))

    def _create_metric_card(self, parent, row: int, col: int, title: str, initial_val: str):
        """Helper to create a small card displaying a key statistic."""
        card = ctk.CTkFrame(parent, fg_color=self.colors["bg_secondary"], corner_radius=8)
        card.grid(row=row, column=col, sticky="nsew", padx=3, pady=3)

        lbl_title = ctk.CTkLabel(card, text=title, font=FONTS["caption"], text_color=self.colors["text_secondary"])
        lbl_title.pack(anchor="w", padx=8, pady=(4, 0))

        lbl_val = ctk.CTkLabel(card, text=initial_val, font=FONTS["body_bold"], text_color=self.colors["text_primary"])
        lbl_val.pack(anchor="w", padx=8, pady=(0, 4))
        return lbl_val

    def display_results(self, report_data: Dict[str, Any]):
        """
        Populate the entire results interface with report data.
        """
        self.current_report_data = report_data
        p = report_data["prediction"]
        m = report_data["linguistic_metrics"]

        verdict = p["verdict"]
        ai_score = p["ai_probability_pct"]
        human_score = p["human_probability_pct"]

        # Color-code verdict
        if "Human" in verdict:
            verdict_color = self.colors["human_emerald"]
            border_color = self.colors["human_bg"]
        elif "AI" in verdict:
            verdict_color = self.colors["ai_rose"]
            border_color = self.colors["ai_bg"]
        else:
            verdict_color = self.colors["borderline_amber"]
            border_color = self.colors["borderline_bg"]

        self.verdict_frame.configure(border_color=border_color, border_width=1.5)
        self.verdict_badge.configure(text=verdict, text_color=verdict_color)
        self.confidence_lbl.configure(text=p["explanation"])

        # Update scores
        self.lbl_human_score.configure(text=f"{human_score:.1f}%")
        self.lbl_ai_score.configure(text=f"{ai_score:.1f}%")
        self.progress_human.set(human_score / 100.0)
        self.progress_ai.set(ai_score / 100.0)

        # Update metric cards
        self.card_lexical.configure(text=f"{m.get('lexical_diversity_ttr', 0.0):.3f}")
        self.card_burstiness.configure(text=f"{m.get('burstiness_sentence_std', 0.0):.1f} std")
        self.card_sentence_len.configure(text=f"{m.get('avg_sentence_length_words', 0.0):.1f} words")
        self.card_words.configure(text=f"{m.get('word_count', 0):,}")
        self.card_punctuation.configure(text=f"{m.get('punctuation_density_pct', 0.0):.1f}%")
        self.card_transitions.configure(text=f"{m.get('transition_word_count', 0)}")

    def _copy_summary(self):
        """Copy analysis verdict and scores to system clipboard."""
        if not self.current_report_data:
            messagebox.showinfo("No Results", "Perform an analysis first before copying results.")
            return

        p = self.current_report_data["prediction"]
        summary_text = (
            f"TextGuard AI Analysis Result:\n"
            f"Verdict: {p['verdict']}\n"
            f"AI Likelihood: {p['ai_probability_pct']:.1f}%\n"
            f"Human Likelihood: {p['human_probability_pct']:.1f}%\n"
            f"Confidence: {p['confidence_level']}\n"
            f"Reasoning: {p['explanation']}"
        )
        self.clipboard_clear()
        self.clipboard_append(summary_text)
        messagebox.showinfo("Copied", "Analysis summary copied to clipboard.")

    def _export_txt(self):
        """Export analysis to TXT file."""
        if not self.current_report_data:
            messagebox.showwarning("No Data", "No analysis data available to export.")
            return

        filepath = filedialog.asksaveasfilename(
            title="Save Text Report",
            defaultextension=".txt",
            filetypes=[("Text File", "*.txt"), ("All Files", "*.*")],
            initialfile="TextGuard_Analysis_Report.txt"
        )
        if filepath:
            try:
                ReportGenerator.export_txt(self.current_report_data, filepath)
                messagebox.showinfo("Export Successful", f"Report saved to:\n{filepath}")
            except Exception as e:
                messagebox.showerror("Export Failed", f"Could not save report: {e}")

    def _export_json(self):
        """Export analysis to JSON file."""
        if not self.current_report_data:
            messagebox.showwarning("No Data", "No analysis data available to export.")
            return

        filepath = filedialog.asksaveasfilename(
            title="Save JSON Report",
            defaultextension=".json",
            filetypes=[("JSON File", "*.json"), ("All Files", "*.*")],
            initialfile="TextGuard_Analysis_Data.json"
        )
        if filepath:
            try:
                ReportGenerator.export_json(self.current_report_data, filepath)
                messagebox.showinfo("Export Successful", f"JSON data saved to:\n{filepath}")
            except Exception as e:
                messagebox.showerror("Export Failed", f"Could not save JSON: {e}")

    def _export_pdf(self):
        """Export analysis to PDF file."""
        if not self.current_report_data:
            messagebox.showwarning("No Data", "No analysis data available to export.")
            return

        filepath = filedialog.asksaveasfilename(
            title="Save PDF Report",
            defaultextension=".pdf",
            filetypes=[("PDF Document", "*.pdf"), ("All Files", "*.*")],
            initialfile="TextGuard_Analysis_Report.pdf"
        )
        if filepath:
            try:
                ReportGenerator.export_pdf(self.current_report_data, filepath)
                messagebox.showinfo("Export Successful", f"PDF report generated at:\n{filepath}")
            except Exception as e:
                messagebox.showerror("Export Failed", f"Could not generate PDF: {e}")
