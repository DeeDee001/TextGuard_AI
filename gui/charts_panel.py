"""
TextGuard AI - Visualizations & Charts Panel
Renders real-time Matplotlib charts embedded directly inside the CustomTkinter GUI.
"""

import customtkinter as ctk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional

from gui.theme import THEME


class ChartsPanel(ctk.CTkFrame):
    """
    Renders dynamic stylometric and classification charts for analyzed text.
    """

    def __init__(self, master, current_theme: str = "dark", **kwargs):
        super().__init__(master, **kwargs)
        self.current_theme = current_theme
        self.colors = THEME[self.current_theme]
        
        self.figure = None
        self.canvas = None
        self._init_layout()

    def _init_layout(self):
        """Build canvas container."""
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Matplotlib Figure with 2 Subplots (Top: Probability Bar, Bottom: Sentence Rhythm)
        self.figure, (self.ax_scores, self.ax_sentences) = plt.subplots(
            nrows=2, ncols=1, figsize=(6.2, 4.8), dpi=100
        )
        self.figure.patch.set_facecolor(self.colors["chart_bg"])
        self.figure.subplots_adjust(hspace=0.45, top=0.92, bottom=0.12, left=0.18, right=0.92)

        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True, padx=4, pady=4)

        self.show_placeholder()

    def show_placeholder(self):
        """Display welcoming guidance before an analysis is triggered."""
        self.ax_scores.clear()
        self.ax_sentences.clear()

        for ax in (self.ax_scores, self.ax_sentences):
            ax.set_facecolor(self.colors["chart_bg"])
            ax.tick_params(colors=self.colors["text_muted"], labelsize=8)
            for spine in ax.spines.values():
                spine.set_color(self.colors["border"])

        self.ax_scores.text(
            0.5, 0.5, "Prediction Visualizations will appear here\nafter running text analysis",
            ha="center", va="center", color=self.colors["text_muted"],
            fontsize=10, transform=self.ax_scores.transAxes
        )
        self.ax_scores.set_xticks([])
        self.ax_scores.set_yticks([])

        self.ax_sentences.text(
            0.5, 0.5, "Sentence Length Progression Chart",
            ha="center", va="center", color=self.colors["text_muted"],
            fontsize=10, transform=self.ax_sentences.transAxes
        )
        self.ax_sentences.set_xticks([])
        self.ax_sentences.set_yticks([])

        self.canvas.draw()

    def update_charts(self, prediction: Dict[str, Any], stats: Dict[str, Any]):
        """
        Draw updated data visualizations from actual prediction and linguistic stats.
        """
        self.ax_scores.clear()
        self.ax_sentences.clear()

        # Styling both axes
        for ax in (self.ax_scores, self.ax_sentences):
            ax.set_facecolor(self.colors["chart_bg"])
            ax.tick_params(colors=self.colors["chart_text"], labelsize=8.5)
            for spine in ax.spines.values():
                spine.set_color(self.colors["chart_grid"])

        # -------------------------------------------------------------
        # 1. Prediction Score Breakdown (Horizontal Bar Chart)
        # -------------------------------------------------------------
        categories = ["Human-Written", "AI-Generated"]
        scores = [prediction.get("human_probability", 0.0), prediction.get("ai_probability", 0.0)]
        bar_colors = [self.colors["chart_bar_human"], self.colors["chart_bar_ai"]]

        bars = self.ax_scores.barh(categories, scores, color=bar_colors, height=0.48, zorder=3)
        self.ax_scores.set_xlim(0, 105)
        self.ax_scores.set_xlabel("Calibrated Probability (%)", color=self.colors["chart_text"], fontsize=8.5)
        self.ax_scores.set_title("Model Probability Breakdown", color=self.colors["chart_text"], fontsize=10, fontweight="bold", pad=8)
        self.ax_scores.grid(axis="x", linestyle="--", alpha=0.3, color=self.colors["chart_grid"], zorder=0)

        # Data labels inside / next to bars
        for bar in bars:
            w = bar.get_width()
            y = bar.get_y() + bar.get_height() / 2
            self.ax_scores.text(
                w + 2.0, y, f"{w:.1f}%",
                va="center", ha="left", color=self.colors["chart_text"],
                fontsize=8.5, fontweight="bold"
            )

        # -------------------------------------------------------------
        # 2. Sentence Length Rhythm / Progression Chart
        # -------------------------------------------------------------
        sentence_lengths = stats.get("sentence_lengths", [])
        if sentence_lengths:
            indices = list(range(1, len(sentence_lengths) + 1))
            self.ax_sentences.plot(
                indices, sentence_lengths, marker="o", markersize=4,
                color=self.colors["chart_line"], linewidth=1.8, label="Sentence Words", zorder=3
            )
            # Add average line
            avg_len = stats.get("avg_sentence_length", 0.0)
            self.ax_sentences.axhline(
                avg_len, color=self.colors["borderline_amber"],
                linestyle=":", linewidth=1.2, label=f"Avg: {avg_len:.1f}w", zorder=2
            )
            self.ax_sentences.set_xlabel("Sentence Index (#)", color=self.colors["chart_text"], fontsize=8.5)
            self.ax_sentences.set_ylabel("Words Count", color=self.colors["chart_text"], fontsize=8.5)
            self.ax_sentences.set_title("Sentence Length Rhythm (Burstiness)", color=self.colors["chart_text"], fontsize=10, fontweight="bold", pad=8)
            self.ax_sentences.grid(True, linestyle="--", alpha=0.3, color=self.colors["chart_grid"], zorder=0)
            self.ax_sentences.legend(fontsize=7.5, facecolor=self.colors["bg_card"], edgecolor=self.colors["border"], labelcolor=self.colors["chart_text"])
        else:
            self.ax_sentences.text(
                0.5, 0.5, "Single sentence or no sentence boundaries detected",
                ha="center", va="center", color=self.colors["text_muted"],
                fontsize=9, transform=self.ax_sentences.transAxes
            )

        self.canvas.draw()

    def set_theme(self, new_theme: str):
        """Update chart background and element colors on theme switch."""
        self.current_theme = new_theme
        self.colors = THEME[self.current_theme]
        self.figure.patch.set_facecolor(self.colors["chart_bg"])
        self.show_placeholder()
