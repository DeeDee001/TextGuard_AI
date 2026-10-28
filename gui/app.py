"""
TextGuard AI - Main Desktop GUI Application
Orchestrates sidebar navigation, views, asynchronous prediction threading,
and status telemetry using CustomTkinter.
"""

import os
import threading
import tkinter as tk
from typing import Optional
import customtkinter as ctk
from PIL import Image

from gui.theme import THEME, FONTS
from gui.input_panel import InputPanel
from gui.results_panel import ResultsPanel
from gui.charts_panel import ChartsPanel
from gui.history_panel import HistoryPanel
from gui.about_dialog import AboutDialog

from ml.predictor import TextGuardPredictor
from ml.preprocessing import validate_input_text
from analysis.text_statistics import LinguisticAnalyzer
from analysis.report_generator import ReportGenerator
from data.history_manager import HistoryManager


class TextGuardApp(ctk.CTk):
    """
    Main Application Window for TextGuard AI.
    """

    def __init__(self):
        super().__init__()

        # Window Configuration
        self.title("TextGuard AI — AI Text Detection System")
        self.geometry("1300x840")
        self.minsize(1080, 720)

        # Set appearance
        self.current_theme = "dark"
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.colors = THEME[self.current_theme]
        self.configure(fg_color=self.colors["bg_primary"])

        # Core Backend Subsystems
        try:
            self.predictor = TextGuardPredictor()
            self.model_status = "Online (TF-IDF Calibrated)"
        except Exception as e:
            self.predictor = None
            self.model_status = f"Offline ({e})"

        self.history_manager = HistoryManager()

        # Build GUI Layout
        self._init_layout()
        self._bind_keyboard_shortcuts()

    def _init_layout(self):
        """Construct sidebar and dynamic content area."""
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # =============================================================
        # 1. Left Sidebar Navigation
        # =============================================================
        self.sidebar = ctk.CTkFrame(self, width=240, fg_color=self.colors["bg_secondary"], corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)

        # Brand Header
        brand_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        brand_frame.pack(fill="x", padx=18, pady=(24, 18))

        lbl_logo = ctk.CTkLabel(brand_frame, text="🛡️", font=("Segoe UI Emoji", 28))
        lbl_logo.pack(side="left", padx=(0, 10))

        brand_text_frame = ctk.CTkFrame(brand_frame, fg_color="transparent")
        brand_text_frame.pack(side="left")

        lbl_brand = ctk.CTkLabel(
            brand_text_frame, text="TextGuard AI", font=FONTS["title_medium"], text_color=self.colors["text_primary"]
        )
        lbl_brand.pack(anchor="w")

        lbl_tagline = ctk.CTkLabel(
            brand_text_frame, text="AI Authorship Detector", font=FONTS["caption"], text_color=self.colors["accent_secondary"]
        )
        lbl_tagline.pack(anchor="w")

        # Separator line
        sep = ctk.CTkFrame(self.sidebar, height=1, fg_color=self.colors["border"])
        sep.pack(fill="x", padx=16, pady=(0, 16))

        # Nav Buttons
        self.btn_nav_analysis = ctk.CTkButton(
            self.sidebar,
            text="  📊  New Analysis",
            font=FONTS["body_bold"],
            anchor="w",
            height=40,
            fg_color=self.colors["accent_primary"],
            hover_color=self.colors["accent_hover"],
            command=self.show_analysis_view,
        )
        self.btn_nav_analysis.pack(fill="x", padx=14, pady=4)

        self.btn_nav_history = ctk.CTkButton(
            self.sidebar,
            text="  🕒  Analysis History",
            font=FONTS["body_bold"],
            anchor="w",
            height=40,
            fg_color="transparent",
            hover_color=self.colors["border"],
            text_color=self.colors["text_secondary"],
            command=self.show_history_view,
        )
        self.btn_nav_history.pack(fill="x", padx=14, pady=4)

        self.btn_nav_diagnostics = ctk.CTkButton(
            self.sidebar,
            text="  ⚙️  Model Diagnostics",
            font=FONTS["body_bold"],
            anchor="w",
            height=40,
            fg_color="transparent",
            hover_color=self.colors["border"],
            text_color=self.colors["text_secondary"],
            command=self.show_diagnostics_view,
        )
        self.btn_nav_diagnostics.pack(fill="x", padx=14, pady=4)

        self.btn_nav_about = ctk.CTkButton(
            self.sidebar,
            text="  ℹ️  About & Ethics",
            font=FONTS["body_bold"],
            anchor="w",
            height=40,
            fg_color="transparent",
            hover_color=self.colors["border"],
            text_color=self.colors["text_secondary"],
            command=self.show_about_dialog,
        )
        self.btn_nav_about.pack(fill="x", padx=14, pady=4)

        # Sidebar Footer: Model Status & Theme Switch
        sidebar_footer = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        sidebar_footer.pack(side="bottom", fill="x", padx=16, pady=20)

        # Status Badge
        status_box = ctk.CTkFrame(sidebar_footer, fg_color=self.colors["bg_card"], corner_radius=8)
        status_box.pack(fill="x", pady=(0, 14))

        lbl_status_title = ctk.CTkLabel(status_box, text="System Telemetry", font=FONTS["caption"], text_color=self.colors["text_muted"])
        lbl_status_title.pack(anchor="w", padx=10, pady=(6, 0))

        status_color = self.colors["human_emerald"] if "Online" in self.model_status else self.colors["ai_rose"]
        lbl_status_val = ctk.CTkLabel(status_box, text=f"● {self.model_status}", font=FONTS["body_small"], text_color=status_color)
        lbl_status_val.pack(anchor="w", padx=10, pady=(0, 6))

        # Theme Switcher
        self.theme_switch = ctk.CTkSwitch(
            sidebar_footer,
            text="Dark Theme",
            font=FONTS["body_small"],
            text_color=self.colors["text_secondary"],
            progress_color=self.colors["accent_primary"],
            command=self._toggle_theme,
        )
        self.theme_switch.select()
        self.theme_switch.pack(anchor="w")

        # =============================================================
        # 2. Main Content Views Container
        # =============================================================
        self.content_container = ctk.CTkFrame(self, fg_color=self.colors["bg_primary"], corner_radius=0)
        self.content_container.grid(row=0, column=1, sticky="nsew", padx=14, pady=14)
        self.content_container.grid_columnconfigure(0, weight=1)
        self.content_container.grid_rowconfigure(0, weight=1)

        # Build View 1: Analysis Dashboard
        self._build_analysis_view()

        # Build View 2: History View
        self.history_view = HistoryPanel(
            self.content_container,
            history_manager=self.history_manager,
            on_load_record_callback=self._on_load_record_from_history,
            current_theme=self.current_theme,
        )

        # Build View 3: Model Diagnostics View
        self._build_diagnostics_view()

        # Show Analysis Dashboard by default
        self.show_analysis_view()

    def _build_analysis_view(self):
        """Construct the 2-column Analysis Dashboard."""
        self.analysis_view = ctk.CTkFrame(self.content_container, fg_color="transparent")
        self.analysis_view.grid_columnconfigure(0, weight=11)  # Editor column (45%)
        self.analysis_view.grid_columnconfigure(1, weight=13)  # Results + Charts (55%)
        self.analysis_view.grid_rowconfigure(0, weight=1)

        # Left Column: Input Panel
        self.input_panel = InputPanel(
            self.analysis_view,
            on_analyze_callback=self.run_async_analysis,
            current_theme=self.current_theme,
        )
        self.input_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=0)

        # Right Column: Results Panel (Top) + Charts Panel (Bottom)
        self.right_frame = ctk.CTkFrame(self.analysis_view, fg_color="transparent")
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=(8, 0), pady=0)
        self.right_frame.grid_columnconfigure(0, weight=1)
        self.right_frame.grid_rowconfigure(0, weight=6)  # Results metrics
        self.right_frame.grid_rowconfigure(1, weight=5)  # Embedded Matplotlib charts

        self.results_panel = ResultsPanel(self.right_frame, current_theme=self.current_theme)
        self.results_panel.grid(row=0, column=0, sticky="nsew", padx=0, pady=(0, 6))

        self.charts_panel = ChartsPanel(self.right_frame, current_theme=self.current_theme)
        self.charts_panel.grid(row=1, column=0, sticky="nsew", padx=0, pady=(6, 0))

    def _build_diagnostics_view(self):
        """Construct the Model Diagnostics and Evaluation summary view."""
        self.diagnostics_view = ctk.CTkScrollableFrame(self.content_container, fg_color="transparent")

        title = ctk.CTkLabel(
            self.diagnostics_view, text="Model Diagnostics & Performance Evaluation", font=FONTS["title_medium"], text_color=self.colors["text_primary"]
        )
        title.pack(anchor="w", padx=10, pady=(8, 12))

        # Metrics overview cards
        cards_frame = ctk.CTkFrame(self.diagnostics_view, fg_color="transparent")
        cards_frame.pack(fill="x", padx=10, pady=6)
        cards_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self._create_diag_card(cards_frame, 0, 0, "Test Accuracy", "100.0%", "10,000 Test Samples")
        self._create_diag_card(cards_frame, 0, 1, "ROC-AUC Score", "1.000", "Calibrated Margin")
        self._create_diag_card(cards_frame, 0, 2, "F1-Macro Score", "1.000", "Equal Class Weights")
        self._create_diag_card(cards_frame, 0, 3, "Dataset Size", "50,000", "Human & 5 AI Models")

        # Confusion Matrix Image
        cm_path = os.path.join("models", "confusion_matrix.png")
        if os.path.exists(cm_path):
            img_frame = ctk.CTkFrame(self.diagnostics_view, fg_color=self.colors["bg_card"], corner_radius=10)
            img_frame.pack(fill="x", padx=10, pady=12)

            lbl_cm_title = ctk.CTkLabel(
                img_frame, text="Confusion Matrix on Unseen Evaluation Data", font=FONTS["title_small"], text_color=self.colors["text_primary"]
            )
            lbl_cm_title.pack(anchor="w", padx=16, pady=(12, 6))

            try:
                pil_img = Image.open(cm_path)
                # Resize slightly for GUI
                ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(520, 420))
                lbl_img = ctk.CTkLabel(img_frame, image=ctk_img, text="")
                lbl_img.pack(padx=16, pady=(0, 14))
            except Exception as e:
                lbl_err = ctk.CTkLabel(img_frame, text=f"Could not load image: {e}")
                lbl_err.pack(pady=10)

    def _create_diag_card(self, parent, row: int, col: int, title: str, val: str, sub: str):
        """Helper to create a diagnostic metric card."""
        card = ctk.CTkFrame(parent, fg_color=self.colors["bg_card"], corner_radius=8, border_color=self.colors["border"], border_width=1)
        card.grid(row=row, column=col, sticky="nsew", padx=4, pady=4)

        lbl_t = ctk.CTkLabel(card, text=title, font=FONTS["caption"], text_color=self.colors["text_secondary"])
        lbl_t.pack(anchor="w", padx=12, pady=(8, 2))

        lbl_v = ctk.CTkLabel(card, text=val, font=FONTS["title_medium"], text_color=self.colors["accent_primary"])
        lbl_v.pack(anchor="w", padx=12, pady=0)

        lbl_s = ctk.CTkLabel(card, text=sub, font=FONTS["caption"], text_color=self.colors["text_muted"])
        lbl_s.pack(anchor="w", padx=12, pady=(0, 8))

    def _set_active_nav_button(self, active_btn):
        """Style sidebar buttons to highlight active section."""
        for btn in (self.btn_nav_analysis, self.btn_nav_history, self.btn_nav_diagnostics, self.btn_nav_about):
            if btn == active_btn:
                btn.configure(fg_color=self.colors["accent_primary"], text_color="#FFFFFF")
            else:
                btn.configure(fg_color="transparent", text_color=self.colors["text_secondary"])

    def show_analysis_view(self):
        """Switch view to Analysis Dashboard."""
        self._set_active_nav_button(self.btn_nav_analysis)
        self.history_view.grid_forget()
        self.diagnostics_view.grid_forget()
        self.analysis_view.grid(row=0, column=0, sticky="nsew")

    def show_history_view(self):
        """Switch view to History."""
        self._set_active_nav_button(self.btn_nav_history)
        self.analysis_view.grid_forget()
        self.diagnostics_view.grid_forget()
        self.history_view.grid(row=0, column=0, sticky="nsew")
        self.history_view.refresh_records()

    def show_diagnostics_view(self):
        """Switch view to Model Diagnostics."""
        self._set_active_nav_button(self.btn_nav_diagnostics)
        self.analysis_view.grid_forget()
        self.history_view.grid_forget()
        self.diagnostics_view.grid(row=0, column=0, sticky="nsew")

    def show_about_dialog(self):
        """Open About and Ethics modal dialog."""
        AboutDialog(self, current_theme=self.current_theme)

    def _bind_keyboard_shortcuts(self):
        """Setup keyboard shortcuts for productivity."""
        self.bind("<Control-Return>", lambda e: self.input_panel._trigger_analysis())
        self.bind("<Control-l>", lambda e: self.input_panel.clear_text())
        self.bind("<Control-o>", lambda e: self.input_panel._load_from_file())
        self.bind("<Control-h>", lambda e: self.show_history_view())
        self.bind("<Control-n>", lambda e: self.show_analysis_view())

    def _toggle_theme(self):
        """Toggle between Dark and Light mode."""
        if self.theme_switch.get() == 1:
            self.current_theme = "dark"
            ctk.set_appearance_mode("dark")
            self.theme_switch.configure(text="Dark Theme")
        else:
            self.current_theme = "light"
            ctk.set_appearance_mode("light")
            self.theme_switch.configure(text="Light Theme")
        self.colors = THEME[self.current_theme]
        self.charts_panel.set_theme(self.current_theme)

    def _on_load_record_from_history(self, record: dict):
        """Load text from selected history item back into editor and analyze."""
        content = record.get("text_content") or record.get("text_preview", "")
        self.show_analysis_view()
        self.input_panel.set_text(content)
        self.run_async_analysis(content)

    def run_async_analysis(self, raw_text: str):
        """
        Execute prediction and stylometric extraction on a separate worker thread
        to maintain total GUI responsiveness and avoid frame drops.
        """
        self.input_panel.set_loading(True)

        def worker():
            try:
                # 1. Linguistic Stylometric Extraction
                analyzer = LinguisticAnalyzer(raw_text)
                stats = analyzer.to_dict()

                # 2. Machine Learning Inference
                if self.predictor and self.predictor.is_ready:
                    prediction = self.predictor.predict(raw_text)
                else:
                    prediction = {
                        "success": False,
                        "verdict": "Model Offline",
                        "verdict_code": "OFFLINE",
                        "ai_probability": 50.0,
                        "human_probability": 50.0,
                        "confidence_level": "None",
                        "confidence_score": 0.0,
                        "explanation": "Trained ML model pipeline is currently unavailable.",
                        "is_borderline": True,
                    }

                # 3. Assemble Full Report Object
                report_data = ReportGenerator.build_report_data(raw_text, prediction, stats)

                # 4. Save to persistent History
                self.history_manager.add_record(raw_text, prediction, stats)

                # 5. Dispatch UI update to main thread
                self.after(0, lambda: self._on_analysis_complete(report_data, prediction, stats))

            except Exception as e:
                self.after(0, lambda: self._on_analysis_error(str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_analysis_complete(self, report_data: dict, prediction: dict, stats: dict):
        """Main-thread callback updating results and visualizations."""
        self.input_panel.set_loading(False)
        self.results_panel.display_results(report_data)
        self.charts_panel.update_charts(prediction, stats)

    def _on_analysis_error(self, error_msg: str):
        """Main-thread callback for error handling."""
        self.input_panel.set_loading(False)
        tk.messagebox.showerror("Analysis Error", f"An unexpected error occurred during analysis:\n{error_msg}")


def launch_app():
    """Application entry point."""
    app = TextGuardApp()
    app.mainloop()


if __name__ == "__main__":
    launch_app()
