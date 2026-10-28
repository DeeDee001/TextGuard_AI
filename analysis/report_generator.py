"""
TextGuard AI - Report Generator Module
Exports comprehensive analysis reports in TXT, JSON, and PDF formats.
Includes linguistic metrics, model verdict, confidence score, and ethical disclaimers.
"""

import datetime
import json
import os
from typing import Dict, Any, Optional

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


class ReportGenerator:
    """
    Generates multi-format analysis reports for auditing, archival,
    and educational evaluation.
    """

    @staticmethod
    def build_report_data(
        raw_text: str,
        prediction: Dict[str, Any],
        stats: Dict[str, Any],
        title: str = "TextGuard AI - Text Authorship Analysis Report"
    ) -> Dict[str, Any]:
        """
        Assemble all analysis components into a unified report dictionary.
        """
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return {
            "title": title,
            "generated_at": now_str,
            "application": "TextGuard AI v1.0.0",
            "institution": "American International University-Bangladesh (AIUB)",
            "course": "CSC 4162 - Programming in Python",
            "model_metadata": {
                "model_name": prediction.get("model_name", "TF-IDF Logistic Classifier"),
                "model_version": prediction.get("model_version", "1.0.0"),
                "is_calibrated": prediction.get("is_calibrated", True),
            },
            "prediction": {
                "verdict": prediction.get("verdict", "Unknown"),
                "verdict_code": prediction.get("verdict_code", "UNKNOWN"),
                "ai_probability_pct": prediction.get("ai_probability", 0.0),
                "human_probability_pct": prediction.get("human_probability", 0.0),
                "confidence_level": prediction.get("confidence_level", "Unknown"),
                "confidence_score": prediction.get("confidence_score", 0.0),
                "is_borderline": prediction.get("is_borderline", False),
                "explanation": prediction.get("explanation", ""),
            },
            "linguistic_metrics": {
                "word_count": stats.get("word_count", 0),
                "character_count_total": stats.get("char_count_total", 0),
                "character_count_no_spaces": stats.get("char_count_no_spaces", 0),
                "sentence_count": stats.get("sentence_count", 0),
                "paragraph_count": stats.get("paragraph_count", 0),
                "avg_word_length": stats.get("avg_word_length", 0.0),
                "avg_sentence_length_words": stats.get("avg_sentence_length", 0.0),
                "burstiness_sentence_std": stats.get("sentence_length_std", 0.0),
                "lexical_diversity_ttr": stats.get("lexical_diversity_ttr", 0.0),
                "root_ttr": stats.get("root_ttr", 0.0),
                "punctuation_count": stats.get("punctuation_count", 0),
                "punctuation_density_pct": stats.get("punctuation_density_pct", 0.0),
                "transition_word_count": stats.get("transition_word_count", 0),
                "flesch_reading_ease": stats.get("flesch_reading_ease", 0.0),
                "flesch_kincaid_grade": stats.get("flesch_kincaid_grade", 0.0),
                "top_repeated_words": stats.get("top_repeated_words", []),
            },
            "input_text": raw_text,
            "disclaimer": (
                "IMPORTANT NOTICE & ETHICAL DISCLAIMER: TextGuard AI is an educational machine learning "
                "text-classification system developed for academic purposes. Predictions and probability scores "
                "represent statistical likelihood based on stylometric n-grams and calibrated classifier margins. "
                "They DO NOT constitute definitive or legal proof of human or artificial intelligence authorship. "
                "Scores may be affected by formatting, domain shifts, non-native writing styles, or paraphrasing. "
                "This report must never be used as sole evidence for disciplinary or punitive actions."
            ),
        }

    @classmethod
    def export_json(cls, report_data: Dict[str, Any], filepath: str) -> str:
        """
        Export report data as formatted JSON.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=4, ensure_ascii=False)
        return filepath

    @classmethod
    def export_txt(cls, report_data: Dict[str, Any], filepath: str) -> str:
        """
        Export report data as a clean, human-readable text document.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        p = report_data["prediction"]
        m = report_data["linguistic_metrics"]
        lines = [
            "=" * 76,
            f"          {report_data['title'].upper()}",
            "=" * 76,
            f"Generated At : {report_data['generated_at']}",
            f"Application  : {report_data['application']}",
            f"Institution  : {report_data['institution']}",
            f"Course       : {report_data['course']}",
            "-" * 76,
            "1. PRIMARY PREDICTION & CLASSIFICATION SCORES",
            "-" * 76,
            f"Verdict             : {p['verdict']}",
            f"AI Likelihood Score : {p['ai_probability_pct']:.2f}%",
            f"Human Likelihood    : {p['human_probability_pct']:.2f}%",
            f"Confidence Level    : {p['confidence_level']} (Score: {p['confidence_score']:.1f}%)",
            f"Classification Model: {report_data['model_metadata']['model_name']} v{report_data['model_metadata']['model_version']}",
            "",
            "Reasoning & Summary:",
            f"  {p['explanation']}",
            "",
            "-" * 76,
            "2. LINGUISTIC & STYLOMETRIC METRICS",
            "-" * 76,
            f"  * Total Word Count              : {m['word_count']:,}",
            f"  * Total Characters              : {m['character_count_total']:,} (without spaces: {m['character_count_no_spaces']:,})",
            f"  * Sentence Count                : {m['sentence_count']:,}",
            f"  * Paragraph Count               : {m['paragraph_count']:,}",
            f"  * Average Word Length           : {m['avg_word_length']:.2f} characters",
            f"  * Average Sentence Length       : {m['avg_sentence_length_words']:.2f} words",
            f"  * Sentence Std Dev (Burstiness) : {m['burstiness_sentence_std']:.2f}",
            f"  * Lexical Diversity (TTR)       : {m['lexical_diversity_ttr']:.3f} (Root TTR: {m['root_ttr']:.2f})",
            f"  * Punctuation Count & Density   : {m['punctuation_count']} chars ({m['punctuation_density_pct']:.2f}%)",
            f"  * Transition Words Count        : {m['transition_word_count']}",
            f"  * Flesch Reading Ease Score     : {m['flesch_reading_ease']:.1f}/100",
            f"  * Flesch-Kincaid Grade Level    : Grade {m['flesch_kincaid_grade']:.1f}",
            "",
            "Top Content Words:",
        ]

        for word, count in m.get("top_repeated_words", []):
            lines.append(f"    - {word:<15}: {count} occurrences")

        lines.extend([
            "",
            "-" * 76,
            "3. ANALYZED TEXT PREVIEW",
            "-" * 76,
            report_data["input_text"],
            "",
            "-" * 76,
            "4. ETHICAL DISCLAIMER & LIMITATIONS",
            "-" * 76,
            report_data["disclaimer"],
            "=" * 76,
        ])

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return filepath

    @classmethod
    def export_pdf(cls, report_data: Dict[str, Any], filepath: str) -> str:
        """
        Export a professional, styled PDF document using ReportLab.
        """
        if not REPORTLAB_AVAILABLE:
            raise RuntimeError("ReportLab is not installed. PDF export is unavailable.")

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        doc = SimpleDocTemplate(
            filepath,
            pagesize=letter,
            rightMargin=45,
            leftMargin=45,
            topMargin=45,
            bottomMargin=45
        )

        styles = getSampleStyleSheet()

        # Custom Styles
        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Heading1"],
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#1E293B"),
            spaceAfter=4,
        )
        subtitle_style = ParagraphStyle(
            "DocSubtitle",
            parent=styles["Normal"],
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#64748B"),
            spaceAfter=12,
        )
        section_style = ParagraphStyle(
            "DocSection",
            parent=styles["Heading2"],
            fontSize=12,
            leading=16,
            textColor=colors.HexColor("#4338CA"),
            spaceBefore=10,
            spaceAfter=6,
        )
        body_style = ParagraphStyle(
            "DocBody",
            parent=styles["Normal"],
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#334155"),
        )
        disclaimer_style = ParagraphStyle(
            "DocDisclaimer",
            parent=styles["Normal"],
            fontSize=7.5,
            leading=10,
            textColor=colors.HexColor("#64748B"),
        )

        elements = []

        # Header Title
        elements.append(Paragraph("<b>TextGuard AI — Text Authorship Analysis Report</b>", title_style))
        elements.append(Paragraph(
            f"Generated: {report_data['generated_at']} | System: {report_data['application']} | "
            f"AIUB (CSC 4162 - Programming in Python)",
            subtitle_style
        ))
        elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#6366F1"), spaceAfter=10))

        # Prediction Verdict Box
        p = report_data["prediction"]
        verdict_color = "#10B981" if "Human" in p["verdict"] else ("#EF4444" if "AI" in p["verdict"] else "#F59E0B")
        
        verdict_data = [
            [
                Paragraph(f"<b>Prediction: {p['verdict']}</b>", ParagraphStyle(
                    "VerdictText", parent=styles["Heading2"], fontSize=13, leading=16, textColor=colors.HexColor(verdict_color)
                )),
                Paragraph(f"<b>AI Score: {p['ai_probability_pct']:.1f}%</b><br/>Human Score: {p['human_probability_pct']:.1f}%", body_style)
            ],
            [
                Paragraph(f"<b>Confidence:</b> {p['confidence_level']} (Score: {p['confidence_score']:.1f}%)", body_style),
                Paragraph(f"<b>Model:</b> {report_data['model_metadata']['model_name']}", body_style)
            ]
        ]
        verdict_table = Table(verdict_data, colWidths=[280, 220])
        verdict_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
            ('PADDING', (0, 0), (-1, -1), 8),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        elements.append(verdict_table)
        elements.append(Spacer(1, 8))

        # Explanation
        elements.append(Paragraph(f"<b>Analysis Summary:</b> {p['explanation']}", body_style))
        elements.append(Spacer(1, 10))

        # Linguistic Metrics Table
        elements.append(Paragraph("<b>Linguistic & Stylometric Measurements</b>", section_style))
        m = report_data["linguistic_metrics"]
        metric_data = [
            ["Metric", "Value", "Metric", "Value"],
            ["Total Words", f"{m['word_count']:,}", "Avg Word Length", f"{m['avg_word_length']:.2f} chars"],
            ["Total Characters", f"{m['character_count_total']:,}", "Avg Sentence Length", f"{m['avg_sentence_length_words']:.2f} words"],
            ["Sentence Count", f"{m['sentence_count']:,}", "Sentence Std Dev (Burstiness)", f"{m['burstiness_sentence_std']:.2f}"],
            ["Paragraph Count", f"{m['paragraph_count']:,}", "Lexical Diversity (TTR)", f"{m['lexical_diversity_ttr']:.3f}"],
            ["Punctuation Density", f"{m['punctuation_density_pct']:.2f}%", "Transition Words Count", f"{m['transition_word_count']}"],
            ["Flesch Reading Ease", f"{m['flesch_reading_ease']:.1f}/100", "Flesch-Kincaid Grade", f"Grade {m['flesch_kincaid_grade']:.1f}"],
        ]
        metric_table = Table(metric_data, colWidths=[140, 110, 140, 110])
        metric_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#EEF2FF")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor("#3730A3")),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8.5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ]))
        elements.append(metric_table)
        elements.append(Spacer(1, 12))

        # Text Sample Excerpt
        elements.append(Paragraph("<b>Submitted Text Sample</b>", section_style))
        sample_excerpt = report_data["input_text"][:600]
        if len(report_data["input_text"]) > 600:
            sample_excerpt += " ... [truncated in summary PDF, see full TXT/JSON]"
        elements.append(Paragraph(f"<i>{sample_excerpt}</i>", body_style))
        elements.append(Spacer(1, 14))

        # Disclaimer
        elements.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#E2E8F0"), spaceAfter=6))
        elements.append(Paragraph(f"<b>Ethical Notice & Disclaimer:</b> {report_data['disclaimer']}", disclaimer_style))

        doc.build(elements)
        return filepath
