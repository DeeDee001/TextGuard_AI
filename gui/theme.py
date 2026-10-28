"""
TextGuard AI - GUI Theme & Styling System
Defines colors, typography, paddings, and aesthetic presets.
"""

# Color Palette - Professional Deep Slate & Indigo Theme
THEME = {
    "dark": {
        "bg_primary": "#0B0F19",       # Deep navy canvas
        "bg_secondary": "#111827",     # Sidebar & header panels
        "bg_card": "#1E293B",          # Elevated result & metric cards
        "bg_card_hover": "#283548",    # Hover state for interactive cards
        "border": "#334155",           # Subtle border lines
        "text_primary": "#F8FAFC",     # Bright primary readable text
        "text_secondary": "#94A3B8",   # Subtle secondary labels
        "text_muted": "#64748B",       # Muted placeholders & captions
        
        # Accents
        "accent_primary": "#6366F1",   # Vibrant Indigo
        "accent_hover": "#4F46E5",     # Darker Indigo for button press/hover
        "accent_secondary": "#8B5CF6", # Soft Violet
        
        # Verdict Colors
        "human_emerald": "#10B981",    # Emerald green for Human-written
        "human_bg": "#064E3B",         # Subtle green badge background
        "ai_rose": "#F43F5E",          # Rose/Coral for AI-generated
        "ai_bg": "#881337",            # Subtle red badge background
        "borderline_amber": "#F59E0B", # Amber for Inconclusive/Borderline
        "borderline_bg": "#78350F",    # Subtle amber badge background
        
        # Chart Colors
        "chart_bg": "#1E293B",
        "chart_text": "#E2E8F0",
        "chart_grid": "#334155",
        "chart_bar_human": "#10B981",
        "chart_bar_ai": "#F43F5E",
        "chart_line": "#818CF8",
    },
    "light": {
        "bg_primary": "#F8FAFC",
        "bg_secondary": "#F1F5F9",
        "bg_card": "#FFFFFF",
        "bg_card_hover": "#E2E8F0",
        "border": "#CBD5E1",
        "text_primary": "#0F172A",
        "text_secondary": "#475569",
        "text_muted": "#94A3B8",
        
        "accent_primary": "#4F46E5",
        "accent_hover": "#4338CA",
        "accent_secondary": "#7C3AED",
        
        "human_emerald": "#059669",
        "human_bg": "#D1FAE5",
        "ai_rose": "#E11D48",
        "ai_bg": "#FFE4E6",
        "borderline_amber": "#D97706",
        "borderline_bg": "#FEF3C7",
        
        "chart_bg": "#FFFFFF",
        "chart_text": "#1E293B",
        "chart_grid": "#E2E8F0",
        "chart_bar_human": "#059669",
        "chart_bar_ai": "#E11D48",
        "chart_line": "#4F46E5",
    }
}

# Typography
FONTS = {
    "title_large": ("Segoe UI", 20, "bold"),
    "title_medium": ("Segoe UI", 16, "bold"),
    "title_small": ("Segoe UI", 13, "bold"),
    "body_bold": ("Segoe UI", 11, "bold"),
    "body_normal": ("Segoe UI", 11),
    "body_small": ("Segoe UI", 10),
    "caption": ("Segoe UI", 9),
    "mono": ("Consolas", 10),
}
