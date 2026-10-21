#!/usr/bin/env python3
"""
DocVerify - UI Design System & Component Library
Clean, enterprise-grade SaaS design system with robust light and dark theme support.

Provides:
- Semantic color palettes for both light and dark modes
- CSS design tokens and base element overrides
- Document choices and verification policies
- Document label resolution and Grad-CAM heatmap overlays
"""

from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.cm as cm

# ==============================================================================
# 1. Enterprise CSS Stylesheet Generator
# ==============================================================================

def get_enterprise_css(theme: str = "dark") -> str:
    """
    Generates centralized CSS design tokens and component styles for the specified theme.
    Supports 'dark' (default) and 'light' modes with zero reliance on client-side JS attributes.
    """
    is_light = (theme == "light")

    if is_light:
        # Light Theme Palette (Clean Enterprise SaaS)
        bg = "#F5F7FA"
        surface = "#FFFFFF"
        surface_alt = "#F8FAFC"
        surface_raised = "#EFF6FF"
        border = "#DDE3EA"
        border_focus = "#2563EB"
        border_strong = "#CBD5E1"
        text_main = "#182230"
        text_secondary = "#667085"
        text_muted = "#667085"
        text_subtle = "#9AA3B3"
        accent = "#2563EB"
        accent_strong = "#1D4ED8"
        accent_hover = "#1E40AF"
        accent_light = "#EFF6FF"
        accent_border = "rgba(37, 99, 235, 0.25)"
        teal = "#0D9488"
        teal_light = "rgba(13, 148, 136, 0.08)"
        success = "#15803D"
        success_bg = "rgba(21, 128, 61, 0.08)"
        success_border = "rgba(21, 128, 61, 0.25)"
        warning = "#B45309"
        warning_bg = "rgba(180, 83, 9, 0.08)"
        warning_border = "rgba(180, 83, 9, 0.25)"
        danger = "#B42318"
        danger_bg = "rgba(180, 35, 24, 0.08)"
        danger_border = "rgba(180, 35, 24, 0.25)"
        shadow_sm = "0 1px 3px rgba(16, 24, 40, 0.05)"
        shadow_md = "0 4px 12px rgba(16, 24, 40, 0.08)"
        shadow_lg = "0 8px 24px rgba(16, 24, 40, 0.12)"
    else:
        # Dark Theme Palette (Deep Navy / Slate Enterprise)
        bg = "#0B1220"
        surface = "#111B2E"
        surface_alt = "#162238"
        surface_raised = "#1B2C47"
        border = "#2A3950"
        border_focus = "#60A5FA"
        border_strong = "#374151"
        text_main = "#F1F5F9"
        text_secondary = "#A6B3C5"
        text_muted = "#A6B3C5"
        text_subtle = "#64748B"
        accent = "#60A5FA"
        accent_strong = "#2563EB"
        accent_hover = "#93C5FD"
        accent_light = "#1B2C47"
        accent_border = "rgba(96, 165, 250, 0.25)"
        teal = "#2DD4BF"
        teal_light = "rgba(45, 212, 191, 0.10)"
        success = "#4ADE80"
        success_bg = "rgba(74, 222, 128, 0.10)"
        success_border = "rgba(74, 222, 128, 0.30)"
        warning = "#FBBF24"
        warning_bg = "rgba(251, 191, 36, 0.10)"
        warning_border = "rgba(251, 191, 36, 0.30)"
        danger = "#F87171"
        danger_bg = "rgba(248, 113, 113, 0.10)"
        danger_border = "rgba(248, 113, 113, 0.30)"
        shadow_sm = "0 1px 3px rgba(0, 0, 0, 0.30)"
        shadow_md = "0 4px 16px rgba(0, 0, 0, 0.40)"
        shadow_lg = "0 8px 32px rgba(0, 0, 0, 0.55)"

    css = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {{
        --dv-bg: {bg};
        --dv-surface: {surface};
        --dv-surface-alt: {surface_alt};
        --dv-surface-raised: {surface_raised};
        --dv-border: {border};
        --dv-border-focus: {border_focus};
        --dv-border-strong: {border_strong};
        --dv-text-main: {text_main};
        --dv-text-secondary: {text_secondary};
        --dv-text-muted: {text_muted};
        --dv-text-subtle: {text_subtle};
        --dv-accent: {accent};
        --dv-accent-strong: {accent_strong};
        --dv-accent-hover: {accent_hover};
        --dv-accent-light: {accent_light};
        --dv-accent-border: {accent_border};
        --dv-teal: {teal};
        --dv-teal-light: {teal_light};
        --dv-success: {success};
        --dv-success-bg: {success_bg};
        --dv-success-border: {success_border};
        --dv-warning: {warning};
        --dv-warning-bg: {warning_bg};
        --dv-warning-border: {warning_border};
        --dv-danger: {danger};
        --dv-danger-bg: {danger_bg};
        --dv-danger-border: {danger_border};
        --dv-radius: 8px;
        --dv-radius-sm: 6px;
        --dv-radius-xs: 4px;
        --dv-shadow-sm: {shadow_sm};
        --dv-shadow-md: {shadow_md};
        --dv-shadow-lg: {shadow_lg};
    }}

    /* Global Typography & Reset */
    html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        color: var(--dv-text-main) !important;
        -webkit-font-smoothing: antialiased;
    }}

    /* Streamlit Root Application Background */
    .stApp {{
        background-color: var(--dv-bg) !important;
        color: var(--dv-text-main) !important;
        min-height: 100vh;
    }}

    [data-testid="stAppViewContainer"] {{
        background-color: var(--dv-bg) !important;
        color: var(--dv-text-main) !important;
    }}

    header[data-testid="stHeader"] {{
        background: transparent !important;
        height: 0 !important;
    }}

    #MainMenu, footer {{
        visibility: hidden;
    }}

    .block-container {{
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        max-width: 1240px !important;
        margin: 0 auto !important;
    }}

    /* Header Container */
    .dv-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 14px 20px;
        background: var(--dv-surface);
        border: 1px solid var(--dv-border);
        border-radius: var(--dv-radius);
        box-shadow: var(--dv-shadow-sm);
        min-height: 64px;
    }}

    .dv-brand-title {{
        display: flex;
        align-items: center;
        gap: 12px;
    }}

    .dv-brand-icon {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 36px;
        height: 36px;
        background: var(--dv-accent-light);
        color: var(--dv-accent);
        border: 1px solid var(--dv-accent-border);
        border-radius: var(--dv-radius-sm);
        flex-shrink: 0;
    }}

    .dv-brand-wordmark {{
        font-size: 1.2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: var(--dv-text-main);
        line-height: 1.2;
    }}

    .dv-brand-subtitle {{
        font-size: 0.78rem;
        color: var(--dv-text-secondary);
        font-weight: 400;
        margin-top: 1px;
    }}

    .dv-header-badges {{
        display: flex;
        align-items: center;
        gap: 8px;
        flex-shrink: 0;
    }}

    .dv-env-badge {{
        background: var(--dv-surface-alt);
        border: 1px solid var(--dv-border);
        border-radius: var(--dv-radius-xs);
        padding: 4px 9px;
        font-size: 0.72rem;
        font-weight: 500;
        color: var(--dv-text-secondary);
    }}

    .dv-privacy-pill {{
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: var(--dv-surface-alt);
        border: 1px solid var(--dv-border);
        border-radius: var(--dv-radius-xs);
        padding: 4px 8px;
        font-size: 0.7rem;
        font-weight: 500;
        color: var(--dv-text-secondary);
    }}

    .dv-status-pill {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: var(--dv-success-bg);
        border: 1px solid var(--dv-success-border);
        border-radius: 9999px;
        padding: 4px 10px;
        font-size: 0.72rem;
        font-weight: 600;
        color: var(--dv-success);
    }}

    .dv-status-dot {{
        width: 6px;
        height: 6px;
        background: var(--dv-success);
        border-radius: 50%;
    }}

    /* Workflow Stage Indicator Tracker */
    .dv-workflow-stages {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: var(--dv-surface);
        border: 1px solid var(--dv-border);
        border-radius: var(--dv-radius);
        padding: 12px 20px;
        margin-bottom: 1.25rem;
        box-shadow: var(--dv-shadow-sm);
    }}

    .dv-workflow-step {{
        display: flex;
        align-items: center;
        gap: 10px;
    }}

    .dv-step-num {{
        width: 28px;
        height: 28px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.8rem;
        font-weight: 700;
        background: var(--dv-surface-alt);
        border: 1px solid var(--dv-border);
        color: var(--dv-text-secondary);
        flex-shrink: 0;
    }}

    .dv-workflow-step.active .dv-step-num {{
        background: var(--dv-accent-strong);
        border-color: var(--dv-accent-strong);
        color: #FFFFFF;
    }}

    .dv-workflow-step.completed .dv-step-num {{
        background: var(--dv-success);
        border-color: var(--dv-success);
        color: #FFFFFF;
    }}

    .dv-step-title {{
        font-size: 0.84rem;
        font-weight: 600;
        color: var(--dv-text-main);
        display: block;
        line-height: 1.2;
    }}

    .dv-step-desc {{
        font-size: 0.72rem;
        color: var(--dv-text-muted);
        display: block;
    }}

    .dv-step-line {{
        flex: 1;
        height: 1px;
        background: var(--dv-border);
        margin: 0 16px;
    }}

    .dv-step-line.active {{
        background: var(--dv-accent);
    }}

    /* Section Stage Headers */
    .dv-section-header {{
        display: flex;
        align-items: center;
        gap: 10px;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 8px;
        border-bottom: 1px solid var(--dv-border);
    }}

    .dv-stage-pill {{
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 3px 8px;
        border-radius: var(--dv-radius-xs);
        background: var(--dv-accent-light);
        color: var(--dv-accent);
        border: 1px solid var(--dv-accent-border);
    }}

    .dv-section-title {{
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--dv-text-main);
        margin: 0;
        letter-spacing: -0.01em;
    }}

    /* Page Typography */
    .dv-page-title {{
        font-size: 1.45rem;
        font-weight: 700;
        color: var(--dv-text-main);
        letter-spacing: -0.02em;
        margin-bottom: 4px;
        margin-top: 1.25rem;
    }}

    .dv-page-subtitle {{
        font-size: 0.88rem;
        color: var(--dv-text-secondary);
        margin-bottom: 1.25rem;
        line-height: 1.45;
    }}

    .dv-card-title {{
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--dv-text-main);
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 8px;
    }}

    .dv-card-title svg {{
        color: var(--dv-accent);
        flex-shrink: 0;
    }}

    .dv-card-desc {{
        font-size: 0.82rem;
        color: var(--dv-text-secondary);
        margin-bottom: 14px;
        line-height: 1.5;
    }}

    .dv-card-subtle {{
        font-size: 0.75rem;
        color: var(--dv-text-muted);
        font-weight: 500;
    }}

    .dv-settings-label {{
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--dv-text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }}

    .dv-policy-desc {{
        font-size: 0.78rem;
        color: var(--dv-text-secondary);
        margin-top: 4px;
        margin-bottom: 8px;
        line-height: 1.4;
    }}

    /* Card Containers */
    .dv-card {{
        background: var(--dv-surface);
        border: 1px solid var(--dv-border);
        border-radius: var(--dv-radius);
        padding: 20px;
        box-shadow: var(--dv-shadow-sm);
        margin-bottom: 1rem;
    }}

    .dv-preview-card {{
        background: var(--dv-surface);
        border: 1px solid var(--dv-border);
        border-radius: var(--dv-radius);
        padding: 16px 18px;
        margin-bottom: 1rem;
        box-shadow: var(--dv-shadow-sm);
    }}

    /* Document Preview Image Sizing & Centering Constraints */
    div[data-testid="stImage"]:has(img[alt*="Loaded Document"]),
    div[data-testid="stImage"]:has(img[caption*="Loaded Document"]) {{
        max-width: 520px !important;
        margin-left: auto !important;
        margin-right: auto !important;
        width: 100% !important;
    }}

    div[data-testid="stImage"]:has(img[alt*="Loaded Document"]) img,
    div[data-testid="stImage"]:has(img[caption*="Loaded Document"]) img {{
        max-width: 520px !important;
        max-height: 380px !important;
        width: auto !important;
        height: auto !important;
        object-fit: contain !important;
        display: block !important;
        margin-left: auto !important;
        margin-right: auto !important;
        border-radius: var(--dv-radius-sm) !important;
        border: 1px solid var(--dv-border) !important;
        box-shadow: var(--dv-shadow-md) !important;
    }}

    .dv-meta-bar {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px 18px;
        background: var(--dv-surface-alt);
        border: 1px solid var(--dv-border);
        border-radius: var(--dv-radius-sm);
        padding: 9px 14px;
        margin-top: 10px;
        font-size: 0.82rem;
    }}

    .dv-meta-item {{
        display: flex;
        align-items: center;
        gap: 6px;
        color: var(--dv-text-secondary);
    }}

    .dv-meta-val {{
        font-weight: 600;
        color: var(--dv-text-main);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
    }}

    .dv-file-note {{
        font-size: 0.75rem;
        color: var(--dv-text-secondary);
        margin-top: 8px;
        line-height: 1.4;
    }}

    /* Streamlit File Uploader Override */
    div[data-testid="stFileUploader"] section {{
        background: var(--dv-surface) !important;
        border: 1.5px dashed var(--dv-border) !important;
        border-radius: var(--dv-radius) !important;
        padding: 24px 20px !important;
        transition: border-color 0.15s ease, background 0.15s ease !important;
    }}

    div[data-testid="stFileUploader"] section:hover {{
        border-color: var(--dv-accent) !important;
        background: var(--dv-accent-light) !important;
    }}

    div[data-testid="stFileUploader"] section small,
    div[data-testid="stFileUploader"] section span,
    div[data-testid="stFileUploader"] section p {{
        color: var(--dv-text-secondary) !important;
    }}

    div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {{
        background: var(--dv-surface-alt) !important;
        border: 1px solid var(--dv-border) !important;
        border-radius: var(--dv-radius-sm) !important;
        color: var(--dv-text-main) !important;
    }}

    div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] small {{
        color: var(--dv-text-secondary) !important;
    }}

    /* Native Buttons */
    div.stButton > button,
    div[data-testid="stButton"] > button,
    button[data-testid="baseButton-secondary"],
    button[data-testid="baseButton-primary"] {{
        border-radius: var(--dv-radius-sm) !important;
        font-family: inherit !important;
        font-weight: 600 !important;
        font-size: 0.86rem !important;
        padding: 0.55rem 1.15rem !important;
        transition: all 0.15s ease !important;
        min-height: 40px !important;
    }}

    div.stButton > button[kind="primary"],
    button[data-testid="baseButton-primary"],
    div[data-testid="stDownloadButton"] > button {{
        background-color: var(--dv-accent-strong) !important;
        border: 1px solid var(--dv-accent-strong) !important;
        color: #FFFFFF !important;
        box-shadow: var(--dv-shadow-sm) !important;
    }}

    div.stButton > button[kind="primary"]:hover,
    button[data-testid="baseButton-primary"]:hover,
    div[data-testid="stDownloadButton"] > button:hover {{
        background-color: var(--dv-accent-hover) !important;
        border-color: var(--dv-accent-hover) !important;
    }}

    div.stButton > button[kind="secondary"],
    button[data-testid="baseButton-secondary"],
    div[data-testid="stButton"] button:not([kind="primary"]):not([data-testid="baseButton-primary"]) {{
        background-color: var(--dv-surface-alt) !important;
        border: 1px solid var(--dv-border) !important;
        color: var(--dv-text-main) !important;
    }}

    div.stButton > button[kind="secondary"]:hover,
    button[data-testid="baseButton-secondary"]:hover,
    div[data-testid="stButton"] button:not([kind="primary"]):not([data-testid="baseButton-primary"]):hover {{
        background-color: var(--dv-surface-raised) !important;
        border-color: var(--dv-border-strong) !important;
        color: var(--dv-accent) !important;
    }}

    div.stButton > button[kind="secondary"] *,
    button[data-testid="baseButton-secondary"] *,
    div[data-testid="stButton"] button:not([kind="primary"]):not([data-testid="baseButton-primary"]) * {{
        color: inherit !important;
    }}

    div.stButton > button:disabled,
    div.stButton > button[disabled] {{
        background-color: var(--dv-surface-alt) !important;
        border: 1px solid var(--dv-border) !important;
        color: var(--dv-text-subtle) !important;
        opacity: 0.6 !important;
        cursor: not-allowed !important;
    }}

    /* Radio Controls */
    div[data-testid="stRadio"] > div {{
        gap: 6px;
    }}

    div[data-testid="stRadio"] label {{
        background: var(--dv-surface-alt) !important;
        border: 1px solid var(--dv-border) !important;
        border-radius: var(--dv-radius-sm) !important;
        padding: 8px 12px !important;
        cursor: pointer !important;
        transition: border-color 0.15s ease, background 0.15s ease !important;
    }}

    div[data-testid="stRadio"] label:hover {{
        border-color: var(--dv-accent-border) !important;
        background: var(--dv-accent-light) !important;
    }}

    div[data-testid="stRadio"] label p,
    div[data-testid="stRadio"] label span {{
        font-size: 0.84rem !important;
        font-weight: 500 !important;
        color: var(--dv-text-main) !important;
    }}

    div[data-testid="stRadio"] label:has(input:checked) {{
        border-color: var(--dv-accent) !important;
        background: var(--dv-accent-light) !important;
    }}

    /* Style the radio circle indicator itself */
    div[data-testid="stRadio"] input[type="radio"] {{
        accent-color: var(--dv-accent) !important;
    }}

    div[data-testid="stRadio"] label [data-checked="true"],
    div[data-testid="stRadio"] label [aria-checked="true"],
    div[data-testid="stRadio"] label div[style*="background-color: rgb(255, 75, 75)"],
    div[data-testid="stRadio"] label div[style*="background-color: #ff4b4b"],
    div[data-testid="stRadio"] label div[style*="background-color:#ff4b4b"] {{
        background-color: var(--dv-accent) !important;
    }}

    div[data-testid="stRadio"] svg {{
        fill: var(--dv-accent) !important;
    }}

    /* Selectbox Overrides */
    div[data-baseweb="select"] > div {{
        background-color: var(--dv-surface) !important;
        border-color: var(--dv-border) !important;
        border-radius: var(--dv-radius-sm) !important;
        color: var(--dv-text-main) !important;
        min-height: 40px !important;
    }}

    div[data-baseweb="select"] span {{
        color: var(--dv-text-main) !important;
    }}

    div[data-baseweb="select"] svg {{
        fill: var(--dv-text-secondary) !important;
    }}

    div[data-baseweb="popover"],
    ul[data-baseweb="menu"] {{
        background-color: var(--dv-surface) !important;
        border: 1px solid var(--dv-border) !important;
        border-radius: var(--dv-radius-sm) !important;
        box-shadow: var(--dv-shadow-md) !important;
    }}

    li[data-baseweb="menu-item"] {{
        background-color: var(--dv-surface) !important;
        color: var(--dv-text-main) !important;
        font-size: 0.84rem !important;
        padding: 8px 12px !important;
    }}

    li[data-baseweb="menu-item"]:hover {{
        background-color: var(--dv-accent-light) !important;
        color: var(--dv-accent) !important;
    }}

    /* Verdict Banner Card */
    .dv-verdict {{
        border-radius: var(--dv-radius);
        padding: 18px 22px;
        margin-bottom: 1.25rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 16px;
        box-shadow: var(--dv-shadow-sm);
    }}

    .dv-verdict-low {{
        background: var(--dv-success-bg);
        border: 1px solid var(--dv-success-border);
    }}
    .dv-verdict-low .dv-verdict-title {{ color: var(--dv-success); }}
    .dv-verdict-low .dv-verdict-badge {{
        background: var(--dv-surface);
        color: var(--dv-success);
        border: 1px solid var(--dv-success-border);
    }}
    .dv-verdict-low .dv-verdict-score-val {{ color: var(--dv-success); }}

    .dv-verdict-med {{
        background: var(--dv-warning-bg);
        border: 1px solid var(--dv-warning-border);
    }}
    .dv-verdict-med .dv-verdict-title {{ color: var(--dv-warning); }}
    .dv-verdict-med .dv-verdict-badge {{
        background: var(--dv-surface);
        color: var(--dv-warning);
        border: 1px solid var(--dv-warning-border);
    }}
    .dv-verdict-med .dv-verdict-score-val {{ color: var(--dv-warning); }}

    .dv-verdict-high {{
        background: var(--dv-danger-bg);
        border: 1px solid var(--dv-danger-border);
    }}
    .dv-verdict-high .dv-verdict-title {{ color: var(--dv-danger); }}
    .dv-verdict-high .dv-verdict-badge {{
        background: var(--dv-surface);
        color: var(--dv-danger);
        border: 1px solid var(--dv-danger-border);
    }}
    .dv-verdict-high .dv-verdict-score-val {{ color: var(--dv-danger); }}

    .dv-verdict-badge {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 3px 8px;
        border-radius: var(--dv-radius-xs);
        margin-bottom: 4px;
    }}

    .dv-verdict-title {{
        font-size: 1.15rem;
        font-weight: 700;
        letter-spacing: -0.015em;
        margin: 0;
        line-height: 1.3;
    }}

    .dv-verdict-desc {{
        font-size: 0.84rem;
        color: var(--dv-text-secondary);
        margin-top: 4px;
        max-width: 650px;
        line-height: 1.45;
    }}

    .dv-verdict-score-box {{
        text-align: right;
        min-width: 130px;
        padding-left: 16px;
        border-left: 1px solid var(--dv-border);
        flex-shrink: 0;
    }}

    .dv-verdict-score-val {{
        font-size: 2.2rem;
        font-weight: 800;
        line-height: 1;
        letter-spacing: -0.03em;
    }}

    .dv-verdict-score-label {{
        font-size: 0.68rem;
        font-weight: 700;
        color: var(--dv-text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }}

    .dv-verdict-score-sub {{
        font-size: 0.68rem;
        color: var(--dv-text-muted);
        margin-top: 2px;
    }}

    /* Metric Tiles */
    .dv-metric-tile {{
        background: var(--dv-surface);
        border: 1px solid var(--dv-border);
        border-radius: var(--dv-radius-sm);
        padding: 12px 14px;
        box-shadow: var(--dv-shadow-sm);
        transition: border-color 0.15s ease, transform 0.15s ease;
    }}

    .dv-metric-tile:hover {{
        border-color: var(--dv-accent-border);
        transform: translateY(-1px);
    }}

    .dv-metric-label {{
        font-size: 0.68rem;
        font-weight: 700;
        color: var(--dv-text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }}

    .dv-metric-value {{
        font-size: 1rem;
        font-weight: 700;
        color: var(--dv-text-main);
        word-break: break-word;
        line-height: 1.3;
    }}

    .dv-metric-note {{
        font-size: 0.7rem;
        color: var(--dv-text-muted);
        margin-top: 3px;
    }}

    /* Evidence Tabs (Regression Guard for Test 16) */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        border-bottom: 1.5px solid var(--dv-border) !important;
        padding-bottom: 0;
        overflow-x: auto;
        flex-wrap: nowrap;
        scrollbar-width: thin;
        background: transparent !important;
    }}

    .stTabs [data-baseweb="tab"] {{
        height: auto !important;
        min-height: 44px !important;
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: unset !important;
        background: transparent !important;
        border-radius: var(--dv-radius-xs) var(--dv-radius-xs) 0 0 !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        color: var(--dv-text-secondary) !important;
        padding: 8px 16px !important;
        border: none !important;
        display: inline-flex !important;
        align-items: center !important;
        gap: 6px !important;
        transition: color 0.15s ease, background 0.15s ease !important;
    }}

    .stTabs [data-baseweb="tab"] p,
    .stTabs [data-baseweb="tab"] span,
    .stTabs [data-baseweb="tab"] div {{
        color: var(--dv-text-secondary) !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }}

    .stTabs [data-baseweb="tab"]:hover {{
        background: var(--dv-accent-light) !important;
    }}

    .stTabs [data-baseweb="tab"]:hover p,
    .stTabs [data-baseweb="tab"]:hover span,
    .stTabs [data-baseweb="tab"]:hover div {{
        color: var(--dv-text-main) !important;
    }}

    .stTabs [data-baseweb="tab"][aria-selected="true"],
    .stTabs [aria-selected="true"] {{
        background: transparent !important;
        border-bottom: 2px solid var(--dv-accent) !important;
        box-shadow: none !important;
    }}

    .stTabs [data-baseweb="tab"][aria-selected="true"] p,
    .stTabs [data-baseweb="tab"][aria-selected="true"] span,
    .stTabs [data-baseweb="tab"][aria-selected="true"] div,
    .stTabs [aria-selected="true"] p,
    .stTabs [aria-selected="true"] span,
    .stTabs [aria-selected="true"] div {{
        color: var(--dv-accent) !important;
        font-weight: 700 !important;
    }}

    .stTabs [data-baseweb="tab-highlight"] {{
        background-color: var(--dv-accent) !important;
    }}

    /* Field Inspection Cards */
    .dv-field-card {{
        background: var(--dv-surface);
        border: 1px solid var(--dv-border);
        border-radius: var(--dv-radius-sm);
        padding: 10px 12px;
        display: flex;
        flex-direction: column;
        gap: 6px;
        box-shadow: var(--dv-shadow-sm);
        margin-bottom: 8px;
    }}

    .dv-field-name {{
        font-size: 0.82rem;
        font-weight: 600;
        color: var(--dv-text-main);
    }}

    .dv-field-badge-low {{
        display: inline-block;
        font-size: 0.68rem;
        font-weight: 600;
        color: var(--dv-success);
        background: var(--dv-success-bg);
        padding: 2px 7px;
        border-radius: var(--dv-radius-xs);
        border: 1px solid var(--dv-success-border);
    }}

    .dv-field-badge-high {{
        display: inline-block;
        font-size: 0.68rem;
        font-weight: 600;
        color: var(--dv-danger);
        background: var(--dv-danger-bg);
        padding: 2px 7px;
        border-radius: var(--dv-radius-xs);
        border: 1px solid var(--dv-danger-border);
    }}

    /* Calibration & Decision Rule */
    .dv-calibration-card {{
        background: var(--dv-surface);
        border: 1px solid var(--dv-border);
        border-radius: var(--dv-radius-sm);
        padding: 16px 18px;
        margin-bottom: 12px;
        box-shadow: var(--dv-shadow-sm);
    }}

    .dv-calibration-label {{
        font-size: 0.72rem;
        font-weight: 700;
        color: var(--dv-text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }}

    .dv-calibration-value {{
        font-size: 1.85rem;
        font-weight: 800;
        color: var(--dv-accent);
        letter-spacing: -0.02em;
        line-height: 1.1;
    }}

    .dv-calibration-note {{
        font-size: 0.78rem;
        color: var(--dv-text-secondary);
        margin-top: 6px;
        line-height: 1.45;
    }}

    .dv-decision-rule {{
        background: var(--dv-surface-alt);
        border-left: 3px solid var(--dv-accent);
        border-radius: 0 var(--dv-radius-sm) var(--dv-radius-sm) 0;
        padding: 10px 14px;
        font-size: 0.78rem;
        color: var(--dv-text-secondary);
        line-height: 1.45;
        margin-top: 8px;
    }}

    /* Analysis Loading Card */
    .dv-analysis-card {{
        background: var(--dv-surface);
        border: 1px solid var(--dv-accent-border);
        border-radius: var(--dv-radius);
        padding: 18px 22px;
        margin: 0.75rem 0;
        box-shadow: var(--dv-shadow-sm);
    }}

    .dv-analysis-title {{
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--dv-accent);
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }}

    .dv-spin {{
        animation: dv-spin 1.2s linear infinite;
    }}

    @keyframes dv-spin {{
        0% {{ transform: rotate(0deg); }}
        100% {{ transform: rotate(360deg); }}
    }}

    .dv-analysis-steps {{
        display: flex;
        flex-direction: column;
        gap: 6px;
    }}

    .dv-step-item {{
        font-size: 0.82rem;
        color: var(--dv-text-secondary);
        display: flex;
        align-items: center;
        gap: 8px;
    }}

    .dv-step-check {{
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: var(--dv-success);
        display: inline-block;
    }}

    .dv-step-active {{
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: var(--dv-accent);
        display: inline-block;
    }}

    .dv-step-pending {{
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: var(--dv-text-subtle);
        display: inline-block;
    }}

    /* Expander Override (Theme Unified) */
    div[data-testid="stExpander"] {{
        background-color: var(--dv-surface) !important;
        border: 1px solid var(--dv-border) !important;
        border-radius: var(--dv-radius-sm) !important;
        box-shadow: var(--dv-shadow-sm) !important;
        overflow: hidden !important;
        margin-top: 1.25rem !important;
    }}

    div[data-testid="stExpander"] details {{
        background-color: var(--dv-surface) !important;
        border: none !important;
    }}

    div[data-testid="stExpander"] summary {{
        background-color: var(--dv-surface) !important;
        color: var(--dv-text-main) !important;
        padding: 10px 14px !important;
        border-bottom: 1px solid var(--dv-border) !important;
    }}

    div[data-testid="stExpander"] summary:hover {{
        background-color: var(--dv-surface-alt) !important;
        color: var(--dv-accent) !important;
    }}

    div[data-testid="stExpander"] summary p,
    div[data-testid="stExpander"] summary span {{
        color: var(--dv-text-main) !important;
        font-weight: 600 !important;
        font-size: 0.86rem !important;
    }}

    div[data-testid="stExpander"] summary svg {{
        fill: var(--dv-text-secondary) !important;
    }}

    div[data-testid="stExpander"] [data-testid="stExpanderDetails"] {{
        background-color: var(--dv-surface) !important;
        color: var(--dv-text-main) !important;
        padding: 14px 18px !important;
    }}

    div[data-testid="stExpander"] [data-testid="stExpanderDetails"] p {{
        color: var(--dv-text-secondary) !important;
        font-size: 0.84rem !important;
        margin-bottom: 6px !important;
    }}

    div[data-testid="stExpander"] [data-testid="stExpanderDetails"] strong {{
        color: var(--dv-text-main) !important;
    }}

    /* Tables */
    div[data-testid="stTable"] table {{
        border-collapse: collapse !important;
        width: 100% !important;
        font-size: 0.84rem !important;
        background-color: var(--dv-surface) !important;
        border: 1px solid var(--dv-border) !important;
        border-radius: var(--dv-radius-sm) !important;
        overflow: hidden !important;
    }}

    div[data-testid="stTable"] th {{
        background-color: var(--dv-surface-alt) !important;
        color: var(--dv-text-secondary) !important;
        font-weight: 600 !important;
        font-size: 0.74rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        padding: 9px 12px !important;
        border-bottom: 1px solid var(--dv-border) !important;
    }}

    div[data-testid="stTable"] td {{
        background-color: var(--dv-surface) !important;
        color: var(--dv-text-main) !important;
        padding: 9px 12px !important;
        border-bottom: 1px solid var(--dv-border) !important;
        vertical-align: middle !important;
    }}

    div[data-testid="stTable"] tr:last-child td {{
        border-bottom: none !important;
    }}

    div[data-testid="stTable"] tr:hover td {{
        background-color: var(--dv-accent-light) !important;
    }}

    /* Alerts */
    div[data-testid="stAlert"] {{
        background-color: var(--dv-surface-alt) !important;
        border: 1px solid var(--dv-border) !important;
        border-radius: var(--dv-radius-sm) !important;
        color: var(--dv-text-main) !important;
        font-size: 0.84rem !important;
    }}

    /* Code styling */
    code {{
        background-color: var(--dv-surface-alt) !important;
        color: var(--dv-accent) !important;
        border: 1px solid var(--dv-border) !important;
        border-radius: 4px !important;
        padding: 1px 5px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.82em !important;
    }}

    /* Footer */
    .dv-footer {{
        text-align: center;
        padding: 24px 0 12px;
        color: var(--dv-text-muted);
        font-size: 0.76rem;
        border-top: 1px solid var(--dv-border);
        margin-top: 2.5rem;
        line-height: 1.6;
    }}

    /* Responsive adjustments */
    @media (max-width: 768px) {{
        .block-container {{
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }}
        .dv-header {{
            flex-direction: column;
            align-items: flex-start;
            gap: 10px;
            padding: 12px 14px;
        }}
        .dv-header-badges {{
            width: 100%;
            justify-content: flex-start;
            flex-wrap: wrap;
        }}
        .dv-workflow-stages {{
            flex-direction: column;
            align-items: flex-start;
            gap: 12px;
            padding: 12px 14px;
        }}
        .dv-step-line {{
            display: none;
        }}
        .dv-verdict {{
            flex-direction: column;
            align-items: flex-start;
        }}
        .dv-verdict-score-box {{
            border-left: none;
            border-top: 1px solid var(--dv-border);
            padding-left: 0;
            padding-top: 12px;
            text-align: left;
            min-width: 0;
            width: 100%;
        }}
        .dv-verdict-score-val {{
            font-size: 1.75rem;
        }}
    }}
</style>
"""
    return css

# Default stylesheet export for backward compatibility
ENTERPRISE_CSS = get_enterprise_css("dark")

# ==============================================================================
# 2. Friendly Display Name Mappings & Helpers
# ==============================================================================
DOCUMENT_TYPE_CHOICES = [
    ("auto", "Automatic Detection (AI Layout Matcher)"),
    ("alb_id", "Albania — National Identity Card"),
    ("aze_passport", "Azerbaijan — International Passport"),
    ("esp_id", "Spain — National Identity Card (DNI)"),
    ("est_id", "Estonia — National Identity Card"),
    ("fin_id", "Finland — National Identity Card"),
    ("grc_passport", "Greece — International Passport"),
    ("lva_passport", "Latvia — International Passport"),
    ("rus_internalpassport", "Russia — Internal Passport"),
    ("srb_passport", "Serbia — International Passport"),
    ("svk_id", "Slovakia — National Identity Card"),
    ("svk_passport", "Slovakia — International Passport")
]

POLICY_CHOICES = [
    ("balanced", "Balanced Risk Policy (Default)", "Optimal decision boundary based on Youden's J statistic (Threshold: 72%)"),
    ("conservative", "High-Recall Anti-Fraud", "Zero tolerance for fraudulent documents; prioritizes fraud capture (Threshold: 50%)"),
    ("strict", "Low False Alarm Policy", "Designed for automated self-service check-in to avoid innocent rejections (Threshold: 80%)")
]


def resolve_document_display_label(template_id: str) -> str:
    """Returns clean, human-friendly document type name without technical IDs."""
    mapping = {
        "alb_id": "Albania — National Identity Card",
        "aze_passport": "Azerbaijan — International Passport",
        "esp_id": "Spain — National Identity Card",
        "est_id": "Estonia — National Identity Card",
        "fin_id": "Finland — National Identity Card",
        "grc_passport": "Greece — International Passport",
        "lva_passport": "Latvia — International Passport",
        "rus_internalpassport": "Russia — Internal Passport",
        "srb_passport": "Serbia — International Passport",
        "svk_id": "Slovakia — National Identity Card",
        "svk_passport": "Slovakia — International Passport",
        "unknown": "Standard Identity Document"
    }
    return mapping.get(str(template_id).lower(), "Standard Identity Document")


def overlay_cam_map(pil_img: Image.Image, cam_map: np.ndarray, alpha: float = 0.55) -> Image.Image:
    """Overlays normalized heatmap on PIL image with jet colormap."""
    w, h = pil_img.size
    colormap = cm.get_cmap("jet")
    colored_cam = colormap(cam_map)
    cam_uint8 = (colored_cam[:, :, :3] * 255).astype(np.uint8)
    cam_pil = Image.fromarray(cam_uint8).resize((w, h), Image.Resampling.BILINEAR)
    return Image.blend(pil_img.convert("RGB"), cam_pil, alpha=alpha)
