#!/usr/bin/env python3
"""
DocVerify AI - Enterprise UI Design System & Component Library
Premium Dark Navy SaaS Theme (#06142F, #0A1F44, #075BFF, #2583FF, #FFFFFF, #EAF2FF, #152B50, #A9B9D3)
Author: Senior UI/UX Designer & Full-Stack Frontend Engineer
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
    Supports 'dark' (default: Deep Navy + Electric Blue) and 'light' (Clean SaaS White + Ice Blue).
    """
    is_light = (theme == "light")

    if is_light:
        # Light Theme Palette (Clean SaaS White & Ice Blue)
        bg = "#FFFFFF"
        bg_alt = "#F8FAFC"
        surface = "#FFFFFF"
        surface_alt = "#F0F5FD"
        surface_raised = "#EAF2FF"
        surface_hero = "#F4F8FF"
        card_white = "#FFFFFF"
        card_white_text = "#06142F"
        card_white_border = "#DDE5F2"
        border = "#DDE5F2"
        border_focus = "#075BFF"
        border_strong = "#CAD6E8"
        text_main = "#06142F"
        text_secondary = "#475569"
        text_muted = "#64748B"
        text_subtle = "#94A3B8"
        accent = "#075BFF"
        accent_bright = "#2583FF"
        accent_hover = "#0047D4"
        accent_light = "#EAF2FF"
        accent_border = "rgba(7, 91, 255, 0.25)"
        slate_blue = "#E2EAF8"
        success = "#0D9488"
        success_bg = "#E6F8F5"
        success_border = "#99F6E4"
        warning = "#D97706"
        warning_bg = "#FEF3C7"
        warning_border = "#FDE68A"
        danger = "#E11D48"
        danger_bg = "#FFE4E6"
        danger_border = "#FECDD3"
        shadow_sm = "0 1px 3px rgba(6, 20, 47, 0.05)"
        shadow_md = "0 6px 18px rgba(6, 20, 47, 0.08)"
        shadow_lg = "0 12px 32px rgba(6, 20, 47, 0.12)"
        shadow_card = "0 20px 40px rgba(6, 20, 47, 0.10)"
        nav_bg = "rgba(255, 255, 255, 0.95)"
    else:
        # Dark Theme Palette (Deep Navy + Electric Blue + White)
        bg = "#06142F"
        bg_alt = "#0A1F44"
        surface = "#0A1F44"
        surface_alt = "#0D2552"
        surface_raised = "#152B50"
        surface_hero = "#071838"
        card_white = "#FFFFFF"
        card_white_text = "#06142F"
        card_white_border = "rgba(255, 255, 255, 0.18)"
        border = "#152B50"
        border_focus = "#075BFF"
        border_strong = "#213C6E"
        text_main = "#FFFFFF"
        text_secondary = "#A9B9D3"
        text_muted = "#7E92B5"
        text_subtle = "#506587"
        accent = "#075BFF"
        accent_bright = "#2583FF"
        accent_hover = "#4094FF"
        accent_light = "#0D275A"
        accent_border = "rgba(7, 91, 255, 0.35)"
        slate_blue = "#152B50"
        success = "#10B981"
        success_bg = "rgba(16, 185, 129, 0.12)"
        success_border = "rgba(16, 185, 129, 0.35)"
        warning = "#F59E0B"
        warning_bg = "rgba(245, 158, 11, 0.12)"
        warning_border = "rgba(245, 158, 11, 0.35)"
        danger = "#F43F5E"
        danger_bg = "rgba(244, 63, 94, 0.12)"
        danger_border = "rgba(244, 63, 94, 0.35)"
        shadow_sm = "0 1px 3px rgba(0, 0, 0, 0.35)"
        shadow_md = "0 8px 24px rgba(0, 0, 0, 0.45)"
        shadow_lg = "0 16px 48px rgba(0, 0, 0, 0.60)"
        shadow_card = "0 24px 48px rgba(0, 0, 0, 0.50)"
        nav_bg = "rgba(6, 20, 47, 0.92)"

    css = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {{
        --dv-bg: {bg};
        --dv-bg-alt: {bg_alt};
        --dv-surface: {surface};
        --dv-surface-alt: {surface_alt};
        --dv-surface-raised: {surface_raised};
        --dv-surface-hero: {surface_hero};
        --dv-card-white: {card_white};
        --dv-card-white-text: {card_white_text};
        --dv-card-white-border: {card_white_border};
        --dv-border: {border};
        --dv-border-focus: {border_focus};
        --dv-border-strong: {border_strong};
        --dv-text-main: {text_main};
        --dv-text-secondary: {text_secondary};
        --dv-text-muted: {text_muted};
        --dv-text-subtle: {text_subtle};
        --dv-accent: {accent};
        --dv-accent-bright: {accent_bright};
        --dv-accent-hover: {accent_hover};
        --dv-accent-light: {accent_light};
        --dv-accent-border: {accent_border};
        --dv-slate-blue: {slate_blue};
        --dv-ice-blue: #EAF2FF;
        --dv-success: {success};
        --dv-success-bg: {success_bg};
        --dv-success-border: {success_border};
        --dv-warning: {warning};
        --dv-warning-bg: {warning_bg};
        --dv-warning-border: {warning_border};
        --dv-danger: {danger};
        --dv-danger-bg: {danger_bg};
        --dv-danger-border: {danger_border};
        --dv-radius: 12px;
        --dv-radius-sm: 8px;
        --dv-radius-lg: 18px;
        --dv-shadow-sm: {shadow_sm};
        --dv-shadow-md: {shadow_md};
        --dv-shadow-lg: {shadow_lg};
        --dv-shadow-card: {shadow_card};
        --dv-nav-bg: {nav_bg};
    }}

    /* Global Typography & Reset */
    html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        color: var(--dv-text-main) !important;
        -webkit-font-smoothing: antialiased;
        text-rendering: optimizeLegibility;
    }}

    /* Streamlit Root Application Background */
    .stApp {{
        background-color: var(--dv-bg) !important;
        color: var(--dv-text-main) !important;
        min-height: 100vh;
    }}

    [data-testid="stAppViewContainer"] {{
        background-color: var(--dv-bg) !important;
        background-image: radial-gradient(circle at 50% 0%, rgba(7, 91, 255, 0.12) 0%, transparent 60%);
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    /* Hide default Streamlit header bar decoration */
    header[data-testid="stHeader"] {{
        background: transparent !important;
        height: 0px !important;
    }}

    /* Main Container Padding */
    .main .block-container {{
        max-width: 1280px !important;
        padding-top: 1.2rem !important;
        padding-bottom: 4rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }}

    /* Modern Navbar Container */
    .dv-navbar {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: var(--dv-nav-bg);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--dv-border);
        border-radius: var(--dv-radius);
        padding: 12px 24px;
        margin-bottom: 24px;
        box-shadow: var(--dv-shadow-md);
        position: relative;
        z-index: 100;
    }}

    .dv-nav-brand {{
        display: flex;
        align-items: center;
        gap: 12px;
        text-decoration: none;
        cursor: pointer;
    }}

    .dv-nav-logo-icon {{
        width: 36px;
        height: 36px;
        border-radius: 10px;
        background: linear-gradient(135deg, #075BFF 0%, #2583FF 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        color: #FFFFFF;
        box-shadow: 0 4px 12px rgba(7, 91, 255, 0.4);
    }}

    .dv-nav-brand-title {{
        font-size: 1.25rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: var(--dv-text-main);
        line-height: 1.1;
    }}

    .dv-nav-brand-badge {{
        font-size: 0.65rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        background: rgba(7, 91, 255, 0.18);
        color: var(--dv-accent-bright);
        border: 1px solid var(--dv-accent-border);
        padding: 2px 7px;
        border-radius: 4px;
        margin-left: 6px;
        display: inline-block;
        vertical-align: middle;
    }}

    .dv-nav-links {{
        display: flex;
        align-items: center;
        gap: 8px;
    }}

    /* Hero Section Styling */
    .dv-hero-wrap {{
        padding: 40px 0 32px 0;
        position: relative;
    }}

    .dv-hero-pill {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(7, 91, 255, 0.12);
        border: 1px solid var(--dv-accent-border);
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        color: var(--dv-accent-bright);
        margin-bottom: 20px;
    }}

    .dv-hero-pill-dot {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #10B981;
        box-shadow: 0 0 10px #10B981;
        animation: pulseDot 2s infinite;
    }}

    @keyframes pulseDot {{
        0%, 100% {{ opacity: 1; transform: scale(1); }}
        50% {{ opacity: 0.5; transform: scale(0.85); }}
    }}

    .dv-hero-h1 {{
        font-size: 3.25rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.035em !important;
        line-height: 1.15 !important;
        color: var(--dv-text-main) !important;
        margin-bottom: 18px !important;
    }}

    .dv-gradient-text {{
        background: linear-gradient(135deg, #FFFFFF 30%, #2583FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}

    .dv-hero-sub {{
        font-size: 1.125rem;
        line-height: 1.6;
        color: var(--dv-text-secondary);
        max-width: 580px;
        margin-bottom: 28px;
    }}

    .dv-hero-features-list {{
        display: flex;
        flex-direction: column;
        gap: 12px;
        margin-bottom: 32px;
    }}

    .dv-hero-feat-item {{
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 0.925rem;
        font-weight: 600;
        color: var(--dv-text-main);
    }}

    .dv-hero-feat-icon {{
        width: 22px;
        height: 22px;
        border-radius: 6px;
        background: rgba(7, 91, 255, 0.15);
        color: var(--dv-accent-bright);
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }}

    /* Premium White Upload Card (Hero Right Side) */
    .dv-upload-card-white {{
        background: var(--dv-card-white) !important;
        color: var(--dv-card-white-text) !important;
        border: 1px solid var(--dv-card-white-border);
        border-radius: var(--dv-radius-lg);
        padding: 32px 28px;
        box-shadow: var(--dv-shadow-card);
        position: relative;
        overflow: hidden;
    }}

    .dv-upload-card-white::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #075BFF 0%, #2583FF 100%);
    }}

    .dv-upload-card-title {{
        font-size: 1.4rem;
        font-weight: 700;
        color: #06142F !important;
        letter-spacing: -0.02em;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 10px;
    }}

    .dv-upload-card-subtitle {{
        font-size: 0.875rem;
        color: #475569 !important;
        line-height: 1.45;
        margin-bottom: 20px;
    }}

    /* Standard SaaS Container Card */
    .dv-card {{
        background: var(--dv-surface);
        border: 1px solid var(--dv-border);
        border-radius: var(--dv-radius);
        padding: 24px;
        box-shadow: var(--dv-shadow-md);
        margin-bottom: 24px;
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    }}

    .dv-card:hover {{
        border-color: var(--dv-accent-border);
    }}

    .dv-card-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 16px;
        border-bottom: 1px solid var(--dv-border);
        padding-bottom: 12px;
    }}

    .dv-card-title-main {{
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--dv-text-main);
        letter-spacing: -0.01em;
        display: flex;
        align-items: center;
        gap: 10px;
    }}

    /* Modern SaaS Workflow Stages */
    .dv-workflow-pipeline {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: var(--dv-surface);
        border: 1px solid var(--dv-border);
        border-radius: var(--dv-radius);
        padding: 16px 24px;
        margin: 20px 0 28px 0;
        box-shadow: var(--dv-shadow-sm);
        overflow-x: auto;
    }}

    .dv-pipeline-step {{
        display: flex;
        align-items: center;
        gap: 10px;
        flex-shrink: 0;
    }}

    .dv-pipeline-num {{
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: var(--dv-surface-raised);
        border: 1px solid var(--dv-border);
        color: var(--dv-text-muted);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.85rem;
        transition: all 0.2s ease;
    }}

    .dv-pipeline-step.active .dv-pipeline-num {{
        background: var(--dv-accent);
        border-color: var(--dv-accent-bright);
        color: #FFFFFF;
        box-shadow: 0 0 12px rgba(7, 91, 255, 0.4);
    }}

    .dv-pipeline-step.completed .dv-pipeline-num {{
        background: var(--dv-success-bg);
        border-color: var(--dv-success-border);
        color: var(--dv-success);
    }}

    .dv-pipeline-text {{
        display: flex;
        flex-direction: column;
    }}

    .dv-pipeline-title {{
        font-size: 0.875rem;
        font-weight: 600;
        color: var(--dv-text-main);
    }}

    .dv-pipeline-desc {{
        font-size: 0.725rem;
        color: var(--dv-text-muted);
    }}

    .dv-pipeline-divider {{
        flex: 1;
        height: 2px;
        background: var(--dv-border);
        margin: 0 16px;
        min-width: 24px;
    }}

    .dv-pipeline-divider.active {{
        background: linear-gradient(90deg, var(--dv-accent) 0%, var(--dv-accent-bright) 100%);
    }}

    /* Circular Risk Score Gauge */
    .dv-gauge-container {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 24px 16px;
    }}

    .dv-gauge-score-value {{
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1;
        color: var(--dv-text-main);
    }}

    .dv-gauge-score-label {{
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--dv-text-muted);
        margin-top: 6px;
    }}

    /* Verdict Card Styles */
    .dv-verdict-box {{
        border-radius: var(--dv-radius);
        padding: 24px;
        border: 1px solid var(--dv-border);
        background: var(--dv-surface);
        box-shadow: var(--dv-shadow-md);
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }}

    .dv-verdict-box.authentic {{
        border-color: var(--dv-success-border);
        background: linear-gradient(180deg, var(--dv-success-bg) 0%, var(--dv-surface) 100%);
    }}

    .dv-verdict-box.review {{
        border-color: var(--dv-warning-border);
        background: linear-gradient(180deg, var(--dv-warning-bg) 0%, var(--dv-surface) 100%);
    }}

    .dv-verdict-box.forged {{
        border-color: var(--dv-danger-border);
        background: linear-gradient(180deg, var(--dv-danger-bg) 0%, var(--dv-surface) 100%);
    }}

    .dv-verdict-pill {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        padding: 4px 12px;
        border-radius: 6px;
        margin-bottom: 12px;
    }}

    .dv-verdict-pill.authentic {{
        background: var(--dv-success);
        color: #FFFFFF;
    }}

    .dv-verdict-pill.review {{
        background: var(--dv-warning);
        color: #FFFFFF;
    }}

    .dv-verdict-pill.forged {{
        background: var(--dv-danger);
        color: #FFFFFF;
    }}

    /* Metric Stat Tile */
    .dv-stat-tile {{
        background: var(--dv-surface);
        border: 1px solid var(--dv-border);
        border-radius: var(--dv-radius);
        padding: 16px 20px;
        box-shadow: var(--dv-shadow-sm);
        display: flex;
        flex-direction: column;
        gap: 4px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }}

    .dv-stat-tile:hover {{
        border-color: var(--dv-accent-border);
        transform: translateY(-2px);
    }}

    .dv-stat-label {{
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--dv-text-muted);
    }}

    .dv-stat-value {{
        font-size: 1.35rem;
        font-weight: 700;
        color: var(--dv-text-main);
        letter-spacing: -0.02em;
    }}

    .dv-stat-hint {{
        font-size: 0.725rem;
        color: var(--dv-text-secondary);
    }}

    /* Supported Document Card */
    .dv-doc-card {{
        background: var(--dv-surface);
        border: 1px solid var(--dv-border);
        border-radius: var(--dv-radius);
        padding: 24px;
        box-shadow: var(--dv-shadow-sm);
        display: flex;
        flex-direction: column;
        gap: 12px;
        height: 100%;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }}

    .dv-doc-card:hover {{
        border-color: var(--dv-accent);
        transform: translateY(-3px);
        box-shadow: var(--dv-shadow-md);
    }}

    .dv-doc-icon {{
        width: 44px;
        height: 44px;
        border-radius: 10px;
        background: var(--dv-surface-raised);
        border: 1px solid var(--dv-border);
        display: flex;
        align-items: center;
        justify-content: center;
        color: var(--dv-accent-bright);
    }}

    .dv-doc-name {{
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--dv-text-main);
    }}

    .dv-doc-desc {{
        font-size: 0.85rem;
        color: var(--dv-text-secondary);
        line-height: 1.5;
    }}

    .dv-doc-tags {{
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-top: auto;
    }}

    .dv-doc-tag {{
        font-size: 0.7rem;
        font-weight: 600;
        background: var(--dv-surface-raised);
        border: 1px solid var(--dv-border);
        color: var(--dv-text-secondary);
        padding: 2px 8px;
        border-radius: 4px;
    }}

    /* Extracted Information Pill */
    .dv-extracted-row {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 16px;
        background: var(--dv-surface-raised);
        border: 1px solid var(--dv-border);
        border-radius: var(--dv-radius-sm);
        margin-bottom: 8px;
    }}

    .dv-extracted-key {{
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--dv-text-secondary);
        display: flex;
        align-items: center;
        gap: 8px;
    }}

    .dv-extracted-val {{
        font-size: 0.9rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        color: var(--dv-text-main);
    }}

    /* Feature Deep Dive Grid */
    .dv-feat-grid-card {{
        background: var(--dv-surface);
        border: 1px solid var(--dv-border);
        border-radius: var(--dv-radius);
        padding: 24px;
        box-shadow: var(--dv-shadow-sm);
        transition: all 0.2s ease;
        height: 100%;
    }}

    .dv-feat-grid-card:hover {{
        border-color: var(--dv-accent);
        transform: translateY(-2px);
    }}

    .dv-feat-icon-box {{
        width: 40px;
        height: 40px;
        border-radius: 8px;
        background: rgba(7, 91, 255, 0.12);
        color: var(--dv-accent-bright);
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 16px;
    }}

    .dv-feat-grid-title {{
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--dv-text-main);
        margin-bottom: 8px;
    }}

    .dv-feat-grid-desc {{
        font-size: 0.85rem;
        color: var(--dv-text-secondary);
        line-height: 1.55;
    }}

    /* Stage Verification Animation Card */
    .dv-stage-item {{
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 14px;
        background: var(--dv-surface);
        border: 1px solid var(--dv-border);
        border-radius: var(--dv-radius-sm);
        margin-bottom: 8px;
        font-size: 0.85rem;
        font-weight: 600;
    }}

    .dv-stage-dot-done {{
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: var(--dv-success);
        box-shadow: 0 0 8px var(--dv-success);
    }}

    .dv-stage-dot-spin {{
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: var(--dv-accent-bright);
        box-shadow: 0 0 8px var(--dv-accent-bright);
        animation: pulseDot 1s infinite;
    }}

    /* Streamlit Button & UI Elements Overrides */
    div.stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, #075BFF 0%, #2583FF 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        padding: 12px 24px !important;
        border: none !important;
        border-radius: var(--dv-radius-sm) !important;
        box-shadow: 0 4px 16px rgba(7, 91, 255, 0.4) !important;
        transition: all 0.2s ease !important;
    }}

    div.stButton > button[kind="primary"]:hover {{
        background: linear-gradient(135deg, #004BD6 0%, #075BFF 100%) !important;
        box-shadow: 0 6px 20px rgba(7, 91, 255, 0.55) !important;
        transform: translateY(-1px) !important;
    }}

    div.stButton > button[kind="secondary"] {{
        background: var(--dv-surface-raised) !important;
        border: 1px solid var(--dv-border) !important;
        color: var(--dv-text-main) !important;
        font-weight: 600 !important;
        border-radius: var(--dv-radius-sm) !important;
        transition: all 0.2s ease !important;
    }}

    div.stButton > button[kind="secondary"]:hover {{
        border-color: var(--dv-accent-border) !important;
        color: var(--dv-accent-bright) !important;
    }}

    /* Streamlit File Uploader Override */
    [data-testid="stFileUploader"] {{
        background: var(--dv-surface);
        border: 2px dashed var(--dv-border);
        border-radius: var(--dv-radius);
        padding: 16px;
        transition: border-color 0.2s ease;
    }}

    [data-testid="stFileUploader"]:hover {{
        border-color: var(--dv-accent);
    }}

    /* Streamlit Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px !important;
        background: var(--dv-surface) !important;
        padding: 6px !important;
        border-radius: var(--dv-radius) !important;
        border: 1px solid var(--dv-border) !important;
    }}

    .stTabs [data-baseweb="tab"] {{
        background: transparent !important;
        border-radius: var(--dv-radius-sm) !important;
        color: var(--dv-text-secondary) !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        padding: 8px 16px !important;
        border: none !important;
        transition: all 0.2s ease !important;
    }}

    .stTabs [aria-selected="true"] {{
        background: var(--dv-accent) !important;
        color: #FFFFFF !important;
        box-shadow: 0 2px 8px rgba(7, 91, 255, 0.35) !important;
    }}

    /* Expander Overrides */
    .streamlit-expanderHeader {{
        background: var(--dv-surface) !important;
        border: 1px solid var(--dv-border) !important;
        border-radius: var(--dv-radius-sm) !important;
        font-weight: 600 !important;
    }}

    /* SaaS Modern Footer */
    .dv-footer {{
        margin-top: 64px;
        padding-top: 32px;
        padding-bottom: 24px;
        border-top: 1px solid var(--dv-border);
        color: var(--dv-text-muted);
        font-size: 0.85rem;
        display: flex;
        flex-direction: column;
        gap: 12px;
        align-items: center;
        text-align: center;
    }}

    .dv-footer-links {{
        display: flex;
        gap: 20px;
        align-items: center;
    }}

    .dv-footer-links span {{
        color: var(--dv-text-secondary);
        font-size: 0.8rem;
    }}
</style>
"""
    return css

# Default stylesheet export for backward compatibility
ENTERPRISE_CSS = get_enterprise_css("dark")

# ==============================================================================
# 2. Friendly Display Name Mappings & Helpers (Preserved & Extended)
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
        "esp_id": "Spain — National Identity Card (DNI)",
        "est_id": "Estonia — National Identity Card",
        "fin_id": "Finland — National Identity Card",
        "grc_passport": "Greece — International Passport",
        "lva_passport": "Latvia — International Passport",
        "rus_internalpassport": "Russia — Internal Passport",
        "srb_passport": "Serbia — International Passport",
        "svk_id": "Slovakia — National Identity Card",
        "svk_passport": "Slovakia — International Passport",
        "aadhaar": "Aadhaar Card (UIDAI)",
        "pan": "PAN Card (Income Tax Dept)",
        "voter_id": "Voter ID Card (Election Commission)",
        "driving_licence": "Driving Licence (MoRTH)",
        "passport": "International Passport",
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


# ==============================================================================
# 3. SVG Visual Component Generators
# ==============================================================================

def render_circular_risk_gauge(risk_score: float, risk_level: str) -> str:
    """
    Renders an animated SVG circular gauge displaying the precise calculated risk score.
    Colors dynamically adapt: Green for Low Risk, Amber for Medium Risk, Red for High Risk.
    """
    score_clamped = max(0.0, min(100.0, float(risk_score)))
    radius = 64
    circumference = 2 * 3.14159 * radius
    offset = circumference - (score_clamped / 100.0) * circumference

    if "LOW" in risk_level or score_clamped < 40.0:
        stroke_color = "#10B981"
        glow_color = "rgba(16, 185, 129, 0.4)"
        status_text = "LOW RISK"
    elif "HIGH" in risk_level or score_clamped >= 72.0:
        stroke_color = "#F43F5E"
        glow_color = "rgba(244, 63, 94, 0.4)"
        status_text = "HIGH RISK"
    else:
        stroke_color = "#F59E0B"
        glow_color = "rgba(245, 158, 11, 0.4)"
        status_text = "MEDIUM RISK"

    return f"""<div style="display:flex; flex-direction:column; align-items:center; justify-content:center; padding:16px;">
<svg width="180" height="180" viewBox="0 0 160 160">
<circle cx="80" cy="80" r="{radius}" fill="none" stroke="rgba(255, 255, 255, 0.08)" stroke-width="12" />
<circle cx="80" cy="80" r="{radius}" fill="none" stroke="{stroke_color}" stroke-width="12" stroke-dasharray="{circumference}" stroke-dashoffset="{offset}" stroke-linecap="round" transform="rotate(-90 80 80)" style="transition: stroke-dashoffset 1s ease; filter: drop-shadow(0 0 8px {glow_color});" />
<text x="80" y="74" text-anchor="middle" dominant-baseline="central" font-family="'Plus Jakarta Sans', sans-serif" font-size="28" font-weight="800" fill="currentColor">{score_clamped:.1f}%</text>
<text x="80" y="100" text-anchor="middle" dominant-baseline="central" font-family="'Plus Jakarta Sans', sans-serif" font-size="11" font-weight="700" fill="{stroke_color}" letter-spacing="1">{status_text}</text>
</svg>
</div>"""


def render_hero_scan_graphic() -> str:
    """
    Renders an attractive SVG illustration of a government identity card being scanned
    with optical bounding boxes and laser beam for the hero section.
    """
    return """<div style="position:relative; width:100%; max-width:440px; margin:0 auto; perspective:1000px;">
<svg viewBox="0 0 400 250" width="100%" height="250" style="border-radius:16px; box-shadow:0 20px 50px rgba(7, 91, 255, 0.25);">
<defs>
<linearGradient id="cardGrad" x1="0" y1="0" x2="1" y2="1">
<stop offset="0%" stop-color="#0A1F44"/>
<stop offset="100%" stop-color="#152B50"/>
</linearGradient>
<linearGradient id="scanBeam" x1="0" y1="0" x2="0" y2="1">
<stop offset="0%" stop-color="#075BFF" stop-opacity="0"/>
<stop offset="85%" stop-color="#2583FF" stop-opacity="0.3"/>
<stop offset="100%" stop-color="#075BFF" stop-opacity="0.9"/>
</linearGradient>
</defs>
<rect x="20" y="20" width="360" height="210" rx="14" fill="url(#cardGrad)" stroke="#213C6E" stroke-width="2"/>
<path d="M 30,80 Q 100,50 180,80 T 360,80" fill="none" stroke="rgba(7, 91, 255, 0.15)" stroke-width="1.5"/>
<path d="M 30,110 Q 120,80 200,110 T 360,110" fill="none" stroke="rgba(7, 91, 255, 0.15)" stroke-width="1.5"/>
<path d="M 30,140 Q 110,160 210,140 T 360,140" fill="none" stroke="rgba(7, 91, 255, 0.15)" stroke-width="1.5"/>
<rect x="42" y="52" width="76" height="96" rx="8" fill="#06142F" stroke="#075BFF" stroke-width="1.5"/>
<circle cx="80" cy="88" r="18" fill="#213C6E"/>
<path d="M 58,136 Q 80,112 102,136" fill="#213C6E"/>
<rect x="86" y="130" width="38" height="16" rx="4" fill="#075BFF"/>
<text x="105" y="141" text-anchor="middle" font-family="'Plus Jakarta Sans', sans-serif" font-size="8" font-weight="700" fill="#FFFFFF">VERIFIED</text>
<rect x="140" y="52" width="130" height="10" rx="3" fill="#FFFFFF" opacity="0.85"/>
<rect x="140" y="68" width="80" height="8" rx="2" fill="#7E92B5"/>
<rect x="140" y="90" width="180" height="6" rx="2" fill="rgba(255,255,255,0.4)"/>
<rect x="140" y="104" width="150" height="6" rx="2" fill="rgba(255,255,255,0.4)"/>
<rect x="140" y="118" width="110" height="6" rx="2" fill="rgba(255,255,255,0.4)"/>
<rect x="140" y="132" width="130" height="6" rx="2" fill="rgba(255,255,255,0.4)"/>
<rect x="20" y="20" width="360" height="85" fill="url(#scanBeam)"/>
<line x1="20" y1="105" x2="380" y2="105" stroke="#075BFF" stroke-width="2.5" filter="drop-shadow(0 0 6px #075BFF)"/>
<rect x="136" y="85" width="190" height="46" rx="4" fill="none" stroke="#2583FF" stroke-width="1.5" stroke-dasharray="4 2"/>
<text x="320" y="80" text-anchor="end" font-family="'JetBrains Mono', monospace" font-size="9" font-weight="600" fill="#2583FF">OCR CONFIDENCE: 98.4%</text>
<rect x="42" y="180" width="316" height="14" rx="3" fill="#06142F" stroke="#213C6E"/>
<text x="52" y="191" font-family="'JetBrains Mono', monospace" font-size="9" fill="#7E92B5" letter-spacing="2">P&amp;lt;UTOALB&amp;lt;&amp;lt;JOHN&amp;lt;&amp;lt;DOE&amp;lt;&amp;lt;&amp;lt;&amp;lt;&amp;lt;&amp;lt;&amp;lt;&amp;lt;&amp;lt;&amp;lt;&amp;lt;&amp;lt;&amp;lt;&amp;lt;&amp;lt;&amp;lt;</text>
</svg>
</div>"""
