#!/usr/bin/env python3
"""
DocVerify AI - Enterprise UI Design System & Component Library
Premium Dark Navy SaaS Theme (#06142F, #0A1F44, #075BFF, #2583FF, #FFFFFF, #EAF2FF, #152B50, #A9B9D3)
Author: Senior UI/UX Designer & Full-Stack Frontend Engineer
"""

from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import base64
from PIL import Image
import numpy as np
import matplotlib.cm as cm

BASE_DIR = Path(__file__).resolve().parent.parent

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
        bg = "#040C1E"
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
        nav_bg = "rgba(4, 12, 30, 0.85)"

    # Load dark navy abstract background SVG for standalone CSS injection
    bg_data_uri = ""
    try:
        bg_svg_file = BASE_DIR / "assets" / "bg_dark_navy.svg"
        if bg_svg_file.exists():
            b64_val = base64.b64encode(bg_svg_file.read_bytes()).decode("utf-8")
            bg_data_uri = f"data:image/svg+xml;base64,{b64_val}"
    except Exception:
        bg_data_uri = ""

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
        background-image: {'url("' + bg_data_uri + '")' if not is_light and bg_data_uri else 'radial-gradient(circle at 50% 0%, rgba(7, 91, 255, 0.12) 0%, transparent 60%)'} !important;
        background-size: cover !important;
        background-position: center top !important;
        background-repeat: no-repeat !important;
        background-attachment: fixed !important;
        color: var(--dv-text-main) !important;
        min-height: 100vh;
    }}

    [data-testid="stAppViewContainer"] {{
        background-color: transparent !important;
        background-image: {'url("' + bg_data_uri + '")' if not is_light and bg_data_uri else 'radial-gradient(circle at 50% 0%, rgba(7, 91, 255, 0.12) 0%, transparent 60%)'} !important;
        background-size: cover !important;
        background-position: center top !important;
        background-repeat: no-repeat !important;
        background-attachment: fixed !important;
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

    /* =========================================================================
       Pixel-Perfect Homepage Reference Styles (Match Selected Reference Design)
       ========================================================================= */

    /* Reference Navigation Brand & Links */
    .dv-ref-brand {{
        display: flex;
        align-items: center;
        gap: 12px;
    }}
    .dv-ref-logo {{
        width: 38px;
        height: 38px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }}
    .dv-ref-title {{
        font-size: 1.4rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #FFFFFF;
        display: flex;
        align-items: center;
    }}
    .dv-ref-title .dv-ai-text {{
        color: #1E6BFF;
        font-weight: 800;
        margin-left: 3px;
    }}
    .dv-ref-sub {{
        font-size: 0.85rem;
        color: #8E9DB7;
        font-weight: 500;
        margin-left: 14px;
        padding-left: 14px;
        border-left: 1px solid rgba(255, 255, 255, 0.14);
        white-space: nowrap;
    }}

    /* Reference Hero Left Elements */
    .dv-ref-hero-left {{
        display: flex;
        flex-direction: column;
        justify-content: center;
        padding: 6px 0;
    }}
    .dv-ref-hero-pill {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(14, 35, 74, 0.85);
        border: 1px solid rgba(26, 61, 124, 0.9);
        padding: 6px 16px;
        border-radius: 24px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #6BB5FF;
        margin-bottom: 20px;
        width: fit-content;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }}
    .dv-ref-hero-pill-sparkle {{
        color: #00D2FF;
        font-size: 0.95rem;
    }}
    .dv-ref-hero-h1 {{
        font-size: 3.4rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.04em !important;
        line-height: 1.1 !important;
        color: #FFFFFF !important;
        margin-bottom: 16px !important;
    }}
    .dv-ref-hero-h1 .dv-trust {{
        color: #1665FF !important;
    }}
    .dv-ref-hero-sub {{
        font-size: 1.05rem;
        line-height: 1.55;
        color: #8E9DB7;
        max-width: 440px;
        margin-bottom: 26px;
        font-weight: 400;
    }}
    .dv-ref-feat-list {{
        display: flex;
        flex-direction: column;
        gap: 16px;
    }}
    .dv-ref-feat-row {{
        display: flex;
        align-items: center;
        gap: 14px;
        font-size: 1.02rem;
        font-weight: 700;
        color: #FFFFFF;
    }}
    .dv-ref-feat-icon-circ {{
        width: 36px;
        height: 36px;
        border-radius: 50%;
        background: rgba(18, 43, 92, 0.85);
        border: 1.5px solid #1E6BFF;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #2583FF;
        flex-shrink: 0;
        box-shadow: 0 0 12px rgba(30, 107, 255, 0.3);
    }}

    /* Reference White Upload Card */
    .dv-ref-upload-card {{
        background: #FFFFFF !important;
        border-radius: 24px !important;
        padding: 28px 24px 22px 24px !important;
        box-shadow: 0 24px 60px rgba(0, 10, 30, 0.45) !important;
        border: 1px solid rgba(255, 255, 255, 0.95) !important;
        color: #06142F !important;
    }}
    .dv-ref-card-header {{
        text-align: center;
        margin-bottom: 16px;
    }}
    .dv-ref-card-icon {{
        width: 44px;
        height: 44px;
        margin: 0 auto 6px auto;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #0066FF;
    }}
    .dv-ref-card-title {{
        font-size: 1.45rem;
        font-weight: 800;
        color: #06142F;
        letter-spacing: -0.02em;
        margin-bottom: 4px;
        line-height: 1.2;
    }}
    .dv-ref-card-subtitle {{
        font-size: 0.85rem;
        color: #5F6E84;
        font-weight: 500;
        line-height: 1.4;
    }}

    /* Dropzone area within card */
    .dv-ref-upload-card [data-testid="stFileUploader"] {{
        background: #F0F5FC !important;
        border: 1.5px dashed #B9CCE8 !important;
        border-radius: 14px !important;
        padding: 14px 10px !important;
        text-align: center !important;
        margin-bottom: 8px !important;
    }}
    .dv-ref-upload-card [data-testid="stFileUploader"] section {{
        background: transparent !important;
        padding: 0 !important;
    }}
    .dv-ref-upload-card [data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"],
    .dv-ref-upload-card [data-testid="stFileUploader"] button {{
        background: #0066FF !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
        padding: 8px 20px !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(0, 102, 255, 0.35) !important;
    }}
    .dv-ref-upload-card [data-testid="stFileUploader"] button:hover {{
        background: #0052CC !important;
    }}
    .dv-ref-upload-card [data-testid="stFileUploader"] span,
    .dv-ref-upload-card [data-testid="stFileUploader"] p,
    .dv-ref-upload-card [data-testid="stFileUploader"] small {{
        color: #475569 !important;
    }}
    .dv-ref-formats-note {{
        font-size: 0.72rem;
        color: #7E90A8;
        text-align: center;
        margin-top: 6px;
        margin-bottom: 14px;
        font-weight: 500;
        letter-spacing: 0.01em;
    }}

    /* Reference Supported Documents Bar */
    .dv-ref-supported-bar {{
        background: #E8F0FA;
        border: 1px solid #D2E0F2;
        border-radius: 18px;
        padding: 14px 28px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-top: 36px;
        margin-bottom: 36px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
    }}
    .dv-ref-sup-title {{
        font-size: 1rem;
        font-weight: 800;
        line-height: 1.15;
        color: #06142F;
        padding-right: 24px;
        border-right: 1px solid #CBDCEE;
    }}
    .dv-ref-sup-item {{
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 0.95rem;
        font-weight: 700;
        color: #06142F;
    }}
    .dv-ref-sup-icon {{
        width: 24px;
        height: 24px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }}

    /* Reference Bottom 3-Column Value Props Bar */
    .dv-ref-props-bar {{
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 32px;
        margin-bottom: 40px;
        padding: 0 12px;
    }}
    .dv-ref-prop-col {{
        display: flex;
        align-items: flex-start;
        gap: 14px;
    }}
    .dv-ref-prop-icon {{
        width: 36px;
        height: 36px;
        border-radius: 50%;
        border: 1.5px solid #1E6BFF;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #2583FF;
        flex-shrink: 0;
        margin-top: 2px;
    }}
    .dv-ref-prop-title {{
        font-size: 1.05rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 4px;
    }}
    .dv-ref-prop-desc {{
        font-size: 0.85rem;
        color: #8E9DB7;
        line-height: 1.45;
    }}

    /* Verify Now CTA Button on Hero Card */
    button[key*="btn_verify_hero"],
    button[key*="btn_verify_disabled_hero"] {{
        background: #0A2558 !important;
        color: #FFFFFF !important;
        font-size: 1.05rem !important;
        font-weight: 800 !important;
        padding: 12px 24px !important;
        border-radius: 12px !important;
        border: none !important;
        width: 100% !important;
        box-shadow: 0 4px 16px rgba(10, 37, 88, 0.35) !important;
        transition: all 0.2s ease !important;
    }}
    button[key*="btn_verify_hero"]:hover {{
        background: #075BFF !important;
        box-shadow: 0 6px 22px rgba(7, 91, 255, 0.5) !important;
        transform: translateY(-1px) !important;
    }}
    button[key*="btn_verify_disabled_hero"]:disabled {{
        background: #0A2558 !important;
        opacity: 0.7 !important;
        color: #A0B2CD !important;
    }}

    /* Top Navbar Link Buttons */
    button[key*="nav_btn_home"] {{
        background: #163260 !important;
        color: #FFFFFF !important;
        border-radius: 20px !important;
        border: 1px solid rgba(255, 255, 255, 0.18) !important;
        font-weight: 700 !important;
        padding: 6px 16px !important;
    }}
    button[key*="nav_btn_get_started"] {{
        background: #FFFFFF !important;
        color: #06142F !important;
        border-radius: 9999px !important;
        font-weight: 700 !important;
        border: none !important;
        padding: 6px 18px !important;
        box-shadow: 0 4px 14px rgba(255, 255, 255, 0.25) !important;
    }}
    button[key*="nav_btn_get_started"]:hover {{
        background: #EAF2FF !important;
        color: #004BD6 !important;
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


def render_ref_brand_logo_html() -> str:
    """
    Renders the exact reference header brand: [ = ] icon, DocVerify AI wordmark,
    and 'AI-Powered Document Authentication' subtitle.
    """
    return """<div class="dv-ref-brand">
<div class="dv-ref-logo">
<svg width="34" height="34" viewBox="0 0 34 34" fill="none">
<rect x="1" y="1" width="32" height="32" rx="8" stroke="#FFFFFF" stroke-width="2" fill="#0A1E48" fill-opacity="0.5"/>
<path d="M 9,13 L 25,13" stroke="#1E6BFF" stroke-width="2.8" stroke-linecap="round"/>
<path d="M 9,21 L 25,21" stroke="#1E6BFF" stroke-width="2.8" stroke-linecap="round"/>
<path d="M 12,8 L 8,8 A 2 2 0 0 0 6,10 L 6,14" stroke="#FFFFFF" stroke-width="2" stroke-linecap="round"/>
<path d="M 22,8 L 26,8 A 2 2 0 0 1 28,10 L 28,14" stroke="#FFFFFF" stroke-width="2" stroke-linecap="round"/>
<path d="M 6,20 L 6,24 A 2 2 0 0 0 8,26 L 12,26" stroke="#FFFFFF" stroke-width="2" stroke-linecap="round"/>
<path d="M 28,20 L 28,24 A 2 2 0 0 1 26,26 L 22,26" stroke="#FFFFFF" stroke-width="2" stroke-linecap="round"/>
</svg>
</div>
<div class="dv-ref-title">
<span>DocVerify</span><span class="dv-ai-text">AI</span>
</div>
<div class="dv-ref-sub">
AI-Powered Document Authentication
</div>
</div>"""


def render_hero_left_section() -> str:
    """
    Renders the Hero left column matching the reference design:
    - ✦ AI Document Verification pill
    - Detect. Verify. Trust. heading
    - Subtitle
    - 3 bullet points with circular blue icon containers
    """
    return """<div class="dv-ref-hero-left">
<div class="dv-ref-hero-pill">
<span class="dv-ref-hero-pill-sparkle">✦</span>
<span>AI Document Verification</span>
</div>
<div class="dv-ref-hero-h1">
Detect. Verify. <span class="dv-trust">Trust.</span>
</div>
<div class="dv-ref-hero-sub">
AI-powered document authentication to detect forgery, tampering and invalid documents instantly.
</div>
<div class="dv-ref-feat-list">
<div class="dv-ref-feat-row">
<div class="dv-ref-feat-icon-circ">
<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2583FF" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round">
<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
<path d="m9 12 2 2 4-4"/>
</svg>
</div>
<span>Authenticity Check</span>
</div>
<div class="dv-ref-feat-row">
<div class="dv-ref-feat-icon-circ">
<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2583FF" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round">
<path d="M4 8V4h4"/>
<path d="M20 8V4h-4"/>
<path d="M4 16v4h4"/>
<path d="M20 16v4h-4"/>
<line x1="8" y1="12" x2="16" y2="12"/>
</svg>
</div>
<span>Tamper &amp; Manipulation Detection</span>
</div>
<div class="dv-ref-feat-row">
<div class="dv-ref-feat-icon-circ">
<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2583FF" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round">
<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
<polyline points="14 2 14 8 20 8"/>
<line x1="16" y1="13" x2="8" y2="13"/>
<line x1="16" y1="17" x2="8" y2="17"/>
</svg>
</div>
<span>Detailed Verification Report</span>
</div>
</div>
</div>"""


def render_cards_stack_svg() -> str:
    """
    Renders the 3D identity card stack illustration (Aadhaar, PAN, Driving License
    with scanning brackets, laser beam, and blue verified badge).
    """
    cards_path = BASE_DIR / "assets" / "cards_stack_illustration.svg"
    if cards_path.exists():
        svg_content = cards_path.read_text(encoding="utf-8")
        return f'<div style="width:100%; display:flex; align-items:center; justify-content:center; padding:10px 0;">{svg_content}</div>'
    return render_hero_scan_graphic()


def render_supported_documents_bar() -> str:
    """
    Renders the horizontal Supported Documents bar matching the reference design:
    - Supported Documents header on the left
    - Aadhaar Card (orange sun emblem)
    - PAN Card
    - Voter ID
    - Driving License
    - Passport
    """
    return """<div class="dv-ref-supported-bar">
<div class="dv-ref-sup-title">
Supported<br/>Documents
</div>
<div class="dv-ref-sup-item">
<div class="dv-ref-sup-icon">
<svg width="22" height="22" viewBox="0 0 24 24" fill="none">
<circle cx="12" cy="12" r="5" fill="#FF6B00"/>
<circle cx="12" cy="12" r="2.5" fill="#FFFFFF"/>
<path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41" stroke="#FF6B00" stroke-width="2" stroke-linecap="round"/>
</svg>
</div>
<span>Aadhaar Card</span>
</div>
<div class="dv-ref-sup-item">
<div class="dv-ref-sup-icon">
<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#06142F" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
<rect x="2" y="5" width="20" height="14" rx="2"/>
<line x1="2" y1="10" x2="22" y2="10"/>
<line x1="6" y1="14" x2="11" y2="14"/>
<circle cx="16.5" cy="14.5" r="1.5"/>
</svg>
</div>
<span>PAN Card</span>
</div>
<div class="dv-ref-sup-item">
<div class="dv-ref-sup-icon">
<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#06142F" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
<rect x="3" y="4" width="18" height="16" rx="2"/>
<rect x="6" y="7" width="5" height="5" rx="1"/>
<line x1="14" y1="8" x2="18" y2="8"/>
<line x1="14" y1="12" x2="18" y2="12"/>
<line x1="6" y1="16" x2="18" y2="16"/>
</svg>
</div>
<span>Voter ID</span>
</div>
<div class="dv-ref-sup-item">
<div class="dv-ref-sup-icon">
<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#06142F" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
<path d="M19 17h2c.6 0 1-.4 1-1v-3c0-.9-.7-1.7-1.5-1.9C18.7 10.6 16 10 16 10s-1.3-1.4-2.2-2.3c-.5-.4-1.1-.7-1.8-.7H5c-.6 0-1.1.4-1.4.9l-1.4 2.9C2.1 11.1 2 11.5 2 12v4c0 .6.4 1 1 1h2"/>
<circle cx="7" cy="17" r="2"/>
<path d="M9 17h6"/>
<circle cx="17" cy="17" r="2"/>
</svg>
</div>
<span>Driving License</span>
</div>
<div class="dv-ref-sup-item">
<div class="dv-ref-sup-icon">
<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#06142F" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
<circle cx="12" cy="12" r="9"/>
<line x1="3" y1="12" x2="21" y2="12"/>
<path d="M12 3a15.3 15.3 0 0 1 4 9 15.3 15.3 0 0 1-4 9 15.3 15.3 0 0 1-4-9 15.3 15.3 0 0 1 4-9z"/>
</svg>
</div>
<span>Passport</span>
</div>
</div>"""


def render_bottom_value_props_bar() -> str:
    """
    Renders the 3-column value props at the bottom of the homepage:
    - Privacy First (shield icon)
    - Fast & Accurate (lightning icon)
    - Detailed Report (chart icon)
    """
    return """<div class="dv-ref-props-bar">
<div class="dv-ref-prop-col">
<div class="dv-ref-prop-icon">
<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#2583FF" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
<path d="m9 12 2 2 4-4"/>
</svg>
</div>
<div>
<div class="dv-ref-prop-title">Privacy First</div>
<div class="dv-ref-prop-desc">Your files are processed temporarily and not stored.</div>
</div>
</div>
<div class="dv-ref-prop-col">
<div class="dv-ref-prop-icon">
<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#2583FF" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
</svg>
</div>
<div>
<div class="dv-ref-prop-title">Fast &amp; Accurate</div>
<div class="dv-ref-prop-desc">Get verification results within seconds.</div>
</div>
</div>
<div class="dv-ref-prop-col">
<div class="dv-ref-prop-icon">
<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#2583FF" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
<line x1="18" y1="20" x2="18" y2="10"/>
<line x1="12" y1="20" x2="12" y2="4"/>
<line x1="6" y1="20" x2="6" y2="14"/>
</svg>
</div>
<div>
<div class="dv-ref-prop-title">Detailed Report</div>
<div class="dv-ref-prop-desc">View evidence, highlighted regions and download a PDF report.</div>
</div>
</div>
</div>"""

