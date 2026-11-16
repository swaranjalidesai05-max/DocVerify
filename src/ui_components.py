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
        # Dark Theme Palette (Deep Navy + Electric Blue + Cyan + White)
        bg = "#061735"
        bg_alt = "#071B3B"
        surface = "#071B3B"
        surface_alt = "#0B2855"
        surface_raised = "#0B2855"
        surface_hero = "#061735"
        card_white = "#FFFFFF"
        card_white_text = "#10254A"
        card_white_border = "rgba(255, 255, 255, 0.95)"
        border = "#0B2855"
        border_focus = "#087BFF"
        border_strong = "rgba(8, 123, 255, 0.35)"
        text_main = "#FFFFFF"
        text_secondary = "#A9BEDD"
        text_muted = "#7E96BF"
        text_subtle = "#526B94"
        accent = "#087BFF"
        accent_bright = "#00C8FF"
        accent_hover = "#0062D6"
        accent_light = "#0D2C5E"
        accent_border = "rgba(8, 123, 255, 0.4)"
        slate_blue = "#0B2855"
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
        nav_bg = "rgba(6, 23, 53, 0.95)"

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
        padding-top: 0.35rem !important;
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
        width: 36px;
        height: 36px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }}
    .dv-ref-title {{
        font-size: 1.45rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #FFFFFF;
        display: flex;
        align-items: center;
    }}
    .dv-ref-title .dv-ai-text {{
        color: #087BFF;
        font-weight: 800;
        margin-left: 4px;
    }}
    .dv-ref-divider {{
        color: rgba(255, 255, 255, 0.25);
        margin: 0 10px;
        font-size: 1rem;
        font-weight: 300;
    }}
    .dv-ref-sub {{
        font-size: 0.85rem;
        color: #A9BEDD;
        font-weight: 500;
        white-space: nowrap;
    }}

    /* Reference Hero Left Elements */
    .dv-ref-hero-left {{
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        padding: 0;
        margin-top: 0;
    }}
    .dv-ref-hero-pill {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(14, 35, 74, 0.85);
        border: 1px solid rgba(8, 123, 255, 0.35);
        padding: 6px 16px;
        border-radius: 24px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #6BB5FF;
        margin-bottom: 18px;
        width: fit-content;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }}
    .dv-ref-hero-pill-sparkle {{
        color: #00C8FF;
        font-size: 0.95rem;
    }}
    .dv-ref-hero-h1 {{
        font-size: 2.85rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.035em !important;
        line-height: 1.15 !important;
        color: #FFFFFF !important;
        margin-bottom: 16px !important;
    }}
    .dv-cyan-highlight {{
        color: #00C8FF !important;
        background: linear-gradient(135deg, #087BFF 0%, #00C8FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .dv-ref-hero-sub {{
        font-size: 0.96rem;
        line-height: 1.55;
        color: #A9BEDD;
        max-width: 440px;
        margin-bottom: 24px;
        font-weight: 400;
    }}
    /* Hero Feature Bullets - Modern Indicators */
    .dv-ref-feat-list {{
        display: flex;
        flex-direction: column;
        gap: 12px;
        margin-top: 6px;
    }}
    .dv-ref-feat-row {{
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 6px 10px;
        border-radius: 10px;
        transition: all 0.22s cubic-bezier(0.16, 1, 0.3, 1);
        cursor: default;
    }}
    .dv-ref-feat-row:hover {{
        background: rgba(8, 123, 255, 0.08);
        transform: translateX(4px);
    }}
    .dv-ref-feat-badge {{
        width: 26px;
        height: 26px;
        border-radius: 7px;
        background: rgba(0, 200, 255, 0.10);
        border: 1px solid rgba(0, 200, 255, 0.35);
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        box-shadow: 0 0 12px rgba(0, 200, 255, 0.22);
        transition: all 0.22s ease;
    }}
    .dv-ref-feat-row:hover .dv-ref-feat-badge {{
        background: rgba(0, 200, 255, 0.18);
        border-color: #00C8FF;
        box-shadow: 0 0 16px rgba(0, 200, 255, 0.40);
        transform: scale(1.06);
    }}
    .dv-ref-feat-label {{
        font-size: 0.94rem;
        font-weight: 600;
        color: #E2E8F0;
        letter-spacing: -0.01em;
        transition: color 0.22s ease;
    }}
    .dv-ref-feat-row:hover .dv-ref-feat-label {{
        color: #FFFFFF;
    }}

    /* Hero Center Illustration - Vibrant Foreground with Deep Midnight Backdrop */
    .dv-hero-illustration-wrap {{
        display: flex;
        align-items: center;
        justify-content: center;
        width: 100%;
        position: relative;
        padding: 0 !important;
        margin-top: 0 !important;
        background: transparent !important;
    }}
    .dv-hero-illustration-dark-bg {{
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 104%;
        height: 104%;
        max-width: 490px;
        max-height: 450px;
        background: radial-gradient(ellipse at 50% 50%, #020713 25%, #040D24 58%, rgba(4, 13, 36, 0) 80%);
        border-radius: 50%;
        pointer-events: none;
        z-index: 1;
    }}
    .dv-hero-illustration-glow {{
        position: absolute;
        top: 52%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 95%;
        height: 90%;
        max-width: 450px;
        max-height: 410px;
        background: radial-gradient(circle at 50% 55%, rgba(0, 190, 255, 0.12) 0%, rgba(8, 91, 255, 0.05) 45%, transparent 70%);
        pointer-events: none;
        z-index: 1;
    }}
    .dv-hero-illustration-img {{
        width: 100%;
        max-width: 475px;
        height: auto;
        display: block;
        object-fit: contain;
        position: relative;
        z-index: 2;
        background: transparent !important;
        border: none !important;
        border-radius: 0 !important;
        box-shadow: none !important;
        opacity: 1 !important;
        filter: drop-shadow(0 0 28px rgba(0, 190, 255, 0.22));
    }}

    /* Unified Hero Upload Card Container - Combined Inside One Card */
    .st-key-dv_hero_upload_card,
    .st-key-dv_hero_upload_card [data-testid="stVerticalBlockBorderWrapper"],
    .st-key-dv_hero_upload_card > div {{
        background: #FFFFFF !important;
        border-radius: 24px !important;
        padding: 24px 22px 20px 22px !important;
        box-shadow: 0 24px 60px rgba(0, 10, 30, 0.45), 0 0 0 1px rgba(255, 255, 255, 0.95) !important;
        border: none !important;
        color: #10254A !important;
    }}
    .st-key-dv_hero_upload_card .dv-ref-card-header {{
        text-align: center;
        margin-bottom: 12px;
    }}
    .st-key-dv_hero_upload_card .dv-ref-card-icon {{
        width: 44px;
        height: 44px;
        margin: 0 auto 6px auto;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #087BFF;
    }}
    .st-key-dv_hero_upload_card .dv-ref-card-title {{
        font-size: 1.45rem;
        font-weight: 800;
        color: #10254A;
        letter-spacing: -0.02em;
        margin-bottom: 4px;
        line-height: 1.2;
    }}
    .st-key-dv_hero_upload_card .dv-ref-card-subtitle {{
        font-size: 0.85rem;
        color: #5F6E84;
        font-weight: 500;
        line-height: 1.4;
    }}

    /* Dropzone area within unified card */
    .st-key-dv_hero_upload_card [data-testid="stFileUploader"] {{
        background: #F0F5FC !important;
        border: 1.5px dashed #B9CCE8 !important;
        border-radius: 14px !important;
        padding: 14px 10px !important;
        text-align: center !important;
        margin-bottom: 6px !important;
        transition: all 0.2s ease !important;
    }}
    .st-key-dv_hero_upload_card [data-testid="stFileUploader"]:hover {{
        border-color: #087BFF !important;
        background: #EAF2FF !important;
    }}
    .st-key-dv_hero_upload_card [data-testid="stFileUploader"] section {{
        background: transparent !important;
        padding: 0 !important;
    }}
    .st-key-dv_hero_upload_card [data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"],
    .st-key-dv_hero_upload_card [data-testid="stFileUploader"] button {{
        background: #087BFF !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
        padding: 8px 22px !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(8, 123, 255, 0.35) !important;
    }}
    .st-key-dv_hero_upload_card [data-testid="stFileUploader"] button:hover {{
        background: #0062D6 !important;
    }}
    .st-key-dv_hero_upload_card [data-testid="stFileUploader"] span,
    .st-key-dv_hero_upload_card [data-testid="stFileUploader"] p,
    .st-key-dv_hero_upload_card [data-testid="stFileUploader"] small {{
        color: #475569 !important;
    }}
    .st-key-dv_hero_upload_card .dv-ref-formats-note {{
        font-size: 0.74rem;
        color: #64748B;
        text-align: center;
        margin-top: 4px;
        margin-bottom: 12px;
        font-weight: 500;
        letter-spacing: 0.01em;
    }}

    /* Expander inside Card */
    .st-key-dv_hero_upload_card [data-testid="stExpander"] {{
        background: #F8FAFC !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 10px !important;
        margin-bottom: 12px !important;
    }}
    .st-key-dv_hero_upload_card [data-testid="stExpander"] summary {{
        color: #10254A !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
    }}
    .st-key-dv_hero_upload_card [data-testid="stExpander"] b,
    .st-key-dv_hero_upload_card [data-testid="stExpander"] label,
    .st-key-dv_hero_upload_card [data-testid="stExpander"] span,
    .st-key-dv_hero_upload_card [data-testid="stExpander"] p {{
        color: #10254A !important;
    }}

    /* Verify Button inside Unified Card */
    .st-key-dv_hero_upload_card [data-testid="baseButton-primary"],
    .st-key-dv_hero_upload_card button[kind="primary"] {{
        background: #087BFF !important;
        color: #FFFFFF !important;
        border-radius: 12px !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        padding: 12px 20px !important;
        box-shadow: 0 6px 18px rgba(8, 123, 255, 0.4) !important;
        border: none !important;
    }}
    .st-key-dv_hero_upload_card [data-testid="baseButton-primary"]:hover,
    .st-key-dv_hero_upload_card button[kind="primary"]:hover {{
        background: #0062D6 !important;
        box-shadow: 0 8px 24px rgba(8, 123, 255, 0.5) !important;
    }}
    .st-key-dv_hero_upload_card [data-testid="baseButton-secondary"],
    .st-key-dv_hero_upload_card button[kind="secondary"] {{
        background: #0B2855 !important;
        color: #A9BEDD !important;
        border-radius: 12px !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        padding: 12px 20px !important;
        border: none !important;
    }}

    /* Reference Supported Documents Section (10 Countries) */
    .dv-ref-sup-bar-10 {{
        background: rgba(11, 40, 85, 0.55);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(8, 123, 255, 0.28);
        border-radius: 18px;
        padding: 14px 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-top: 32px;
        margin-bottom: 32px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
        gap: 8px;
        overflow-x: auto;
    }}
    .dv-sup-title-box {{
        display: flex;
        align-items: center;
        gap: 12px;
        padding-right: 16px;
        border-right: 1px solid rgba(8, 123, 255, 0.25);
        flex-shrink: 0;
    }}
    .dv-sup-globe-badge {{
        width: 38px;
        height: 38px;
        border-radius: 50%;
        background: rgba(8, 123, 255, 0.18);
        border: 1.5px solid #087BFF;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #00C8FF;
        flex-shrink: 0;
    }}
    .dv-sup-main-title {{
        font-size: 0.98rem;
        font-weight: 800;
        color: #FFFFFF;
        line-height: 1.15;
    }}
    .dv-sup-sub-title {{
        font-size: 0.74rem;
        color: #A9BEDD;
        margin-top: 2px;
    }}
    .dv-sup-card-link {{
        text-decoration: none !important;
        color: inherit !important;
        display: flex;
        flex: 1;
        min-width: 0;
    }}
    .dv-sup-card-item {{
        display: flex;
        flex-direction: column;
        gap: 4px;
        padding: 8px 10px;
        border-right: 1px solid rgba(8, 123, 255, 0.15);
        border-radius: 8px;
        flex-shrink: 0;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
        cursor: pointer;
    }}
    .dv-sup-card-item:hover {{
        background: rgba(8, 123, 255, 0.15);
        transform: translateY(-2px);
    }}
    .dv-sup-card-item:last-child {{
        border-right: none;
    }}
    .dv-sup-card-top {{
        display: flex;
        align-items: center;
        gap: 7px;
    }}
    .dv-sup-flag-box {{
        width: 24px;
        height: 16px;
        border-radius: 3px;
        overflow: hidden;
        flex-shrink: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.25);
    }}
    .dv-sup-flag-svg {{
        width: 100%;
        height: 100%;
        display: block;
        object-fit: cover;
    }}
    .dv-sup-code-tag {{
        font-size: 0.72rem;
        font-weight: 800;
        color: #A9BEDD;
        letter-spacing: 0.05em;
    }}
    .dv-sup-country-name {{
        font-size: 0.82rem;
        font-weight: 700;
        color: #FFFFFF;
        white-space: nowrap;
    }}
    .dv-sup-doc-type {{
        font-size: 0.7rem;
        color: #A9BEDD;
        white-space: nowrap;
    }}

    /* Bottom 3-Feature Section */
    .dv-ref-bottom-features {{
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 0;
        margin-top: 8px;
        margin-bottom: 40px;
        padding: 0 8px;
    }}
    .dv-ref-bottom-col {{
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 10px 24px;
        border-right: 1px solid rgba(8, 123, 255, 0.2);
    }}
    .dv-ref-bottom-col:last-child {{
        border-right: none;
    }}
    .dv-ref-bottom-icon {{
        width: 40px;
        height: 40px;
        border-radius: 50%;
        border: 1.5px solid #087BFF;
        background: rgba(11, 40, 85, 0.5);
        display: flex;
        align-items: center;
        justify-content: center;
        color: #00C8FF;
        flex-shrink: 0;
        box-shadow: 0 0 12px rgba(8, 123, 255, 0.3);
    }}
    .dv-ref-bottom-title {{
        font-size: 1.05rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 3px;
    }}
    .dv-ref-bottom-desc {{
        font-size: 0.84rem;
        color: #A9BEDD;
        line-height: 1.4;
    }}

    /* Upload Card CTA button */
    button[key*="btn_verify_hero"],
    button[key*="btn_verify_disabled_hero"] {{
        background: #0B2855 !important;
        color: #FFFFFF !important;
        font-size: 1.02rem !important;
        font-weight: 800 !important;
        padding: 12px 24px !important;
        border-radius: 12px !important;
        border: none !important;
        width: 100% !important;
        box-shadow: 0 4px 16px rgba(11, 40, 85, 0.4) !important;
        transition: all 0.2s ease !important;
    }}
    button[key*="btn_verify_hero"]:hover {{
        background: #087BFF !important;
        box-shadow: 0 6px 20px rgba(8, 123, 255, 0.5) !important;
        transform: translateY(-1px) !important;
    }}
    button[key*="btn_verify_disabled_hero"]:disabled {{
        background: #0B2855 !important;
        opacity: 0.7 !important;
        color: #7E96BF !important;
    }}

    /* Top Navbar Link Buttons */
    button[key*="nav_btn_home"] {{
        background: #087BFF !important;
        color: #FFFFFF !important;
        border-radius: 20px !important;
        border: none !important;
        font-weight: 700 !important;
        padding: 6px 18px !important;
        box-shadow: 0 2px 10px rgba(8, 123, 255, 0.4) !important;
    }}
    button[key*="nav_btn_get_started"] {{
        background: #087BFF !important;
        color: #FFFFFF !important;
        border-radius: 9999px !important;
        font-weight: 700 !important;
        border: none !important;
        padding: 6px 20px !important;
        box-shadow: 0 4px 14px rgba(8, 123, 255, 0.4) !important;
    }}
    button[key*="nav_btn_get_started"]:hover {{
        background: #0062D6 !important;
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
    vertical divider, and 'AI-Powered Document Authentication' subtitle.
    """
    return """<div class="dv-ref-brand">
<div class="dv-ref-logo">
<svg width="36" height="36" viewBox="0 0 34 34" fill="none">
<rect x="1" y="1" width="32" height="32" rx="8" stroke="#FFFFFF" stroke-width="2" fill="#061735" fill-opacity="0.6"/>
<path d="M 9,13 L 25,13" stroke="#087BFF" stroke-width="2.8" stroke-linecap="round"/>
<path d="M 9,21 L 25,21" stroke="#087BFF" stroke-width="2.8" stroke-linecap="round"/>
<path d="M 12,8 L 8,8 A 2 2 0 0 0 6,10 L 6,14" stroke="#FFFFFF" stroke-width="2" stroke-linecap="round"/>
<path d="M 22,8 L 26,8 A 2 2 0 0 1 28,10 L 28,14" stroke="#FFFFFF" stroke-width="2" stroke-linecap="round"/>
<path d="M 6,20 L 6,24 A 2 2 0 0 0 8,26 L 12,26" stroke="#FFFFFF" stroke-width="2" stroke-linecap="round"/>
<path d="M 28,20 L 28,24 A 2 2 0 0 1 26,26 L 22,26" stroke="#FFFFFF" stroke-width="2" stroke-linecap="round"/>
</svg>
</div>
<div class="dv-ref-title">
<span>DocVerify</span><span class="dv-ai-text">AI</span>
</div>
<span class="dv-ref-divider">|</span>
<div class="dv-ref-sub">
AI-Powered Document Authentication
</div>
</div>"""


def render_hero_left_section() -> str:
    """
    Renders the Hero left column matching Reference Image 1:
    - ✦ AI Document Verification badge
    - Main heading: Verify International Identity Documents with AI
    - Description about detecting forged, tampered and invalid documents
    - Three feature points:
      - Authenticity Verification
      - Tamper & Manipulation Detection
      - Detailed Verification Report
    """
    return """<div class="dv-ref-hero-left">
<div class="dv-ref-hero-pill">
<span class="dv-ref-hero-pill-sparkle">✦</span>
<span>AI Document Verification</span>
</div>
<div class="dv-ref-hero-h1">
Verify International<br/>
Identity Documents<br/>
<span class="dv-cyan-highlight">with AI</span>
</div>
<div class="dv-ref-hero-sub">
Detect forged, tampered or invalid identity documents from multiple countries using advanced computer vision and machine learning.
</div>
<div class="dv-ref-feat-list">
<div class="dv-ref-feat-row">
<div class="dv-ref-feat-badge">
<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#00C8FF" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round">
<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
<polyline points="9 12 11 14 15 10"/>
</svg>
</div>
<span class="dv-ref-feat-label">AI-Powered Authenticity Verification</span>
</div>
<div class="dv-ref-feat-row">
<div class="dv-ref-feat-badge">
<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#00C8FF" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round">
<circle cx="12" cy="12" r="10"/>
<path d="m9 12 2 2 4-4"/>
<line x1="12" y1="2" x2="12" y2="6"/>
<line x1="12" y1="18" x2="12" y2="22"/>
<line x1="2" y1="12" x2="6" y2="12"/>
<line x1="18" y1="12" x2="22" y2="12"/>
</svg>
</div>
<span class="dv-ref-feat-label">Tamper &amp; Manipulation Detection</span>
</div>
<div class="dv-ref-feat-row">
<div class="dv-ref-feat-badge">
<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#00C8FF" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round">
<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
<polyline points="14 2 14 8 20 8"/>
<polyline points="9 13 11 15 15 11"/>
</svg>
</div>
<span class="dv-ref-feat-label">Detailed Verification Reports</span>
</div>
</div>
</div>"""


def render_hero_illustration_html() -> str:
    """
    Renders Reference Image 2 as the central hero illustration asset.
    Preserves original proportions, dotted world map, passport, identity card,
    scanning brackets and security shield with ambient blue glow, seamlessly blended
    into the dark navy background without harsh rectangular image boundaries.
    """
    opt_path = BASE_DIR / "assets" / "hero_illustration_seamless.png"
    if not opt_path.exists():
        opt_path = BASE_DIR / "assets" / "hero_illustration_opt.png"
    if not opt_path.exists():
        opt_path = BASE_DIR / "assets" / "hero_illustration_blended.png"
    if not opt_path.exists():
        opt_path = BASE_DIR / "assets" / "hero_illustration.jpg"

    if opt_path.exists():
        b64_val = base64.b64encode(opt_path.read_bytes()).decode("utf-8")
        return f"""<div class="dv-hero-illustration-wrap">
<div class="dv-hero-illustration-dark-bg"></div>
<div class="dv-hero-illustration-glow"></div>
<img src="data:image/png;base64,{b64_val}" class="dv-hero-illustration-img" alt="DocVerify AI Hero Illustration Asset" />
</div>"""
    return ""


def render_cards_stack_svg() -> str:
    """
    Backward-compatible alias pointing directly to Reference Image 2 hero illustration.
    """
    return render_hero_illustration_html()


def render_supported_documents_bar() -> str:
    """
    Renders the horizontal Supported Documents bar matching Reference Image 1:
    Contains all 10 supported document standards from 10 international jurisdictions
    with actual, high-precision country flags:
    1. Albania – National ID (ALB)
    2. Azerbaijan – Passport (AZE)
    3. Spain – National ID (DNI) (ESP)
    4. Estonia – National ID (EST)
    5. Finland – National ID (FIN)
    6. Greece – Passport (GRC)
    7. Latvia – Passport (LVA)
    8. Russia – Internal Passport (RUS)
    9. Serbia – Passport (SRB)
    10. Slovenia – National ID (SVN)
    """
    cards_data = [
        ("ALB", "Albania", "National ID", "alb_id",
         '<rect width="28" height="20" fill="#E41E20"/><path d="M14 5.5l1.2 1.8 2.2-.4-1.3 1.8 1.8 1.4-2.1.2.2 2.2-1.9-1.2-1.1 1.9-1.1-1.9-1.9 1.2.2-2.2-2.1-.2 1.8-1.4-1.3-1.8 2.2.4L14 5.5z" fill="#000000"/><circle cx="12.8" cy="6.2" r="0.8" fill="#000000"/><circle cx="15.2" cy="6.2" r="0.8" fill="#000000"/>'),
        ("AZE", "Azerbaijan", "Passport", "aze_passport",
         '<rect width="28" height="6.66" fill="#0092BC"/><rect y="6.66" width="28" height="6.67" fill="#E00034"/><rect y="13.33" width="28" height="6.67" fill="#009933"/><path d="M13.2 8.5a2.2 2.2 0 1 0 0 3 1.7 1.7 0 1 1 0-3z" fill="#FFFFFF"/><polygon points="15.5,9.2 16,9.8 16.8,9.7 16.4,10.3 16.8,11 16,10.8 15.5,11.5 15.3,10.7 14.5,10.7 15.1,10.2 14.7,9.6 15.4,9.8" fill="#FFFFFF"/>'),
        ("ESP", "Spain", "National ID (DNI)", "esp_id",
         '<rect width="28" height="5" fill="#AA151B"/><rect y="5" width="28" height="10" fill="#F1BF00"/><rect y="15" width="28" height="5" fill="#AA151B"/><path d="M8 7.5h2.8v3a1.4 1.4 0 0 1-2.8 0v-3z" fill="#AA151B"/><circle cx="9.4" cy="6.5" r="0.8" fill="#F1BF00"/>'),
        ("EST", "Estonia", "National ID", "est_id",
         '<rect width="28" height="6.66" fill="#0072CE"/><rect y="6.66" width="28" height="6.67" fill="#000000"/><rect y="13.33" width="28" height="6.67" fill="#FFFFFF"/>'),
        ("FIN", "Finland", "National ID", "fin_id",
         '<rect width="28" height="20" fill="#FFFFFF"/><rect x="8" width="4" height="20" fill="#002F6C"/><rect y="8" width="28" height="4" fill="#002F6C"/>'),
        ("GRC", "Greece", "Passport", "grc_passport",
         '<rect width="28" height="20" fill="#0D5EAF"/><rect y="2.22" width="28" height="2.22" fill="#FFFFFF"/><rect y="6.66" width="28" height="2.22" fill="#FFFFFF"/><rect y="11.11" width="28" height="2.22" fill="#FFFFFF"/><rect y="15.55" width="28" height="2.22" fill="#FFFFFF"/><rect width="11" height="11" fill="#0D5EAF"/><rect x="4" width="3" height="11" fill="#FFFFFF"/><rect y="4" width="11" height="3" fill="#FFFFFF"/>'),
        ("LVA", "Latvia", "Passport", "lva_passport",
         '<rect width="28" height="8" fill="#9E3039"/><rect y="8" width="28" height="4" fill="#FFFFFF"/><rect y="12" width="28" height="8" fill="#9E3039"/>'),
        ("RUS", "Russia", "Internal Passport", "rus_internalpassport",
         '<rect width="28" height="6.66" fill="#FFFFFF"/><rect y="6.66" width="28" height="6.67" fill="#0039A6"/><rect y="13.33" width="28" height="6.67" fill="#D52B1E"/>'),
        ("SRB", "Serbia", "Passport", "srb_passport",
         '<rect width="28" height="6.66" fill="#C6363C"/><rect y="6.66" width="28" height="6.67" fill="#0C4076"/><rect y="13.33" width="28" height="6.67" fill="#FFFFFF"/><path d="M9 6h3v4.5a1.5 1.5 0 0 1-3 0V6z" fill="#C6363C"/><polygon points="10.5,4 11.5,5.5 9.5,5.5" fill="#FFD700"/>'),
        ("SVN", "Slovenia", "National ID", "svk_id",
         '<rect width="28" height="6.66" fill="#FFFFFF"/><rect y="6.66" width="28" height="6.67" fill="#005CE6"/><rect y="13.33" width="28" height="6.67" fill="#ED1C24"/><path d="M8 4h3v3.5a1.5 1.5 0 0 1-3 0V4z" fill="#005CE6" stroke="#ED1C24" stroke-width="0.5"/><polygon points="9.5,4.5 10,5.5 9,5.5" fill="#FFD700"/>'),
    ]

    items_html = []
    for code, name, doc_type, tpl_key, svg_inner in cards_data:
        item = f"""<a href="?tpl={tpl_key}" class="dv-sup-card-link" title="Click to inspect {name} standard">
<div class="dv-sup-card-item">
<div class="dv-sup-card-top">
<div class="dv-sup-flag-box"><svg viewBox="0 0 28 20" class="dv-sup-flag-svg">{svg_inner}</svg></div>
<span class="dv-sup-code-tag">{code}</span>
</div>
<div class="dv-sup-country-name">{name}</div>
<div class="dv-sup-doc-type">{doc_type}</div>
</div>
</a>"""
        items_html.append(item)

    cards_joined = "".join(items_html)

    return f"""<div class="dv-ref-sup-bar-10">
<div class="dv-sup-title-box">
<div class="dv-sup-globe-badge">
<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#00C8FF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
<circle cx="12" cy="12" r="10"/>
<line x1="2" y1="12" x2="22" y2="12"/>
<path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>
</svg>
</div>
<div>
<div class="dv-sup-main-title">Supported Documents</div>
<div class="dv-sup-sub-title">10 Document Types from 10 Countries</div>
</div>
</div>
{cards_joined}
</div>"""


def render_bottom_value_props_bar() -> str:
    """
    Renders the 3 horizontally aligned feature blocks matching Reference Image 1:
    - Fast Verification (lightning icon): Get results within seconds.
    - Privacy First (padlock icon): Your files are processed temporarily and not stored.
    - Comprehensive Report (analytics icon): View evidence, risk score and download a PDF report.
    """
    return """<div class="dv-ref-bottom-features">
<div class="dv-ref-bottom-col">
<div class="dv-ref-bottom-icon">
<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#00C8FF" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round">
<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
</svg>
</div>
<div>
<div class="dv-ref-bottom-title">Fast Verification</div>
<div class="dv-ref-bottom-desc">Get results within seconds.</div>
</div>
</div>
<div class="dv-ref-bottom-col">
<div class="dv-ref-bottom-icon">
<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#00C8FF" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round">
<rect x="3" y="11" width="18" height="11" rx="2" ry="2"/>
<path d="M7 11V7a5 5 0 0 1 10 0v4"/>
</svg>
</div>
<div>
<div class="dv-ref-bottom-title">Privacy First</div>
<div class="dv-ref-bottom-desc">Your files are processed temporarily and not stored.</div>
</div>
</div>
<div class="dv-ref-bottom-col">
<div class="dv-ref-bottom-icon">
<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#00C8FF" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round">
<line x1="18" y1="20" x2="18" y2="10"/>
<line x1="12" y1="20" x2="12" y2="4"/>
<line x1="6" y1="20" x2="6" y2="14"/>
</svg>
</div>
<div>
<div class="dv-ref-bottom-title">Comprehensive Report</div>
<div class="dv-ref-bottom-desc">View evidence, risk score and download a PDF report.</div>
</div>
</div>
</div>"""


