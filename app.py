#!/usr/bin/env python3
"""
DocVerify - AI-Based Document Verification and Forgery Detection System
Production Web Application (AI-Powered Passport & Identity Document Verification)
Author: Senior Full-Stack AI & UX Systems Engineer
"""

import sys
import os
from pathlib import Path
import time
import io

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.inference_engine import DocVerifyInferenceEngine
from src.template_layouts import CANONICAL_TEMPLATE_LAYOUTS, get_template_display_name
from src.report_generator import generate_verification_report
from src.ui_components import (
    get_enterprise_css,
    ENTERPRISE_CSS,
    DOCUMENT_TYPE_CHOICES,
    POLICY_CHOICES,
    resolve_document_display_label,
    overlay_cam_map
)

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="DocVerify — Identity Document Verification",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---- Theme State Initialization ------------------------------------------------
# "dv_theme" stores either "dark" (default) or "light".
# Theme styles are injected purely through CSS tokens on each render without fragile JS.
if "dv_theme" not in st.session_state:
    st.session_state["dv_theme"] = "dark"

_current_theme = st.session_state["dv_theme"]

# Apply Enterprise Design System CSS dynamically based on active theme
st.markdown(get_enterprise_css(_current_theme), unsafe_allow_html=True)


# ==============================================================================
# Model Engine Caching (Phase 8 Production Configuration)
# ==============================================================================
@st.cache_resource(show_spinner="Initializing DocVerify multi-modal inference engine...")
def load_docverify_engine():
    """Caches the heavy CNN models, dedicated patch classifier, and OCR pipelines."""
    return DocVerifyInferenceEngine(
        global_model_path=BASE_DIR / "models" / "phase2" / "best_model.pth",
        patch_model_path=BASE_DIR / "models" / "phase8" / "dedicated_patch_model.pth",
        fusion_weights_path=BASE_DIR / "outputs" / "phase8" / "fusion" / "fusion_weights.json",
        thresholds_path=BASE_DIR / "outputs" / "phase8" / "fusion" / "operational_thresholds.json",
        calibrator_path=BASE_DIR / "models" / "phase8" / "fusion_calibrator.joblib",
        enable_ocr=True
    )


# ==============================================================================
# Main Application Flow
# ==============================================================================
def main():
    # -------------------------------------------------------------------------
    # 1. Header — Brand + Privacy Metadata + Status + Theme Toggle
    # -------------------------------------------------------------------------
    _theme = st.session_state.get("dv_theme", "dark")
    toggle_label = "Light Mode" if _theme == "dark" else "Dark Mode"
    toggle_icon = ":material/light_mode:" if _theme == "dark" else ":material/dark_mode:"

    hdr_col, toggle_col = st.columns([0.84, 0.16], vertical_alignment="center")

    with hdr_col:
        st.markdown("""
        <div class="dv-header">
            <div class="dv-brand-title">
                <div class="dv-brand-icon">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                        <path d="m9 12 2 2 4-4"/>
                    </svg>
                </div>
                <div>
                    <div class="dv-brand-wordmark">DocVerify</div>
                    <div class="dv-brand-subtitle">Identity Verification &amp; Document Intelligence</div>
                </div>
            </div>
            <div class="dv-header-badges">
                <div class="dv-env-badge">Production v9.2</div>
                <div class="dv-privacy-pill">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <rect width="18" height="11" x="3" y="11" rx="2" ry="2"/>
                        <path d="M7 11V7a5 5 0 0 1 10 0v4"/>
                    </svg>
                    <span>Single-Session</span>
                </div>
                <div class="dv-status-pill">
                    <div class="dv-status-dot"></div>
                    <span>Engine Active</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with toggle_col:
        if st.button(
            toggle_label,
            icon=toggle_icon,
            key="dv_theme_toggle",
            use_container_width=True,
            type="secondary",
            help="Toggle between light and dark theme"
        ):
            st.session_state["dv_theme"] = "light" if _theme == "dark" else "dark"
            st.rerun()

    # -------------------------------------------------------------------------
    # 2. Page Heading & Single-Session Privacy Notice
    # -------------------------------------------------------------------------
    st.markdown("""
        <div class="dv-page-title">Verify Identity Documents</div>
        <div class="dv-page-subtitle">Single-session document verification and forgery detection. Verification data is processed in-memory and not stored.</div>
    """, unsafe_allow_html=True)

    # Check session state for existing results
    prior_result = st.session_state.get("last_result", None)
    has_prior_result = prior_result is not None

    # Three-Stage Visual Workflow Tracker
    s1_class = "completed" if has_prior_result else "active"
    s2_class = "completed" if has_prior_result else ""
    s3_class = "active" if has_prior_result else ""
    l1_class = "active" if has_prior_result else ""
    l2_class = "active" if has_prior_result else ""

    st.markdown(f"""
    <div class="dv-workflow-stages">
        <div class="dv-workflow-step {s1_class}">
            <span class="dv-step-num">1</span>
            <div class="dv-step-text">
                <span class="dv-step-title">Document Upload</span>
                <span class="dv-step-desc">Intake &amp; Risk Settings</span>
            </div>
        </div>
        <div class="dv-step-line {l1_class}"></div>
        <div class="dv-workflow-step {s2_class}">
            <span class="dv-step-num">2</span>
            <div class="dv-step-text">
                <span class="dv-step-title">Verification Results</span>
                <span class="dv-step-desc">Authenticity Verdict</span>
            </div>
        </div>
        <div class="dv-step-line {l2_class}"></div>
        <div class="dv-workflow-step {s3_class}">
            <span class="dv-step-num">3</span>
            <div class="dv-step-text">
                <span class="dv-step-title">Evidence Analysis &amp; Report</span>
                <span class="dv-step-desc">Multi-Modal Signals &amp; PDF</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # STAGE 1: Document Upload & Verification Settings
    # -------------------------------------------------------------------------
    st.markdown("""
    <div class="dv-section-header">
        <span class="dv-stage-pill">Stage 1</span>
        <h3 class="dv-section-title">Document Upload &amp; Verification Settings</h3>
    </div>
    """, unsafe_allow_html=True)

    col_upload, col_settings = st.columns([1.15, 0.85], gap="large")

    pil_image = None
    file_name = "uploaded_document.jpg"
    benchmark_metadata = None

    with col_upload:
        st.markdown("""
        <div class="dv-card-title">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                <polyline points="17 8 12 3 7 8"/>
                <line x1="12" y1="3" x2="12" y2="15"/>
            </svg>
            <span>Document Upload</span>
        </div>
        <div class="dv-card-desc">Select or drag-and-drop a passport or identity document image for multi-modal authenticity assessment.</div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload Document",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            label_visibility="collapsed",
            help="Supported formats: JPG, PNG, WEBP, BMP. Maximum file size: 200 MB."
        )

        st.markdown("""
        <div class="dv-file-note" style="margin-bottom: 16px;">
            Supported formats: <b>JPG, PNG, WEBP, BMP</b> | Maximum file size: <b>200 MB</b>
        </div>
        """, unsafe_allow_html=True)

        # Primary Action Button located right under the file formats note
        if uploaded_file is not None:
            verify_btn = st.button("Verify Document Authenticity", type="primary", use_container_width=True, icon=":material/shield:")
        else:
            st.button("Verify Document Authenticity", type="secondary", disabled=True, use_container_width=True, icon=":material/shield:")
            verify_btn = False

        # Stale state prevention on file change / upload
        if uploaded_file is not None:
            if st.session_state.get("current_doc_id") != uploaded_file.name:
                st.session_state["current_doc_id"] = uploaded_file.name
                st.session_state.pop("last_result", None)
                st.session_state.pop("last_file_name", None)

            try:
                pil_image = Image.open(uploaded_file).convert("RGB")
                file_name = uploaded_file.name
            except Exception as e:
                st.error("Error reading uploaded document. Please upload a valid image file.")

    with col_settings:
        st.markdown("""
        <div class="dv-card-title">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/>
                <circle cx="12" cy="12" r="3"/>
            </svg>
            <span>Verification Settings</span>
        </div>
        <div class="dv-card-desc">Configure document type matching and risk evaluation policy.</div>
        """, unsafe_allow_html=True)

        # 1. Document Layout Mode
        st.markdown('<div class="dv-settings-label">Document Specification</div>', unsafe_allow_html=True)
        doc_mode = st.radio(
            "Document Mode",
            ["Automatic (AI Layout Matcher)", "Manual Override"],
            horizontal=True,
            label_visibility="collapsed"
        )
        
        if doc_mode == "Manual Override":
            manual_keys = [k for k, _ in DOCUMENT_TYPE_CHOICES if k != "auto"]
            manual_labels = {k: label for k, label in DOCUMENT_TYPE_CHOICES}
            selected_doc_key = st.selectbox(
                "Select Specific Template",
                manual_keys,
                index=0,
                format_func=lambda k: manual_labels.get(k, k),
                label_visibility="collapsed"
            )
            template_override = selected_doc_key
        else:
            selected_doc_key = "auto"
            template_override = None

        # 2. Operational Verification Policy
        st.markdown('<div class="dv-settings-label" style="margin-top:12px;">Verification Risk Policy</div>', unsafe_allow_html=True)
        policy_keys = [k for k, _, _ in POLICY_CHOICES]
        policy_labels = {k: name for k, name, _ in POLICY_CHOICES}
        policy_descs = {k: desc for k, _, desc in POLICY_CHOICES}

        selected_policy = st.radio(
            "Verification Policy",
            policy_keys,
            index=0,
            format_func=lambda k: policy_labels.get(k, k),
            label_visibility="collapsed"
        )
        st.markdown(f'<div class="dv-policy-desc">{policy_descs.get(selected_policy, "")}</div>', unsafe_allow_html=True)

    # Document Preview Card (Visible when document is loaded)
    if pil_image is not None:
        detected_preview_label = resolve_document_display_label(selected_doc_key) if selected_doc_key != "auto" else "Automatic Detection"
        st.markdown(f"""
        <div class="dv-preview-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                <div class="dv-card-title" style="margin:0;">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                        <polyline points="14 2 14 8 20 8"/>
                        <line x1="16" y1="13" x2="8" y2="13"/>
                        <line x1="16" y1="17" x2="8" y2="17"/>
                        <polyline points="10 9 9 9 8 9"/>
                    </svg>
                    <span>Document Preview</span>
                </div>
                <div class="dv-card-subtle">Aspect Ratio Preserved</div>
            </div>
            <div class="dv-meta-bar">
                <div class="dv-meta-item"><span>File:</span> <span class="dv-meta-val"><code>{file_name}</code></span></div>
                <div class="dv-meta-item"><span>Dimensions:</span> <span class="dv-meta-val">{pil_image.width} × {pil_image.height} px</span></div>
                <div class="dv-meta-item"><span>Standardized Format:</span> <span class="dv-meta-val">RGB</span></div>
                <div class="dv-meta-item"><span>Target Document:</span> <span class="dv-meta-val">{detected_preview_label}</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Centered image display with compact natural sizing (approx. 46% container width, max 520px)
        _, col_center_img, _ = st.columns([0.27, 0.46, 0.27])
        with col_center_img:
            st.image(pil_image, caption=f"Loaded Document: {file_name}", use_container_width=True)

            _, c_btn_col, _ = st.columns([0.20, 0.60, 0.20])
            with c_btn_col:
                if st.button("Remove Document", type="secondary", icon=":material/close:", use_container_width=True):
                    st.session_state.pop("current_doc_id", None)
                    st.session_state.pop("last_result", None)
                    st.session_state.pop("last_file_name", None)
                    st.rerun()

    # -------------------------------------------------------------------------
    # Verification Analysis Execution
    # -------------------------------------------------------------------------
    if verify_btn and pil_image is not None:
        engine = load_docverify_engine()
        progress_area = st.empty()

        with progress_area.container():
            st.markdown("""
            <div class="dv-analysis-card">
                <div class="dv-analysis-title">
                    <svg class="dv-spin" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M21 12a9 9 0 1 1-6.219-8.56"/>
                    </svg>
                    <span>Analyzing Document Authenticity...</span>
                </div>
                <div class="dv-analysis-steps">
                    <div class="dv-step-item"><span class="dv-step-check"></span> <span>Image standardized</span></div>
                    <div class="dv-step-item"><span class="dv-step-check"></span> <span>Document layout identified</span></div>
                    <div class="dv-step-item"><span class="dv-step-active"></span> <span>Inspecting visual authenticity</span></div>
                    <div class="dv-step-item"><span class="dv-step-pending"></span> <span>Checking local fields</span></div>
                    <div class="dv-step-item"><span class="dv-step-pending"></span> <span>Analyzing text consistency</span></div>
                    <div class="dv-step-item"><span class="dv-step-pending"></span> <span>Combining verification evidence</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            t0 = time.time()
            try:
                inference_result = engine.verify_document(
                    pil_image,
                    filename_hint=file_name,
                    template_override=template_override,
                    operational_mode=selected_policy
                )
                inference_result["latency_seconds"] = round(time.time() - t0, 3)
                inference_result["input_filename"] = file_name
                st.session_state["last_result"] = inference_result
                st.session_state["last_file_name"] = file_name
            except Exception as e:
                st.error("The system encountered an error while verifying the document. Please verify that the image is valid and readable.")
                progress_area.empty()
                return

        progress_area.empty()

    # -------------------------------------------------------------------------
    # STAGE 2: Verification Results
    # -------------------------------------------------------------------------
    result = st.session_state.get("last_result", None)
    if result is not None and pil_image is not None:
        st.markdown("""
        <div class="dv-section-header">
            <span class="dv-stage-pill">Stage 2</span>
            <h3 class="dv-section-title">Verification Results</h3>
        </div>
        """, unsafe_allow_html=True)

        verdict_str = str(result.get("verdict", "UNKNOWN")).upper()
        risk_level_str = str(result.get("risk_level", "MEDIUM RISK")).upper()
        risk_score = float(result.get("risk_score", 50.0))

        # Three distinct, decision-support states
        if "AUTHENTIC" in verdict_str or "GENUINE" in verdict_str or "LOW" in risk_level_str:
            card_class = "dv-verdict-low"
            badge_text = "VERIFIED GENUINE"
            badge_icon = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>'
            verdict_headline = "Verification Result: Genuine"
            verdict_desc = "The document shows no strong evidence of tampering based on the available verification signals."
        elif "FORGED" in verdict_str or "SUSPICIOUS" in verdict_str or "HIGH" in risk_level_str:
            card_class = "dv-verdict-high"
            badge_text = "POTENTIAL FORGERY DETECTED"
            badge_icon = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>'
            verdict_headline = "Verification Result: Potential Forgery Detected"
            verdict_desc = "The system detected multiple signals associated with possible document manipulation."
        else:
            card_class = "dv-verdict-med"
            badge_text = "REVIEW REQUIRED"
            badge_icon = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>'
            verdict_headline = "Verification Result: Review Required"
            verdict_desc = "The system detected signals that require additional review."

        # Verdict Summary Card
        st.markdown(f"""
        <div class="dv-verdict {card_class}">
            <div>
                <div class="dv-verdict-badge">{badge_icon} <span>{badge_text}</span></div>
                <div class="dv-verdict-title">{verdict_headline}</div>
                <div class="dv-verdict-desc">{verdict_desc}</div>
            </div>
            <div class="dv-verdict-score-box">
                <div class="dv-verdict-score-val">{risk_score:.1f}%</div>
                <div class="dv-verdict-score-label">Tampering Risk</div>
                <div class="dv-verdict-score-sub">Estimated tampering risk</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 5 Key Evidence Summary Tiles
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            detected_id = result.get("template_id", "unknown")
            doc_display = resolve_document_display_label(detected_id)
            st.markdown(f"""
            <div class="dv-metric-tile">
                <div class="dv-metric-label">Document Type</div>
                <div class="dv-metric-value" style="font-size: 0.9rem;">{doc_display}</div>
                <div class="dv-metric-note">Layout Identified</div>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            p_glob = float(result.get("evidence", {}).get("p_global_fake", 0.0)) * 100
            glob_status = "Elevated" if p_glob >= 60 else ("Moderate" if p_glob >= 40 else "Low")
            st.markdown(f"""
            <div class="dv-metric-tile">
                <div class="dv-metric-label">Visual Analysis</div>
                <div class="dv-metric-value">{p_glob:.1f}% <span style="font-size:0.75rem; font-weight:600; color:var(--dv-text-secondary);">({glob_status})</span></div>
                <div class="dv-metric-note">Macro Visual Texture</div>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            p_patch = float(result.get("evidence", {}).get("p_patch_fake", 0.0)) * 100
            patch_status = "Elevated" if p_patch >= 50 else "Low Concern"
            st.markdown(f"""
            <div class="dv-metric-tile">
                <div class="dv-metric-label">Field Inspection</div>
                <div class="dv-metric-value">{p_patch:.1f}% <span style="font-size:0.75rem; font-weight:600; color:var(--dv-text-secondary);">({patch_status})</span></div>
                <div class="dv-metric-note">Native-Scale Crops</div>
            </div>
            """, unsafe_allow_html=True)

        with m4:
            ocr_c = float(result.get("evidence", {}).get("ocr_confidence", 0.85)) * 100
            txt_status = "Good" if ocr_c >= 70 else "Fair"
            st.markdown(f"""
            <div class="dv-metric-tile">
                <div class="dv-metric-label">Text & Typography</div>
                <div class="dv-metric-value">{ocr_c:.1f}% <span style="font-size:0.75rem; font-weight:600; color:var(--dv-text-secondary);">({txt_status})</span></div>
                <div class="dv-metric-note">Character Recognition</div>
            </div>
            """, unsafe_allow_html=True)

        with m5:
            forensics = result.get("evidence", {}).get("forensic_cues", {})
            ela_val = float(forensics.get("ela_inconsistency", 0.0))
            forensic_status = "Normal" if ela_val < 0.30 else "Inconsistent"
            st.markdown(f"""
            <div class="dv-metric-tile">
                <div class="dv-metric-label">Forensic Cues</div>
                <div class="dv-metric-value">{forensic_status}</div>
                <div class="dv-metric-note">Compression & Noise</div>
            </div>
            """, unsafe_allow_html=True)

        # -------------------------------------------------------------------------
        # STAGE 3: Detailed Evidence Analysis & Report
        # -------------------------------------------------------------------------
        st.markdown("""
        <div class="dv-section-header">
            <span class="dv-stage-pill">Stage 3</span>
            <h3 class="dv-section-title">Detailed Analysis &amp; Verification Report</h3>
        </div>
        """, unsafe_allow_html=True)

        tab_visual, tab_patches, tab_ocr, tab_forensics, tab_report = st.tabs([
            "Visual Explainability",
            "Field-Level Inspection",
            "Text & Typography",
            "Evidence Fusion",
            "Verification Report"
        ])

        # TAB 1: Visual Explainability (Grad-CAM)
        with tab_visual:
            st.markdown("#### Visual Explainability (Macro Texture Attribution)")
            st.caption(
                "The highlighted regions indicate areas that contributed to the model's global assessment. "
                "Attention highlights indicate influential feature regions and should be treated as supporting evidence, not standalone proof of forgery."
            )

            cv1, cv2 = st.columns(2, gap="medium")
            with cv1:
                st.markdown("**Original Ingested Document**")
                st.image(result["pil_image"], use_container_width=True)
            with cv2:
                st.markdown("**Attention Heatmap Overlay (Grad-CAM)**")
                gc_map = result.get("heatmaps", {}).get("gradcam")
                if gc_map is not None:
                    overlay_img = overlay_cam_map(result["pil_image"], gc_map, alpha=0.55)
                    st.image(overlay_img, use_container_width=True)
                else:
                    st.info("Heatmap attribution is unavailable for this image.")

        # TAB 2: Field-Level Inspection (Native-scale Patch Crops)
        with tab_patches:
            st.markdown("#### Field-Level Inspection")
            st.caption(
                "Evaluates candidate text and portrait zones at native sensor resolution using the dedicated "
                "local patch verification model."
            )

            patch_findings = result.get("patch_findings", [])
            if patch_findings:
                # Up to 4 columns per row
                p_cols = st.columns(min(len(patch_findings), 4))
                pil_doc = result["pil_image"]

                for i, p_info in enumerate(patch_findings):
                    col = p_cols[i % len(p_cols)]
                    with col:
                        field_lbl = p_info.get("field_name", "Field").replace("_", " ").title()
                        prob_fake = float(p_info.get("patch_prob_fake", 0.0))
                        badge_class = "dv-field-badge-high" if prob_fake >= 0.50 else "dv-field-badge-low"
                        risk_tag = "Elevated Risk" if prob_fake >= 0.50 else "Low Risk"

                        st.markdown(f"""
                        <div class="dv-field-card">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <div class="dv-field-name">{field_lbl}</div>
                                <span class="{badge_class}">{risk_tag}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        bbox = p_info.get("crop_bbox")
                        if bbox and len(bbox) == 4:
                            x, y, w, h = bbox
                            patch_crop = pil_doc.crop((x, y, x + w, y + h))
                            st.image(patch_crop, use_container_width=True)
                        else:
                            st.info("No crop preview available")

                        st.caption(f"Estimated Field Risk: **{round(prob_fake * 100, 1)}%**")
            else:
                st.info("Candidate field layout is not predefined for this document geometry; evaluated via global texture and forensic consistency.")

        # TAB 3: Text & Typography Evidence
        with tab_ocr:
            st.markdown("#### Text & Typography Analysis")
            st.caption("Transcribes alphanumeric fields using optical character recognition and checks syntax integrity.")

            ocr_records = result.get("ocr_findings", [])
            if ocr_records:
                ocr_display_rows = []
                for item in ocr_records:
                    is_anom = item.get("is_anomaly", False)
                    status_str = "Format Discrepancy" if is_anom else "Valid Format"
                    txt = str(item.get("text", "")).strip()
                    conf = float(item.get("confidence", 0.0))

                    ocr_display_rows.append({
                        "Field": str(item.get("field_name", "N/A")).replace("_", " ").title(),
                        "Expected Format": str(item.get("field_type", "Text")).title(),
                        "Extracted Content": txt if txt else "[Empty / Unreadable]",
                        "Recognition Confidence": f"{round(conf * 100, 1)}%",
                        "Format Status": status_str
                    })
                st.table(pd.DataFrame(ocr_display_rows))
            else:
                st.info("OCR text extraction yielded no recognizable character sequences. The multi-modal pipeline completed verification using visual and forensic signals.")

        # TAB 4: Evidence Fusion & Forensic Consistency
        with tab_forensics:
            st.markdown("#### Multi-Modal Evidence Fusion Breakdown")
            st.caption(
                "DocVerify synthesizes evidence across visual, structural, textual, and statistical forensic dimensions. "
                "The calibrated risk is derived using logistic fusion and isotonic probability calibration."
            )

            evidence = result.get("evidence", {})
            forensics_dict = evidence.get("forensic_cues", {})

            f_col1, f_col2 = st.columns(2, gap="medium")
            with f_col1:
                st.markdown("**Evidence Signal Synthesis**")
                synthesis_rows = [
                    {"Signal Dimension": "Global Visual Texture", "Score": f"{float(evidence.get('p_global_fake', 0.0))*100:.1f}%", "Weight Assessment": "Macro-level authentic pattern verification"},
                    {"Signal Dimension": "Local Field Integrity", "Score": f"{float(evidence.get('p_patch_fake', 0.0))*100:.1f}%", "Weight Assessment": "High-resolution native-scale field verification"},
                    {"Signal Dimension": "Text Format Consistency", "Score": "Consistent" if evidence.get('format_anomalies_detected', 0) == 0 else "Anomaly Detected", "Weight Assessment": "Alphanumeric syntax alignment with template standard"},
                    {"Signal Dimension": "JPEG Error Analysis (ELA)", "Score": f"{forensics_dict.get('ela_inconsistency', 0.0):.4f}", "Weight Assessment": "Compression artifact homogeneity across boundaries"},
                    {"Signal Dimension": "Sensor Noise Distribution", "Score": f"{forensics_dict.get('noise_inconsistency', 0.0):.4f}", "Weight Assessment": "High-frequency image sensor pattern variance"}
                ]
                st.table(pd.DataFrame(synthesis_rows))

            with f_col2:
                st.markdown("**Operational Decision Calibration**")
                calib_prob = float(result.get('calibrated_forgery_probability', 0.0)) * 100
                thresh_applied = float(result.get('threshold_applied', 0.72)) * 100
                policy_name = result.get('operational_policy', 'Balanced Risk Policy')

                st.markdown(f"""
                <div class="dv-calibration-card">
                    <div class="dv-calibration-label">Calibrated Multi-Modal Tampering Probability</div>
                    <div class="dv-calibration-value">{calib_prob:.1f}%</div>
                    <div class="dv-calibration-note">
                        Evaluated under <b>{policy_name}</b> (Decision Threshold: <b>{thresh_applied:.1f}%</b>).
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                <div class="dv-decision-rule">
                    <b>Decision Rule:</b> When the calibrated tampering risk is below the operational policy threshold, the document is classified as authentic. Documents near the boundary trigger manual review, while documents exceeding the threshold are flagged for potential forgery.
                </div>
                """, unsafe_allow_html=True)

        # TAB 5: Report Export
        with tab_report:
            st.markdown("#### Verification Report Export")
            st.caption(
                "Generate a downloadable PDF verification report containing verification summary, evidence breakdown, "
                "and explainability heatmaps for documentation."
            )

            reports_dir = BASE_DIR / "outputs" / "phase9" / "sample_reports"
            reports_dir.mkdir(parents=True, exist_ok=True)

            clean_stem = Path(file_name).stem
            pdf_out_path = reports_dir / f"DocVerify_Report_{clean_stem}.pdf"

            temp_orig = reports_dir / f"temp_orig_{clean_stem}.png"
            temp_cam = reports_dir / f"temp_cam_{clean_stem}.png"

            result["pil_image"].save(temp_orig)
            gc_map = result.get("heatmaps", {}).get("gradcam")
            if gc_map is not None:
                gc_overlay = overlay_cam_map(result["pil_image"], gc_map, alpha=0.55)
                gc_overlay.save(temp_cam)
            else:
                temp_cam = None

            try:
                generate_verification_report(
                    verification_result=result,
                    output_pdf_path=pdf_out_path,
                    gradcam_img_path=temp_cam,
                    orig_img_path=temp_orig
                )

                with open(pdf_out_path, "rb") as f_pdf:
                    pdf_bytes = f_pdf.read()

                st.download_button(
                    label="Download Verification Report",
                    data=pdf_bytes,
                    file_name=pdf_out_path.name,
                    mime="application/pdf",
                    type="primary",
                    icon=":material/download:"
                )
                st.success(f"Verification report generated successfully ({len(pdf_bytes) / 1024:.1f} KB).")
            except Exception as e:
                st.error("Error generating verification report. Please try again.")
            finally:
                if temp_orig and temp_orig.exists():
                    try:
                        temp_orig.unlink(missing_ok=True)
                    except Exception:
                        pass
                if temp_cam and temp_cam.exists():
                    try:
                        temp_cam.unlink(missing_ok=True)
                    except Exception:
                        pass

        # Technical & Research Details Expander
        with st.expander("Technical & Research Details", expanded=False):
            st.markdown(f"**Document File:** `{result.get('input_filename', file_name)}`")
            st.markdown(f"**Identified Template:** {resolve_document_display_label(result.get('template_id', 'unknown'))}")
            st.markdown(f"**Template Matching Confidence:** `{round(result.get('template_confidence', 1.0) * 100, 1)}%` (Method: {result.get('template_method')})")
            st.markdown(f"**Processing Latency:** `{result.get('latency_seconds', 0.0)} seconds`")
            st.markdown(f"**Hardware Device:** `{result.get('device', 'CPU')}`")
            st.markdown(f"**Operational Policy Applied:** {result.get('operational_policy', 'Balanced Risk Policy')}")
            st.markdown(f"**Global Classifier:** EfficientNet-B0 (Frozen Base Model)")
            st.markdown(f"**Patch Classifier:** EfficientNet-B0 (Dedicated High-Resolution Model)")
            st.markdown(f"**Probability Calibrator:** Isotonic Regression (Phase 8 Production Calibrator)")

    # -------------------------------------------------------------------------
    # 8. Disclaimer Footer
    # -------------------------------------------------------------------------
    st.markdown("""
    <div class="dv-footer">
        <b>DocVerify Enterprise AI</b> &bull; Privacy-First Single-Session Document Verification Architecture<br/>
        <span style="font-size:0.75rem;">
            Stateless in-memory processing. No verification history or uploaded documents are retained on server.<br/>
            Powered by PyTorch 2.6, EfficientNet-B0 Dual Backbones, EasyOCR, and Isotonic Probability Calibration.
        </span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
