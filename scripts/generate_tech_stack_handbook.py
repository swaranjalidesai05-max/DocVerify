#!/usr/bin/env python3
"""
DocVerify - Master Comprehensive Project Knowledge, Architecture, Engineering & Viva Handbook Generator
Generates an all-inclusive, highly structured PDF reference document covering everything from basic project
fundamentals to advanced scientific and technical details, designed for defense in university reviews,
examinations, presentations, and team knowledge transfer.
"""

import sys
from pathlib import Path
import time

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable, KeepTogether
)
from reportlab.pdfgen import canvas

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_PDF_PATH = PROJECT_ROOT / "outputs" / "DocVerify_TechStack_Architecture_and_Metrics_Handbook.pdf"


class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        # Top Header line & title
        self.setFont("Helvetica-Bold", 7.5)
        self.setFillColor(colors.HexColor("#1e3a8a"))
        self.drawString(54, 750, "DOCVERIFY: AI-BASED IDENTITY DOCUMENT VERIFICATION & FORGERY DETECTION")
        self.setFont("Helvetica", 7.5)
        self.setFillColor(colors.HexColor("#64748b"))
        self.drawRightString(558, 750, "MASTER PROJECT KNOWLEDGE & VIVA HANDBOOK")
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(54, 744, 558, 744)

        # Bottom Footer line & page numbers
        self.line(54, 46, 558, 46)
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))
        self.drawString(54, 34, "DocVerify Engineering & Academic Research Reference • All Rights Reserved")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 34, page_str)
        self.restoreState()


def build_master_handbook():
    OUTPUT_PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT_PDF_PATH),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Typography styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#475569"),
        spaceAfter=12
    )
    section_h1 = ParagraphStyle(
        "SectionH1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=12.5,
        leading=16,
        textColor=colors.HexColor("#1e3a8a"),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )
    section_h2 = ParagraphStyle(
        "SectionH2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13.5,
        textColor=colors.HexColor("#334155"),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        "DocBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=6
    )
    bullet_style = ParagraphStyle(
        "DocBullet",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.25,
        leading=11.5,
        textColor=colors.HexColor("#1e293b"),
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=3
    )
    qna_q = ParagraphStyle(
        "QnA_Q",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.75,
        leading=12,
        textColor=colors.HexColor("#1e3a8a"),
        spaceBefore=6,
        spaceAfter=2,
        keepWithNext=True
    )
    qna_a = ParagraphStyle(
        "QnA_A",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.25,
        leading=11.5,
        textColor=colors.HexColor("#1e293b"),
        leftIndent=8,
        spaceAfter=6
    )

    # Table styles
    th_style = ParagraphStyle(
        "TH",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10.5,
        textColor=colors.white
    )
    td_style = ParagraphStyle(
        "TD",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.75,
        leading=10.5,
        textColor=colors.HexColor("#0f172a")
    )
    td_bold = ParagraphStyle(
        "TDBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.75,
        leading=10.5,
        textColor=colors.HexColor("#0f172a")
    )
    td_status_use = ParagraphStyle(
        "TDUse",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#15803d")
    )
    td_status_rej = ParagraphStyle(
        "TDRej",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#b91c1c")
    )

    story = []

    # =========================================================================
    # TITLE & HEADER
    # =========================================================================
    story.append(Paragraph("DocVerify — Master Project Knowledge & Viva Handbook", title_style))
    story.append(Paragraph(
        "Complete Technical Compendium: Basic Project Overview, Problem Statement, Dataset Foundations, "
        "Roadmap Evolution (Phase 1–9), Technology Stack Decisions, Multi-Modal Scoring, Evaluation Metrics, "
        "and Comprehensive Viva / Faculty Defense Guide.", subtitle_style
    ))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1e40af"), spaceBefore=0, spaceAfter=12))

    # =========================================================================
    # PART 1: PROJECT FOUNDATIONS & BASIC OVERVIEW
    # =========================================================================
    story.append(Paragraph("1. Project Foundations & Basic Overview", section_h1))
    story.append(Paragraph(
        "<b>What is DocVerify?</b> DocVerify is an AI-powered verification and forgery detection system specifically engineered for "
        "evaluating the authenticity of international passports and national identity cards. It solves the critical limitation of traditional automated "
        "verification systems by combining full-document macro visual texture analysis, uncompressed native-sensor patch inspection, "
        "optical character recognition (OCR) typography analysis, regex syntactic rule checking, and digital image forensics (Error Level Analysis "
        "and camera sensor noise consistency) into a statistically calibrated decision engine.", body_style
    ))
    story.append(Paragraph(
        "<b>Core Problem Statement:</b> With the advent of digital photo-editing tools and generative AI, document forgery has shifted from physical "
        "tampering to sophisticated digital manipulations—such as inpainting, number rewriting, and face swapping. Standard Deep Convolutional Neural Networks "
        "(CNNs) fail to catch these attacks reliably when full documents are resized down to small inputs (e.g. 224×224 px), because tiny altered digits "
        "become blurred out. Furthermore, single-modality AI models suffer from high false alarm rates. DocVerify introduces a <b>multi-modal defense</b> "
        "where visual, textual, and forensic signals verify each other.", body_style
    ))
    story.append(Spacer(1, 4))

    # =========================================================================
    # PART 2: DATASET FOUNDATIONS & BENCHMARK SETUP
    # =========================================================================
    story.append(Paragraph("2. Dataset Foundations: The SIDTD Benchmark", section_h1))
    story.append(Paragraph(
        "DocVerify is built on the rigorous, peer-reviewed <b>SIDTD (Synthetic & Identity Document Tampering Dataset)</b> standard. "
        "The project does not use random internet pictures or unverified datasets. The following matrix outlines the dataset parameters:", body_style
    ))

    dataset_summary_data = [
        [Paragraph("Dataset Parameter", th_style), Paragraph("Specification / Value", th_style), Paragraph("Significance & Engineering Importance", th_style)],
        [Paragraph("<b>Total Dataset Volume</b>", td_bold), Paragraph("<b>2,222 Total Documents</b><br/>(1,000 Genuine, 1,222 Forged)", td_style), Paragraph("Provides balanced, realistic distribution across authentic and manipulated government-issued IDs.", td_style)],
        [Paragraph("<b>Document Templates</b>", td_bold), Paragraph("<b>10 International Templates</b> across 10 countries", td_style), Paragraph("Passports: Azerbaijan, Greece, Latvia, Russia, Serbia, Slovakia.<br/>National IDs: Albania, Spain, Estonia, Finland.", td_style)],
        [Paragraph("<b>Dataset Split Scheme</b>", td_bold), Paragraph("<b>Identity-Disjoint Split:</b><br/>Train: 1,565 | Val: 337 | Test: 320", td_style), Paragraph("<b>Zero Subject Leakage:</b> No person, identity card, or base ID in the training set appears in validation or test splits. Eliminates artificial memorization.", td_style)],
        [Paragraph("<b>Manipulation Types</b>", td_bold), Paragraph("• <b>Inpaint_and_Rewrite</b> (1,077)<br/>• <b>Crop_and_Replace</b> (145)", td_style), Paragraph("Covers realistic real-world attacks: digital text alteration (erasing numbers/dates and re-typing) and photo replacement.", td_style)],
        [Paragraph("<b>Manipulated Fields</b>", td_bold), Paragraph("Names, Surnames, Birth Dates, Expiry Dates, ID Numbers, Gender, Nationality", td_style), Paragraph("Focuses on the high-value alphanumeric fields targeted by fraudsters in banking, KYC, immigration, and border control.", td_style)]
    ]
    t_data = Table(dataset_summary_data, colWidths=[120, 140, 244])
    t_data.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t_data)
    story.append(Spacer(1, 10))

    # =========================================================================
    # PART 3: PHASE-BY-PHASE DEVELOPMENT ROADMAP (PHASE 1 TO 9)
    # =========================================================================
    story.append(Paragraph("3. Complete Project Evolution: Phase-by-Phase Roadmap (Phase 1–9)", section_h1))
    story.append(Paragraph(
        "DocVerify was developed through an iterative, hypothesis-driven engineering methodology. Understanding what transpired in every phase "
        "is critical to understanding why the current architecture exists:", body_style
    ))

    roadmap_data = [
        [Paragraph("Phase", th_style), Paragraph("Milestone & Objective", th_style), Paragraph("Key Findings, Discoveries & Technical Transformations", th_style)],
        [Paragraph("<b>Phase 1</b>", td_bold), Paragraph("Dataset Audit & Ingestion", td_style), Paragraph("Discovered CMYK color-space issues causing PIL crashes. Built verified `dataset_manifest.csv` with MD5 hashes, color standardization, and identity clusters.", td_style)],
        [Paragraph("<b>Phase 2</b>", td_bold), Paragraph("Global Baseline Model", td_style), Paragraph("Trained global EfficientNet-B0 on full document images. Reached <b>79.69% accuracy</b> (ROC-AUC: 0.9112). Established the permanent, frozen global feature backbone.", td_style)],
        [Paragraph("<b>Phase 3</b>", td_bold), Paragraph("Explainability & Failure Analysis", td_style), Paragraph("Implemented initial Grad-CAM. Discovered that the global model struggled with small text manipulations and suffered from spatial stretching when resizing rectangular IDs to squares.", td_style)],
        [Paragraph("<b>Phase 4</b>", td_bold), Paragraph("Multi-Modal Prototype", td_style), Paragraph("Introduced experimental Tesseract OCR, YOLO proposals, and multi-modal signals. Identified that naive score averaging produced erratic decisions and high false alarms.", td_style)],
        [Paragraph("<b>Phase 5</b>", td_bold), Paragraph("Local Field Extraction", td_style), Paragraph("Prototyped crop extraction for candidate fields. Discovered that YOLO bounding boxes suffered from jitter and boundary cuts. Developed deterministic canonical geometry matching.", td_style)],
        [Paragraph("<b>Phase 6</b>", td_bold), Paragraph("Leakage Audit & Frozen Test", td_style), Paragraph("Discovered that filename parsing (`alb_id_01.jpg`) acted as an oracle shortcut. Sealed oracle leakage by removing all filename hints. Froze the permanent 320-sample test set.", td_style)],
        [Paragraph("<b>Phase 7</b>", td_bold), Paragraph("Fusion & Calibration R&D", td_style), Paragraph("Trained logistic regression fusion model. Introduced Isotonic calibration. Identified deep learning overconfidence issues and calibrated empirical risk probabilities.", td_style)],
        [Paragraph("<b>Phase 8</b>", td_bold), Paragraph("Final Remediated Benchmark", td_style), Paragraph("Remediated dedicated patch model with 2.0x crop expansion. Achieved benchmark jump from <b>79.69% to 91.25% accuracy</b> (ROC-AUC: 0.9677, McNemar's p < 0.001). 14/14 automated tests passed.", td_style)],
        [Paragraph("<b>Phase 9</b>", td_bold), Paragraph("Enterprise UI & Polish", td_style), Paragraph("Enterprise CSS redesign. Fixed tab clipping and text visibility. Corrected threshold boundary condition (`>` vs `>=`). Resolved 100% risk clamping via continuous proportional scaling.", td_style)]
    ]
    t_roadmap = Table(roadmap_data, colWidths=[55, 125, 324])
    t_roadmap.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t_roadmap)
    story.append(Spacer(1, 10))

    # =========================================================================
    # PART 4: MASTER TECHNOLOGY STACK (RETAINED VS. REJECTED)
    # =========================================================================
    story.append(Paragraph("4. Master Technology Stack: Retained vs. Replaced Technologies", section_h1))
    story.append(Paragraph(
        "The following structured matrix details every technology evaluated across the project's development lifecycle, documenting whether it is actively deployed "
        "or rejected, along with the precise technical justification.", body_style
    ))

    tech_table_data = [
        [Paragraph("Component / Domain", th_style), Paragraph("Evaluated Technology", th_style), Paragraph("Status", th_style), Paragraph("Technical Justification & Lifecycle Rationale", th_style)],
        
        [Paragraph("<b>Object Detection / Layout Proposal</b>", td_style),
         Paragraph("<b>YOLO (v5 / v8)</b>", td_bold),
         Paragraph("REJECTED", td_status_rej),
         Paragraph("<b>Overkill & Unstable Boundaries:</b> YOLO proposals suffered from coordinate jitter, frequently slicing text or portrait boundaries and causing downstream OCR failures. Replaced with exact canonical geometry matching.", td_style)],

        [Paragraph("<b>Document Layout Matching</b>", td_style),
         Paragraph("<b>Autonomous Geometry & Aspect Ratio Alignment</b>", td_bold),
         Paragraph("ACTIVE", td_status_use),
         Paragraph("<b>Deterministic & Fast:</b> Computes physical document aspect ratios and field coordinates mathematically from predefined canonical templates. Executes 10x faster with zero memory overhead and zero boundary drift.", td_style)],

        [Paragraph("<b>Physical Liveness / Anti-Spoofing</b>", td_style),
         Paragraph("<b>PAD (Presentation Attack Detection)</b>", td_bold),
         Paragraph("REJECTED", td_status_rej),
         Paragraph("<b>Scope & Dataset Misalignment:</b> PAD is tailored for biometric screen/paper replay attacks, whereas the SIDTD dataset targets digital tampering (altered digits, swapped photos). Generated heavy false positives on clean flatbed scans.", td_style)],

        [Paragraph("<b>Digital Image Forensics</b>", td_style),
         Paragraph("<b>ELA, Sensor Noise & Edge Variance</b>", td_bold),
         Paragraph("ACTIVE", td_status_use),
         Paragraph("<b>Signal-Level Splicing Detection:</b> Replaced PAD. Detects differential JPEG re-compression (ELA), camera sensor pattern discrepancies across quadrants, and Laplacian edge blurring around digitally pasted elements.", td_style)],

        [Paragraph("<b>Text Extraction (OCR)</b>", td_style),
         Paragraph("<b>Tesseract OCR</b>", td_bold),
         Paragraph("REJECTED", td_status_rej),
         Paragraph("<b>External Dependency & Brittle:</b> Required OS-level binary installations (`apt`/`.exe`), had fragile threading, and struggled with complex background Guilloche security line patterns.", td_style)],

        [Paragraph("<b>Text Extraction (OCR)</b>", td_style),
         Paragraph("<b>EasyOCR (PyTorch CRAFT + CRNN)</b>", td_bold),
         Paragraph("ACTIVE", td_status_use),
         Paragraph("<b>Python Native & Resilient:</b> Runs directly within PyTorch on CPU or GPU. Delivers superior character recognition across skewed identity cards and multi-lingual character sets.", td_style)],

        [Paragraph("<b>Global Feature Extractor</b>", td_style),
         Paragraph("<b>EfficientNet-B0 Backbone (Frozen)</b>", td_bold),
         Paragraph("ACTIVE", td_status_use),
         Paragraph("<b>High Accuracy / Low Overhead:</b> 4.01M parameters. Retrained in Phase 2 on global identity textures, then frozen to prevent catastrophic forgetting. Captures full-document security patterns and macro inconsistencies.", td_style)],

        [Paragraph("<b>Local Patch Inspection</b>", td_style),
         Paragraph("<b>Dedicated EfficientNet-B0 Patch Classifier</b>", td_bold),
         Paragraph("ACTIVE", td_status_use),
         Paragraph("<b>Prevents Resolution Loss:</b> Downscaling high-res documents (e.g. 2167×1360) to 224×224 blurs tiny altered numbers. The patch model inspects text/photo fields at native camera scale with 2.0x context expansion.", td_style)],

        [Paragraph("<b>Visual Explainability (XAI)</b>", td_style),
         Paragraph("<b>Naive Standard Grad-CAM</b>", td_bold),
         Paragraph("REJECTED", td_status_rej),
         Paragraph("<b>Geometric Spatial Distortion:</b> Resizing rectangular ID cards (1.6:1 ratio) to square 224×224 inputs caused Grad-CAM heatmaps to stretch and point to incorrect coordinates on the original image.", td_style)],

        [Paragraph("<b>Visual Explainability (XAI)</b>", td_style),
         Paragraph("<b>CorrectedGradCAM (Inverse Coordinate Mapping)</b>", td_bold),
         Paragraph("ACTIVE", td_status_use),
         Paragraph("<b>Mathematically Exact Attribution:</b> Projects activation gradients backwards through exact aspect ratio transforms directly onto the native document image, preserving pixel alignment.", td_style)],

        [Paragraph("<b>Multi-Modal Evidence Fusion</b>", td_style),
         Paragraph("<b>Logistic Fusion + Scaler Normalization</b>", td_bold),
         Paragraph("ACTIVE", td_status_use),
         Paragraph("<b>Statistically Learned Weights:</b> Replaced naive score averaging. Assigns mathematically optimal weights (+2.92 to global, +1.56 to patch) learned on validation data, accounting for feature correlations.", td_style)],

        [Paragraph("<b>Probability Calibration</b>", td_style),
         Paragraph("<b>Isotonic Regression + Continuous Scaling</b>", td_bold),
         Paragraph("ACTIVE", td_status_use),
         Paragraph("<b>Calibrated Confidence:</b> Prevents deep learning overconfidence. Converts raw logits into statistically verified probabilities (Brier: 0.0657, ECE: 0.0484) with smooth proportional risk scaling.", td_style)],

        [Paragraph("<b>Decision Policy Layer</b>", td_style),
         Paragraph("<b>Monotonically Ordered Multi-Policy Thresholds</b>", td_bold),
         Paragraph("ACTIVE", td_status_use),
         Paragraph("<b>Operational Flexibility:</b> Provides 3 operational thresholds (Conservative KYC: 0.50, Balanced: 0.72, Strict: 0.80) to accommodate different enterprise risk tolerances.", td_style)],

        [Paragraph("<b>User Interface</b>", td_style),
         Paragraph("<b>Streamlit 1.42 + Custom Enterprise CSS</b>", td_bold),
         Paragraph("ACTIVE", td_status_use),
         Paragraph("<b>Enterprise UX Standards:</b> Overrides default Streamlit styling to eliminate tab clipping, enforce readable typography, provide responsive containers, and display clean status badges.", td_style)],

        [Paragraph("<b>Audit Certificate Generation</b>", td_style),
         Paragraph("<b>ReportLab 5.0.1 Flowable PDF Engine</b>", td_bold),
         Paragraph("ACTIVE", td_status_use),
         Paragraph("<b>Compliance Record Export:</b> Generates professional, multi-page verification reports complete with session metadata, extracted text tables, and dual heatmap overlays.", td_style)]
    ]

    t_tech = Table(tech_table_data, colWidths=[95, 110, 55, 244])
    t_tech.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t_tech)
    story.append(Spacer(1, 10))

    # =========================================================================
    # PART 5: COMPLETE SCORE CATALOG (ALL 9 SCORES)
    # =========================================================================
    story.append(Paragraph("5. Complete Score Catalog: Every Score Calculated by DocVerify", section_h1))
    story.append(Paragraph(
        "DocVerify calculates a total of <b>9 distinct scores</b> during each inference execution: 7 multi-modal evidence scores and 2 final decision scores. "
        "No single clue is trusted in isolation; all signals are synthesized mathematically.", body_style
    ))

    scores_table_data = [
        [Paragraph("Score Identifier", th_style), Paragraph("Scale / Range", th_style), Paragraph("Extraction Source", th_style), Paragraph("Functional Role & Investigative Objective", th_style)],
        
        [Paragraph("<b>p_global_fake</b><br/>(Global Visual Score)", td_bold),
         Paragraph("0.0000 – 1.0000", td_style),
         Paragraph("Frozen EfficientNet-B0 Full Document Classifier", td_style),
         Paragraph("Evaluates macro-level visual features: background Guilloche patterns, micro-line security printing, overall color balance, and border integrity.", td_style)],

        [Paragraph("<b>p_patch_fake</b><br/>(Local Field Score)", td_bold),
         Paragraph("0.0000 – 1.0000", td_style),
         Paragraph("Dedicated Patch Model on 2.0x Field Crops", td_style),
         Paragraph("Inspects sensitive zones (name, birth date, document ID number, portrait photo) at native camera resolution to catch micro-tampering blurred out by global downsampling.", td_style)],

        [Paragraph("<b>ela_inconsistency</b><br/>(Error Level Analysis)", td_bold),
         Paragraph("0.0000 – 1.0000+<br/>(Normalized Std)", td_style),
         Paragraph("JPEG Re-compression Error Variance", td_style),
         Paragraph("Measures compression discrepancy across image quadrants. When forgers paste modified text onto an existing document, the modified region compresses differently from the background.", td_style)],

        [Paragraph("<b>noise_inconsistency</b><br/>(Sensor Pattern Variance)", td_bold),
         Paragraph("0.0000 – 50.0000+<br/>(Variance Std)", td_style),
         Paragraph("High-Pass Median Filter Residual Variance", td_style),
         Paragraph("Detects multi-source image splicing. Every scanner and camera sensor introduces a unique noise fingerprint. Discrepancies across quadrants expose spliced portraits or fields.", td_style)],

        [Paragraph("<b>edge_variance</b><br/>(Laplacian Sharpness)", td_bold),
         Paragraph("0.0 – 5000.0+<br/>(Raw Variance)", td_style),
         Paragraph("Laplacian Filter Edge Gradient Distribution", td_style),
         Paragraph("Quantifies edge sharpness. Forgers frequently apply subtle Gaussian blur or feathering to soften cut boundaries. Abnormal edge distributions flag artificial tampering.", td_style)],

        [Paragraph("<b>ocr_confidence</b><br/>(Typography Score)", td_bold),
         Paragraph("0.0000 – 1.0000<br/>(Mean Conf)", td_style),
         Paragraph("EasyOCR Character Recognition Confidence", td_style),
         Paragraph("Assesses typographic consistency. Legitimate identity cards utilize crisp, standardized government typefaces. Unofficial or modified fonts trigger low recognition confidence.", td_style)],

        [Paragraph("<b>format_anomaly</b><br/>(Syntactic Prior Check)", td_bold),
         Paragraph("0 (Valid) or<br/>1 (Anomaly)", td_style),
         Paragraph("Regex Pattern Rules (Dates, Alphanumeric IDs)", td_style),
         Paragraph("Validates expected structural formats (e.g. `DD/MM/YYYY`, strictly numeric ID sequences). Forgers making typographic syntax errors are flagged immediately.", td_style)],

        [Paragraph("<b>raw_forgery_probability</b><br/>(Fused Logit)", td_bold),
         Paragraph("0.0000 – 1.0000", td_style),
         Paragraph("Trained Multi-Modal Logistic Regression", td_style),
         Paragraph("Combines all 7 standardized features using validation-learned coefficients (+2.92 global, +1.56 patch, -1.68 ELA) into a unified probability logit.", td_style)],

        [Paragraph("<b>risk_score</b><br/>(Calibrated User Score)", td_bold),
         Paragraph("0.0% – 100.0%<br/>(Proportional)", td_style),
         Paragraph("Isotonic Calibrator + Continuous Scaling", td_style),
         Paragraph("The primary user-facing tampering metric. Maps raw confidence into true statistical probabilities and provides continuous scaling, reserving 100% strictly for near-certain forgeries.", td_style)]
    ]

    t_scores = Table(scores_table_data, colWidths=[105, 75, 115, 209])
    t_scores.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t_scores)
    story.append(Spacer(1, 10))

    # =========================================================================
    # PART 6: SCIENTIFIC EVALUATION METRICS MATRIX
    # =========================================================================
    story.append(Paragraph("6. Scientific Evaluation Metrics & Phase 8 Benchmark Results", section_h1))
    story.append(Paragraph(
        "DocVerify was subjected to rigorous empirical evaluation on the <b>frozen 320-document test dataset</b> (151 genuine, 169 forged) "
        "across 10 international document templates. The system evaluated performance across four statistical dimensions:", body_style
    ))

    metrics_table_data = [
        [Paragraph("Evaluation Metric", th_style), Paragraph("Benchmark Value", th_style), Paragraph("Mathematical Definition", th_style), Paragraph("Why This Specific Metric Is Essential for DocVerify", th_style)],
        
        [Paragraph("<b>Accuracy</b>", td_bold),
         Paragraph("<b>91.25%</b><br/>(292 / 320)", td_style),
         Paragraph("(TP + TN) / Total", td_style),
         Paragraph("Provides the general baseline correctness across the entire test set. However, accuracy alone is insufficient for security applications due to class balance dependencies.", td_style)],

        [Paragraph("<b>Precision</b><br/>(Positive Predictive Value)", td_bold),
         Paragraph("<b>91.23%</b><br/>(156 / 171)", td_style),
         Paragraph("TP / (TP + FP)", td_style),
         Paragraph("<b>Prevents False Rejections:</b> When the AI triggers a fraud alert, how often is it actually forged? High precision guarantees innocent citizens are not falsely denied access.", td_style)],

        [Paragraph("<b>Recall</b><br/>(Sensitivity / True Positive)", td_bold),
         Paragraph("<b>92.31%</b><br/>(156 / 169)", td_style),
         Paragraph("TP / (TP + FN)", td_style),
         Paragraph("<b>Prevents Security Breaches:</b> What percentage of all forged documents were caught? High recall ensures manipulated passports do not slip past border or KYC security.", td_style)],

        [Paragraph("<b>F1-Score</b><br/>(Harmonic Balance)", td_bold),
         Paragraph("<b>91.76%</b><br/>(0.9176)", td_style),
         Paragraph("2 × (P × R) / (P + R)", td_style),
         Paragraph("<b>Core Verification Measure:</b> Harmonic mean balancing precision and recall. Proves that the model achieves high fraud detection without generating excessive false alarms.", td_style)],

        [Paragraph("<b>ROC-AUC</b><br/>(Area Under ROC Curve)", td_bold),
         Paragraph("<b>0.9677</b><br/>(96.77%)", td_style),
         Paragraph("Integral of TPR vs. FPR across all thresholds", td_style),
         Paragraph("<b>Threshold-Independent Quality:</b> Demonstrates that the multi-modal features separate authentic and forged documents exceptionally well regardless of the chosen operational threshold.", td_style)],

        [Paragraph("<b>Brier Score</b><br/>(Probability Error)", td_bold),
         Paragraph("<b>0.0657</b><br/>(Very Low)", td_style),
         Paragraph("(1/N) × ∑ (p_pred - y_true)²", td_style),
         Paragraph("<b>Probability Accuracy:</b> Traditional metrics ignore confidence quality. Brier score measures mean squared probability error; 0.0657 confirms predicted probabilities match true outcomes.", td_style)],

        [Paragraph("<b>Expected Calibration Error (ECE)</b>", td_bold),
         Paragraph("<b>0.0484</b><br/>(4.84%)", td_style),
         Paragraph("Weighted difference between confidence and accuracy across 10 bins", td_style),
         Paragraph("<b>Eliminates Overconfidence:</b> Uncalibrated neural networks often output 99% confidence on borderline guesses. An ECE under 5% guarantees that a 70% risk score represents ~70% empirical fraud.", td_style)],

        [Paragraph("<b>McNemar's Test</b><br/>(Statistical Significance)", td_bold),
         Paragraph("<b>p = 0.000003</b><br/>(p < 0.001)", td_style),
         Paragraph("Paired contingency: (b - c)² / (b + c)", td_style),
         Paragraph("<b>Scientific Proof of Superiority:</b> Compares Phase 8 directly against the Phase 2 Global Baseline (79.69% accuracy). The p-value confirms the 11.56% accuracy leap is statistically significant.", td_style)]
    ]

    t_metrics = Table(metrics_table_data, colWidths=[95, 75, 105, 229])
    t_metrics.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t_metrics)
    story.append(Spacer(1, 10))

    # =========================================================================
    # PART 7: NOTE ON EXCLUDED METRICS (WHY IOU IS NOT USED)
    # =========================================================================
    story.append(Paragraph("7. Note on Excluded Metrics: Why IoU (Intersection over Union) Is Not Applicable", section_h1))
    story.append(Paragraph(
        "Reviewers occasionally ask why object detection metrics such as <b>IoU (Intersection over Union)</b> are not reported in DocVerify:", body_style
    ))
    story.append(Paragraph("• <b>Classification vs. Detection Paradigm:</b> DocVerify is designed as a document verification and classification system. Its task is to output an authentic/tampered verdict, forensic risk scores, and attention heatmaps—not bounding-box coordinates.", bullet_style))
    story.append(Paragraph("• <b>Elimination of Box Proposals:</b> Because YOLO was dropped in favor of canonical template geometry, field bounding boxes are calculated deterministically from standard aspect ratios rather than predicted as regression targets.", bullet_style))
    story.append(Paragraph("• <b>Zero Bounding Box Drift:</b> Without bounding-box regression heads, measuring overlap via IoU is mathematically non-applicable. All statistical rigor is properly focused on classification metrics (F1, Precision, Recall), calibration metrics (Brier, ECE), and discriminative power (ROC-AUC).", bullet_style))
    story.append(Spacer(1, 10))

    # =========================================================================
    # PART 8: COMPLETE FACULTY & VIVA Q&A DEFENSE GUIDE
    # =========================================================================
    story.append(Paragraph("8. Complete Viva & Faculty Defense Guide (Frequently Asked Questions)", section_h1))
    story.append(Paragraph(
        "The following questions and model answers prepare project teammates to address any technical, theoretical, or practical inquiry "
        "from evaluators, professors, or external examiners with authority:", body_style
    ))

    viva_qa = [
        ("Q1: What is the main innovation of DocVerify compared to regular CNN classifiers?",
         "DocVerify rejects the naive assumption that document verification is just image classification. High-resolution identity documents lose critical micro-tampering details when downscaled to 224x224. DocVerify introduces a multi-modal fusion architecture that evaluates global visual features, uncompressed 2.0x local field patches, OCR typography consistency, syntax rules, and JPEG/sensor noise forensics in a single calibrated model."),

        ("Q2: Why did you freeze the Global EfficientNet-B0 backbone instead of fine-tuning it end-to-end?",
         "Freezing the Phase 2 global backbone (trained on 1,565 identity documents) preserves foundational macro-texture feature representations and prevents catastrophic forgetting. It also isolates the patch and fusion stages, ensuring reproducible, modular scientific evaluation without gradient interference."),

        ("Q3: How does DocVerify prevent shortcut learning and oracle leakage?",
         "In Phase 6, we discovered that parsing filenames (like 'alb_id_01.jpg') allowed the model to bypass actual visual inspection. We completely eliminated all filename hints from inference. The system autonomously identifies documents by measuring physical aspect ratios and geometry against canonical standards with an explicit UNKNOWN fallback."),

        ("Q4: What is the difference between Raw Probability and Calibrated Probability?",
         "Deep learning models output uncalibrated softmax logits that often exhibit extreme overconfidence (e.g. 99% confident on ambiguous samples). Calibrated probability uses Isotonic Regression fitted on the validation set to align predicted risk with actual empirical frequency (reducing Brier score to 0.0657 and ECE to 0.0484)."),

        ("Q5: Why did you change the threshold comparison from >= to > in Phase 9?",
         "The Isotonic calibrator maps a cluster of borderline genuine documents on the validation split to the flat step value of exactly 0.7200. Using >= caused genuine documents at this boundary to be classified as forgeries. Using strict > and introducing an explicit review zone (threshold - 0.15) resolved these boundary false positives while maintaining 100% fake detection."),

        ("Q6: Why does the system no longer collapse high-risk documents to a flat 100.0%?",
         "The isotonic calibrator's top step for scores above 0.707 previously mapped to 1.0 (100%). We introduced a continuous proportional scaling layer that scales risk dynamically across [72.0%, 99.5%] based on the raw fusion score. Only documents approaching mathematical certainty (p_raw >= 0.999) reach 100.0%."),

        ("Q7: How does DocVerify ensure user data privacy and GDPR compliance?",
         "Unlike cloud-based solutions (e.g. AWS Textract, Google Cloud Vision), DocVerify runs 100% locally on-premise using PyTorch and EasyOCR. Sensitive identity scans, names, and passport numbers never leave the host server or network perimeter."),

        ("Q8: What are the primary operational modes and how do they differ?",
         "DocVerify provides 3 monotonically ordered operational policies: Conservative KYC (threshold = 0.50, high recall 95.3% for zero tolerance to fraud), Balanced Risk (threshold = 0.72, optimal Youden's J statistic with 91.25% accuracy), and Strict Anti-Fraud (threshold = 0.80, high precision 98.0% for low false alarms)."),

        ("Q9: What happens if an image is corrupted, blurry, or not an identity card?",
         "The pipeline contains multi-layered guards: Safe Pillow RGB conversion handles CMYK color spaces; dimension aspect-ratio analysis routes unusual images to an 'UNKNOWN' template fallback; and OCR timeouts degrade gracefully without crashing the visual fusion pipeline.")
    ]

    for q, a in viva_qa:
        story.append(Paragraph(q, qna_q))
        story.append(Paragraph(a, qna_a))

    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cbd5e1"), spaceBefore=4, spaceAfter=8))
    story.append(Paragraph("<i>DocVerify Master Project Knowledge & Viva Handbook • Published for University Examination & Review Defense</i>", ParagraphStyle("Signoff", fontName="Helvetica-Oblique", fontSize=7.5, textColor=colors.HexColor("#64748b"), alignment=1)))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Master Handbook PDF generated successfully at: {OUTPUT_PDF_PATH}")
    return OUTPUT_PDF_PATH


if __name__ == "__main__":
    build_master_handbook()
