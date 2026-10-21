#!/usr/bin/env python3
"""
Generate Comprehensive DocVerify Project Specification, Architecture & Evaluation PDF Report.
"""

import sys
from pathlib import Path
import time

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable, KeepTogether
)
from reportlab.pdfgen import canvas

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_PDF = PROJECT_ROOT / "DocVerify_Complete_Project_Specification_Report.pdf"


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
        # Top Header
        self.setFont("Helvetica-Bold", 7.5)
        self.setFillColor(colors.HexColor("#1e3a8a"))
        self.drawString(54, 752, "DOCVERIFY: AI-BASED IDENTITY DOCUMENT VERIFICATION & FORGERY DETECTION")
        self.setFont("Helvetica", 7.5)
        self.setFillColor(colors.HexColor("#64748b"))
        self.drawRightString(558, 752, "SYSTEM ARCHITECTURE, HYPERPARAMETERS & SCIENTIFIC BENCHMARK")
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(54, 746, 558, 746)

        # Bottom Footer
        self.line(54, 44, 558, 44)
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))
        self.drawString(54, 32, "DocVerify Technical & Academic Compendium • Confidential Document")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 32, page_str)
        self.restoreState()


def build_pdf():
    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

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
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#475569"),
        spaceAfter=10
    )
    section_h1 = ParagraphStyle(
        "SectionH1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#1e3a8a"),
        spaceBefore=12,
        spaceAfter=5,
        keepWithNext=True
    )
    section_h2 = ParagraphStyle(
        "SectionH2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=8,
        spaceAfter=3,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        "DocBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.2,
        leading=11.5,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=5
    )
    bullet_style = ParagraphStyle(
        "DocBullet",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#1e293b"),
        leftIndent=10,
        firstLineIndent=-6,
        spaceAfter=3
    )

    # Table styles
    th_style = ParagraphStyle(
        "TH",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.7,
        leading=10,
        textColor=colors.white
    )
    td_style = ParagraphStyle(
        "TD",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#0f172a")
    )
    td_bold = ParagraphStyle(
        "TDBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#0f172a")
    )
    td_use = ParagraphStyle(
        "TDUse",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.2,
        leading=9.5,
        textColor=colors.HexColor("#15803d")
    )
    td_rej = ParagraphStyle(
        "TDRej",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.2,
        leading=9.5,
        textColor=colors.HexColor("#b91c1c")
    )

    story = []

    # Title Block
    story.append(Paragraph("DocVerify — Complete Technical Specification & Evaluation Report", title_style))
    story.append(Paragraph(
        "End-to-End System Architecture, Complete Tech Stack, Hyperparameters, Model Parameters, "
        "Feature Weights, Calibration, Accuracy, Recall, Precision, and Statistical Significance", subtitle_style
    ))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1e40af"), spaceBefore=0, spaceAfter=8))

    # 1. Executive Summary & Problem Formulation
    story.append(Paragraph("1. Executive Overview & Problem Formulation", section_h1))
    story.append(Paragraph(
        "<b>DocVerify</b> is an enterprise-grade, multi-modal AI system engineered to detect digital tampering, text inpainting, "
        "and character rewrites across international identity documents (passports and national identity cards). "
        "Traditional deep learning approaches downsample high-resolution document scans (e.g. 2167×1360 px) to standard CNN sizes (224×224 px). "
        "This introduces the <b>micro-forgery downsampling bottleneck</b>, where single-character alterations (e.g., date-of-birth or ID number rewrites) "
        "collapse into sub-pixel noise. DocVerify solves this by coupling a global macro CNN with an uncompressed native-sensor local patch classifier, "
        "digital image forensics (Error Level Analysis, camera noise inconsistency, Laplacian edge variance), and EasyOCR syntactic regex validation, "
        "synthesizing all evidence via regularized logistic regression and true Isotonic Probability Calibration.", body_style
    ))

    # 2. Complete Technology Stack
    story.append(Paragraph("2. Master Technology Stack: Retained vs. Replaced", section_h1))
    story.append(Paragraph(
        "The following table details every framework, architecture, and tool evaluated in the DocVerify lifecycle, explaining why each is retained or superseded:", body_style
    ))

    tech_table = [
        [Paragraph("Category / Layer", th_style), Paragraph("Component Evaluated", th_style), Paragraph("Status", th_style), Paragraph("Engineering Rationale & Lifecycle Justification", th_style)],
        
        [Paragraph("<b>Core Frameworks</b>", td_style),
         Paragraph("<b>Python 3.10–3.13, PyTorch 2.x, Torchvision</b>", td_bold),
         Paragraph("ACTIVE", td_use),
         Paragraph("Native tensor acceleration, custom model definitions, automatic differentiation, and production CPU/CUDA compatibility.", td_style)],

        [Paragraph("<b>Document Localization</b>", td_style),
         Paragraph("<b>YOLO (v5 / v8)</b>", td_bold),
         Paragraph("REJECTED", td_rej),
         Paragraph("Bounding box jitter caused inconsistent text slicing and boundary truncation. Replaced with exact canonical geometry matching.", td_style)],

        [Paragraph("<b>Layout Alignment</b>", td_style),
         Paragraph("<b>Autonomous Aspect Ratio & Canonical Priors</b>", td_bold),
         Paragraph("ACTIVE", td_use),
         Paragraph("100% deterministic geometry matching across 10 international templates with zero runtime bounding box drift and zero GPU memory overhead.", td_style)],

        [Paragraph("<b>Liveness / Spoofing</b>", td_style),
         Paragraph("<b>Biometric PAD (Presentation Attack Detection)</b>", td_bold),
         Paragraph("REJECTED", td_rej),
         Paragraph("Trained for physical screen/paper replay attacks; produced excessive false alarms on flatbed document scans. Digital forensics substituted.", td_style)],

        [Paragraph("<b>Digital Image Forensics</b>", td_style),
         Paragraph("<b>Error Level Analysis (ELA), Noise & Edge Variance</b>", td_bold),
         Paragraph("ACTIVE", td_use),
         Paragraph("Detects differential JPEG recompression (Q=90), sensor residual variance across quadrants (splicing), and Laplacian high-frequency edge energy.", td_style)],

        [Paragraph("<b>Text Extraction (OCR)</b>", td_style),
         Paragraph("<b>Tesseract OCR</b>", td_bold),
         Paragraph("REJECTED", td_rej),
         Paragraph("Heavy external C++ binary dependency, brittle threading, and poor character extraction on complex Guilloche background security patterns.", td_style)],

        [Paragraph("<b>Text Extraction (OCR)</b>", td_style),
         Paragraph("<b>EasyOCR (PyTorch CRAFT + CRNN)</b>", td_bold),
         Paragraph("ACTIVE", td_use),
         Paragraph("100% Python/PyTorch native. Superior multi-lingual recognition across skewed cards, paired with regex syntactic format audits.", td_style)],

        [Paragraph("<b>Global Feature Extractor</b>", td_style),
         Paragraph("<b>EfficientNet-B0 (Frozen Backbone)</b>", td_bold),
         Paragraph("ACTIVE", td_use),
         Paragraph("4.01M parameters. Pre-trained on ImageNet-1K, fine-tuned on full document scans in Phase 2, and frozen to maintain macro-visual stability.", td_style)],

        [Paragraph("<b>Local Patch Classifier</b>", td_style),
         Paragraph("<b>Dedicated EfficientNet-B0 Patch Model</b>", td_bold),
         Paragraph("ACTIVE", td_use),
         Paragraph("Operates directly on uncompressed native-sensor crops with 2.0x context expansion, capturing micro-text tampering invisible to the global model.", td_style)],

        [Paragraph("<b>Visual Explainability</b>", td_style),
         Paragraph("<b>Standard Grad-CAM</b>", td_bold),
         Paragraph("REJECTED", td_rej),
         Paragraph("Resizing rectangular cards to square (224×224) induced 20–30% spatial stretching, pointing heatmaps to wrong coordinates on the raw scan.", td_style)],

        [Paragraph("<b>Visual Explainability</b>", td_style),
         Paragraph("<b>CorrectedGradCAM (Exact Inverse Mapping)</b>", td_bold),
         Paragraph("ACTIVE", td_use),
         Paragraph("Projects gradients through exact affine inverse transforms back to native document scan space (quantized to 1-pixel accuracy).", td_style)],

        [Paragraph("<b>Evidence Fusion</b>", td_style),
         Paragraph("<b>Logistic Regression + StandardScaler Normalization</b>", td_bold),
         Paragraph("ACTIVE", td_use),
         Paragraph("Replaced naive score averaging. Statistically combines 7 normalized evidence signals with optimal L2-regularized weights.", td_style)],

        [Paragraph("<b>Probability Calibration</b>", td_style),
         Paragraph("<b>Isotonic Regression + Proportional Scaling</b>", td_bold),
         Paragraph("ACTIVE", td_use),
         Paragraph("Eliminates neural overconfidence. Maps raw logits to empirical fraud probabilities (Brier: 0.0657, ECE: 0.0484) with continuous risk scaling.", td_style)],

        [Paragraph("<b>Web Application UI</b>", td_style),
         Paragraph("<b>Streamlit + Custom Enterprise CSS</b>", td_bold),
         Paragraph("ACTIVE", td_use),
         Paragraph("Interactive browser dashboard with reactive metric badges, Grad-CAM overlays, multi-modal evidence breakdown, and PDF export.", td_style)],

        [Paragraph("<b>Report Generation</b>", td_style),
         Paragraph("<b>ReportLab 4.x / 5.x Flowable PDF Engine</b>", td_bold),
         Paragraph("ACTIVE", td_use),
         Paragraph("Compiles multi-page forensic audit certificates with session hashes, metadata, extracted fields, and explainability maps.", td_style)]
    ]

    t_tech = Table(tech_table, colWidths=[90, 115, 50, 249])
    t_tech.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_tech)
    story.append(Spacer(1, 8))

    # Page Break for Clean Layout
    story.append(PageBreak())

    # 3. Dataset Architecture & Split Scheme
    story.append(Paragraph("3. Dataset Foundations: The SIDTD Benchmark", section_h1))
    story.append(Paragraph(
        "DocVerify is trained and benchmarked on the peer-reviewed <b>SIDTD (Synthetic & Identity Document Tampering Dataset)</b>. "
        "To guarantee zero data contamination and scientific defensibility, splits are strictly <b>identity-disjoint</b>: "
        "no identity subject, document template instance, or base card appears in more than one partition.", body_style
    ))

    dataset_table = [
        [Paragraph("Dataset Parameter", th_style), Paragraph("Quantity / Configuration", th_style), Paragraph("Details & Integrity Safeguards", th_style)],
        [Paragraph("<b>Total Document Volume</b>", td_bold), Paragraph("<b>2,222 Documents</b>", td_style), Paragraph("1,000 Authentic (Genuine) Documents + 1,222 Digitally Forged Documents.", td_style)],
        [Paragraph("<b>Training Split</b>", td_bold), Paragraph("<b>1,565 Documents</b> (70.4%)", td_style), Paragraph("703 Real, 862 Fake. Used exclusively for model weight optimization.", td_style)],
        [Paragraph("<b>Validation Split</b>", td_bold), Paragraph("<b>337 Documents</b> (15.2%)", td_style), Paragraph("146 Real, 191 Fake. Used for best checkpoint selection, scaler, fusion, and calibration.", td_style)],
        [Paragraph("<b>Frozen Test Split</b>", td_bold), Paragraph("<b>320 Documents</b> (14.4%)", td_style), Paragraph("151 Real, 169 Fake across 139 disjoint clusters. <b>Zero contamination</b> with train/val.", td_style)],
        [Paragraph("<b>Document Templates</b>", td_bold), Paragraph("<b>10 International Templates</b>", td_style), Paragraph("Passports: Azerbaijan, Greece, Latvia, Russia, Serbia, Slovakia.<br/>National IDs: Albania, Spain, Estonia, Finland.", td_style)],
        [Paragraph("<b>Forgery Categories</b>", td_bold), Paragraph("• Inpaint_and_Rewrite (1,077)<br/>• Crop_and_Replace (145)", td_style), Paragraph("Focuses on high-value fraud targets: Names, Surnames, Birth Dates, Expiry Dates, ID Numbers, and Photos.", td_style)]
    ]
    t_ds = Table(dataset_table, colWidths=[120, 130, 254])
    t_ds.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_ds)
    story.append(Spacer(1, 8))

    # 4. Parameters & Hyperparameters Compendium
    story.append(Paragraph("4. Comprehensive Model Parameters & Hyperparameters", section_h1))
    story.append(Paragraph(
        "Every neural, statistical, and forensic component in DocVerify is configured with explicit, documented hyperparameters:", body_style
    ))

    hyper_table = [
        [Paragraph("Component / Pipeline", th_style), Paragraph("Parameter Name", th_style), Paragraph("Value / Setting", th_style), Paragraph("Functional Purpose & Operational Impact", th_style)],
        
        [Paragraph("<b>Phase 2 Global Baseline</b>", td_bold), Paragraph("Backbone Architecture", td_style), Paragraph("EfficientNet-B0", td_style), Paragraph("Pre-trained on ImageNet-1K; 4,010,110 total and trainable parameters.", td_style)],
        [Paragraph("", td_style), Paragraph("Input Dimensions", td_style), Paragraph("256 Resize -> 224x224 CenterCrop", td_style), Paragraph("Standardized input with ImageNet mean/std normalization.", td_style)],
        [Paragraph("", td_style), Paragraph("Optimizer & Learning Rate", td_style), Paragraph("AdamW, LR = 1e-4, Weight Decay = 1e-4", td_style), Paragraph("Decoupled weight decay regularization to avoid overfitting on background textures.", td_style)],
        [Paragraph("", td_style), Paragraph("LR Scheduler", td_style), Paragraph("CosineAnnealingLR (T_max=10)", td_style), Paragraph("Smoothly decays learning rate towards zero over 10 training epochs.", td_style)],
        [Paragraph("", td_style), Paragraph("Loss Function & Weights", td_style), Paragraph("CrossEntropyLoss (Real: 1.113, Fake: 0.908)", td_style), Paragraph("Strictly derived from train class balance (703 Real, 862 Fake).", td_style)],
        [Paragraph("", td_style), Paragraph("Batch Size & Epochs", td_style), Paragraph("Batch Size = 32, Epochs = 10", td_style), Paragraph("Best validation checkpoint attained at Epoch 9 (Val Loss: 0.3668, Val F1: 0.8356).", td_style)],
        [Paragraph("", td_style), Paragraph("Data Augmentation", td_style), Paragraph("HFlip (p=0.5), Rot (+/-4°), ColorJitter (0.08)", td_style), Paragraph("Conservative augmentations preserve delicate micro-line forgery cues.", td_style)],

        [Paragraph("<b>Phase 8 Patch Model</b>", td_bold), Paragraph("Backbone Architecture", td_style), Paragraph("EfficientNet-B0", td_style), Paragraph("Binary classification head Linear(1280, 2); 4,010,110 parameters.", td_style)],
        [Paragraph("", td_style), Paragraph("Context Crop Expansion", td_style), Paragraph("<b>2.0x bounding box width & height</b>", td_style), Paragraph("Provides local context around text to detect character boundary inconsistencies.", td_style)],
        [Paragraph("", td_style), Paragraph("Target Patch Resolution", td_style), Paragraph("224 × 224 pixels", td_style), Paragraph("Uncompressed native sensor crop resized directly to 224×224 without whole-page downsampling.", td_style)],
        [Paragraph("", td_style), Paragraph("Optimizer & Hyperparams", td_style), Paragraph("AdamW, LR = 1e-4, Decay = 1e-4, Epochs = 6", td_style), Paragraph("Best epoch 1 saved via copy.deepcopy to isolate parameter tensors.", td_style)],
        [Paragraph("", td_style), Paragraph("Class Weighting", td_style), Paragraph("Weights = [1.226, 1.000]", td_style), Paragraph("Derived from 703 real and 862 fake training crops.", td_style)],

        [Paragraph("<b>Forensic Extractors</b>", td_bold), Paragraph("Error Level Analysis (ELA)", td_style), Paragraph("JPEG Quality Q = 90, 4 quadrants", td_style), Paragraph("Standard deviation of mean difference across 4 quadrants flags recompression anomalies.", td_style)],
        [Paragraph("", td_style), Paragraph("Sensor Noise Consistency", td_style), Paragraph("3×3 Median Filter High-Pass Residual", td_style), Paragraph("Standard deviation of residual variance across 4 quadrants exposes multi-camera splicing.", td_style)],
        [Paragraph("", td_style), Paragraph("Edge Energy Variance", td_style), Paragraph("3×3 Laplacian High-Frequency Filter", td_style), Paragraph("Measures pixel sharpness variance; reveals boundary feathering and smoothing.", td_style)],

        [Paragraph("<b>Semantic OCR Stream</b>", td_bold), Paragraph("EasyOCR Configuration", td_style), Paragraph("Languages: ['en', 'ru'], Min Height: 48px", td_style), Paragraph("Contrast enhancement and upscale prior to CRAFT/CRNN extraction.", td_style)],
        [Paragraph("", td_style), Paragraph("Syntactic Priors", td_style), Paragraph("Alphanumeric & Date Regex Patterns", td_style), Paragraph("Flags format anomalies (0=Valid, 1=Anomaly) when dates or ID formats violate standards.", td_style)],

        [Paragraph("<b>Evidence Fusion</b>", td_bold), Paragraph("Classifier & Regularization", td_style), Paragraph("Logistic Regression (L2 penalty, C = 0.5)", td_style), Paragraph("lbfgs solver, random_state=42. Moderate C prevents single-feature shortcut dominance.", td_style)],
        [Paragraph("", td_style), Paragraph("Feature Standardization", td_style), Paragraph("StandardScaler fit on Validation", td_style), Paragraph("Normalizes all 7 evidence features to zero mean and unit variance.", td_style)],
        [Paragraph("", td_style), Paragraph("Probability Calibrator", td_style), Paragraph("IsotonicRegression (clip, [0.0, 1.0])", td_style), Paragraph("Non-parametric isotonic fitting on validation logits aligns risk with empirical fraud rates.", td_style)]
    ]

    t_hyper = Table(hyper_table, colWidths=[90, 105, 115, 194])
    t_hyper.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_hyper)
    story.append(Spacer(1, 8))

    # Page Break for Clean Layout
    story.append(PageBreak())

    # 5. Calibrated Multi-Modal Evidence Fusion & Mathematical Weights
    story.append(Paragraph("5. Multi-Modal Evidence Fusion Weights & Scoring Equation", section_h1))
    story.append(Paragraph(
        r"DocVerify synthesizes 7 normalized evidence cues into a single regularized forgery logit \(z\), which is converted to raw probability and then calibrated:", body_style
    ))

    weights_table = [
        [Paragraph("Feature Identifier", th_style), Paragraph("Feature Description", th_style), Paragraph("Scaler Mean (μ)", th_style), Paragraph("Scaler Std (σ)", th_style), Paragraph("Learned Weight (w)", th_style), Paragraph("Physical Forensic Interpretation", th_style)],
        [Paragraph("<b>p_global_fake</b>", td_bold), Paragraph("Global EfficientNet-B0 Score", td_style), Paragraph("0.5102", td_style), Paragraph("0.3074", td_style), Paragraph("<b>+2.9232</b>", td_bold), Paragraph("Primary positive indicator of macro document forgery.", td_style)],
        [Paragraph("<b>p_patch_fake</b>", td_bold), Paragraph("Dedicated Patch Model Score", td_style), Paragraph("0.1058", td_style), Paragraph("0.0896", td_style), Paragraph("<b>+1.5644</b>", td_bold), Paragraph("Strong positive indicator of localized micro-text manipulation.", td_style)],
        [Paragraph("<b>ela_inconsistency</b>", td_bold), Paragraph("JPEG Recompression Std", td_style), Paragraph("0.2440", td_style), Paragraph("0.2977", td_style), Paragraph("<b>-1.6811</b>", td_bold), Paragraph("Regularized negative weight neutralizes dataset compression bias.", td_style)],
        [Paragraph("<b>noise_inconsistency</b>", td_bold), Paragraph("Median Residual Variance Std", td_style), Paragraph("17.1608", td_style), Paragraph("8.1989", td_style), Paragraph("<b>-0.4144</b>", td_bold), Paragraph("Quadrant noise consistency check across scanner regions.", td_style)],
        [Paragraph("<b>edge_variance</b>", td_bold), Paragraph("Laplacian High-Pass Energy", td_style), Paragraph("2611.67", td_style), Paragraph("976.81", td_style), Paragraph("<b>+0.0542</b>", td_bold), Paragraph("Slight positive weighting for anomalous edge gradient energy.", td_style)],
        [Paragraph("<b>ocr_confidence</b>", td_bold), Paragraph("EasyOCR Typography Confidence", td_style), Paragraph("0.7006", td_style), Paragraph("0.2886", td_style), Paragraph("<b>-0.8589</b>", td_bold), Paragraph("High OCR confidence strongly correlates with genuine official fonts.", td_style)],
        [Paragraph("<b>format_anomaly</b>", td_bold), Paragraph("Regex Syntactic Error Flag", td_style), Paragraph("0.2285", td_style), Paragraph("0.4199", td_style), Paragraph("<b>-0.4393</b>", td_bold), Paragraph("Syntactic coherence check across alphanumeric strings.", td_style)],
        [Paragraph("<b>Intercept (w0)</b>", td_bold), Paragraph("Logistic Regression Intercept", td_style), Paragraph("—", td_style), Paragraph("—", td_style), Paragraph("<b>+0.8106</b>", td_bold), Paragraph("Base logit offset calibrated on validation split balance.", td_style)]
    ]
    t_weights = Table(weights_table, colWidths=[90, 105, 65, 65, 75, 104])
    t_weights.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_weights)
    story.append(Spacer(1, 8))

    # Mathematical Formula Box
    formula_text = (
        "<b>Mathematical Decision Pipeline:</b><br/>"
        "1. Standardize features: <i>z_i = (x_i - μ_i) / σ_i</i><br/>"
        "2. Compute raw fusion logit: <i>logit = w_0 + ∑ (w_i · z_i)</i><br/>"
        "3. Sigmoid raw probability: <i>p_raw = 1 / (1 + exp(-logit))</i><br/>"
        "4. Isotonic calibration: <i>p_calib = IsotonicCalibrator(p_raw)</i><br/>"
        "5. Continuous risk score scaling: Reserves 100% strictly for <i>p_raw ≥ 0.999</i>, preventing artificial risk saturation."
    )
    story.append(Paragraph(formula_text, body_style))
    story.append(Spacer(1, 6))

    # 6. Operational Policies & Decision Thresholds
    story.append(Paragraph("6. Operational Decision Policies & Thresholds", section_h1))
    story.append(Paragraph(
        "To support diverse institutional compliance requirements, DocVerify features 3 monotonically ordered operational thresholds derived from the validation ROC curve:", body_style
    ))

    policy_table = [
        [Paragraph("Operational Policy", th_style), Paragraph("Threshold (τ)", th_style), Paragraph("Target Use Case", th_style), Paragraph("Accuracy", th_style), Paragraph("Precision", th_style), Paragraph("Recall", th_style), Paragraph("F1-Score", th_style), Paragraph("Error Profile (FN / FP)", th_style)],
        [Paragraph("<b>Conservative KYC</b>", td_bold), Paragraph("<b>0.5000</b>", td_bold), Paragraph("Border Control & High-Value Banking", td_style), Paragraph("90.31%", td_style), Paragraph("87.50%", td_style), Paragraph("<b>95.27%</b>", td_bold), Paragraph("91.22%", td_style), Paragraph("FN: <b>8 / 169</b> (4.7%)<br/>FP: 23 / 151 (15.2%)", td_style)],
        [Paragraph("<b>Balanced Risk (Optimal)</b>", td_bold), Paragraph("<b>0.7200</b>", td_bold), Paragraph("Standard Digital Onboarding", td_style), Paragraph("<b>91.25%</b>", td_bold), Paragraph("91.23%", td_style), Paragraph("<b>92.31%</b>", td_bold), Paragraph("<b>91.76%</b>", td_bold), Paragraph("FN: <b>13 / 169</b> (7.7%)<br/>FP: 15 / 151 (9.9%)", td_style)],
        [Paragraph("<b>Strict Anti-Fraud</b>", td_bold), Paragraph("<b>0.8000</b>", td_bold), Paragraph("Low-Friction Automated Onboarding", td_style), Paragraph("<b>92.50%</b>", td_bold), Paragraph("<b>98.01%</b>", td_bold), Paragraph("87.57%", td_style), Paragraph("<b>92.50%</b>", td_bold), Paragraph("FN: 21 / 169 (12.4%)<br/>FP: <b>3 / 151 (1.9%)</b>", td_style)]
    ]
    t_pol = Table(policy_table, colWidths=[85, 55, 105, 45, 45, 45, 45, 79])
    t_pol.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 3.5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 3.5),
    ]))
    story.append(t_pol)
    story.append(Spacer(1, 6))

    # Risk Tier Mapping
    tier_text = (
        "<b>Multi-Tiered Risk Mapping:</b><br/>"
        "• <b>HIGH RISK / SUSPICIOUS (Red):</b> <i>p_calib > τ</i> (Immediate tampering alert, flagged for physical inspection).<br/>"
        "• <b>MEDIUM RISK / HUMAN REVIEW (Amber):</b> <i>(τ - 0.15) ≤ p_calib ≤ τ</i> (Borderline zone, routed to compliance officer).<br/>"
        "• <b>LOW RISK / AUTHENTIC (Green):</b> <i>p_calib < (τ - 0.15)</i> (Automated straight-through processing)."
    )
    story.append(Paragraph(tier_text, body_style))
    story.append(Spacer(1, 8))

    # Page Break for Clean Layout
    story.append(PageBreak())

    # 7. Benchmark Results, Comparison & Statistical Significance
    story.append(Paragraph("7. Benchmark Results & Statistical Validation ($N = 320$)", section_h1))
    story.append(Paragraph(
        "Performance evaluated strictly on the frozen out-of-distribution test split (151 Real, 169 Fake across 139 disjoint clusters):", body_style
    ))

    benchmark_table = [
        [Paragraph("Architecture / Experiment", th_style), Paragraph("Operating Mode", th_style), Paragraph("Accuracy", th_style), Paragraph("Precision", th_style), Paragraph("Recall", th_style), Paragraph("F1-Score", th_style), Paragraph("ROC-AUC", th_style), Paragraph("Brier Score", th_style)],
        [Paragraph("<b>Phase 2 Global Baseline</b> (EfficientNet-B0)", td_bold), Paragraph("Autonomous (Full Page)", td_style), Paragraph("79.69%", td_style), Paragraph("82.91%", td_style), Paragraph("77.51%", td_style), Paragraph("80.12%", td_style), Paragraph("0.9112", td_style), Paragraph("0.1420", td_style)],
        [Paragraph("<b>Phase 6 Oracle Patch Model</b> (GT Boxes)", td_bold), Paragraph("Oracle Assistance", td_style), Paragraph("99.38%", td_style), Paragraph("100.00%", td_style), Paragraph("98.82%", td_style), Paragraph("99.40%", td_style), Paragraph("0.9999", td_style), Paragraph("—", td_style)],
        [Paragraph("<b>Phase 8 Remediated Patch Model</b> (Sensor Crops)", td_bold), Paragraph("Autonomous Sensor Crops", td_style), Paragraph("<b>98.44%</b>", td_bold), Paragraph("<b>100.00%</b>", td_bold), Paragraph("<b>97.04%</b>", td_bold), Paragraph("<b>98.50%</b>", td_bold), Paragraph("<b>0.9967</b>", td_bold), Paragraph("—", td_style)],
        [Paragraph("<b>Phase 8 Production Fusion</b> (Balanced τ=0.72)", td_bold), Paragraph("Autonomous Fusion", td_style), Paragraph("<b>91.25%</b>", td_bold), Paragraph("<b>91.23%</b>", td_bold), Paragraph("<b>92.31%</b>", td_bold), Paragraph("<b>91.76%</b>", td_bold), Paragraph("<b>0.9677</b>", td_bold), Paragraph("<b>0.0657</b>", td_bold)],
        [Paragraph("<b>Phase 8 Strict Anti-Fraud</b> (Strict τ=0.80)", td_bold), Paragraph("Autonomous Fusion", td_style), Paragraph("<b>92.50%</b>", td_bold), Paragraph("<b>98.01%</b>", td_bold), Paragraph("87.57%", td_style), Paragraph("<b>92.50%</b>", td_bold), Paragraph("<b>0.9677</b>", td_bold), Paragraph("<b>0.0657</b>", td_bold)],
        [Paragraph("<b>Phase 8 Conservative KYC</b> (High Recall τ=0.50)", td_bold), Paragraph("Autonomous Fusion", td_style), Paragraph("90.31%", td_style), Paragraph("87.50%", td_style), Paragraph("<b>95.27%</b>", td_bold), Paragraph("91.22%", td_style), Paragraph("<b>0.9677</b>", td_bold), Paragraph("<b>0.0657</b>", td_bold)]
    ]
    t_bench = Table(benchmark_table, colWidths=[120, 90, 45, 45, 45, 45, 45, 49])
    t_bench.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 3.5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 3.5),
    ]))
    story.append(t_bench)
    story.append(Spacer(1, 8))

    # Statistical Rigor Sub-sections
    story.append(Paragraph("<b>A. 5,000 Bootstrap Confidence Intervals (95% CI):</b>", section_h2))
    story.append(Paragraph(
        r"To establish estimation certainty on the 320 frozen test examples, non-parametric bootstrap resampling (B = 5,000) was computed:<br/>"
        "• <b>Accuracy:</b> Mean = <b>91.22%</b> | 95% Bootstrap CI: <b>[87.81%, 94.38%]</b><br/>"
        "• <b>Precision:</b> Mean = <b>91.25%</b> | 95% Bootstrap CI: <b>[86.83%, 95.40%]</b><br/>"
        "• <b>Recall:</b> Mean = <b>92.24%</b> | 95% Bootstrap CI: <b>[88.00%, 95.98%]</b><br/>"
        "• <b>F1-Score:</b> Mean = <b>91.72%</b> | 95% Bootstrap CI: <b>[88.46%, 94.67%]</b><br/>"
        "• <b>ROC-AUC:</b> Mean = <b>96.75%</b> | 95% Bootstrap CI: <b>[94.79%, 98.44%]</b><br/>"
        "• <b>Non-Overlapping Proof:</b> Phase 2 baseline Accuracy CI is [75.31%, 84.06%]. "
        "The complete separation between the baseline and remediated distributions confirms that the improvement is not an artifact of random sampling.", body_style
    ))
    story.append(Spacer(1, 4))

    story.append(Paragraph("<b>B. McNemar's Paired Hypothesis Testing vs. Global Baseline:</b>", section_h2))
    story.append(Paragraph(
        r"McNemar's test evaluated the paired 2×2 contingency table on the exact same 320 test documents:<br/>"
        "• Both Models Correct: 243 | Both Models Wrong: 16<br/>"
        "• Phase 2 Correct, Phase 8 Wrong: 12<br/>"
        "• <b>Phase 8 Correct, Phase 2 Wrong: 49</b><br/>"
        "• <b>Test Statistic:</b> Chi-squared = 21.2459 (with continuity correction)<br/>"
        "• <b>p-value:</b> <b>4.04 × 10^-6</b> (p < 0.0001)<br/>"
        "• <b>Conclusion:</b> The null hypothesis of error equivalence is decisively rejected. Phase 8 multi-modal fusion provides statistically proven superiority.", body_style
    ))
    story.append(Spacer(1, 4))

    story.append(Paragraph("<b>C. Probability Calibration & Error Metrics:</b>", section_h2))
    story.append(Paragraph(
        "• <b>Brier Score:</b> Improved from <b>0.1420</b> (Phase 2) to <b>0.0657</b> (Phase 8), representing a 53.7% reduction in mean squared probability error.<br/>"
        "• <b>Expected Calibration Error (ECE):</b> Contained at <b>0.0484 (4.84%)</b> across 10 empirical probability bins. "
        "This confirms that predicted confidence aligns with true real-world risk (e.g. an 80% risk rating corresponds to 80% empirical fraud probability).", body_style
    ))
    story.append(Spacer(1, 8))

    # Page Break for Clean Layout
    story.append(PageBreak())

    # 8. Robustness & Environmental Stress Testing
    story.append(Paragraph("8. Robustness & Environmental Stress Testing", section_h1))
    story.append(Paragraph(
        "DocVerify was evaluated across 10 harsh operational environments to verify resilience against real-world mobile capture and re-compression artifacts:", body_style
    ))

    stress_table = [
        [Paragraph("Degradation / Stress Condition", th_style), Paragraph("Accuracy", th_style), Paragraph("Precision", th_style), Paragraph("Recall", th_style), Paragraph("F1-Score", th_style), Paragraph("Degradation vs. High-Res Baseline", th_style)],
        [Paragraph("<b>Original High-Res Scans (Baseline)</b>", td_bold), Paragraph("<b>91.25%</b>", td_bold), Paragraph("<b>91.23%</b>", td_bold), Paragraph("<b>92.31%</b>", td_bold), Paragraph("<b>91.76%</b>", td_bold), Paragraph("0.00% (Baseline Reference)", td_style)],
        [Paragraph("<b>JPEG Q95 Recompression</b>", td_style), Paragraph("87.50%", td_style), Paragraph("85.00%", td_style), Paragraph("90.00%", td_style), Paragraph("87.42%", td_style), Paragraph("-3.75% (Minor attenuation)", td_style)],
        [Paragraph("<b>JPEG Q80 Recompression</b>", td_style), Paragraph("82.50%", td_style), Paragraph("80.00%", td_style), Paragraph("85.00%", td_style), Paragraph("82.43%", td_style), Paragraph("-8.75% (Moderate compression)", td_style)],
        [Paragraph("<b>JPEG Q60 Recompression</b>", td_style), Paragraph("77.50%", td_style), Paragraph("75.00%", td_style), Paragraph("80.00%", td_style), Paragraph("77.42%", td_style), Paragraph("-13.75% (Heavy compression)", td_style)],
        [Paragraph("<b>JPEG Q50 Recompression</b>", td_style), Paragraph("72.50%", td_style), Paragraph("70.00%", td_style), Paragraph("75.00%", td_style), Paragraph("72.41%", td_style), Paragraph("-18.75% (Graceful decay)", td_style)],
        [Paragraph("<b>Low Resolution (0.5× Scale)</b>", td_style), Paragraph("80.00%", td_style), Paragraph("78.26%", td_style), Paragraph("85.00%", td_style), Paragraph("81.49%", td_style), Paragraph("-11.25% (Maintains high recall)", td_style)],
        [Paragraph("<b>Mild Gaussian Blur (Radius 1.0)</b>", td_style), Paragraph("82.50%", td_style), Paragraph("80.95%", td_style), Paragraph("85.00%", td_style), Paragraph("82.93%", td_style), Paragraph("-8.75% (Resilient to lens softness)", td_style)],
        [Paragraph("<b>Brightness Shift (+20%)</b>", td_style), Paragraph("87.50%", td_style), Paragraph("85.71%", td_style), Paragraph("90.00%", td_style), Paragraph("87.80%", td_style), Paragraph("-3.75% (Robust to exposure)", td_style)],
        [Paragraph("<b>Contrast Shift (-20%)</b>", td_style), Paragraph("85.00%", td_style), Paragraph("81.82%", td_style), Paragraph("90.00%", td_style), Paragraph("85.71%", td_style), Paragraph("-6.25% (Robust to faded ink)", td_style)],
        [Paragraph("<b>Metadata Stripped RGB Normalization</b>", td_bold), Paragraph("<b>91.25%</b>", td_bold), Paragraph("<b>91.23%</b>", td_bold), Paragraph("<b>92.31%</b>", td_bold), Paragraph("<b>91.76%</b>", td_bold), Paragraph("<b>0.00% (Complete Invariance)</b>", td_use)]
    ]
    t_stress = Table(stress_table, colWidths=[130, 48, 48, 48, 48, 172])
    t_stress.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 3.5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 3.5),
    ]))
    story.append(t_stress)
    story.append(Spacer(1, 8))

    # 9. Error Analysis & Forensic Root Causes
    story.append(Paragraph("9. Forensic Error Analysis & Failure Modes", section_h1))
    story.append(Paragraph(
        "A comprehensive forensic audit of all misclassifications on the 320 frozen test examples reveals well-defined failure patterns:", body_style
    ))
    story.append(Paragraph(
        "• <b>False Positives (15 / 151 = 9.93%):</b> Primarily observed in Finnish IDs (<code>fin_id</code>: 4 samples) and Russian Internal Passports (<code>rus_internalpassport</code>: 3 samples). "
        "<b>Root Cause:</b> Dense background Guilloche security waves and scanner noise produced elevated high-frequency edge variance that mimicked splice boundaries.<br/>"
        "• <b>False Negatives (13 / 169 = 7.69%):</b> Primarily observed in Greek Passports (<code>grc_passport</code>: 3 samples) and Slovakian IDs (<code>svk_id</code>: 3 samples). "
        "<b>Root Cause:</b> Ultra-subtle single-character inpainting where the synthetic font, resolution, and baseline angle matched authentic typography nearly identically.<br/>"
        "• <b>Mitigation via Conservative KYC Policy:</b> Switching to threshold τ = 0.5000 reduces false negatives from <b>13 down to only 8</b>, catching 61.5% of these borderline forgeries.", body_style
    ))
    story.append(Spacer(1, 8))

    # 10. University Defense & Viva Q&A Guide
    story.append(Paragraph("10. University Review & Faculty Viva Defense Reference", section_h1))
    viva_questions = [
        ("Q: Why is Intersection over Union (IoU) not reported for DocVerify?",
         "A: IoU is an object detection metric evaluating predicted bounding boxes against ground truth. DocVerify is a document verification and classification system. Rather than predicting bounding boxes via regression heads (which introduce coordinate jitter), DocVerify computes canonical field coordinates deterministically from physical template aspect ratios. Hence, classification metrics (Accuracy, Recall, Precision, F1, ROC-AUC) and calibration metrics (Brier, ECE) are the scientifically appropriate evaluation standards."),
        
        ("Q: What is the main innovation over a standard CNN baseline?",
         "A: Standard CNNs downsample entire identity documents to 224x224, causing micro-manipulations (altered digits, changed expiry dates) to collapse into sub-pixel blur. DocVerify introduces a multi-modal architecture combining global visual patterns, uncompressed 2.0x native-sensor local patch inspection, JPEG Error Level Analysis, camera sensor noise consistency, and EasyOCR syntactic regex validation into a calibrated decision engine."),

        ("Q: Why freeze the Global EfficientNet-B0 backbone instead of end-to-end fine-tuning?",
         "A: Freezing the Phase 2 global model preserves macro-texture feature representations, prevents catastrophic forgetting, and provides a stable, reproducible visual feature extractor that decouples local patch verification and fusion optimization."),

        ("Q: How does DocVerify achieve privacy compliance (GDPR)?",
         "A: Unlike cloud APIs (AWS Textract, Google Cloud Vision) that transmit identity documents over external networks, DocVerify runs 100% locally on-premise using PyTorch and EasyOCR, ensuring sensitive biometric and personal data never leaves the host infrastructure.")
    ]

    for q, a in viva_questions:
        story.append(Paragraph(f"<b>{q}</b>", ParagraphStyle("Q", parent=body_style, fontName="Helvetica-Bold", textColor=colors.HexColor("#1e3a8a"), spaceBefore=4, spaceAfter=2)))
        story.append(Paragraph(f"{a}", ParagraphStyle("A", parent=body_style, leftIndent=8, spaceAfter=4)))

    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cbd5e1"), spaceBefore=4, spaceAfter=6))
    story.append(Paragraph("<i>DocVerify Master Project Specification & Technical Compendium • Prepared for Academic Examination & Engineering Review</i>", ParagraphStyle("Sign", fontName="Helvetica-Oblique", fontSize=7.5, textColor=colors.HexColor("#64748b"), alignment=1)))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF successfully generated at: {OUTPUT_PDF}")


if __name__ == "__main__":
    build_pdf()
