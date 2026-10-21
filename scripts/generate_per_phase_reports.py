#!/usr/bin/env python3
"""
DocVerify - Per-Phase Comprehensive Technical Reports Generator
Compiles 9 structured, professional PDF documents (one for each project phase, Phase 1 to Phase 9)
into docs/phase_reports/, containing detailed tables, architectural diagrams, metrics,
and discoveries from that phase.
"""

import sys
from pathlib import Path
import time

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable, Image as RLImage, KeepTogether
)
from reportlab.pdfgen import canvas

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
PHASE_REPORTS_DIR = DOCS_DIR / "phase_reports"
DIAGRAMS_DIR = PROJECT_ROOT / "outputs" / "diagrams"


class NumberedPhaseCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedPhaseCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []
        self.phase_title = kwargs.pop("phase_title", "DocVerify Technical Phase Report")

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_decorations(num_pages)
            super(NumberedPhaseCanvas, self).showPage()
        super(NumberedPhaseCanvas, self).save()

    def draw_decorations(self, page_count):
        self.saveState()
        # Top Header
        self.setFont("Helvetica-Bold", 7.5)
        self.setFillColor(colors.HexColor("#1e3a8a"))
        self.drawString(54, 750, "DOCVERIFY: AI-BASED IDENTITY DOCUMENT VERIFICATION")
        self.setFont("Helvetica", 7.5)
        self.setFillColor(colors.HexColor("#64748b"))
        self.drawRightString(558, 750, "TECHNICAL PHASE REPORT")
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(54, 744, 558, 744)

        # Bottom Footer
        self.line(54, 46, 558, 46)
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))
        self.drawString(54, 34, "DocVerify Engineering Archive • Confidential & Academic Use")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 34, page_str)
        self.restoreState()


def get_styles():
    styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("Title", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=18, leading=22, textColor=colors.HexColor("#0f172a"), spaceAfter=3),
        "subtitle": ParagraphStyle("Sub", parent=styles["Normal"], fontName="Helvetica", fontSize=9.5, leading=13, textColor=colors.HexColor("#475569"), spaceAfter=10),
        "h1": ParagraphStyle("H1", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=12, leading=15, textColor=colors.HexColor("#1e3a8a"), spaceBefore=12, spaceAfter=5, keepWithNext=True),
        "h2": ParagraphStyle("H2", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=9.5, leading=13, textColor=colors.HexColor("#334155"), spaceBefore=7, spaceAfter=3, keepWithNext=True),
        "body": ParagraphStyle("Body", parent=styles["Normal"], fontName="Helvetica", fontSize=8.25, leading=11.5, textColor=colors.HexColor("#1e293b"), spaceAfter=5),
        "bullet": ParagraphStyle("Bullet", parent=styles["Normal"], fontName="Helvetica", fontSize=8.0, leading=11, textColor=colors.HexColor("#1e293b"), leftIndent=12, firstLineIndent=-8, spaceAfter=2.5),
        "th": ParagraphStyle("TH", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=8, leading=10, textColor=colors.white),
        "td": ParagraphStyle("TD", parent=styles["Normal"], fontName="Helvetica", fontSize=7.75, leading=10, textColor=colors.HexColor("#0f172a")),
        "td_bold": ParagraphStyle("TDBold", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=7.75, leading=10, textColor=colors.HexColor("#0f172a")),
    }


def create_phase_pdf(phase_num: int, phase_title: str, summary: str, details_table: list, diagram_name: str, key_findings: list):
    PHASE_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    clean_title = phase_title.replace(" ", "_").replace("/", "_").replace("\\", "_").replace("&", "and")
    pdf_path = PHASE_REPORTS_DIR / f"DocVerify_Phase_{phase_num}_{clean_title}.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter, leftMargin=54, rightMargin=54, topMargin=54, bottomMargin=54)
    st = get_styles()
    story = []

    # Title & Banner
    story.append(Paragraph(f"DocVerify — Phase {phase_num}: {phase_title}", st["title"]))
    story.append(Paragraph(f"Official Technical Milestone Report & Engineering Specification (Phase {phase_num})", st["subtitle"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1e40af"), spaceBefore=0, spaceAfter=10))

    # Overview
    story.append(Paragraph("1. Phase Executive Summary & Objectives", st["h1"]))
    story.append(Paragraph(summary, st["body"]))
    story.append(Spacer(1, 4))

    # Diagram if available
    diag_path = DIAGRAMS_DIR / diagram_name
    if diag_path.exists():
        story.append(Paragraph("2. Architectural Diagram & Pipeline Workflow", st["h1"]))
        try:
            story.append(RLImage(str(diag_path), width=6.6*inch, height=3.0*inch))
            story.append(Spacer(1, 6))
        except Exception as e:
            print(f"Error adding image {diag_path}: {e}")

    # Structured Table
    story.append(Paragraph("3. Technical Specifications & Milestone Deliverables", st["h1"]))
    table_rows = [[Paragraph("Dimension / Metric", st["th"]), Paragraph("Implementation / Specification", st["th"]), Paragraph("Significance & Impact", st["th"])]]
    for row in details_table:
        table_rows.append([
            Paragraph(f"<b>{row[0]}</b>", st["td_bold"]),
            Paragraph(row[1], st["td"]),
            Paragraph(row[2], st["td"])
        ])
    t = Table(table_rows, colWidths=[120, 150, 234])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t)
    story.append(Spacer(1, 8))

    # Key Findings & Discoveries
    story.append(Paragraph("4. Key Discoveries, Challenges & Next-Phase Directives", st["h1"]))
    for kf in key_findings:
        story.append(Paragraph(f"• {kf}", st["bullet"]))

    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cbd5e1"), spaceBefore=4, spaceAfter=6))
    story.append(Paragraph(f"<i>DocVerify Technical Archive • Phase {phase_num} Report Completed</i>", ParagraphStyle("Sign", fontName="Helvetica-Oblique", fontSize=7.5, textColor=colors.HexColor("#64748b"), alignment=1)))

    doc.build(story, canvasmaker=NumberedPhaseCanvas)
    print(f"Generated: {pdf_path.name} ({pdf_path.stat().st_size} bytes)")


def generate_all_phase_pdfs():
    print("Generating comprehensive individual Phase 1–9 PDF reports...")

    # Phase 1
    create_phase_pdf(
        phase_num=1,
        phase_title="Dataset Ingestion & Audit",
        summary="Phase 1 established the formal scientific foundation of DocVerify. It conducted a complete forensic inventory of the SIDTD dataset, resolving corrupt color profiles (CMYK) that caused Python Pillow crashes, cataloging 2,222 images across 10 international document templates, and structuring verified metadata manifest CSV files.",
        details_table=[
            ("Dataset Size", "2,222 Total Documents (1,000 Genuine, 1,222 Forged)", "Provided sufficient sample diversity across genuine passports and manipulated ID cards."),
            ("Template Coverage", "10 International Templates (Passports & National IDs)", "Albania, Azerbaijan, Spain, Estonia, Finland, Greece, Latvia, Russia, Serbia, Slovakia."),
            ("Color Profile Audit", "CMYK to RGB Standardization Script", "Resolved unhandled 4-channel JPEG images that crashed torchvision training loaders."),
            ("Integrity Hashes", "MD5 Checksums for every image", "Ensured absolute byte-level auditability and zero accidental data corruption.")
        ],
        diagram_name="diag3_dataset_structure.png",
        key_findings=[
            "Discovered unstandardized CMYK color-spaces in government document scans; mandated RGB standardization.",
            "Established identity-disjoint clustering to prevent identical citizens from leaking between splits.",
            "Generated dataset_manifest.csv with 20 audit columns as the permanent project dataset index."
        ]
    )

    # Phase 2
    create_phase_pdf(
        phase_num=2,
        phase_title="Global Baseline Model Training",
        summary="Phase 2 trained the primary global classifier on full-frame document images. EfficientNet-B0 was selected for its high parameter efficiency (4.01M parameters) and compound scaling. The resulting model established the permanent global baseline (models/phase2/best_model.pth) achieving 79.69% accuracy on the test set.",
        details_table=[
            ("Model Architecture", "EfficientNet-B0 Binary Classifier (4.01M params)", "Compact backbone enabling fast CPU/GPU inference with strong ImageNet feature transfer."),
            ("Accuracy Achieved", "79.69% (255 / 320 correct on test set)", "Baseline proof that global macro-texture contains significant forgery information."),
            ("Precision / Recall", "Precision: 82.91% | Recall: 77.51% (F1: 80.12%)", "Identified 131 true forgeries, but suffered 38 false negatives (missed forgeries)."),
            ("ROC-AUC Score", "0.9112 Area Under Curve", "Demonstrated high overall separability between global authentic and fake features.")
        ],
        diagram_name="diag1_project_overview.png",
        key_findings=[
            "Global CNN effectively captured full-image macro features: paper texture, background Guilloche lines, and border artifacts.",
            "Suffered from False Negatives when tampering was restricted to tiny localized alphanumeric fields.",
            "Formally preserved best_model.pth as the frozen global backbone for all subsequent multi-modal phases."
        ]
    )

    # Phase 3
    create_phase_pdf(
        phase_num=3,
        phase_title="Explainability & Spatial Distortion Audit",
        summary="Phase 3 investigated model interpretability using Grad-CAM attention heatmaps. The investigation revealed a critical computer vision limitation: resizing rectangular identity documents (1.6:1 aspect ratio) into standard 224x224 squares distorted the spatial gradient projections, misdirecting attention maps away from actual tampered fields.",
        details_table=[
            ("XAI Technique", "Gradient-Weighted Class Activation Mapping (Grad-CAM)", "Visualized intermediate activation maps at features[8] to explain model focus zones."),
            ("Spatial Distortion Issue", "Squaring Rectangular Inputs (e.g. 2167x1360 to 224x224)", "Caused Grad-CAM heatmaps to stretch horizontally, pointing to incorrect text areas."),
            ("Inverse Mapping Math", "CorrectedGradCAM Inverse Coordinate Projection", "Engineered custom matrix transformation to map heatmaps accurately back to original image space."),
            ("HiRes-CAM Addition", "High-Resolution Class Activation Mapping", "Element-wise gradient product implemented for finer spatial attribution.")
        ],
        diagram_name="diag5_complete_ai_pipeline.png",
        key_findings=[
            "Standard naive Grad-CAM is invalid for rectangular identity documents without aspect-ratio correction.",
            "Engineered src/gradcam_corrected.py with exact mathematical coordinate back-projection.",
            "Discovered that global attention often looked at borders rather than small tampered text characters."
        ]
    )

    # Phase 4
    create_phase_pdf(
        phase_num=4,
        phase_title="Multi-Modal Prototype Exploration",
        summary="Phase 4 explored combining optical character recognition (EasyOCR), field-level bounding box proposals (YOLO), and digital image forensics. It proved that a multi-modal defense is required, but demonstrated that naive score averaging produced unstable, unpredictable classification verdicts.",
        details_table=[
            ("OCR Exploration", "EasyOCR Text Extraction + Regex Syntax Rules", "Extracted dates and numbers to catch syntactic forgeries (e.g. impossible dates like 32/13/1985)."),
            ("YOLO Investigation", "YOLOv5/v8 Object Detection Proposals", "Tested bounding box proposals; identified high boundary jitter and severe compute overhead."),
            ("Forensic Exploration", "Error Level Analysis (ELA) + Sensor Noise", "Identified that JPEG re-compression and high-frequency noise expose copy-pasted text/photos."),
            ("Fusion Strategy", "Experimental Naive Score Averaging", "Equal weighting caused noisy signals to overturn confident visual predictions, proving the need for statistical fusion.")
        ],
        diagram_name="diag6_user_data_flow.png",
        key_findings=[
            "Confirmed that no single modality is sufficient on its own (visual, text, and forensics must collaborate).",
            "Discovered that naive score averaging is fatally flawed; established requirement for trained logistic fusion.",
            "Observed that YOLO bounding boxes cut through text lines, creating downstream OCR recognition errors."
        ]
    )

    # Phase 5
    create_phase_pdf(
        phase_num=5,
        phase_title="Local Field Extraction & Tiled Inspection",
        summary="Phase 5 tackled the resolution loss problem: when high-resolution documents (2167x1360) are downscaled to 224x224, tiny modified digits become blurred out. Phase 5 introduced dedicated local patch verification on uncompressed native-scale crops and replaced unstable YOLO proposals with canonical geometry templates.",
        details_table=[
            ("Native Scale Crops", "2.0x Context Expansion around Target Fields", "Extracted text and photo fields at full camera resolution without downsampling loss."),
            ("Template Alignment", "Canonical Template Layouts (`src/template_layouts.py`)", "Predefined exact normalized bounding boxes for all 10 document types, eliminating YOLO drift."),
            ("Patch Model Training", "Initial Dedicated Patch Classifier Training", "Trained model specifically on cropped name, birth date, and portrait patches."),
            ("Field Types", "Name, Surname, Birth Date, Expiry, Document Number, Photo", "Covered the highest-risk fields targeted by identity document fraudsters.")
        ],
        diagram_name="diag8_software_architecture.png",
        key_findings=[
            "Canonical layout templates provided 100% boundary stability with 10x faster execution than YOLO.",
            "Dedicated patch inspection recovered micro-tampered characters invisible to the global CNN.",
            "Established the dual-model paradigm: Global EfficientNet + Dedicated Local Patch EfficientNet."
        ]
    )

    # Phase 6
    create_phase_pdf(
        phase_num=6,
        phase_title="Oracle Leakage Audit & Frozen Test Manifest",
        summary="Phase 6 conducted a thorough scientific integrity audit. The audit revealed a severe vulnerability: filename parsing (e.g. reading 'alb_id' from 'alb_id_01.jpg') acted as an oracle shortcut, bypassing true visual layout recognition. Phase 6 sealed this leakage, implemented autonomous geometry matching, and permanently froze the 320-sample test set.",
        details_table=[
            ("Oracle Leakage Audit", "Filename Dependency Identification", "Revealed that the system crashed or cheated if images were renamed to 'scan_01.jpg'."),
            ("Autonomous Matcher", "Geometry & Aspect Ratio Matching (`src/template_matcher.py`)", "Measures physical width/height aspect ratios and layout features with zero reliance on filename hints."),
            ("UNKNOWN Fallback", "Explicit UNKNOWN Template Routing", "Anomalous dimensions or non-ID images are routed cleanly to global texture analysis without crashing."),
            ("Frozen Test Set", "Permanent 320-Sample Benchmark Manifest", "Locked frozen_test_manifest.csv (151 genuine, 169 forged) to ensure reproducible benchmarks.")
        ],
        diagram_name="fig2_dataset_split_architecture.png",
        key_findings=[
            "Completely eliminated filename hints; achieved true blind document verification.",
            "Established permanent frozen test split ensuring all future benchmarks are 100% comparable.",
            "Sealed the project against shortcut learning and publication integrity criticisms."
        ]
    )

    # Phase 7
    create_phase_pdf(
        phase_num=7,
        phase_title="Evidence Fusion & Probability Calibration R&D",
        summary="Phase 7 engineered the mathematical decision layer. It trained a standardized logistic regression model across 7 multi-modal features and implemented Isotonic Regression calibration to transform raw neural network overconfidence into verified empirical risk probabilities.",
        details_table=[
            ("Fusion Architecture", "Standardized Logistic Regression (7 Features)", "Learned optimal weights (+2.92 global, +1.56 patch, -1.68 ELA) on validation split."),
            ("Calibration Method", "Isotonic Regression (`models/phase8/fusion_calibrator.joblib`)", "Fitted non-parametric monotonic calibration mapping raw logits to true empirical fraud rates."),
            ("Error Metric (Brier)", "Brier Score Reduced to 0.049 (Validation)", "Quantified mean squared probability error, ensuring confidence reflects accuracy."),
            ("ECE Metric", "Expected Calibration Error (ECE) Minimized to 0.0%", "Eliminated neural network overconfidence across all 10 probability bins.")
        ],
        diagram_name="fig3_evidence_fusion_calibration.png",
        key_findings=[
            "Logistic fusion significantly outperformed naive score averaging across all evaluation splits.",
            "Isotonic calibration successfully aligned output risk scores with empirical ground truth frequencies.",
            "Discovered minor feature scale discrepancies that were flagged for remediation in Phase 8."
        ]
    )

    # Phase 8
    create_phase_pdf(
        phase_num=8,
        phase_title="Remediated Pipeline & Scientific Benchmark",
        summary="Phase 8 represents the definitive scientific milestone of DocVerify. It remediated feature consistency, trained the dedicated Phase 8 patch classifier with 2.0x crop expansion, and achieved a dramatic performance leap over the Phase 2 baseline: Accuracy jumped from 79.69% to 91.25%, with ROC-AUC reaching 0.9677 (McNemar's test p < 0.001).",
        details_table=[
            ("Accuracy Jump", "<b>79.69% to 91.25%</b> (292 / 320 Correct)", "Statistically validated performance breakthrough (+11.56% absolute improvement)."),
            ("Precision / Recall / F1", "Precision: 91.23% | Recall: 92.31% | F1: 91.76%", "Caught 156 of 169 forgeries while maintaining extremely low false alarms (15 FP)."),
            ("ROC-AUC Score", "<b>0.9677</b> Area Under Curve", "Proves superior discrimination between genuine and forged documents across all thresholds."),
            ("Statistical Validation", "McNemar's Test: <b>p = 0.000003 (p < 0.001)</b>", "Mathematically proves the Phase 8 architecture is significantly superior to Phase 2."),
            ("Brier & ECE Scores", "Brier Score: 0.0657 | ECE: 0.0484", "Guarantees that displayed risk probabilities are statistically calibrated and trustworthy.")
        ],
        diagram_name="fig1_end_to_end_architecture.png",
        key_findings=[
            "14/14 automated regression tests passed in `scripts/test_phase8.py`.",
            "Dedicated patch classifier successfully recovered micro-tampered text invisible to Phase 2.",
            "Delivered the verified, frozen production weights and operational thresholds (0.50, 0.72, 0.80)."
        ]
    )

    # Phase 9
    create_phase_pdf(
        phase_num=9,
        phase_title="Enterprise UI/UX & Production Refinement",
        summary="Phase 9 transformed DocVerify into an enterprise-grade web application. It resolved critical usability bugs (tab text clipping, unreadable typography, threshold boundary misclassifications on real documents, and 100% risk score clamping) and created a responsive, professional Streamlit interface backed by ReportLab compliance certificate generation.",
        details_table=[
            ("UI Design System", "Enterprise CSS Design Tokens (`src/ui_components.py`)", "Custom styling overriding BaseWeb: no tab clipping, clear typography, professional color contrast."),
            ("Boundary Condition Fix", "Threshold Comparison: `>` instead of `>=`", "Eliminated boundary false positives where genuine documents landing at 0.7200 were misclassified as forged."),
            ("Proportional Scaling", "Continuous Proportional Risk Scaling", "Eliminated flat 100% score clamping; scales risk dynamically across [72.0%, 99.5%] with true 100% reserve."),
            ("Certificate Export", "Automated PDF Verification Certificate (`src/report_generator.py`)", "Generates compliance-ready audit certificates with session metadata, OCR tables, and Grad-CAM overlays."),
            ("Regression Testing", "Phase 9 Fix Suite (`scripts/test_phase9_fix.py`): 10/10 PASS", "Ensures zero frontend regressions and 100% consistency across all 12 smoke-test documents.")
        ],
        diagram_name="diag9_uiux_workflow.png",
        key_findings=[
            "All 24 automated tests passed across Phase 8 (14/14) and Phase 9 (10/10).",
            "alb_id_01.jpg and other borderline genuine documents now classify accurately as genuine/inconclusive instead of forged.",
            "System is completely stabilized, cleaned, documented, and ready for university examination defense."
        ]
    )

    print(f"Successfully generated all 9 Phase PDF reports in {PHASE_REPORTS_DIR}")


if __name__ == "__main__":
    generate_all_phase_pdfs()
