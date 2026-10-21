#!/usr/bin/env python3
"""
DocVerify - Phase 9 Fix: Controlled Smoke Test on Known Real and Fake Documents
"""

import sys
from pathlib import Path
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference_engine import DocVerifyInferenceEngine

def main():
    manifest = pd.read_csv(PROJECT_ROOT / "outputs" / "dataset_manifest" / "dataset_manifest.csv")
    engine = DocVerifyInferenceEngine(enable_ocr=True)

    test_files = [
        # Reals
        ("alb_id_01.jpg", "GENUINE"),
        ("alb_id_03.jpg", "GENUINE"),
        ("esp_id_00.jpg", "GENUINE"),
        ("aze_passport_00.jpg", "GENUINE"),
        ("grc_passport_00.jpg", "GENUINE"),
        ("srb_passport_00.jpg", "GENUINE"),
        # Fakes
        ("alb_id_03_fake_6_69.jpg", "FORGED"),
        ("alb_id_00_fake_6_25.jpg", "FORGED"),
        ("esp_id_00_fake_2_26.jpg", "FORGED"),
        ("aze_passport_00_fake_3_104.jpg", "FORGED"),
        ("grc_passport_00_fake_4_111.jpg", "FORGED"),
        ("srb_passport_00_fake_5_114.jpg", "FORGED")
    ]

    records = []
    print("=" * 95)
    print(f"{'Image Filename':32s} | {'Ground Truth':8s} | {'Global':6s} | {'Patch':6s} | {'Raw':6s} | {'Risk':6s} | {'Verdict'}")
    print("=" * 95)

    for fname, gt in test_files:
        m = manifest[manifest["image_filename"] == fname]
        if m.empty:
            print("Not in manifest:", fname)
            continue
        r = m.iloc[0]
        p = Path(r["absolute_path"])
        if not p.exists():
            p = PROJECT_ROOT / r["image_path"]

        img = Image.open(p)
        res = engine.verify_document(img, filename_hint=fname, operational_mode="balanced")

        glob_val = res["evidence"]["p_global_fake"]
        patch_val = res["evidence"]["p_patch_fake"]
        ocr_val = res["evidence"]["ocr_confidence"]
        ela_val = res["evidence"]["forensic_cues"]["ela_inconsistency"]
        noise_val = res["evidence"]["forensic_cues"]["noise_inconsistency"]
        raw_val = res["raw_forgery_probability"]
        risk_val = res["risk_score"]
        verdict = res["verdict"]

        records.append({
            "Image": fname,
            "Ground Truth": gt,
            "Template": res["template_id"],
            "Global": f"{glob_val:.4f}",
            "Patch": f"{patch_val:.4f}",
            "OCR": f"{ocr_val:.4f}",
            "Forensic ELA": f"{ela_val:.4f}",
            "Forensic Noise": f"{noise_val:.4f}",
            "Fusion Raw": f"{raw_val:.4f}",
            "Calibrated Risk": f"{risk_val}%",
            "Final Verdict": verdict
        })

        print(f"{fname:32s} | {gt:8s} | {glob_val:6.4f} | {patch_val:6.4f} | {raw_val:6.4f} | {risk_val:5.1f}% | {verdict}")

    df_out = pd.DataFrame(records)
    out_csv = PROJECT_ROOT / "outputs" / "phase9_fix" / "ui_model_smoke_test.csv"
    df_out.to_csv(out_csv, index=False)
    print("=" * 95)
    print("Saved smoke test results to:", out_csv)


if __name__ == "__main__":
    main()
