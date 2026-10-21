#!/usr/bin/env python3
"""
DocVerify - Dataset Audit Script (Phase 1)
Author: Senior ML & CV Engineer
Purpose: Audits the SIDTD dataset (CSVs, images, annotations), checks integrity,
         measures class balance, inspects dimensions, detects duplicates and data leakage,
         and generates comprehensive audit outputs (JSON summary, markdown report, visualizations).

CRITICAL CONSTRAINTS:
- Non-destructive: Does NOT alter, move, rename, delete, or overwrite original dataset files.
- Reproducible: Uses pathlib and parameterized directories.
"""

import json
import re
import hashlib
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def get_base_doc(filename_or_path: str) -> str:
    """Extracts the base document/identity identifier.
    E.g.: 'alb_id_00_fake_6_25.jpg' -> 'alb_id_00'
          'alb_id_00.jpg'           -> 'alb_id_00'
    """
    stem = Path(filename_or_path).stem
    m = re.match(r'^(.*?_\d+)(_fake.*)?$', stem)
    if m:
        return m.group(1)
    return stem


def audit_dataset(
    dataset_dir: Path,
    output_dir: Path,
    generate_viz: bool = True
) -> dict:
    print(f"[1/7] Initializing dataset audit on {dataset_dir}...")
    
    split_dir = dataset_dir / "split_normal"
    template_dir = dataset_dir / "template"
    images_dir = template_dir / "Images"
    ann_dir = template_dir / "Annotations"
    
    reals_img_dir = images_dir / "reals"
    fakes_img_dir = images_dir / "fakes"
    reals_ann_dir = ann_dir / "reals"
    fakes_ann_dir = ann_dir / "fakes"
    
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {}

    # ----------------------------------------------------
    # 1. Directory Structure Verification
    # ----------------------------------------------------
    print("[2/7] Checking directory structure & physical file counts...")
    real_images = sorted(list(reals_img_dir.glob("*.jpg")))
    fake_images = sorted(list(fakes_img_dir.glob("*.jpg")))
    real_annotations = sorted(list(reals_ann_dir.glob("*.json")))
    fake_annotations = sorted(list(fakes_ann_dir.glob("*.json")))
    
    summary["filesystem_counts"] = {
        "reals_images": len(real_images),
        "fakes_images": len(fake_images),
        "total_images": len(real_images) + len(fake_images),
        "reals_annotations": len(real_annotations),
        "fakes_annotations": len(fake_annotations),
        "total_annotations": len(real_annotations) + len(fake_annotations),
    }

    # ----------------------------------------------------
    # 2. CSV Split Analysis
    # ----------------------------------------------------
    print("[3/7] Reading and analyzing split CSV files...")
    splits = {}
    for s_name in ["train", "val", "test"]:
        csv_file = split_dir / f"{s_name}_split_SIDTD.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            splits[s_name] = df
        else:
            splits[s_name] = None
            print(f"Warning: {csv_file} does not exist.")

    split_analysis = {}
    combined_csv_rows = 0
    all_csv_paths = []
    
    for s_name, df in splits.items():
        if df is not None:
            combined_csv_rows += len(df)
            counts = df["label_name"].value_counts().to_dict()
            all_csv_paths.extend(df["image_path"].tolist())
            split_analysis[s_name] = {
                "rows": len(df),
                "columns": list(df.columns),
                "label_counts": counts,
                "null_values": int(df.isnull().sum().sum()),
                "unique_paths": int(df["image_path"].nunique()),
                "classes": df["class_name"].value_counts().to_dict(),
            }
    
    summary["split_analysis"] = split_analysis
    summary["csv_total_rows"] = combined_csv_rows

    # Label representation verification
    train_df = splits.get("train")
    if train_df is not None:
        label_map = (
            train_df[["label_name", "label"]]
            .drop_duplicates()
            .set_index("label_name")["label"]
            .to_dict()
        )
        summary["label_mapping"] = label_map
    else:
        summary["label_mapping"] = {}

    # Discrepancy between CSV paths and disk images
    all_csv_filenames = set(Path(p).name for p in all_csv_paths)
    disk_image_filenames = set(f.name for f in real_images + fake_images)
    
    summary["csv_vs_disk_discrepancy"] = {
        "csv_unique_filenames": len(all_csv_filenames),
        "disk_filenames": len(disk_image_filenames),
        "csv_files_present_on_disk": len(all_csv_filenames & disk_image_filenames),
        "csv_files_missing_from_disk": len(all_csv_filenames - disk_image_filenames),
        "disk_files_not_referenced_in_csv": len(disk_image_filenames - all_csv_filenames),
        "reals_in_csv_matching_disk": len(set(f.name for f in real_images) & all_csv_filenames),
        "fakes_in_csv_matching_disk": len(set(f.name for f in fake_images) & all_csv_filenames),
    }

    # ----------------------------------------------------
    # 3. Image Analysis (Integrity, Color, Dimensions, Hash)
    # ----------------------------------------------------
    print("[4/7] Scanning all images for integrity, color modes, dimensions, and MD5 hashes...")
    image_stats = {
        "corrupt_count": 0,
        "modes": Counter(),
        "dimensions": Counter(),
        "aspect_ratios": [],
        "file_sizes_bytes": [],
        "template_dimensions": defaultdict(set),
        "template_modes": defaultdict(set),
    }
    
    md5_to_files = defaultdict(list)
    
    for img_path in tqdm(real_images + fake_images, desc="Auditing Images"):
        try:
            sz = img_path.stat().st_size
            image_stats["file_sizes_bytes"].append(sz)
            
            # MD5 calculation
            hasher = hashlib.md5()
            with open(img_path, "rb") as f:
                hasher.update(f.read())
            img_hash = hasher.hexdigest()
            md5_to_files[img_hash].append(img_path.name)
            
            # Pillow load (in memory only)
            with Image.open(img_path) as im:
                w, h = im.size
                image_stats["dimensions"][(w, h)] += 1
                image_stats["modes"][im.mode] += 1
                image_stats["aspect_ratios"].append(round(w / h, 3))
                
                # Document template name (e.g. alb_id)
                template_name = "_".join(img_path.stem.split("_")[:2])
                image_stats["template_dimensions"][template_name].add((w, h))
                image_stats["template_modes"][template_name].add(im.mode)
                
        except Exception as e:
            image_stats["corrupt_count"] += 1
            print(f"Error reading {img_path.name}: {e}")

    # Duplicates
    exact_duplicates = {k: v for k, v in md5_to_files.items() if len(v) > 1}
    summary["image_analysis"] = {
        "corrupt_count": image_stats["corrupt_count"],
        "modes": dict(image_stats["modes"]),
        "resolutions": [
            {"width": w, "height": h, "count": count}
            for (w, h), count in image_stats["dimensions"].most_common()
        ],
        "file_size_summary_kb": {
            "min": round(min(image_stats["file_sizes_bytes"]) / 1024, 2),
            "max": round(max(image_stats["file_sizes_bytes"]) / 1024, 2),
            "mean": round(np.mean(image_stats["file_sizes_bytes"]) / 1024, 2),
        },
        "exact_duplicates": [
            {"hash": h, "files": files} for h, files in exact_duplicates.items()
        ],
        "template_modes": {k: list(v) for k, v in image_stats["template_modes"].items()},
        "template_dimensions": {
            k: [list(dim) for dim in v] for k, v in image_stats["template_dimensions"].items()
        },
    }

    # ----------------------------------------------------
    # 4. Annotation Analysis
    # ----------------------------------------------------
    print("[5/7] Analyzing VIA templates and fake JSON annotations...")
    
    # 4A. Real VIA templates
    via_templates = {}
    via_summary = {}
    for ann_p in real_annotations:
        tname = ann_p.stem
        with open(ann_p, "r", encoding="utf-8", errors="replace") as fp:
            data = json.load(fp)
        img_meta = data.get("_via_img_metadata", {})
        via_templates[tname] = {}
        fields_in_template = Counter()
        for entry in img_meta.values():
            fname = entry.get("filename")
            regions = entry.get("regions", [])
            via_templates[tname][fname] = regions
            for r in regions:
                fn = r.get("region_attributes", {}).get("field_name")
                if fn:
                    fields_in_template[fn] += 1
        via_summary[tname] = {
            "annotated_real_images": len(img_meta),
            "field_counts": dict(fields_in_template),
        }
    
    # 4B. Fake JSONs
    fake_ctypes = Counter()
    fake_fields = Counter()
    fake_loaders = Counter()
    matched_to_via_field = 0
    
    pat_src = re.compile(r'^(.*)_(\d+)\.jpg$')
    
    for f_ann in fake_annotations:
        with open(f_ann, "r", encoding="utf-8", errors="replace") as fp:
            data = json.load(fp)
        ctype = data.get("ctype")
        field = data.get("field")
        src = data.get("src")
        loader = data.get("loader")
        
        fake_ctypes[ctype] += 1
        fake_fields[field] += 1
        fake_loaders[loader] += 1
        
        # Check if matched in VIA
        m = pat_src.match(src) if src else None
        if m:
            tname, idx = m.group(1), m.group(2)
            src_fn = f"{idx}.jpg"
            if tname in via_templates and src_fn in via_templates[tname]:
                for r in via_templates[tname][src_fn]:
                    if r.get("region_attributes", {}).get("field_name") == field:
                        matched_to_via_field += 1
                        break

    summary["annotation_analysis"] = {
        "real_via_templates": via_summary,
        "fake_annotation_count": len(fake_annotations),
        "fake_ctypes": dict(fake_ctypes),
        "fake_loaders": dict(fake_loaders),
        "fake_fields_top15": dict(fake_fields.most_common(15)),
        "manipulated_field_to_via_bbox_match_rate": (
            matched_to_via_field / len(fake_annotations) if fake_annotations else 0
        ),
    }

    # ----------------------------------------------------
    # 5. Data Leakage & Document Overlap Investigation
    # ----------------------------------------------------
    print("[6/7] Investigating base document identity leakage across splits...")
    leakage_analysis = {}
    if splits.get("train") is not None and splits.get("val") is not None and splits.get("test") is not None:
        train_df = splits["train"].copy()
        val_df = splits["val"].copy()
        test_df = splits["test"].copy()
        
        train_df["base_doc"] = train_df["image_path"].apply(get_base_doc)
        val_df["base_doc"] = val_df["image_path"].apply(get_base_doc)
        test_df["base_doc"] = test_df["image_path"].apply(get_base_doc)
        
        train_docs = set(train_df["base_doc"])
        val_docs = set(val_df["base_doc"])
        test_docs = set(test_df["base_doc"])
        
        leakage_analysis = {
            "train_unique_base_docs": len(train_docs),
            "val_unique_base_docs": len(val_docs),
            "test_unique_base_docs": len(test_docs),
            "overlap_train_val": len(train_docs & val_docs),
            "overlap_train_test": len(train_docs & test_docs),
            "overlap_val_test": len(val_docs & test_docs),
            "train_val_overlap_pct": round(len(train_docs & val_docs) / len(val_docs) * 100, 2),
            "train_test_overlap_pct": round(len(train_docs & test_docs) / len(test_docs) * 100, 2),
            "sample_leaked_docs": sorted(list(train_docs & test_docs))[:10],
        }
    summary["data_leakage"] = leakage_analysis

    # ----------------------------------------------------
    # 6. Dataset Visualizations
    # ----------------------------------------------------
    if generate_viz:
        print("[7/7] Generating audit visualization artifacts...")
        sns.set_theme(style="whitegrid")
        
        # A. Class distribution chart
        plt.figure(figsize=(8, 5))
        counts = [summary["filesystem_counts"]["reals_images"], summary["filesystem_counts"]["fakes_images"]]
        labels = ["Reals (1,000)", "Fakes (1,222)"]
        colors = ["#2ecc71", "#e74c3c"]
        bars = plt.bar(labels, counts, color=colors, width=0.5)
        plt.title("Disk Image Count by Class (SIDTD Template Subset)", fontsize=14, fontweight="bold")
        plt.ylabel("Number of Images", fontsize=12)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 20, f"{yval}", ha="center", va="bottom", fontsize=11, fontweight="bold")
        plt.ylim(0, 1500)
        plt.tight_layout()
        plt.savefig(viz_dir / "class_distribution.png", dpi=200)
        plt.close()

        # B. Split distribution chart
        if split_analysis:
            plt.figure(figsize=(10, 5))
            split_df = pd.DataFrame([
                {"Split": "Train", "Reals": split_analysis["train"]["label_counts"].get("reals", 0), "Fakes": split_analysis["train"]["label_counts"].get("fakes", 0)},
                {"Split": "Val", "Reals": split_analysis["val"]["label_counts"].get("reals", 0), "Fakes": split_analysis["val"]["label_counts"].get("fakes", 0)},
                {"Split": "Test", "Reals": split_analysis["test"]["label_counts"].get("reals", 0), "Fakes": split_analysis["test"]["label_counts"].get("fakes", 0)},
            ])
            ax = split_df.plot(x="Split", y=["Reals", "Fakes"], kind="bar", color=["#2ecc71", "#e74c3c"], figsize=(9, 5), width=0.6)
            plt.title("CSV Predefined Split Distribution (Train / Val / Test)", fontsize=14, fontweight="bold")
            plt.ylabel("Number of Samples", fontsize=12)
            plt.xticks(rotation=0, fontsize=11)
            plt.legend(title="Class", fontsize=11)
            for p in ax.patches:
                height = p.get_height()
                if height > 0:
                    ax.annotate(f"{int(height)}", (p.get_x() + p.get_width() / 2., height + 25), ha="center", va="center", fontsize=10, fontweight="bold")
            plt.ylim(0, 1900)
            plt.tight_layout()
            plt.savefig(viz_dir / "split_distribution.png", dpi=200)
            plt.close()

        # C. Forgery Types & Methods Distribution
        plt.figure(figsize=(8, 4))
        ctype_keys = list(fake_ctypes.keys())
        ctype_vals = list(fake_ctypes.values())
        bars = plt.barh(ctype_keys, ctype_vals, color="#3498db", height=0.5)
        plt.title("Manipulation Techniques in Fake Annotations", fontsize=14, fontweight="bold")
        plt.xlabel("Annotation Count", fontsize=12)
        for bar in bars:
            wval = bar.get_width()
            plt.text(wval + 20, bar.get_y() + bar.get_height()/2.0, f"{wval}", ha="left", va="center", fontsize=11, fontweight="bold")
        plt.xlim(0, 1300)
        plt.tight_layout()
        plt.savefig(viz_dir / "forgery_methods.png", dpi=200)
        plt.close()

        # D. Base Document Leakage Visual
        if leakage_analysis:
            plt.figure(figsize=(8, 4.5))
            leak_labels = ["Val Base Docs in Train", "Test Base Docs in Train"]
            leak_pcts = [leakage_analysis["train_val_overlap_pct"], leakage_analysis["train_test_overlap_pct"]]
            bars = plt.bar(leak_labels, leak_pcts, color="#e67e22", width=0.4)
            plt.title("Data Leakage: Percentage of Evaluation Identities Present in Train", fontsize=13, fontweight="bold")
            plt.ylabel("% Overlap with Train Base Identities", fontsize=11)
            for bar in bars:
                hval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2.0, hval + 2, f"{hval}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
            plt.ylim(0, 115)
            plt.tight_layout()
            plt.savefig(viz_dir / "data_leakage_overlap.png", dpi=200)
            plt.close()

        # E. Example Real & Fake Image with Bounding Box Overlay
        try:
            sample_fake_img_path = fakes_img_dir / "alb_id_00_fake_6_25.jpg"
            sample_fake_ann_path = fakes_ann_dir / "alb_id_00_fake_6_25.json"
            
            with open(sample_fake_ann_path, "r") as fp:
                f_meta = json.load(fp)
            
            # Ground truth bounding box for expiry_date on alb_id 00:
            # x=1456, y=846, width=397, height=43
            with Image.open(sample_fake_img_path) as im:
                im_rgb = im.convert("RGB")
                draw = ImageDraw.Draw(im_rgb)
                # Bbox from VIA: [1456, 846, 1456+397, 846+43]
                x, y, w, h = 1456, 846, 397, 43
                draw.rectangle([x, y, x + w, y + h], outline="red", width=6)
                
                # Save annotated visual
                annotated_out = viz_dir / "sample_fake_annotation_overlay.png"
                im_rgb.save(annotated_out)
        except Exception as e:
            print(f"Warning: Could not render sample overlay: {e}")

    # Save summary JSON
    summary_path = output_dir / "dataset_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Dataset audit completed! Summary written to: {summary_path}")
    
    return summary


if __name__ == "__main__":
    DATASET_DIR = Path("E:/DocverifyAi/Dataset")
    OUTPUT_DIR = Path("E:/DocverifyAi/outputs/dataset_audit")
    audit_dataset(DATASET_DIR, OUTPUT_DIR, generate_viz=True)
