#!/usr/bin/env python3
"""
DocVerify - Build Verified Dataset Manifest & Identity-Disjoint Split (Phase 2)
Author: Senior ML & Research Engineer

Purpose:
1. Scans all 2,222 physical images on disk (1,000 reals, 1,222 fakes).
2. Parses real VIA annotations and fake JSON manipulation descriptors.
3. Groups documents into connected identity clusters (handling both direct inpainting
   and donor-recipient Crop_and_Replace relationships) to guarantee zero data leakage.
4. Generates a reproducible, stratified identity-disjoint train (~70%), val (~15%), test (~15%) split.
5. Saves dataset_manifest.csv with complete metadata:
   - image_path, image_filename, label (0/1), label_name, base_document_id, identity_cluster_id
   - template_country, template_type, forgery_type, manipulated_field, annotation_path
   - width, height, color_mode, split, is_exact_duplicate
"""

import json
import re
import hashlib
import random
from pathlib import Path
from collections import defaultdict, Counter

import pandas as pd
from PIL import Image
from tqdm import tqdm


def build_manifest(
    dataset_dir: Path,
    output_dir: Path,
    seed: int = 42
) -> pd.DataFrame:
    print(f"[1/5] Initializing manifest creation from {dataset_dir} (Seed={seed})...")
    
    reals_img_dir = dataset_dir / "template" / "Images" / "reals"
    fakes_img_dir = dataset_dir / "template" / "Images" / "fakes"
    reals_ann_dir = dataset_dir / "template" / "Annotations" / "reals"
    fakes_ann_dir = dataset_dir / "template" / "Annotations" / "fakes"
    
    real_images = sorted(list(reals_img_dir.glob("*.jpg")))
    fake_images = sorted(list(fakes_img_dir.glob("*.jpg")))
    
    print(f"Found {len(real_images)} real images and {len(fake_images)} fake images.")
    
    # ----------------------------------------------------
    # Step A: Load Real VIA Annotations for Bounding Boxes
    # ----------------------------------------------------
    print("[2/5] Loading real VIA annotations...")
    via_templates = {}
    for ann_p in sorted(reals_ann_dir.glob("*.json")):
        tname = ann_p.stem
        with open(ann_p, "r", encoding="utf-8", errors="replace") as fp:
            data = json.load(fp)
        meta = data.get("_via_img_metadata", {})
        via_templates[tname] = {}
        for entry in meta.values():
            fn = entry.get("filename")  # e.g. 00.jpg
            via_templates[tname][fn] = entry.get("regions", [])

    # ----------------------------------------------------
    # Step B: Build Union-Find for Connected Identity Clusters
    # ----------------------------------------------------
    print("[3/5] Resolving base document identities & connected clusters...")
    parent = {}
    
    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]
        
    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    # Initialize all real document stems
    for r in real_images:
        parent[r.stem] = r.stem

    # Link fake documents
    fake_metadata = {}
    pat_filename = re.compile(r'^(.*?_\d+)(_fake.*)?$')
    
    for f in fake_images:
        ann_p = fakes_ann_dir / f"{f.stem}.json"
        with open(ann_p, "r", encoding="utf-8", errors="replace") as fp:
            meta = json.load(fp)
        fake_metadata[f.name] = meta
        
        src_stem = Path(meta["src"]).stem
        m = pat_filename.match(f.stem)
        prefix_stem = m.group(1) if m else None
        
        for node in [src_stem, prefix_stem]:
            if node and node not in parent:
                parent[node] = node
        if prefix_stem:
            union(src_stem, prefix_stem)
            
        second_src = meta.get("second_src")
        if second_src and second_src not in [None, "None"]:
            second_stem = Path(second_src).stem
            if second_stem not in parent:
                parent[second_stem] = second_stem
            union(src_stem, second_stem)

    # ----------------------------------------------------
    # Step C: Assign Identity-Disjoint Splits
    # ----------------------------------------------------
    # Map each cluster ID to its document template (e.g. alb_id, esp_id)
    cluster_to_template = {}
    for r in real_images:
        c_id = find(r.stem)
        t = "_".join(r.stem.split("_")[:-1])
        cluster_to_template[c_id] = t

    # Group clusters by template for stratified assignment
    template_to_clusters = defaultdict(list)
    for c_id, t in cluster_to_template.items():
        template_to_clusters[t].append(c_id)

    random.seed(seed)
    cluster_to_split = {}
    
    for t, c_list in sorted(template_to_clusters.items()):
        c_list = sorted(c_list)
        random.shuffle(c_list)
        n = len(c_list)
        n_train = int(round(0.70 * n))
        n_val = int(round(0.15 * n))
        
        train_c = c_list[:n_train]
        val_c = c_list[n_train:n_train + n_val]
        test_c = c_list[n_train + n_val:]
        
        for c in train_c:
            cluster_to_split[c] = "train"
        for c in val_c:
            cluster_to_split[c] = "val"
        for c in test_c:
            cluster_to_split[c] = "test"

    # ----------------------------------------------------
    # Step D: Extract Image Metadata & Compile Manifest Rows
    # ----------------------------------------------------
    print("[4/5] Extracting image metadata and assembling manifest rows...")
    manifest_rows = []
    seen_hashes = {}
    
    # Process Real Images
    for r in tqdm(real_images, desc="Manifest (Reals)"):
        base_id = r.stem  # e.g. alb_id_00
        cluster_id = find(base_id)
        assigned_split = cluster_to_split[cluster_id]
        
        # MD5 Hash check
        h = hashlib.md5(r.read_bytes()).hexdigest()
        is_dup = (h in seen_hashes)
        seen_hashes[h] = r.name
        
        # Image details
        with Image.open(r) as im:
            w, h_px = im.size
            mode = im.mode
            
        t_country = r.stem.split("_")[0]
        t_type = "_".join(r.stem.split("_")[1:-1]) if len(r.stem.split("_")) > 2 else r.stem.split("_")[1]
        
        manifest_rows.append({
            "image_path": str(r.relative_to(dataset_dir.parent)).replace("\\", "/"),
            "absolute_path": str(r.resolve()).replace("\\", "/"),
            "image_filename": r.name,
            "label": 0,
            "label_name": "reals",
            "base_document_id": base_id,
            "identity_cluster_id": cluster_id,
            "template": f"{t_country}_{t_type}",
            "country": t_country,
            "doc_type": t_type,
            "forgery_type": "None",
            "manipulated_field": "None",
            "annotation_path": f"Dataset/template/Annotations/reals/{t_country}_{t_type}.json",
            "bounding_box_json": "None",
            "width": w,
            "height": h_px,
            "color_mode": mode,
            "md5_hash": h,
            "is_exact_duplicate": is_dup,
            "split": assigned_split
        })

    # Process Fake Images
    pat_src = re.compile(r'^(.*)_(\d+)\.jpg$')
    for f in tqdm(fake_images, desc="Manifest (Fakes)"):
        meta = fake_metadata[f.name]
        src_stem = Path(meta["src"]).stem
        cluster_id = find(src_stem)
        assigned_split = cluster_to_split[cluster_id]
        
        # MD5 Hash check
        h = hashlib.md5(f.read_bytes()).hexdigest()
        is_dup = (h in seen_hashes)
        seen_hashes[h] = f.name
        
        # Image details
        with Image.open(f) as im:
            w, h_px = im.size
            mode = im.mode
            
        # Parse ground-truth bounding box from VIA
        bbox_str = "None"
        m = pat_src.match(meta["src"])
        if m:
            tname, idx = m.group(1), m.group(2)
            src_fn = f"{idx}.jpg"
            if tname in via_templates and src_fn in via_templates[tname]:
                for reg in via_templates[tname][src_fn]:
                    if reg.get("region_attributes", {}).get("field_name") == meta["field"]:
                        bbox_str = json.dumps(reg.get("shape_attributes", {}))
                        break

        t_country = src_stem.split("_")[0]
        t_type = "_".join(src_stem.split("_")[1:-1]) if len(src_stem.split("_")) > 2 else src_stem.split("_")[1]

        manifest_rows.append({
            "image_path": str(f.relative_to(dataset_dir.parent)).replace("\\", "/"),
            "absolute_path": str(f.resolve()).replace("\\", "/"),
            "image_filename": f.name,
            "label": 1,
            "label_name": "fakes",
            "base_document_id": src_stem,
            "identity_cluster_id": cluster_id,
            "template": f"{t_country}_{t_type}",
            "country": t_country,
            "doc_type": t_type,
            "forgery_type": meta.get("ctype", "Unknown"),
            "manipulated_field": meta.get("field", "Unknown"),
            "annotation_path": f"Dataset/template/Annotations/fakes/{f.stem}.json",
            "bounding_box_json": bbox_str,
            "width": w,
            "height": h_px,
            "color_mode": mode,
            "md5_hash": h,
            "is_exact_duplicate": is_dup,
            "split": assigned_split
        })

    # ----------------------------------------------------
    # Step E: Save and Verify
    # ----------------------------------------------------
    print("[5/5] Saving manifest and verifying split properties...")
    df = pd.DataFrame(manifest_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / "dataset_manifest.csv"
    df.to_csv(out_csv, index=False)
    print(f"Manifest successfully generated with {len(df)} rows at: {out_csv}")
    
    # Validation report
    tr_docs = set(df[df["split"] == "train"]["identity_cluster_id"])
    val_docs = set(df[df["split"] == "val"]["identity_cluster_id"])
    te_docs = set(df[df["split"] == "test"]["identity_cluster_id"])
    
    print("\n--- SPLIT VERIFICATION ---")
    print(f"Train samples: {len(df[df['split'] == 'train'])} | Reals: {len(df[(df['split'] == 'train') & (df['label'] == 0)])} | Fakes: {len(df[(df['split'] == 'train') & (df['label'] == 1)])}")
    print(f"Val samples:   {len(df[df['split'] == 'val'])} | Reals: {len(df[(df['split'] == 'val') & (df['label'] == 0)])} | Fakes: {len(df[(df['split'] == 'val') & (df['label'] == 1)])}")
    print(f"Test samples:  {len(df[df['split'] == 'test'])} | Reals: {len(df[(df['split'] == 'test') & (df['label'] == 0)])} | Fakes: {len(df[(df['split'] == 'test') & (df['label'] == 1)])}")
    print(f"Train & Val identity overlap: {len(tr_docs & val_docs)}")
    print(f"Train & Test identity overlap: {len(tr_docs & te_docs)}")
    print(f"Val & Test identity overlap:   {len(val_docs & te_docs)}")
    
    assert len(tr_docs & val_docs) == 0, "FATAL: Train/Val identity leakage detected!"
    assert len(tr_docs & te_docs) == 0, "FATAL: Train/Test identity leakage detected!"
    assert len(val_docs & te_docs) == 0, "FATAL: Val/Test identity leakage detected!"
    print("VERIFICATION SUCCESSFUL: 100% Identity-Disjoint guarantee confirmed.\n")
    
    return df


if __name__ == "__main__":
    DATASET_DIR = Path("E:/DocverifyAi/Dataset")
    OUTPUT_DIR = Path("E:/DocverifyAi/outputs/dataset_manifest")
    build_manifest(DATASET_DIR, OUTPUT_DIR, seed=42)
