import os
import sys
import json
import argparse
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd

# =========================
# CONFIG & CONSTANTS
# =========================
CSV_BACKGROUND_CLASS = "No finding"

OFFICIAL_14_CLASS_NAMES = [
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Consolidation", "ILD", "Infiltration", "Lung Opacity",
    "Nodule/Mass", "Other lesion", "Pleural effusion", "Pleural thickening",
    "Pneumothorax", "Pulmonary fibrosis",
]

DROP_8_CLASS_NAMES = [
    "Clavicle fracture", "Edema", "Emphysema", "Enlarged PA",
    "Lung cavity", "Lung cyst", "Mediastinal shift", "Rib fracture",
]

ALL_22_CLASS_NAMES = sorted(OFFICIAL_14_CLASS_NAMES + DROP_8_CLASS_NAMES)


def is_background_class(class_name):
    normalized = str(class_name).strip().lower().replace("_", " ")
    return normalized in {"no finding", "background", "negative"}

# =========================
# DATA LOADING & MERGING
# =========================
def load_and_merge_data(ref_csv: Path, json_train: Path, json_val: Path, class_mode="14", skip_box_thr=0.0):
    # 1. Load Reference CSV to get the ground truth IDs
    print(f"Lấy danh sách ID gốc từ: {ref_csv.name}")
    df_ref = pd.read_csv(ref_csv)
    expected_ids = set(df_ref["image_id"].astype(str).str.strip().unique())
    print(f" -> Có tổng cộng {len(expected_ids)} image_ids cần tìm.")

    # 2. Load JSONs
    print(f"Đọc tọa độ Bounding Boxes từ: {json_train.name} & {json_val.name}")
    with open(json_train, 'r') as f: data_train = json.load(f)
    with open(json_val, 'r') as f:   data_val = json.load(f)

    # Merge categories
    cat_id_to_name = {c['id']: c['name'] for c in data_train.get('categories', [])}
    for c in data_val.get('categories', []):
        cat_id_to_name[c['id']] = c['name']

    merged_images = data_train.get('images', []) + data_val.get('images', [])
    merged_anns = data_train.get('annotations', []) + data_val.get('annotations', [])

    # Map JSON's internal integer IDs to the actual hash string ID (from file_name)
    int_id_to_str_id = {}
    json_str_ids = set()
    images_meta = {}

    for img in merged_images:
        # file_name is usually "images/train/abc.jpg"
        str_id = Path(img["file_name"]).stem.strip()
        int_id_to_str_id[img["id"]] = str_id
        json_str_ids.add(str_id)
        
        if str_id in expected_ids:
            images_meta[str_id] = {
                "file_name": img["file_name"],
                "width": img["width"],
                "height": img["height"]
            }

    # 3. YELL IF MISSING IDs!
    missing_ids = expected_ids - json_str_ids
    if missing_ids:
        print("\n" + "🔥" * 30)
        print("KHÔNG THỂ CHẤP NHẬN ĐƯỢC! THIẾU IMAGE ID TRONG FILE JSON!")
        print(f"Số lượng ID bị thiếu: {len(missing_ids)}")
        print(f"Một vài ID bị thiếu: {list(missing_ids)[:5]}")
        print("🔥" * 30 + "\n")
        sys.exit(1)
    
    print("✅ TẤT CẢ CÁC ID TRONG CSV ĐỀU ĐÃ CÓ MẶT TRONG FILE JSON!")

    # 4. Filter JSON to match the exact subset from CSV
    rows = []
    for ann in merged_anns:
        str_id = int_id_to_str_id.get(ann["image_id"])
        if str_id not in expected_ids:
            continue
            
        c_name = cat_id_to_name.get(ann["category_id"], "Unknown")
        
        # Merge rare classes if mode is 14
        if class_mode == "14" and c_name in DROP_8_CLASS_NAMES:
            c_name = "Other lesion"
            
        x, y, w, h = ann["bbox"]
        rows.append({
            "sample_id": str_id,
            "class_name": c_name,
            "x_min": float(x),
            "y_min": float(y),
            "x_max": float(x + w),
            "y_max": float(y + h),
            "box_score": 1.0
        })

    df = pd.DataFrame(rows)
    df["is_background"] = df["class_name"].map(is_background_class)
    
    ann_df = df[~df["is_background"]].copy()
    ann_df = ann_df.dropna(subset=["x_min", "y_min", "x_max", "y_max"]).copy()
    ann_df = ann_df[
        (ann_df["x_max"] > ann_df["x_min"]) & 
        (ann_df["y_max"] > ann_df["y_min"]) &
        (ann_df["box_score"] >= skip_box_thr)
    ].copy()
    
    class_names = sorted(ann_df["class_name"].unique().tolist())
    print(f"Trích xuất thành công: {len(class_names)} Abnormal Classes. Số lượng BBoxes gốc: {len(ann_df)}")
    
    return ann_df, list(expected_ids), class_names, images_meta

# =========================
# WBF LOGIC
# =========================
def compute_iou_xyxy(box_a, box_b):
    x1, y1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    x2, y2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0: return 0.0
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def weighted_fuse_group(group_df, iou_thr=0.5):
    if len(group_df) <= 1:
        return group_df.copy()

    work = group_df.copy()
    work["_score_sort"] = pd.to_numeric(work["box_score"], errors="coerce").fillna(1.0)
    work = work.sort_values("_score_sort", ascending=False).reset_index(drop=True)

    clusters = []
    for idx, row in work.iterrows():
        box = row[["x_min", "y_min", "x_max", "y_max"]].to_numpy(dtype=float)
        score = float(row.get("box_score", 1.0))
        best_cluster, best_iou = None, 0.0

        for c_idx, cluster in enumerate(clusters):
            iou = compute_iou_xyxy(box, cluster["box"])
            if iou >= iou_thr and iou > best_iou:
                best_iou = iou
                best_cluster = c_idx

        if best_cluster is None:
            clusters.append({"indices": [idx], "boxes": [box], "scores": [score], "box": box.copy()})
        else:
            c = clusters[best_cluster]
            c["indices"].append(idx)
            c["boxes"].append(box)
            c["scores"].append(score)
            w = np.maximum(np.asarray(c["scores"], dtype=float), 1e-6)
            c["box"] = np.average(np.asarray(c["boxes"], dtype=float), axis=0, weights=w)

    rows = []
    for c in clusters:
        fused = work.loc[c["indices"]].iloc[0].copy()
        w = np.maximum(np.asarray(c["scores"], dtype=float), 1e-6)
        fused_box = np.average(np.asarray(c["boxes"], dtype=float), axis=0, weights=w)
        fused["x_min"], fused["y_min"] = float(fused_box[0]), float(fused_box[1])
        fused["x_max"], fused["y_max"] = float(fused_box[2]), float(fused_box[3])
        fused["box_score"] = float(np.mean(c["scores"]))
        rows.append(fused.drop(labels=["_score_sort"], errors="ignore"))

    return pd.DataFrame(rows)

def apply_wbf(ann_df, iou_thr=0.5):
    fused_groups = []
    grouped = ann_df.groupby(["sample_id", "class_name"], sort=False, group_keys=False)
    for _, group in grouped:
        fused_groups.append(weighted_fuse_group(group, iou_thr))
    if not fused_groups:
        return ann_df.copy()
    
    fused_df = pd.concat(fused_groups, ignore_index=True).reset_index(drop=True)
    print(f"WBF: Gom hộp rút gọn từ {len(ann_df)} -> {len(fused_df)} BBoxes (IoU: {iou_thr})")
    return fused_df

# =========================
# SPLITTING LOGIC
# =========================
def compute_target_sizes(n_items, split_ratios):
    raw = {k: n_items * v for k, v in split_ratios.items()}
    sizes = {k: int(np.floor(v)) for k, v in raw.items()}
    remaining = n_items - sum(sizes.values())
    order = sorted(split_ratios.keys(), key=lambda k: (raw[k] - sizes[k], k), reverse=True)
    for k in order[:remaining]:
        sizes[k] += 1
    return sizes

def build_multilabel_stratified_split(ann_df, sample_ids, split_ratios, seed=42):
    rng = np.random.default_rng(seed)
    
    grouped = ann_df.groupby("sample_id")["class_name"].apply(lambda x: sorted(set(x))).to_dict()
    image_class_sets = {s: grouped.get(s, [CSV_BACKGROUND_CLASS]) for s in sample_ids}
    
    split_names = list(split_ratios.keys())
    target_sizes = compute_target_sizes(len(sample_ids), split_ratios)
    
    class_total = Counter(c for classes in image_class_sets.values() for c in classes)
    target_class_counts = {s: {c: class_total[c] * split_ratios[s] for c in class_total} for s in split_names}
    
    current_sizes = {s: 0 for s in split_names}
    current_class_counts = {s: Counter() for s in split_names}
    split_sets = {s: set() for s in split_names}
    
    ordered_ids = sorted(
        sample_ids,
        key=lambda s: (
            -sum(1.0 / max(class_total[c], 1) for c in image_class_sets[s]),
            -len(image_class_sets[s]),
            rng.random(),
        )
    )
    
    for s_id in ordered_ids:
        classes = image_class_sets[s_id]
        cands = [s for s in split_names if current_sizes[s] < target_sizes[s]] or split_names
        best_split, best_score = None, None
        
        for s in cands:
            size_def = (target_sizes[s] - current_sizes[s]) / max(target_sizes[s], 1)
            class_def, over_pen = 0.0, 0.0
            for c in classes:
                target = max(target_class_counts[s][c], 1e-9)
                before = current_class_counts[s][c]
                after = before + 1
                class_def += max(target - before, 0.0) / target
                over_pen += max(after - target, 0.0) / target
            
            score = size_def + class_def - over_pen + rng.random() * 1e-6
            if best_score is None or score > best_score:
                best_score, best_split = score, s
                
        split_sets[best_split].add(s_id)
        current_sizes[best_split] += 1
        for c in classes:
            current_class_counts[best_split][c] += 1
            
    return split_sets

# =========================
# COCO EXPORT
# =========================
def export_coco_json(ann_df, split_sample_ids, split_name, out_dir, images_meta, class_mode):
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"instances_{split_name}.json"
    
    class_list = OFFICIAL_14_CLASS_NAMES if class_mode == "14" else ALL_22_CLASS_NAMES
    cat_to_id = {name: i+1 for i, name in enumerate(class_list)}
    
    coco = {
        "info": {"description": f"VinBigData - {split_name} split (WBF + Stratified)"},
        "images": [],
        "annotations": [],
        "categories": [{"id": v, "name": k} for k, v in cat_to_id.items()]
    }
    
    ann_id = 1
    sample_to_int_id = {}
    
    split_sample_ids = sorted(list(split_sample_ids))
    for i, s_id in enumerate(split_sample_ids, start=1):
        sample_to_int_id[s_id] = i
        meta = images_meta.get(s_id, {"file_name": str(s_id)+".jpg", "width": 512, "height": 512})
        coco["images"].append({
            "id": i,
            "file_name": meta["file_name"],
            "width": meta["width"],
            "height": meta["height"]
        })
        
    split_df = ann_df[ann_df["sample_id"].isin(split_sample_ids)]
    for _, row in split_df.iterrows():
        c_name = str(row["class_name"]).strip()
        if c_name not in cat_to_id:
            continue
            
        x1, y1 = float(row["x_min"]), float(row["y_min"])
        x2, y2 = float(row["x_max"]), float(row["y_max"])
        w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
        
        coco["annotations"].append({
            "id": ann_id,
            "image_id": sample_to_int_id[row["sample_id"]],
            "category_id": cat_to_id[c_name],
            "bbox": [x1, y1, w, h],
            "area": w * h,
            "iscrowd": 0
        })
        ann_id += 1
        
    with open(json_path, "w") as f:
        json.dump(coco, f, indent=2)
    print(f"✅ Xuất mâm {split_name.upper()}: {len(coco['images'])} ảnh, {len(coco['annotations'])} boxes -> {json_path}")

# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser("Faster R-CNN: WBF & Stratified Split on Adaptive Preprocessed JSONs")
    parser.add_argument("--ref_csv", type=str, required=True, help="Path to original CSV for exact ID matching")
    parser.add_argument("--in_train_json", type=str, required=True, help="Input adaptive train JSON")
    parser.add_argument("--in_val_json", type=str, required=True, help="Input adaptive val JSON")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for new COCO JSONs")
    parser.add_argument("--class_mode", type=str, default="14", choices=["14", "22"], help="14 or 22 class mode")
    parser.add_argument("--iou_thr", type=float, default=0.5, help="WBF IoU threshold")
    parser.add_argument("--skip_box_thr", type=float, default=0.0, help="Minimum score to keep box")
    parser.add_argument("--seed", type=int, default=42, help="Seed for splitting")
    args = parser.parse_args()
    
    ref_csv = Path(args.ref_csv)
    in_train = Path(args.in_train_json)
    in_val = Path(args.in_val_json)
    out_dir = Path(args.out_dir)
    
    ann_df, sample_ids, class_names, images_meta = load_and_merge_data(
        ref_csv=ref_csv, 
        json_train=in_train,
        json_val=in_val,
        class_mode=args.class_mode, 
        skip_box_thr=args.skip_box_thr
    )
    
    # ÉP WBF
    fused_df = apply_wbf(ann_df, iou_thr=args.iou_thr)
    
    # CHIA BÀI 70 / 10 / 20
    split_ratios = {"train": 0.70, "val": 0.10, "test": 0.20}
    print(f"\nPhân bổ lại Data (Stratified) theo tỉ lệ: {split_ratios} (Seed: {args.seed})")
    split_sets = build_multilabel_stratified_split(fused_df, sample_ids, split_ratios, seed=args.seed)
    
    # Xuất COCO JSONs
    print("\n[ĐANG XUẤT RA FILE JSON]")
    for split_name, ids in split_sets.items():
        export_coco_json(
            ann_df=fused_df, 
            split_sample_ids=ids, 
            split_name=split_name, 
            out_dir=out_dir, 
            images_meta=images_meta,
            class_mode=args.class_mode
        )

if __name__ == "__main__":
    main()
