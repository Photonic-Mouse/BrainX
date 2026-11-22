import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from collections import Counter, defaultdict

# Root directory
REPO_ROOT = Path(".").resolve()
DATASET_DIR = REPO_ROOT / "Dataset"
SPLITS_CSV = DATASET_DIR / "splits" / "splits.csv"
AUDIT_DIR = REPO_ROOT / "results" / "audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

print("=== STARTING STAGE 1 AUDIT SCRIPT ===")
print(f"Repo Root: {REPO_ROOT}")
print(f"Dataset Dir: {DATASET_DIR}")
print(f"Splits CSV: {SPLITS_CSV}")

# ----------------------------------------------------
# 1. ACTUAL IMAGES ON DISK vs EXPECTED
# ----------------------------------------------------
expected_class_counts = {
    "Mild_Demented": 896,
    "Moderate_Demented": 64,
    "Non_Demented": 3200,
    "Very_Mild_Demented": 2240
}
valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

disk_images = {}
disk_counts = {}
for class_folder, exp_count in expected_class_counts.items():
    folder_path = DATASET_DIR / class_folder
    if not folder_path.exists():
        print(f"ERROR: Missing folder {folder_path}")
        continue
    files = [f for f in folder_path.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions]
    disk_images[class_folder] = files
    disk_counts[class_folder] = len(files)

total_disk_images = sum(disk_counts.values())
print(f"Disk Counts: {disk_counts}, Total: {total_disk_images}")

# ----------------------------------------------------
# 2. READ SPLITS.CSV AND CHECK ROW COUNTS BY SPLIT & CLASS
# ----------------------------------------------------
df = pd.read_csv(SPLITS_CSV)
print(f"Total rows in manifest: {len(df)}")

expected_split_counts = {
    "train": {"Mild_Demented": 627, "Moderate_Demented": 45, "Non_Demented": 2240, "Very_Mild_Demented": 1568, "total": 4480},
    "validation": {"Mild_Demented": 134, "Moderate_Demented": 10, "Non_Demented": 480, "Very_Mild_Demented": 336, "total": 960},
    "test": {"Mild_Demented": 135, "Moderate_Demented": 9, "Non_Demented": 480, "Very_Mild_Demented": 336, "total": 960}
}

split_audit = {}
for s, exp_dict in expected_split_counts.items():
    sub_df = df[df["split"] == s]
    counts = sub_df["class_name"].value_counts().to_dict()
    total = len(sub_df)
    match_total = (total == exp_dict["total"])
    class_matches = {}
    for c, c_exp in exp_dict.items():
        if c == "total": continue
        c_actual = counts.get(c, 0)
        class_matches[c] = (c_actual == c_exp, c_actual, c_exp)
    split_audit[s] = {
        "actual_total": total,
        "expected_total": exp_dict["total"],
        "total_match": match_total,
        "class_details": class_matches
    }

print("Split count audit results:")
for s, res in split_audit.items():
    print(f"  {s}: Total {res['actual_total']}/{res['expected_total']} (Match: {res['total_match']})")
    for c, (m, act, exp) in res["class_details"].items():
        print(f"    - {c}: Actual {act}, Expected {exp}, Match: {m}")

# ----------------------------------------------------
# 3. VERIFY FILENAME, CLASS_NAME, LABEL, AND RESOLUTION TO DISK
# ----------------------------------------------------
label_map = {
    "Mild_Demented": 0,
    "Moderate_Demented": 1,
    "Non_Demented": 2,
    "Very_Mild_Demented": 3
}

resolution_errors = []
manifest_inventory = []

for idx, row in df.iterrows():
    raw_path = str(row["image_path"])
    c_name = row["class_name"]
    lbl = int(row["label"])
    s_name = row["split"]
    
    # Extract filename
    fname = Path(raw_path.replace("\\", "/")).name
    
    # Expected relative path
    rel_path = Path("Dataset") / c_name / fname
    full_path = REPO_ROOT / rel_path
    
    # Check label agreement
    if label_map.get(c_name) != lbl:
        resolution_errors.append(f"Row {idx}: Label mismatch (class_name={c_name}, label={lbl})")
    
    # Check disk existence
    exists = full_path.exists()
    if not exists:
        resolution_errors.append(f"Row {idx}: File not found at {full_path}")
        
    manifest_inventory.append({
        "row_idx": idx,
        "filename": fname,
        "class_name": c_name,
        "label": lbl,
        "split": s_name,
        "relative_path": str(rel_path).replace("\\", "/"),
        "exists_on_disk": exists
    })

print(f"Total resolution / agreement errors: {len(resolution_errors)}")

# ----------------------------------------------------
# 4. COMPUTE STABLE IDENTITY BEFORE/AFTER CHECKSUM
# ----------------------------------------------------
def get_filename(p):
    return Path(str(p).replace("\\", "/")).name

# Stable identity is hash of (filename, class_name, label, split) in original row order
stable_rows_before = [
    f"{get_filename(row['image_path'])}|{row['class_name']}|{row['label']}|{row['split']}"
    for _, row in df.iterrows()
]
stable_hash_before = hashlib.sha256("\n".join(stable_rows_before).encode("utf-8")).hexdigest()
print(f"Stable identity SHA-256 before repair: {stable_hash_before}")

# ----------------------------------------------------
# 5. IMAGE PROPERTIES, DIMENSIONS, CORRUPTION, INTENSITIES, WATERMARKS
# ----------------------------------------------------
print("Auditing image physical properties across all 6,400 images...")
corrupted_files = []
dimensions_counter = Counter()
modes_counter = Counter()
intensity_stats = defaultdict(list)
black_border_ratios = []

# Difference hash helper for near-duplicate analysis
def dhash(image, hash_size=8):
    # Grayscale, resize to (hash_size + 1, hash_size)
    img = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.array(img, dtype=np.float32)
    diff = pixels[:, 1:] > pixels[:, :-1]
    return hashlib.sha256(diff.tobytes()).hexdigest(), diff

file_hashes = {}
dhashes = defaultdict(list)
file_metadata = []

for item in manifest_inventory:
    fpath = REPO_ROOT / item["relative_path"]
    try:
        with Image.open(fpath) as img:
            img.verify()
        
        with Image.open(fpath) as img:
            w, h = img.size
            mode = img.mode
            dimensions_counter[(w, h)] += 1
            modes_counter[mode] += 1
            
            # Convert to numpy for intensity analysis
            arr = np.array(img.convert("L"))
            min_val = int(arr.min())
            max_val = int(arr.max())
            mean_val = float(arr.mean())
            std_val = float(arr.std())
            
            # Black border check: fraction of zero pixels on 4 borders (5 pixels depth)
            border_mask = np.zeros_like(arr, dtype=bool)
            border_mask[:5, :] = True
            border_mask[-5:, :] = True
            border_mask[:, :5] = True
            border_mask[:, -5:] = True
            border_zeros = np.mean(arr[border_mask] == 0)
            black_border_ratios.append(border_zeros)
            
            # File SHA-256
            with open(fpath, "rb") as f:
                fbytes = f.read()
                f_sha256 = hashlib.sha256(fbytes).hexdigest()
            
            # Perceptual hash (dHash)
            dh_str, _ = dhash(img)
            
            file_hashes[item["relative_path"]] = f_sha256
            dhashes[dh_str].append(item)
            
            intensity_stats[item["class_name"]].append({
                "min": min_val,
                "max": max_val,
                "mean": mean_val,
                "std": std_val
            })
            
            file_metadata.append({
                **item,
                "sha256": f_sha256,
                "dhash": dh_str,
                "width": w,
                "height": h,
                "mode": mode,
                "min_intensity": min_val,
                "max_intensity": max_val,
                "mean_intensity": mean_val,
                "std_intensity": std_val,
                "border_zero_ratio": float(border_zeros)
            })
            
    except Exception as e:
        corrupted_files.append((item["relative_path"], str(e)))

print(f"Dimensions observed: {dict(dimensions_counter)}")
print(f"Modes observed: {dict(modes_counter)}")
print(f"Corrupted files: {len(corrupted_files)}")
print(f"Average border zero ratio: {np.mean(black_border_ratios):.4f}")

# ----------------------------------------------------
# 6. EXACT SHA-256 DUPLICATES & CROSS-SPLIT CONTAMINATION
# ----------------------------------------------------
sha_groups = defaultdict(list)
for item in file_metadata:
    sha_groups[item["sha256"]].append(item)

exact_duplicate_groups = {h: grp for h, grp in sha_groups.items() if len(grp) > 1}
print(f"Exact SHA-256 duplicate groups: {len(exact_duplicate_groups)}")
total_exact_duplicate_files = sum(len(grp) for grp in exact_duplicate_groups.values())
print(f"Total files involved in exact duplicate groups: {total_exact_duplicate_files}")

# Check cross-split exact duplicates
cross_split_exact_duplicates = []
for h, grp in exact_duplicate_groups.items():
    splits_in_grp = set(x["split"] for x in grp)
    classes_in_grp = set(x["class_name"] for x in grp)
    if len(splits_in_grp) > 1:
        cross_split_exact_duplicates.append({
            "sha256": h,
            "classes": list(classes_in_grp),
            "splits": list(splits_in_grp),
            "members": [(x["filename"], x["class_name"], x["split"]) for x in grp]
        })

print(f"Cross-split exact duplicate groups: {len(cross_split_exact_duplicates)}")

# Perceptual near-duplicates
perceptual_dup_groups = {dh: grp for dh, grp in dhashes.items() if len(grp) > 1}
cross_split_perceptual = []
for dh, grp in perceptual_dup_groups.items():
    splits_in_grp = set(x["split"] for x in grp)
    if len(splits_in_grp) > 1:
        cross_split_perceptual.append({
            "dhash": dh,
            "splits": list(splits_in_grp),
            "members": [(x["filename"], x["class_name"], x["split"]) for x in grp]
        })
print(f"Cross-split perceptual near-duplicate groups: {len(cross_split_perceptual)}")

# ----------------------------------------------------
# 7. METADATA & PARTICIPANT / SCAN IDENTIFIER AUDIT
# ----------------------------------------------------
# Check filename patterns
filenames = [item["filename"] for item in manifest_inventory]
prefix_counts = Counter(f.split("_")[0] for f in filenames)
print(f"Filename prefix counts: {dict(prefix_counts)}")

# Check for OASIS patterns or patient numbers
# Pattern: mild_xxx.jpg, moderate_xxx.jpg, non_xxx.jpg, verymild_xxx.jpg
# Are there any EXIF metadata or header tags?
exif_found_count = 0
for item in manifest_inventory[:100]:
    fpath = REPO_ROOT / item["relative_path"]
    with Image.open(fpath) as img:
        exif = img.getexif()
        if exif:
            exif_found_count += 1
print(f"EXIF tags found in sample of 100 images: {exif_found_count}")

# ----------------------------------------------------
# 8. VERIFY SAVED ARITHMETIC IN RESULTS FILES
# ----------------------------------------------------
results_verification = {}
for m_name, folder in [("baseline", "baseline"), ("clahe", "clahe"), ("he", "he")]:
    report_file = REPO_ROOT / "results" / folder / "classification_report.txt"
    preds_file = REPO_ROOT / "results" / folder / "test_predictions.csv"
    
    if report_file.exists() and preds_file.exists():
        p_df = pd.read_csv(preds_file)
        acc = (p_df["actual"] == p_df["predicted"]).mean()
        total_p = len(p_df)
        with open(report_file, "r") as rf:
            txt = rf.read()
        results_verification[m_name] = {
            "predictions_count": total_p,
            "calculated_accuracy": float(acc),
            "report_snippet": txt.splitlines()[:2]
        }
print("Results verification:", results_verification)

# ----------------------------------------------------
# 9. EXECUTE SAFE PATH REPAIR ON SPLITS.CSV
# ----------------------------------------------------
print("Performing path-only repair on splits.csv...")
# Update only image_path column to relative path: "Dataset/<class_name>/<filename>"
df_repaired = df.copy()
repaired_paths = []
for idx, row in df_repaired.iterrows():
    c_name = row["class_name"]
    fname = Path(str(row["image_path"]).replace("\\", "/")).name
    rel_p = f"Dataset/{c_name}/{fname}"
    repaired_paths.append(rel_p)

df_repaired["image_path"] = repaired_paths

# Verify stable identity AFTER repair
stable_rows_after = [
    f"{get_filename(row['image_path'])}|{row['class_name']}|{row['label']}|{row['split']}"
    for _, row in df_repaired.iterrows()
]
stable_hash_after = hashlib.sha256("\n".join(stable_rows_after).encode("utf-8")).hexdigest()
print(f"Stable identity SHA-256 after repair:  {stable_hash_after}")

assert stable_hash_before == stable_hash_after, "FATAL: Stable identity checksum mismatch!"
print("-> CHECKSUM VERIFIED: Exactly identical stable identity!")

# Confirm every repaired relative path exists on disk
all_exist = all((REPO_ROOT / p).exists() for p in repaired_paths)
assert all_exist, "FATAL: Some repaired paths do not exist on disk!"
print("-> ALL 6,400 REPAIRED PATHS EXIST ON DISK!")

# Save repaired splits.csv
df_repaired.to_csv(SPLITS_CSV, index=False)
print(f"Repaired splits.csv saved to {SPLITS_CSV}")

# ----------------------------------------------------
# 10. SAVE ARTIFACTS
# ----------------------------------------------------
# File inventory
inventory_csv = AUDIT_DIR / "file_inventory.csv"
pd.DataFrame(file_metadata).to_csv(inventory_csv, index=False)
print(f"Saved inventory to {inventory_csv}")

# Duplicate groups
dup_json = AUDIT_DIR / "duplicate_groups.json"
dup_data = {
    "total_exact_duplicate_groups": len(exact_duplicate_groups),
    "total_files_in_exact_duplicates": total_exact_duplicate_files,
    "cross_split_exact_duplicates_count": len(cross_split_exact_duplicates),
    "cross_split_exact_duplicates": cross_split_exact_duplicates,
    "cross_split_perceptual_duplicates_count": len(cross_split_perceptual),
    "cross_split_perceptual_duplicates": cross_split_perceptual
}
with open(dup_json, "w") as f:
    json.dump(dup_data, f, indent=2)
print(f"Saved duplicate groups to {dup_json}")

# Path repair summary
repair_json = AUDIT_DIR / "path_repair_summary.json"
repair_data = {
    "manifest_path": str(SPLITS_CSV),
    "total_rows": len(df_repaired),
    "stable_hash_before": stable_hash_before,
    "stable_hash_after": stable_hash_after,
    "hashes_match": (stable_hash_before == stable_hash_after),
    "sample_before": df.iloc[0]["image_path"],
    "sample_after": df_repaired.iloc[0]["image_path"],
    "all_files_exist_on_disk": True
}
with open(repair_json, "w") as f:
    json.dump(repair_data, f, indent=2)
print(f"Saved path repair summary to {repair_json}")

print("=== STAGE 1 AUDIT SCRIPT COMPLETED SUCCESSFULLY ===")
