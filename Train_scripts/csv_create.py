from pathlib import Path
import pandas as pd
import numpy as np

# =========================
# PATHS
# =========================
TRAIN_ANN_CSV = Path("/kaggle/input/datasets/benxelua/correct-label/annotations/annotations_train.csv")
TEST_ANN_CSV  = Path("/kaggle/input/datasets/benxelua/correct-label/annotations/annotations_test.csv")

OUT_DIR = Path("/kaggle/working/vindr_merged_all_1000nf")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "annotations_all_merged_other_1000nf.csv"

SEED = 42
N_NO_FINDING_TOTAL = 1000

RARE_8_CLASS_NAMES = [
    "Clavicle fracture",
    "Edema",
    "Emphysema",
    "Enlarged PA",
    "Lung cavity",
    "Lung cyst",
    "Mediastinal shift",
    "Rib fracture",
]

MERGED_CLASS_NAME = "Other lesion"
NO_FINDING_NAME = "No finding"

# =========================
# LOAD + MERGE TRAIN TEST
# =========================
df_train = pd.read_csv(TRAIN_ANN_CSV)
df_test  = pd.read_csv(TEST_ANN_CSV)

df_train["original_split"] = "train"
df_test["original_split"] = "test"

df_all = pd.concat([df_train, df_test], ignore_index=True)

print("Original train shape:", df_train.shape)
print("Original test shape :", df_test.shape)
print("Merged shape        :", df_all.shape)
print("Merged unique images:", df_all["image_id"].nunique())

# =========================
# STEP 1: MERGE 8 RARE CLASSES -> Other lesion
# =========================
df_all.loc[df_all["class_name"].isin(RARE_8_CLASS_NAMES), "class_name"] = MERGED_CLASS_NAME

# =========================
# STEP 2: KEEP ALL ABNORMAL + ONLY 1000 NO-FINDING-ONLY IMAGES
# =========================
def get_no_finding_only_ids(df, no_finding_name="No finding"):
    abnormal_ids = set(
        df[df["class_name"].fillna("").str.lower() != no_finding_name.lower()]["image_id"].unique()
    )
    all_ids = set(df["image_id"].unique())
    no_finding_only_ids = sorted(all_ids - abnormal_ids)
    return no_finding_only_ids, abnormal_ids

nf_only_ids, abnormal_ids = get_no_finding_only_ids(df_all, NO_FINDING_NAME)

print("\nBefore filtering:")
print("Abnormal images       :", len(abnormal_ids))
print("No-finding-only images:", len(nf_only_ids))

rng = np.random.default_rng(SEED)

if len(nf_only_ids) > N_NO_FINDING_TOTAL:
    kept_nf_ids = set(rng.choice(nf_only_ids, size=N_NO_FINDING_TOTAL, replace=False).tolist())
else:
    kept_nf_ids = set(nf_only_ids)

kept_ids = set(abnormal_ids) | kept_nf_ids
df_final = df_all[df_all["image_id"].isin(kept_ids)].copy()

# =========================
# STEP 3: DROP NO FINDING ROWS FROM ABNORMAL IMAGES
# =========================
def drop_no_finding_rows_from_abnormal_images(df, no_finding_name="No finding"):
    df = df.copy()

    abnormal_ids = set(
        df[df["class_name"].fillna("").str.lower() != no_finding_name.lower()]["image_id"].unique()
    )

    mask_drop = (
        df["image_id"].isin(abnormal_ids) &
        (df["class_name"].fillna("").str.lower() == no_finding_name.lower())
    )

    return df[~mask_drop].copy()

df_final = drop_no_finding_rows_from_abnormal_images(df_final, NO_FINDING_NAME)

# =========================
# STEP 4: SUMMARY
# =========================
def image_level_summary(df, name, no_finding_name="No finding"):
    abnormal_ids = set(
        df[df["class_name"].fillna("").str.lower() != no_finding_name.lower()]["image_id"].unique()
    )
    all_ids = set(df["image_id"].unique())
    nf_only_ids = all_ids - abnormal_ids

    print(f"\n{name} image-level summary")
    print("Total images      :", len(all_ids))
    print("Abnormal images   :", len(abnormal_ids))
    print("No-finding-only   :", len(nf_only_ids))

print("\nAfter filtering:")
print("Final rows        :", len(df_final))
print("Final unique imgs :", df_final["image_id"].nunique())

print("\nClass counts:")
print(df_final["class_name"].value_counts(dropna=False))

image_level_summary(df_final, "FINAL", NO_FINDING_NAME)

print("\nOriginal split counts after filtering:")
print(df_final[["image_id", "original_split"]].drop_duplicates()["original_split"].value_counts())

# =========================
# STEP 5: SAVE ONE CSV
# =========================
df_final.to_csv(OUT_CSV, index=False)

print("\nSaved:")
print(OUT_CSV)