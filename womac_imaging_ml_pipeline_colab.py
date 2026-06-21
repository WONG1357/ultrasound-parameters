# %% [markdown]
# # WOMAC Imaging ML Pipeline
#
# This notebook fixes robust patient ID matching, duplicate-key merge inflation,
# MRI wide-to-long conversion, leakage-free ultrasound feature processing,
# patient-safe train/test splitting, model evaluation, baselines, and
# output saving.
#
# It can run in Google Colab or a local Jupyter environment. Configure paths in
# the setup cell below or with environment variables.

# %%
try:
    from google.colab import drive

    drive.mount("/content/drive")
except Exception:
    print("Not running in Google Colab, or Google Drive mount is unavailable. Continuing without Drive mount.")

# %%
import json
import os
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import KernelPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    f1_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TEST_SIZE = 0.25
N_TREES = 400

BASE_DIR = Path(os.environ.get("WOMAC_BASE_DIR", ".")).expanduser()
US_PATH = Path(
    os.environ.get("WOMAC_US_PATH", str(BASE_DIR / "data/processed/ultrasound_radiomics_features_updated.xlsx"))
).expanduser()
GT_PATH = Path(
    os.environ.get("WOMAC_GT_PATH", str(BASE_DIR / "data/raw/ground_truth_with_bmi_womac.xlsx"))
).expanduser()
MRI_PATH = Path(
    os.environ.get("WOMAC_MRI_PATH", str(BASE_DIR / "data/raw/mri_muscle_features.xlsx"))
).expanduser()
OUTPUT_DIR = Path(os.environ.get("WOMAC_OUTPUT_DIR", str(BASE_DIR / "outputs"))).expanduser()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SHOW_REQUIRED_OUTPUTS_ONLY = True
SAVE_CONFUSION_MATRIX_IMAGES = False
REQUIRED_OUTPUT_FILENAMES = {
    "model_results.csv",
    "baseline_results.csv",
    "classification_reports.csv",
    "patient_safe_split_diagnostics.csv",
    "feature_transform_diagnostics.csv",
    "selectedall_strict_test_only_kpca_mri_correlations.csv",
    "selectedall_cross_fitted_kpca_mri_correlations.csv",
    "selectedall_cross_fitted_kpca_scores.csv",
    "mri_matching_diagnostics.csv",
}

print("Output directory:", OUTPUT_DIR)
for p in [US_PATH, GT_PATH, MRI_PATH]:
    print(p, "exists:", p.exists())

# %% [markdown]
# ## Utility functions

# %%
def clean_raw_id(x):
    """Uppercase alphanumeric cleaned ID, preserving meaningful letters and digits."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().upper()
    if s.endswith(".0"):
        s = s[:-2]
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s if s else np.nan


def digits_preserve_zeros(x):
    """All digits in the identifier, preserving leading zeros."""
    s = clean_raw_id(x)
    if pd.isna(s):
        return np.nan
    d = "".join(re.findall(r"\d+", s))
    return d if d else np.nan


def digits_integer_string(x):
    """All digits converted through int, removing leading zeros."""
    d = digits_preserve_zeros(x)
    if pd.isna(d):
        return np.nan
    try:
        return str(int(d))
    except Exception:
        return np.nan


def ug_normalized(x, width=4):
    """Normalize any usable ID to UG####, e.g. 38, 0038, ug0038 -> UG0038."""
    d = digits_integer_string(x)
    if pd.isna(d):
        return np.nan
    return f"UG{int(d):0{width}d}"


ID_STRATEGIES = {
    "id_raw_cleaned": clean_raw_id,
    "id_digits_preserve_zeros": digits_preserve_zeros,
    "id_digits_integer": digits_integer_string,
    "id_ug_normalized": ug_normalized,
}


def add_id_formats(df, source_col, prefix):
    out = df.copy()
    for name, fn in ID_STRATEGIES.items():
        out[f"{prefix}_{name}"] = out[source_col].map(fn)
    return out


def normalize_side(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().upper()
    if s in {"L", "LEFT", "LT"}:
        return "Left"
    if s in {"R", "RIGHT", "RT"}:
        return "Right"
    if "LEFT" in s or re.search(r"(^|[^A-Z])L([^A-Z]|$)", s):
        return "Left"
    if "RIGHT" in s or re.search(r"(^|[^A-Z])R([^A-Z]|$)", s):
        return "Right"
    return np.nan


def first_existing(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    raise KeyError(f"None of these columns exist: {candidates}")


def save_df(df, filename):
    path = OUTPUT_DIR / filename
    if SHOW_REQUIRED_OUTPUTS_ONLY and filename not in REQUIRED_OUTPUT_FILENAMES:
        return path
    df.to_csv(path, index=False)
    print("Saved:", path)
    return path


def save_df_excel(df, filename):
    path = OUTPUT_DIR / filename
    if SHOW_REQUIRED_OUTPUTS_ONLY and filename not in REQUIRED_OUTPUT_FILENAMES:
        return path
    df.to_excel(path, index=False)
    print("Saved:", path)
    return path


def save_json(obj, filename):
    path = OUTPUT_DIR / filename
    if SHOW_REQUIRED_OUTPUTS_ONLY and filename not in REQUIRED_OUTPUT_FILENAMES:
        return path
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print("Saved:", path)
    return path


def cleanup_nonrequired_outputs():
    for path in OUTPUT_DIR.glob("*"):
        if path.is_file() and (
            path.name.startswith("confusion_matrix_")
            or path.name not in REQUIRED_OUTPUT_FILENAMES
        ):
            path.unlink()


cleanup_nonrequired_outputs()

# %% [markdown]
# ## Load data

# %%
us = pd.read_excel(US_PATH)
gt = pd.read_excel(GT_PATH)
mri_wide = pd.read_excel(MRI_PATH)

print("Ultrasound shape:", us.shape)
print("Ground truth shape:", gt.shape)
print("MRI wide shape:", mri_wide.shape)
print("US columns:", list(us.columns[:12]), "...")
print("GT columns:", list(gt.columns[:12]), "...")
print("MRI columns:", list(mri_wide.columns[:12]), "...")

# %% [markdown]
# ## Standardize IDs, side labels, and MRI wide-to-long

# %%
US_ID_COL = first_existing(us.columns, ["Patient_Number", "PatientID", "ID", "Patient ID"])
GT_ID_COL = first_existing(gt.columns, ["Patient_Number", "PatientID", "ID", "Patient ID"])
MRI_ID_COL = first_existing(mri_wide.columns, ["ID", "Patient_Number", "PatientID", "Patient ID"])

US_SIDE_COL = first_existing(us.columns, ["Leg_Side", "Side", "Leg", "left_right"])
GT_SIDE_COL = first_existing(gt.columns, ["Leg_Side", "Side", "Leg", "left_right"])

us = add_id_formats(us, US_ID_COL, "us")
gt = add_id_formats(gt, GT_ID_COL, "gt")
mri_wide = add_id_formats(mri_wide, MRI_ID_COL, "mri")

us["side_norm"] = us[US_SIDE_COL].map(normalize_side)
gt["side_norm"] = gt[GT_SIDE_COL].map(normalize_side)

# The supplied MRI workbook has one missing header for RF_imat_percent_R.
if "Unnamed: 24" in mri_wide.columns and "RF_imat_percent_R" not in mri_wide.columns:
    mri_wide = mri_wide.rename(columns={"Unnamed: 24": "RF_imat_percent_R"})


def mri_wide_to_long(df):
    id_cols = [MRI_ID_COL] + [f"mri_{name}" for name in ID_STRATEGIES]
    id_cols = [c for c in id_cols if c in df.columns]
    non_feature_cols = set(id_cols + ["MRI No.", "BMI"])

    rows = []
    for _, r in df.iterrows():
        for suffix, side in [("_L", "Left"), ("_R", "Right")]:
            row = {c: r[c] for c in id_cols}
            row["side_norm"] = side
            for col in df.columns:
                if col in non_feature_cols:
                    continue
                if str(col).endswith(suffix):
                    base_name = str(col)[: -len(suffix)]
                    if not base_name.startswith("RF"):
                        continue
                    feature_name = "MRI_" + base_name
                    row[feature_name] = r[col]
            rows.append(row)

    long = pd.DataFrame(rows)
    feature_cols = [c for c in long.columns if c.startswith("MRI_")]
    long = long.dropna(how="all", subset=feature_cols)
    return long, feature_cols


mri_long, mri_feature_cols = mri_wide_to_long(mri_wide)
print("MRI long shape:", mri_long.shape)
print("MRI feature count:", len(mri_feature_cols))

# %% [markdown]
# ## Duplicate key diagnostics and protected ultrasound-ground-truth merge
#
# Duplicate patient-side rows exist in both ultrasound and ground truth. A plain
# merge on patient-side can create an accidental Cartesian product. We add an
# occurrence counter within each patient-side key and merge on patient-side plus
# occurrence number.

# %%
def duplicate_key_diagnostics(df, id_col, side_col, name):
    key_counts = (
        df.groupby([id_col, side_col], dropna=False)
        .size()
        .reset_index(name="row_count")
        .sort_values("row_count", ascending=False)
    )
    diag = {
        "dataset": name,
        "rows": int(len(df)),
        "unique_patient_ids": int(df[id_col].nunique(dropna=True)),
        "unique_patient_side_keys": int(key_counts.shape[0]),
        "duplicate_patient_side_keys": int((key_counts["row_count"] > 1).sum()),
        "max_rows_per_patient_side": int(key_counts["row_count"].max()),
    }
    return diag, key_counts


merge_trials = []
merged_by_strategy = {}

for strategy_name in ID_STRATEGIES:
    us_key = f"us_{strategy_name}"
    gt_key = f"gt_{strategy_name}"

    us_tmp = us.copy()
    gt_tmp = gt.copy()
    us_tmp["_merge_id"] = us_tmp[us_key]
    gt_tmp["_merge_id"] = gt_tmp[gt_key]

    us_tmp["_key_occ"] = us_tmp.groupby(["_merge_id", "side_norm"], dropna=False).cumcount()
    gt_tmp["_key_occ"] = gt_tmp.groupby(["_merge_id", "side_norm"], dropna=False).cumcount()

    merged = us_tmp.merge(
        gt_tmp,
        on=["_merge_id", "side_norm", "_key_occ"],
        how="inner",
        suffixes=("_us", "_gt"),
        validate="one_to_one",
    )
    target_cols_found = [c for c in ["Total Score", "WOMAC Pain", "WOMAC Function"] if c in merged.columns]
    nonmissing_targets = int(merged[target_cols_found].notna().any(axis=1).sum()) if target_cols_found else 0
    merge_trials.append(
        {
            "strategy": strategy_name,
            "merged_rows_with_occurrence_counter": int(len(merged)),
            "nonmissing_any_core_target_rows": nonmissing_targets,
            "us_rows": int(len(us)),
            "gt_rows": int(len(gt)),
            "inflation_vs_us": float(len(merged) / max(len(us), 1)),
            "inflation_vs_gt": float(len(merged) / max(len(gt), 1)),
        }
    )
    merged_by_strategy[strategy_name] = merged

merge_trials_df = pd.DataFrame(merge_trials).sort_values(
    ["merged_rows_with_occurrence_counter", "nonmissing_any_core_target_rows"],
    ascending=False,
)
best_us_gt_strategy = merge_trials_df.iloc[0]["strategy"]
base = merged_by_strategy[best_us_gt_strategy].copy()
base["patient_group"] = base["_merge_id"].map(ug_normalized)

print("Selected ultrasound-ground-truth merge strategy:", best_us_gt_strategy)
print("Protected merged shape:", base.shape)

plain_merge_rows = []
for strategy_name in ID_STRATEGIES:
    us_key = f"us_{strategy_name}"
    gt_key = f"gt_{strategy_name}"
    plain_rows = us.merge(
        gt,
        left_on=[us_key, "side_norm"],
        right_on=[gt_key, "side_norm"],
        how="inner",
        suffixes=("_us", "_gt"),
    ).shape[0]
    plain_merge_rows.append({"strategy": strategy_name, "plain_merge_rows_without_occurrence_counter": plain_rows})

plain_merge_df = pd.DataFrame(plain_merge_rows)
duplicate_summary = []
for strategy_name in ID_STRATEGIES:
    us_diag, us_counts = duplicate_key_diagnostics(us, f"us_{strategy_name}", "side_norm", f"ultrasound_{strategy_name}")
    gt_diag, gt_counts = duplicate_key_diagnostics(gt, f"gt_{strategy_name}", "side_norm", f"ground_truth_{strategy_name}")
    duplicate_summary.extend([us_diag, gt_diag])
    save_df(us_counts, f"duplicate_keys_ultrasound_{strategy_name}.csv")
    save_df(gt_counts, f"duplicate_keys_ground_truth_{strategy_name}.csv")

duplicate_summary_df = pd.DataFrame(duplicate_summary)
save_df(merge_trials_df, "us_gt_merge_strategy_diagnostics.csv")
save_df(plain_merge_df, "plain_merge_inflation_diagnostics.csv")
save_df(duplicate_summary_df, "duplicate_key_diagnostics_summary.csv")

# %% [markdown]
# ## MRI matching strategy selection

# %%
mri_match_trials = []
mri_merged_by_strategy = {}

for strategy_name in ID_STRATEGIES:
    base_key = "_merge_id" if strategy_name == best_us_gt_strategy else f"gt_{strategy_name}"
    mri_key = f"mri_{strategy_name}"

    tmp = base.copy()
    tmp["_mri_match_id"] = tmp[base_key]
    mri_tmp = mri_long.copy()
    mri_tmp["_mri_match_id"] = mri_tmp[mri_key]

    merged = tmp.merge(
        mri_tmp[["_mri_match_id", "side_norm"] + mri_feature_cols],
        on=["_mri_match_id", "side_norm"],
        how="left",
        validate="many_to_one",
    )
    matched_rows = int(merged[mri_feature_cols].notna().any(axis=1).sum())
    matched_keys = int(
        merged.loc[merged[mri_feature_cols].notna().any(axis=1), ["_mri_match_id", "side_norm"]]
        .drop_duplicates()
        .shape[0]
    )
    mri_match_trials.append(
        {
            "strategy": strategy_name,
            "base_key_column": base_key,
            "mri_key_column": mri_key,
            "base_rows": int(len(base)),
            "mri_long_rows": int(len(mri_long)),
            "matched_base_rows": matched_rows,
            "matched_unique_patient_side_keys": matched_keys,
            "match_rate_base_rows": matched_rows / max(len(base), 1),
        }
    )
    mri_merged_by_strategy[strategy_name] = merged

mri_match_df = pd.DataFrame(mri_match_trials).sort_values(
    ["matched_base_rows", "matched_unique_patient_side_keys"], ascending=False
)
best_mri_strategy = mri_match_df.iloc[0]["strategy"]
data = mri_merged_by_strategy[best_mri_strategy].copy()

print("Selected MRI matching strategy:", best_mri_strategy)
print("MRI matched rows:", int(data[mri_feature_cols].notna().any(axis=1).sum()), "/", len(data))
print("Final fused data shape:", data.shape)

mri_raw_id_col = f"mri_{best_mri_strategy}"
base_id_col_for_mri = "_merge_id" if best_mri_strategy == best_us_gt_strategy else f"gt_{best_mri_strategy}"
mri_subject_ids = set(mri_wide[mri_raw_id_col].dropna().astype(str))
base_subject_ids = set(base[base_id_col_for_mri].dropna().astype(str))
matched_subject_ids = set(data.loc[data[mri_feature_cols].notna().any(axis=1), "_mri_match_id"].dropna().astype(str))
matched_side_rows = data.loc[data[mri_feature_cols].notna().any(axis=1)].copy()

mri_matching_diagnostics_df = pd.DataFrame(
    [
        {
            "selected_us_gt_strategy": best_us_gt_strategy,
            "selected_mri_strategy": best_mri_strategy,
            "raw_mri_subjects": int(len(mri_subject_ids)),
            "raw_mri_nonmissing_RF_imat_ratio_L": int(pd.to_numeric(mri_wide.get("RF_imat_ratio_L"), errors="coerce").notna().sum())
            if "RF_imat_ratio_L" in mri_wide.columns
            else 0,
            "raw_mri_nonmissing_RF_imat_ratio_R": int(pd.to_numeric(mri_wide.get("RF_imat_ratio_R"), errors="coerce").notna().sum())
            if "RF_imat_ratio_R" in mri_wide.columns
            else 0,
            "expected_max_side_rows": int(len(mri_subject_ids) * 2),
            "mri_long_rows": int(len(mri_long)),
            "mri_rows_successfully_matched_to_ultrasound_womac": int(len(matched_side_rows)),
            "matched_left_rows": int(matched_side_rows["side_norm"].eq("Left").sum()),
            "matched_right_rows": int(matched_side_rows["side_norm"].eq("Right").sum()),
            "mri_ids_not_matched_to_ultrasound_womac": json.dumps(sorted(mri_subject_ids - matched_subject_ids)),
            "ultrasound_womac_ids_not_matched_to_mri": json.dumps(sorted(base_subject_ids - matched_subject_ids)),
            "rows_dropped_possible_reasons": (
                "MRI IDs not present in protected ultrasound/WOMAC merge; side-specific MRI rows missing after merge; "
                "or later target/correlation analyses drop rows with missing WOMAC severity, missing side, or missing RF IMAT ratio."
            ),
        }
    ]
)

mri_match_summary = {
    "selected_us_gt_strategy": best_us_gt_strategy,
    "selected_mri_strategy": best_mri_strategy,
    "mri_matched_rows": int(data[mri_feature_cols].notna().any(axis=1).sum()),
    "total_rows": int(len(data)),
    "mri_feature_count": int(len(mri_feature_cols)),
}
save_df(mri_match_df, "mri_matching_strategy_diagnostics.csv")
save_df(mri_matching_diagnostics_df, "mri_matching_diagnostics.csv")
save_json(mri_match_summary, "mri_matching_summary.json")

# %% [markdown]
# ## Targets and severity bins

# %%
TARGET_SOURCE_COLS = {
    "Total": "Total Score",
    "Pain": "WOMAC Pain",
    "Function": "WOMAC Function",
}

for target_name, col in TARGET_SOURCE_COLS.items():
    if col not in data.columns:
        raise KeyError(f"Missing target column: {col}")
    data[target_name] = pd.to_numeric(data[col], errors="coerce")

data["PainFunction"] = data["Pain"] + data["Function"]


def severity_bin(values, target):
    values = pd.to_numeric(values, errors="coerce")
    if target == "Pain":
        bins = [-np.inf, 4, 9, 14, np.inf]
    elif target == "PainFunction":
        bins = [-np.inf, 10, 21, 32, np.inf]
    else:  # Total and Function
        bins = [-np.inf, 9, 19, 29, np.inf]
    labels = ["Mild", "Medium", "Severe", "Intensed"]
    return pd.cut(values, bins=bins, labels=labels)


TARGETS = ["Total", "Pain", "Function", "PainFunction"]
for t in TARGETS:
    data[f"{t}_Severity"] = severity_bin(data[t], t)

target_summary = []
for t in TARGETS:
    y = data[f"{t}_Severity"]
    target_summary.append({"target": t, "nonmissing_rows": int(y.notna().sum()), **y.value_counts(dropna=False).to_dict()})
target_summary_df = pd.DataFrame(target_summary)
save_df(target_summary_df, "target_severity_distribution.csv")

# %% [markdown]
# ## Leakage-free feature sets and transformers

# %%
TARGET_AND_LABEL_COLS = set(TARGETS + [f"{t}_Severity" for t in TARGETS])
TARGET_AND_LABEL_COLS.update(["Total Score", "WOMAC Pain", "WOMAC Stiffness", "WOMAC Function"])

METADATA_PATTERNS = [
    "patient", "label", "side", "sex", "age", "bmi", "height", "date", "time",
    "id_", "_id", "merge", "group", "occ", "mri no"
]


def is_metadata_col(col):
    c = str(col).lower()
    return any(p in c for p in METADATA_PATTERNS)


radiomics_cols = [
    c for c in data.columns
    if str(c).startswith("original_") and pd.api.types.is_numeric_dtype(data[c])
]

known_morphology_cols = [
    "Area", "Perim.", "Major", "Minor", "Angle", "Circ.", "Feret", "FeretX", "FeretY",
    "FeretAngle", "MinFeret", "AR", "Round", "Solidity", "Normalized_Area", "MT_adjusted",
    "EI_original", "EI_adjusted", "Norm_Area_H^2", "Norm_Area_H", "Norm_Area_BMI^0.67",
    "Norm_Area_log(area/BMI)", "Norm_Area_from_R", "MT_residual", "MT_new",
]

morphology_cols = [
    c for c in known_morphology_cols
    if c in data.columns and pd.api.types.is_numeric_dtype(data[c]) and c not in TARGET_AND_LABEL_COLS
]

# Fallback: numeric ground-truth columns that are not targets or metadata.
if not morphology_cols:
    morphology_cols = [
        c for c in data.columns
        if pd.api.types.is_numeric_dtype(data[c])
        and c not in TARGET_AND_LABEL_COLS
        and not str(c).startswith("original_")
        and not str(c).startswith("MRI_")
        and not is_metadata_col(c)
    ]

mri_reference_cols = [
    c for c in mri_feature_cols
    if c in data.columns and "imat_ratio" in str(c).lower()
]
mri_cols = mri_reference_cols

print("Morphology feature count:", len(morphology_cols))
print("Radiomics feature count:", len(radiomics_cols))
print("MRI reference feature count:", len(mri_reference_cols))

save_json(
    {
        "morphology_cols": morphology_cols,
        "radiomics_cols": radiomics_cols,
        "mri_reference_cols": mri_reference_cols,
    },
    "feature_columns.json",
)


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drop highly correlated columns using only training data."""

    def __init__(self, threshold=0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        if X_df.shape[1] <= 1:
            self.keep_indices_ = np.arange(X_df.shape[1])
            return self
        corr = X_df.corr().abs().fillna(0)
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        drop_cols = [col for col in upper.columns if any(upper[col] > self.threshold)]
        self.keep_indices_ = np.array([i for i, c in enumerate(X_df.columns) if c not in drop_cols], dtype=int)
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.keep_indices_]


def kpca_components(n_samples, n_features, max_components=10):
    if n_samples < 3 or n_features < 1:
        return 1
    return max(1, min(max_components, n_features, n_samples - 1))


def make_kpca_pipeline(n_components):
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold(threshold=0.0)),
            ("corr", CorrelationFilter(threshold=0.95)),
            ("scaler", MinMaxScaler()),
            ("kpca", KernelPCA(n_components=n_components, kernel="rbf", random_state=RANDOM_STATE)),
        ]
    )


def make_scaled_pipeline():
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold(threshold=0.0)),
            ("corr", CorrelationFilter(threshold=0.95)),
            ("scaler", MinMaxScaler()),
        ]
    )


def fit_transform_block(train_df, test_df, cols, prefix, use_kpca=True):
    if not cols:
        return pd.DataFrame(index=train_df.index), pd.DataFrame(index=test_df.index), {"input_cols": 0, "output_cols": 0}

    X_train = train_df[cols].apply(pd.to_numeric, errors="coerce")
    X_test = test_df[cols].apply(pd.to_numeric, errors="coerce")

    if use_kpca:
        n_comp = kpca_components(X_train.shape[0], X_train.shape[1])
        pipe = make_kpca_pipeline(n_comp)
        out_cols = [f"{prefix}{i+1}" for i in range(n_comp)]
    else:
        pipe = make_scaled_pipeline()
        out_cols = None

    Xtr = pipe.fit_transform(X_train)
    Xte = pipe.transform(X_test)

    if out_cols is None:
        out_cols = [f"{prefix}{i+1}" for i in range(Xtr.shape[1])]

    return (
        pd.DataFrame(Xtr, index=train_df.index, columns=out_cols),
        pd.DataFrame(Xte, index=test_df.index, columns=out_cols),
        {"input_cols": len(cols), "output_cols": len(out_cols)},
    )


def build_feature_sets(train_df, test_df):
    morph_tr, morph_te, morph_info = fit_transform_block(train_df, test_df, morphology_cols, "Morph_KPCA_", True)
    rad_tr, rad_te, rad_info = fit_transform_block(train_df, test_df, radiomics_cols, "Radiomics_KPCA_", True)

    feature_sets = {
        "Morph_KPCA": (morph_tr, morph_te),
        "Radiomics_KPCA": (rad_tr, rad_te),
        "Combined_KPCA": (
            pd.concat([morph_tr, rad_tr], axis=1),
            pd.concat([morph_te, rad_te], axis=1),
        ),
    }
    info = {"morphology": morph_info, "radiomics": rad_info}
    return feature_sets, info

# %% [markdown]
# ## Model training, baselines, reports, and confusion matrices

# %%
def majority_class_baseline(y_train, y_test):
    majority = y_train.value_counts().idxmax()
    y_pred = pd.Series([majority] * len(y_test), index=y_test.index)
    return majority, y_pred


def metric_row(y_true, y_pred):
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "weighted_precision": precision_w,
        "weighted_recall": recall_w,
        "weighted_f1": f1_w,
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def save_confusion_matrix(y_true, y_pred, labels, target, model_name, cohort):
    if not SAVE_CONFUSION_MATRIX_IMAGES:
        return None
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{target} - {model_name} - {cohort}")
    plt.tight_layout()
    safe_model = re.sub(r"[^A-Za-z0-9_]+", "_", model_name)
    path = OUTPUT_DIR / f"confusion_matrix_{target}_{safe_model}_{cohort}.png"
    plt.savefig(path, dpi=200)
    plt.close()
    return path


all_results = []
all_baselines = []
all_reports = []
split_diagnostics = []
feature_diagnostics = []

LABEL_ORDER = ["Mild", "Medium", "Severe", "Intensed"]

for cohort_name, cohort_filter in [
    ("imputed_all_rows", pd.Series(True, index=data.index)),
    ("mri_complete_cases", data[mri_cols].notna().all(axis=1) if mri_cols else pd.Series(False, index=data.index)),
]:
    cohort_data = data.loc[cohort_filter].copy()
    print("\nCohort:", cohort_name, "rows:", len(cohort_data))

    for target in TARGETS:
        target_col = f"{target}_Severity"
        df_t = cohort_data.loc[cohort_data[target_col].notna()].copy()
        df_t[target_col] = df_t[target_col].astype(str)

        class_counts = df_t[target_col].value_counts()
        if len(df_t) < 10 or class_counts.shape[0] < 2:
            print(f"Skipping {cohort_name} / {target}: insufficient rows/classes")
            continue

        groups = df_t["patient_group"].astype(str)
        splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        train_idx, test_idx = next(splitter.split(df_t, df_t[target_col], groups=groups))
        train_df = df_t.iloc[train_idx].copy()
        test_df = df_t.iloc[test_idx].copy()

        train_groups = set(train_df["patient_group"].astype(str))
        test_groups = set(test_df["patient_group"].astype(str))
        overlap = train_groups.intersection(test_groups)
        assert len(overlap) == 0, f"Patient overlap detected for {cohort_name} / {target}: {overlap}"

        y_train = train_df[target_col].astype(str)
        y_test = test_df[target_col].astype(str)

        labels = [x for x in LABEL_ORDER if x in set(y_train).union(set(y_test))]

        split_diagnostics.append(
            {
                "cohort": cohort_name,
                "target": target,
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "train_patients": int(len(train_groups)),
                "test_patients": int(len(test_groups)),
                "patient_overlap": int(len(overlap)),
                "train_class_counts": y_train.value_counts().to_dict(),
                "test_class_counts": y_test.value_counts().to_dict(),
            }
        )

        baseline_class, baseline_pred = majority_class_baseline(y_train, y_test)
        baseline_metrics = metric_row(y_test, baseline_pred)
        baseline_row = {
            "cohort": cohort_name,
            "target": target,
            "model": "majority_class_baseline",
            "majority_class": baseline_class,
            **baseline_metrics,
        }
        all_baselines.append(baseline_row)
        save_confusion_matrix(y_test, baseline_pred, labels, target, "majority_class_baseline", cohort_name)

        all_reports.append(
            {
                "cohort": cohort_name,
                "target": target,
                "model": "majority_class_baseline",
                "classification_report": classification_report(y_test, baseline_pred, labels=labels, zero_division=0),
            }
        )

        feature_sets, f_info = build_feature_sets(train_df, test_df)
        feature_diagnostics.append({"cohort": cohort_name, "target": target, **f_info})

        for model_name, (X_train, X_test) in feature_sets.items():
            if X_train.shape[1] == 0:
                print(f"Skipping {cohort_name} / {target} / {model_name}: no features")
                continue

            clf = RandomForestClassifier(
                n_estimators=N_TREES,
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=-1,
            )
            clf.fit(X_train, y_train)
            y_pred = pd.Series(clf.predict(X_test), index=y_test.index)

            row = {
                "cohort": cohort_name,
                "target": target,
                "model": model_name,
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "n_features": int(X_train.shape[1]),
                **metric_row(y_test, y_pred),
            }
            row["baseline_accuracy"] = baseline_metrics["accuracy"]
            row["baseline_balanced_accuracy"] = baseline_metrics["balanced_accuracy"]
            row["delta_accuracy_vs_baseline"] = row["accuracy"] - baseline_metrics["accuracy"]
            row["delta_balanced_accuracy_vs_baseline"] = row["balanced_accuracy"] - baseline_metrics["balanced_accuracy"]
            row["delta_macro_f1_vs_baseline"] = row["macro_f1"] - baseline_metrics["macro_f1"]
            all_results.append(row)

            all_reports.append(
                {
                    "cohort": cohort_name,
                    "target": target,
                    "model": model_name,
                    "classification_report": classification_report(y_test, y_pred, labels=labels, zero_division=0),
                }
            )
            save_confusion_matrix(y_test, y_pred, labels, target, model_name, cohort_name)

results_df = pd.DataFrame(all_results).sort_values(
    ["cohort", "target", "balanced_accuracy", "macro_f1"], ascending=[True, True, False, False]
)
baseline_df = pd.DataFrame(all_baselines)
reports_df = pd.DataFrame(all_reports)
split_diag_df = pd.DataFrame(split_diagnostics)
feature_diag_df = pd.DataFrame(feature_diagnostics)

display(results_df)

save_df(results_df, "model_results.csv")
save_df(baseline_df, "baseline_results.csv")
save_df(reports_df, "classification_reports.csv")
save_df(split_diag_df, "patient_safe_split_diagnostics.csv")
save_df(feature_diag_df, "feature_transform_diagnostics.csv")

# %% [markdown]
# ## Leakage-free all-ultrasound side-specific KPCA and MRI correlation

# %%
def unique_existing(cols):
    return [c for c in dict.fromkeys(cols) if c in data.columns]


def bh_fdr(p_values):
    p = pd.to_numeric(pd.Series(p_values), errors="coerce")
    out = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.dropna()
    if valid.empty:
        return out
    try:
        from statsmodels.stats.multitest import multipletests

        out.loc[valid.index] = multipletests(valid.values, method="fdr_bh")[1]
        return out
    except Exception:
        ranked = valid.sort_values()
        m = len(ranked)
        adjusted = ranked * m / np.arange(1, m + 1)
        adjusted = adjusted.iloc[::-1].cummin().iloc[::-1].clip(upper=1.0)
        out.loc[adjusted.index] = adjusted
        return out


def find_mri_imat_ratio_reference(df, side):
    side_suffix = "_L" if side == "Left" else "_R"
    exact_wide = f"RF_imat_ratio{side_suffix}"
    return exact_wide if exact_wide in df.columns else None


def fit_side_kpca(train_side, test_side, ultrasound_features):
    ultrasound_features = [
        c for c in unique_existing(ultrasound_features)
        if c in train_side.columns and c in test_side.columns
    ]
    if not ultrasound_features or train_side.empty or test_side.empty:
        return None, None

    X_train = train_side[ultrasound_features].apply(pd.to_numeric, errors="coerce")
    X_test = test_side[ultrasound_features].apply(pd.to_numeric, errors="coerce")
    nonempty_cols = [c for c in ultrasound_features if X_train[c].notna().any()]
    if not nonempty_cols:
        return None, None

    X_train = X_train[nonempty_cols]
    X_test = X_test[nonempty_cols]

    imputer = SimpleImputer(strategy="median")
    variance = VarianceThreshold(threshold=0.0)
    corr = CorrelationFilter(threshold=0.95)
    scaler = MinMaxScaler()
    kpca = KernelPCA(n_components=1, kernel="rbf", random_state=RANDOM_STATE)

    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)
    try:
        X_train_var = variance.fit_transform(X_train_imp)
        X_test_var = variance.transform(X_test_imp)
    except ValueError:
        return None, None

    var_names = np.array(nonempty_cols, dtype=object)[variance.get_support()]
    if len(var_names) == 0:
        return None, None

    X_train_corr = corr.fit_transform(pd.DataFrame(X_train_var, columns=var_names, index=train_side.index))
    X_test_corr = corr.transform(pd.DataFrame(X_test_var, columns=var_names, index=test_side.index))
    if X_train_corr.shape[1] == 0:
        return None, None

    X_train_scaled = scaler.fit_transform(X_train_corr)
    X_test_scaled = scaler.transform(X_test_corr)

    train_scores = kpca.fit_transform(X_train_scaled)[:, 0]
    test_scores = kpca.transform(X_test_scaled)[:, 0]
    return pd.Series(train_scores, index=train_side.index), pd.Series(test_scores, index=test_side.index)


candidate_ultrasound_cols = unique_existing(morphology_cols + radiomics_cols)


def correlate_score_frame(score_frame, analysis_type, sample_size_note):
    rows = []
    for (cohort_name, target, side, kpca_variable, mri_variable), grp in score_frame.groupby(
        ["cohort", "target", "side", "kpca_variable", "mri_variable"], dropna=False
    ):
        corr_input = grp.dropna(subset=["kpca_score", "mri_value"])
        if len(corr_input) >= 3 and corr_input["kpca_score"].nunique() > 1 and corr_input["mri_value"].nunique() > 1:
            rho, p_value = spearmanr(corr_input["kpca_score"], corr_input["mri_value"])
        else:
            rho, p_value = np.nan, np.nan
        rows.append(
            {
                "analysis_type": analysis_type,
                "cohort": cohort_name,
                "target": target,
                "side": side,
                "kpca_variable": kpca_variable,
                "mri_variable": mri_variable,
                "method": "spearman",
                "rho": rho,
                "p_value": p_value,
                "n_samples": int(len(corr_input)),
                "sample_size_note": sample_size_note,
                "ultrasound_features_json": grp["ultrasound_features_json"].dropna().iloc[0]
                if grp["ultrasound_features_json"].notna().any()
                else "[]",
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_fdr_bh"] = bh_fdr(out["p_value"])
        out["significant_fdr_0_05"] = out["p_fdr_bh"] < 0.05
    return out


# 1. Strict held-out test-set correlation. This is methodologically strict, but the
# MRI sample is small, so each held-out side-specific correlation can have only a few cases.
strict_score_frames = []
strict_cohort_filter = data[mri_cols].notna().all(axis=1) if mri_cols else pd.Series(False, index=data.index)
strict_cohort_data = data.loc[strict_cohort_filter].copy()
print("\nSelectedAll strict_test_only cohort: mri_complete_cases rows:", len(strict_cohort_data))

for target in TARGETS:
    target_col = f"{target}_Severity"
    df_t = strict_cohort_data.loc[strict_cohort_data[target_col].notna()].copy()
    df_t[target_col] = df_t[target_col].astype(str)

    class_counts = df_t[target_col].value_counts()
    if len(df_t) < 10 or class_counts.shape[0] < 2:
        print(f"Skipping strict_test_only / {target}: insufficient rows/classes")
        continue

    groups = df_t["patient_group"].astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(df_t, df_t[target_col], groups=groups))
    train_df = df_t.iloc[train_idx].copy()
    test_df = df_t.iloc[test_idx].copy()

    for side, score_col in [("Left", "SelectedAll_KPCA_L"), ("Right", "SelectedAll_KPCA_R")]:
        mri_col = find_mri_imat_ratio_reference(df_t, side)
        if not mri_col:
            print(f"Skipping strict_test_only / {target} / {side}: no MRI IMAT ratio reference column")
            continue

        train_side = train_df.loc[train_df["side_norm"].map(normalize_side).eq(side)].copy()
        test_side = test_df.loc[test_df["side_norm"].map(normalize_side).eq(side)].copy()
        _, test_scores = fit_side_kpca(train_side, test_side, candidate_ultrasound_cols)
        if test_scores is None:
            print(f"Skipping strict_test_only / {target} / {side}: not enough side-specific ultrasound features")
            continue

        strict_score_frames.append(
            pd.DataFrame(
                {
                    "analysis_type": "strict_test_only",
                    "cohort": "mri_complete_cases",
                    "target": target,
                    "fold": "single_group_shuffle_split",
                    "side": side,
                    "patient_group": test_side["patient_group"].astype(str),
                    "dataset_split": "test",
                    "kpca_variable": score_col,
                    "kpca_score": test_scores,
                    "mri_variable": mri_col,
                    "mri_value": pd.to_numeric(test_side[mri_col], errors="coerce"),
                    "ultrasound_features_json": json.dumps(candidate_ultrasound_cols),
                }
            )
        )

strict_scores_df = pd.concat(strict_score_frames, ignore_index=True) if strict_score_frames else pd.DataFrame()
strict_corr_df = correlate_score_frame(
    strict_scores_df,
    "strict_test_only",
    "Held-out test-set only; sample size may be small because the MRI file has 21 subjects.",
) if not strict_scores_df.empty else pd.DataFrame()


# 2. Cross-fitted MRI-complete correlation. This is the main MRI-complete analysis:
# every correlation uses out-of-fold KPCA scores for held-out patients only.
cross_fitted_score_frames = []
cross_cohort_data = data.loc[strict_cohort_filter].copy()
print("\nSelectedAll cross_fitted_mri_complete cohort rows:", len(cross_cohort_data))

for target in TARGETS:
    target_col = f"{target}_Severity"
    df_t = cross_cohort_data.loc[cross_cohort_data[target_col].notna()].copy()
    df_t[target_col] = df_t[target_col].astype(str)

    class_counts = df_t[target_col].value_counts()
    groups = df_t["patient_group"].astype(str)
    unique_groups = groups.nunique(dropna=True)
    if len(df_t) < 10 or class_counts.shape[0] < 2 or unique_groups < 2:
        print(f"Skipping cross_fitted_mri_complete / {target}: insufficient rows/classes/groups")
        continue

    n_splits = min(5, unique_groups)
    splitter = GroupKFold(n_splits=n_splits)

    for fold_idx, (train_idx, heldout_idx) in enumerate(splitter.split(df_t, df_t[target_col], groups=groups), start=1):
        train_df = df_t.iloc[train_idx].copy()
        heldout_df = df_t.iloc[heldout_idx].copy()
        y_train = train_df[target_col].astype(str)

        if y_train.nunique() < 2:
            print(f"Skipping cross_fitted_mri_complete / {target} / fold {fold_idx}: training fold has one class")
            continue

        for side, score_col in [("Left", "SelectedAll_KPCA_L"), ("Right", "SelectedAll_KPCA_R")]:
            mri_col = find_mri_imat_ratio_reference(df_t, side)
            if not mri_col:
                print(f"Skipping cross_fitted_mri_complete / {target} / {side}: no MRI IMAT ratio reference column")
                continue

            train_side = train_df.loc[train_df["side_norm"].map(normalize_side).eq(side)].copy()
            heldout_side = heldout_df.loc[heldout_df["side_norm"].map(normalize_side).eq(side)].copy()
            _, heldout_scores = fit_side_kpca(train_side, heldout_side, candidate_ultrasound_cols)
            if heldout_scores is None:
                print(f"Skipping cross_fitted_mri_complete / {target} / {side} / fold {fold_idx}: not enough side-specific ultrasound features")
                continue

            cross_fitted_score_frames.append(
                pd.DataFrame(
                    {
                        "analysis_type": "cross_fitted_mri_complete",
                        "cohort": "mri_complete_cases",
                        "target": target,
                        "fold": fold_idx,
                        "side": side,
                        "patient_group": heldout_side["patient_group"].astype(str),
                        "dataset_split": "out_of_fold",
                        "kpca_variable": score_col,
                        "kpca_score": heldout_scores,
                        "mri_variable": mri_col,
                        "mri_value": pd.to_numeric(heldout_side[mri_col], errors="coerce"),
                        "ultrasound_features_json": json.dumps(candidate_ultrasound_cols),
                    }
                )
            )

cross_fitted_scores_df = pd.concat(cross_fitted_score_frames, ignore_index=True) if cross_fitted_score_frames else pd.DataFrame()
cross_fitted_corr_df = correlate_score_frame(
    cross_fitted_scores_df,
    "cross_fitted_mri_complete",
    "Main MRI-complete analysis; KPCA scores are out-of-fold for held-out patients and use all matched MRI-complete side rows.",
) if not cross_fitted_scores_df.empty else pd.DataFrame()

display(cross_fitted_corr_df)
display(strict_corr_df)

save_df(strict_corr_df, "selectedall_strict_test_only_kpca_mri_correlations.csv")
save_df(cross_fitted_corr_df, "selectedall_cross_fitted_kpca_mri_correlations.csv")
save_df(cross_fitted_scores_df, "selectedall_cross_fitted_kpca_scores.csv")
cleanup_nonrequired_outputs()

# %% [markdown]
# ## Final summary

# %%
print("\nDone.")
print("Selected US-GT ID strategy:", best_us_gt_strategy)
print("Selected MRI ID strategy:", best_mri_strategy)
print("Protected US-GT merged rows:", len(base))
print("MRI matched rows:", int(data[mri_cols].notna().any(axis=1).sum()), "/", len(data))
print("Outputs saved to:", OUTPUT_DIR)
print("Inline displays are limited to model results, cross-fitted KPCA-MRI correlations, and strict test-only KPCA-MRI correlations.")
