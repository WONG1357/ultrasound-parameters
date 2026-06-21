# WOMAC Ultrasound Imaging ML Pipeline

This repository contains a reproducible workflow for studying associations between ultrasound-derived muscle features, WOMAC severity, and MRI RF IMAT ratio references.

The pipeline starts at image feature acquisition and ends with leakage-safe WOMAC severity modeling plus side-specific KPCA correlation against MRI reference data.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Place local data under the ignored `data/` folders, or pass custom paths using command-line arguments and environment variables.

Recommended local layout:

```text
data/
├── raw/
│   ├── ground_truth_with_bmi_womac.xlsx
│   ├── mri_muscle_features.xlsx
│   ├── ultrasound_images/
│   └── eroded_masks/
└── processed/
    └── ultrasound_radiomics_features_updated.xlsx
```

The main notebook uses these default relative paths:

```text
data/processed/ultrasound_radiomics_features_updated.xlsx
data/raw/ground_truth_with_bmi_womac.xlsx
data/raw/mri_muscle_features.xlsx
outputs/
```

You can override them without editing the notebook:

```bash
export WOMAC_BASE_DIR=/path/to/project
export WOMAC_US_PATH=/path/to/ultrasound_radiomics_features_updated.xlsx
export WOMAC_GT_PATH=/path/to/ground_truth_with_bmi_womac.xlsx
export WOMAC_MRI_PATH=/path/to/mri_muscle_features.xlsx
export WOMAC_OUTPUT_DIR=/path/to/outputs
```

## Data Acquisition Overview

The project uses four main data sources:

1. Ultrasound images
   - Raw B-mode ultrasound images.
   - One image should correspond to one patient/side or one ROI-level observation.

2. Ultrasound masks
   - Binary ROI masks for the ultrasound muscle region.
   - Used for PyRadiomics texture extraction.
   - In the current radiomics script, mask files are expected to begin with `mask_bw_`.

3. Morphological measurements from Fiji/ImageJ
   - Direct geometric and intensity-related measurements exported from Fiji.
   - Examples from the workbook include `Area`, `Perim.`, `Major`, `Minor`, `Angle`, `Circ.`, `Feret`, `FeretX`, `FeretY`, `FeretAngle`, `MinFeret`, `AR`, `Round`, and `Solidity`.

4. Clinical and reference outcomes
   - WOMAC targets: `WOMAC Pain`, `WOMAC Stiffness`, `WOMAC Function`, `Total Score`.
   - Demographics/body size: `Sex_MF`, `Sex_bin`, `BMI`, `Age`, `Height`.
   - MRI reference variables: `RF_imat_ratio_L`, `RF_imat_ratio_R`.

The WOMAC/morphology workbook may also include engineered fields that are not simple Fiji exports, such as:

```text
Normalized_Area
MT_adjusted
EI_original
EI_adjusted
Norm_Area_H^2
Norm_Area_H
Norm_Area_BMI^0.67
Norm_Area_log(area/BMI)
Norm_Area_from_R
MT_residual
MT_new
```

These should be treated as preprocessed or derived variables. The simple/direct morphology measurements should be exported from Fiji first, then downstream scripts or spreadsheets can compute the derived fields.

## Batch Morphology Extraction In Fiji/ImageJ

Fiji is used to massively acquire simple morphology measurements from ultrasound ROIs.

### 1. Organize Input Files

Use a consistent folder structure:

```text
project_data/
├── ultrasound_images/
│   ├── UG001_L.png
│   ├── UG001_R.png
│   └── ...
├── roi_masks/
│   ├── UG001_L_mask.png
│   ├── UG001_R_mask.png
│   └── ...
└── fiji_exports/
```

Recommended naming fields:

```text
Patient_Number
Leg_Side
Image_ID
```

For example:

```text
UG001_L.png
UG001_R.png
UG002_L.png
UG002_R.png
```

### 2. Set Fiji Measurements

In Fiji:

1. Open `Analyze > Set Measurements...`
2. Enable the simple/direct parameters needed for this project:
   - `Area`
   - `Perimeter`
   - `Fit ellipse`, which gives `Major`, `Minor`, and `Angle`
   - `Shape descriptors`, which gives circularity/roundness/aspect ratio/solidity-related measurements
   - `Feret's diameter`, which gives `Feret`, `FeretX`, `FeretY`, `FeretAngle`, and `MinFeret`
   - `Mean gray value` if extracting echogenicity/intensity from the grayscale image
3. Set decimal places consistently, for example 3 or 4.

Direct Fiji-style output columns expected by the downstream workbook include:

```text
Area
Perim.
Major
Minor
Angle
Circ.
Feret
FeretX
FeretY
FeretAngle
MinFeret
AR
Round
Solidity
```

If echogenicity is measured in Fiji, export the mean gray value and then rename or process it downstream as `EI_original` and/or `EI_adjusted`.

### 3. Batch Processing Strategy

For large image sets, use a Fiji macro instead of opening images manually.

The core idea is:

1. Loop through every image file in a folder.
2. Open the matching ROI mask or thresholded ROI.
3. Apply the mask/selection to the ultrasound image.
4. Run `Analyze > Measure`.
5. Add patient ID and side metadata from the file name.
6. Save one combined CSV.

Example Fiji macro template:

```javascript
imageDir = "/path/to/ultrasound_images/";
maskDir = "/path/to/roi_masks/";
outputCsv = "/path/to/fiji_exports/morphology_measurements.csv";

run("Set Measurements...", "area mean perimeter fit shape feret's redirect=None decimal=4");

list = getFileList(imageDir);
if (File.exists(outputCsv)) {
    File.delete(outputCsv);
}

for (i = 0; i < list.length; i++) {
    filename = list[i];
    if (!(endsWith(filename, ".png") || endsWith(filename, ".jpg") || endsWith(filename, ".tif"))) {
        continue;
    }

    imagePath = imageDir + filename;
    base = File.nameWithoutExtension(filename);
    maskPath = maskDir + base + "_mask.png";

    if (!File.exists(maskPath)) {
        print("Missing mask for: " + filename);
        continue;
    }

    open(imagePath);
    imageTitle = getTitle();

    open(maskPath);
    maskTitle = getTitle();
    setAutoThreshold("Default dark no-reset");
    run("Convert to Mask");
    run("Create Selection");

    selectWindow(imageTitle);
    run("Restore Selection");
    run("Measure");

    close(imageTitle);
    close(maskTitle);
}

saveAs("Results", outputCsv);
print("Saved morphology measurements to: " + outputCsv);
```

After export, add or parse metadata columns:

```text
Patient_Number
Leg_Side
```

Then compute derived variables outside Fiji, such as height-normalized area, BMI-normalized area, adjusted EI, and residualized muscle thickness.

### 4. Fiji Quality Checks

Before using the exported morphology table:

- Confirm every image has exactly one intended ROI measurement.
- Check that units are consistent across all images.
- Confirm left/right side labels are encoded consistently.
- Check for duplicated patient-side rows.
- Inspect outliers in `Area`, `Perim.`, `Major`, `Minor`, and `Feret`.
- Confirm masks align with the ultrasound image and do not include labels, scale bars, or background.

## Radiomic Texture Feature Extraction

Texture features are extracted from the ultrasound image and binary ROI mask using PyRadiomics.

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the included script with default repo-relative folders:

```bash
python scripts/extract_radiomics.py
```

Or provide custom folders:

```bash
python scripts/extract_radiomics.py \
  --image-folder data/raw/ultrasound_images \
  --mask-folder data/raw/eroded_masks \
  --output-csv data/processed/ultrasound_radiomics_features.csv \
  --mask-prefix mask_bw_ \
  --label 255
```

Mask names should start with:

```text
mask_bw_
```

For example:

```text
mask_bw_UG001_L.png
```

The script removes `mask_bw_` and searches for the matching source image:

```text
UG001_L.*
```

### PyRadiomics Code Used

```python
import os
import glob
import argparse
import SimpleITK as sitk
from radiomics import featureextractor
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--image-folder", default="data/raw/ultrasound_images")
parser.add_argument("--mask-folder", default="data/raw/eroded_masks")
parser.add_argument("--output-csv", default="data/processed/ultrasound_radiomics_features.csv")
parser.add_argument("--mask-prefix", default="mask_bw_")
parser.add_argument("--label", type=int, default=255)
args = parser.parse_args()

image_folder = args.image_folder
mask_folder = args.mask_folder

settings = {
    'binWidth': 25,
    'resampledPixelSpacing': None,
    'interpolator': sitk.sitkBSpline,
    'force2D': True,
    'force2Ddimension': 0,
    'label' : args.label
}

extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
extractor.enableAllFeatures()

feature_list = []
print(f"Scanning mask folder: {mask_folder} ...")

mask_files = [f for f in os.listdir(mask_folder) if f.startswith(args.mask_prefix)]

if not mask_files:
    print(f"Warning: no mask files beginning with '{args.mask_prefix}' were found.")

for mask_filename in mask_files:
    try:
        mask_name_no_ext = os.path.splitext(mask_filename)[0]
        target_id = mask_name_no_ext.replace(args.mask_prefix, '', 1)
        search_pattern = os.path.join(image_folder, f"{target_id}.*")
        potential_images = glob.glob(search_pattern)

        if not potential_images:
            print(f"Skipping: no source image found for ID '{target_id}'")
            continue

        image_path = potential_images[0]
        mask_path = os.path.join(mask_folder, mask_filename)

        image = sitk.ReadImage(image_path)
        mask = sitk.ReadImage(mask_path)

        if image.GetNumberOfComponentsPerPixel() > 1:
            image = sitk.VectorIndexSelectionCast(image, 0)

        if mask.GetNumberOfComponentsPerPixel() > 1:
            mask = sitk.VectorIndexSelectionCast(mask, 0)

        result = extractor.execute(image, mask)

        patient_features = {'PatientID': target_id}
        for key, value in result.items():
            if key.startswith("original_"):
                if isinstance(value, np.ndarray):
                    patient_features[key] = float(value)
                else:
                    patient_features[key] = value

        feature_list.append(patient_features)

    except Exception as e:
        print(f"Error processing {mask_filename}: {str(e)}")

if feature_list:
    df = pd.DataFrame(feature_list)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Successfully extracted features for {len(df)} patients.")
    print(f"Saved to: {args.output_csv}")
else:
    print("No features extracted. Check paths and filename rules.")
```

The radiomics workbook used by the model contains `PatientID`, `Patient_Number`, `Leg_Side`, and many `original_*` PyRadiomics columns, including shape2D, first-order, GLCM, GLDM, GLRLM, GLSZM, and NGTDM features.

## Data Integration

The Colab notebook combines:

```text
ultrasound_radiomics_features_updated.xlsx
ground truth with BMI WOMAC.xlsx
MRI_muscle_features.xlsx
```

The notebook preserves multiple patient-ID normalization strategies:

```text
id_raw_cleaned
id_digits_preserve_zeros
id_digits_integer
id_ug_normalized
```

For ultrasound/WOMAC merging, duplicate patient-side rows are handled by adding a row occurrence counter within each patient-side group:

```python
us_tmp["_key_occ"] = us_tmp.groupby(["_merge_id", "side_norm"], dropna=False).cumcount()
gt_tmp["_key_occ"] = gt_tmp.groupby(["_merge_id", "side_norm"], dropna=False).cumcount()
```

The protected merge then uses:

```python
on=["_merge_id", "side_norm", "_key_occ"]
```

This prevents accidental many-to-many Cartesian inflation when both source tables contain duplicate patient-side rows.

## WOMAC Severity Modeling

The main notebook:

- Defines WOMAC targets: `Total`, `Pain`, `Function`, and `PainFunction`.
- Converts target scores into severity bins.
- Uses patient-safe `GroupShuffleSplit`, so the same patient cannot appear in both training and test sets.
- Evaluates ultrasound morphology/radiomics feature sets:
  - `Morph_KPCA`
  - `Radiomics_KPCA`
  - `Combined_KPCA`
- Removes MRI features from all WOMAC severity classifiers.

MRI is kept only as an external reference for side-specific correlation analysis.

## MRI Reference Analysis

MRI reference variables:

```text
RF_imat_ratio_L
RF_imat_ratio_R
```

The notebook supports the merged long-format equivalent:

```text
MRI_RF_imat_ratio
```

One MRI correlation analyses are implemented.

### 1. Cross-Fitted MRI-Complete Analysis

Outputs:

```text
selected5_cross_fitted_kpca_mri_correlations.csv
selected5_cross_fitted_kpca_scores.csv
```

This is the main MRI-complete analysis.

For each target and fold:

1. Use patient-grouped cross-validation.
2. Select the top 5 morphology/radiomics features using Random Forest on training patients only.
3. Fit imputation, scaling, and side-specific KPCA on training patients only.
4. Transform held-out patients only.
5. Store out-of-fold KPCA scores.
6. Correlate out-of-fold KPCA scores with side-matched MRI RF IMAT ratio.

Left scores are correlated with left MRI RF IMAT ratio, and right scores are correlated with right MRI RF IMAT ratio.

## Main Outputs

The notebook saves:

```text
model_results.csv
baseline_results.csv
classification_reports.csv
patient_safe_split_diagnostics.csv
feature_transform_diagnostics.csv
selected5_strict_test_only_kpca_mri_correlations.csv
selected5_cross_fitted_kpca_mri_correlations.csv
selected5_cross_fitted_kpca_scores.csv
selected5_rf_feature_selection_diagnostics.csv
mri_matching_diagnostics.csv
```

`mri_matching_diagnostics.csv` reports:

- raw MRI subject count
- non-missing `RF_imat_ratio_L`
- non-missing `RF_imat_ratio_R`
- expected maximum side rows
- MRI rows successfully matched to ultrasound/WOMAC data
- matched left/right rows
- MRI IDs not matched to ultrasound/WOMAC
- ultrasound/WOMAC IDs not matched to MRI
- possible row-drop reasons

## Reproducibility Notes

- Do not commit patient data, raw images, masks, or generated outputs.
- Keep consistent patient IDs and side labels from acquisition onward.
- Confirm ImageJ/Fiji measurement settings before batch export.
- Use the same ROI masks for morphology and radiomics where possible.
- Treat MRI variables as external references, not model predictors.
- Use patient-grouped splits whenever evaluating model performance or cross-fitted associations.


## Data Summary

The MRI dataset contained 21 subjects. Each subject had left- and right-leg MRI RF IMAT ratio values:

| MRI variable      | Available values |
| ----------------- | ---------------: |
| `RF_imat_ratio_L` |               21 |
| `RF_imat_ratio_R` |               21 |

Because each subject has two legs, the expected maximum number of side-specific MRI rows was:

```text
21 subjects × 2 sides = 42 side-specific rows
```

After matching MRI data with the ultrasound/WOMAC dataset, the pipeline obtained:

| Matching result                                | Count |
| ---------------------------------------------- | ----: |
| Expected MRI side rows                         |    42 |
| Successfully matched MRI-ultrasound/WOMAC rows |    37 |
| Matched left-side rows                         |    19 |
| Matched right-side rows                        |    18 |

Three MRI subjects did not match successfully:

```text
UG0048, UG0061, UG0094
```

This means the raw MRI data was complete, but several MRI rows were lost during matching with the ultrasound/WOMAC data.

---

## WOMAC Classification Results

The pipeline evaluated ultrasound morphology/radiomics KPCA models for WOMAC severity classification.

The model sets included:

* `Morph_KPCA`
* `Radiomics_KPCA`
* `Combined_KPCA`

MRI-based model inputs were removed from the classification comparison.

A majority-class baseline model was used for comparison. This baseline does not use any imaging features. It simply predicts the most common WOMAC severity class in the training set, usually `Mild`.

### Main Finding

The classification models did not consistently outperform the majority-class baseline.

This is mainly because the WOMAC severity classes were highly imbalanced. Most test cases belonged to the `Mild` class, while `Medium`, `Severe`, and `Intensed` cases were rare.

For example, in the imputed-all-rows cohort, the Total WOMAC test set was distributed approximately as:

| WOMAC severity | Test count |
| -------------- | ---------: |
| Mild           |         37 |
| Medium         |          5 |
| Severe         |          3 |
| Intensed       |          1 |

Because of this imbalance, a simple model that always predicts `Mild` can achieve high accuracy. Therefore, accuracy alone is misleading. Balanced accuracy and macro F1 are more informative.

For Total WOMAC in the imputed-all-rows cohort:

| Model          | Accuracy | Balanced accuracy | Macro F1 |
| -------------- | -------: | ----------------: | -------: |
| Baseline       |    0.804 |             0.250 |    0.223 |
| Morph_KPCA     |    0.783 |             0.243 |    0.220 |
| Radiomics_KPCA |    0.761 |             0.236 |    0.216 |
| Combined_KPCA  |    0.783 |             0.243 |    0.220 |

The KPCA models performed similarly to or worse than the majority-class baseline.

### Interpretation

The ultrasound KPCA classification models were not able to reliably separate WOMAC severity classes in the current dataset. This does not necessarily mean the ultrasound features are uninformative. Rather, the classification task is limited by:

* small sample size,
* strong class imbalance,
* very few non-Mild cases,
* and limited test-set representation of severe categories.

The current classification results should therefore be interpreted cautiously.

---

## Feature Transformation Summary

The pipeline reduced high-dimensional morphology and radiomics features using KPCA.

The approximate feature transformation was:

| Feature group | Input features | KPCA output features |
| ------------- | -------------: | -------------------: |
| Morphology    |             25 |                   10 |
| Radiomics     |            102 |                   10 |

This dimensionality reduction helps reduce feature complexity, but KPCA components are not directly interpretable. For this reason, a later analysis selected the top 5 important features using Random Forest before applying KPCA.

---

## Random Forest Top-5 Feature Selection

A leakage-aware Random Forest feature selection procedure was used to select the 5 most important ultrasound-derived features for WOMAC severity prediction.

Candidate features included:

* morphology features,
* radiomics features.

The top 5 features were selected across morphology and radiomics combined, not separately from each group.

Feature selection was based only on WOMAC severity prediction. MRI variables were not used during feature selection.

### Repeatedly Selected Features

Across folds and targets, several features appeared repeatedly.

Commonly selected morphology-related features included:

```text
FeretY
FeretAngle
Angle
Circ.
Normalized_Area
FeretX
```

Commonly selected radiomics-related features included:

```text
EI_original
original_firstorder_90Percentile
original_firstorder_Minimum
original_firstorder_Range
original_glcm_ClusterProminence
original_glrlm_LongRunHighGrayLevelEmphasis
original_ngtdm_Strength
```

### Interpretation

The selected features were biologically reasonable because they relate to:

* muscle shape,
* muscle size,
* muscle orientation,
* echo intensity,
* intensity distribution,
* and texture.

However, selected features varied across folds, suggesting that feature importance estimates remain unstable in this small dataset.

---

## Side-Specific KPCA and MRI Correlation Analysis

The most important part of the analysis was the side-specific KPCA–MRI correlation.

Because MRI RF IMAT ratio is defined separately for left and right legs, the pipeline created side-specific KPCA scores:

```text
Selected5_KPCA_L
Selected5_KPCA_R
```

These were correlated with side-specific MRI RF IMAT ratio:

```text
Selected5_KPCA_L vs RF_imat_ratio_L
Selected5_KPCA_R vs RF_imat_ratio_R
```

Spearman correlation was used because the relationship between KPCA scores and MRI IMAT ratio may be monotonic but not necessarily linear.

Benjamini-Hochberg FDR correction was applied across all correlation tests.

---

## Cross-Fitted MRI-Complete Correlation

A better analysis used cross-fitted KPCA scores.

In the cross-fitted analysis:

1. Patients were split into folds.
2. For each fold, feature selection, scaling, and KPCA fitting were performed only on training patients.
3. KPCA scores were generated for held-out patients.
4. Out-of-fold KPCA scores were combined.
5. These cross-fitted KPCA scores were correlated with MRI RF IMAT ratio.

This approach allowed more of the MRI-complete data to be used while still reducing leakage.

### Cross-Fitted Correlation Results

| Target       |  Side |  n | Spearman rho | p-value | FDR p-value | Significant after FDR? |
| ------------ | ----: | -: | -----------: | ------: | ----------: | ---------------------- |
| Total        |  Left | 10 |       -0.661 |  0.0376 |      0.2108 | No                     |
| Pain         | Right | 18 |        0.430 |  0.0751 |      0.2108 | No                     |
| PainFunction | Right | 18 |       -0.425 |  0.0790 |      0.2108 | No                     |
| Pain         |  Left | 10 |       -0.440 |  0.2028 |      0.4055 | No                     |
| PainFunction |  Left | 10 |        0.306 |  0.3901 |      0.5573 | No                     |
| Total        | Right | 18 |        0.204 |  0.4180 |      0.5573 | No                     |
| Function     |  Left | 10 |        0.122 |  0.7364 |      0.7881 | No                     |
| Function     | Right | 18 |       -0.068 |  0.7881 |      0.7881 | No                     |

### Main Finding

No KPCA–MRI correlation remained statistically significant after FDR correction.

The strongest nominal association was:

```text
Total WOMAC, left leg:
rho = -0.661
p = 0.0376
FDR p = 0.2108
```

This suggests a moderate-to-strong negative relationship between the left-leg ultrasound-derived selected-feature KPCA score and left-leg MRI RF IMAT ratio for Total WOMAC. However, because the result did not survive FDR correction, it should be treated as exploratory rather than confirmatory.

Moderate but non-significant trends were also observed for:

```text
Pain, right leg:
rho = 0.430
p = 0.0751

PainFunction, right leg:
rho = -0.425
p = 0.0790
```

---

## Duplicate Patient-Side Rows

One important limitation was that some patient-side rows appeared more than once in the cross-fitted KPCA score output.

For each target, the cross-fitted score file contained approximately:

| Side  | Rows | Unique patients |
| ----- | ---: | --------------: |
| Left  |   10 |               7 |
| Right |   18 |              16 |

This means some patient-side observations were duplicated.

Example duplicated left-side patient IDs included:

```text
UG0072
UG0075
UG0076
```

Example duplicated right-side patient IDs included:

```text
UG0074
UG0091
```

This matters because correlation assumes independent observations. If one patient-side appears more than once, that patient-side may receive extra weight in the correlation.

After averaging duplicate patient-side KPCA scores, the strongest pattern remained left-leg Total WOMAC, but the result was still not statistically significant:

```text
Total WOMAC, left leg:
rho ≈ -0.750
p ≈ 0.052
unique patients = 7
```

This suggests that the left-leg Total WOMAC signal may be biologically interesting, but the number of independent observations is too small for a firm conclusion.

---

## Overall Interpretation

The results suggest the following:

1. The standard WOMAC classification models are currently weak and do not consistently outperform the majority-class baseline.
2. This weakness is likely caused by small sample size and severe class imbalance.
3. Random Forest feature selection repeatedly selected biologically plausible morphology, echo-intensity, and radiomics texture features.
4. The MRI-complete cross-fitted correlation analysis showed exploratory side-specific associations, especially for left-leg Total WOMAC.
5. No KPCA–MRI association remained statistically significant after FDR correction.
6. Duplicate patient-side rows and small effective sample size limit the reliability of the correlation results.

The most important exploratory signal was:

```text
Left-leg Total WOMAC selected-feature KPCA score vs left-leg MRI RF IMAT ratio
rho = -0.661
p = 0.0376
FDR p = 0.2108
```

This should be described as a nominal, non-FDR-significant association.

---

## Recommended Conclusion

The current analysis does not provide confirmatory evidence that ultrasound-derived morphology/radiomics KPCA features predict WOMAC severity or significantly correlate with MRI RF IMAT ratio.

However, the results suggest possible exploratory relationships between selected ultrasound feature patterns and MRI-derived muscle fat infiltration, particularly for Total WOMAC in the left leg.

Further work should focus on:

* increasing sample size,
* improving class balance,
* aggregating duplicate patient-side rows before correlation,
* checking why some matched left-side rows are lost before correlation,
* and validating the observed KPCA–MRI trends in an independent cohort.


