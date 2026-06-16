# WOMAC Ultrasound Imaging ML Pipeline

This repository contains a reproducible workflow for studying associations between ultrasound-derived muscle features, WOMAC severity, and MRI RF IMAT ratio references.

The pipeline starts at image feature acquisition and ends with leakage-safe WOMAC severity modeling plus side-specific KPCA correlation against MRI reference data.

## Repository Contents

```text
.
├── womac_imaging_ml_pipeline_colab.ipynb   # Main Google Colab notebook
├── womac_imaging_ml_pipeline_colab.py      # Python export of the notebook
├── scripts/
│   └── extract_radiomics.py                # PyRadiomics texture extraction script
├── data/
│   ├── raw/                                # Local raw inputs, not committed
│   └── processed/                          # Local processed tables, not committed
├── outputs/                                # Local outputs, not committed
├── requirements.txt
└── README.md
```

Patient data, Excel workbooks, images, masks, and generated outputs are intentionally ignored by Git.

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

Run the included script:

```bash
python scripts/extract_radiomics.py
```

The current script expects:

```python
image_folder = "/Users/yichun/Downloads/unannotated images"
mask_folder = "/Users/yichun/Downloads/eroded mask"
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
import SimpleITK as sitk
from radiomics import featureextractor
import pandas as pd
import numpy as np

image_folder = '/Users/yichun/Downloads/unannotated images'
mask_folder = '/Users/yichun/Downloads/eroded mask'

settings = {
    'binWidth': 25,
    'resampledPixelSpacing': None,
    'interpolator': sitk.sitkBSpline,
    'force2D': True,
    'force2Ddimension': 0,
    'label' : 255
}

extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
extractor.enableAllFeatures()

feature_list = []
print(f"Scanning mask folder: {mask_folder} ...")

mask_files = [f for f in os.listdir(mask_folder) if f.startswith('mask_bw_')]

if not mask_files:
    print("Warning: no mask files beginning with 'mask_bw_' were found.")

for mask_filename in mask_files:
    try:
        mask_name_no_ext = os.path.splitext(mask_filename)[0]
        target_id = mask_name_no_ext.replace('mask_bw_', '')
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
    output_csv = 'ultrasound_radiomics_features.csv'
    df.to_csv(output_csv, index=False)
    print(f"Successfully extracted features for {len(df)} patients.")
    print(f"Saved to: {output_csv}")
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

Two MRI correlation analyses are implemented.

### 1. Strict Test-Only Analysis

Output:

```text
selected5_strict_test_only_kpca_mri_correlations.csv
```

This uses only the held-out test split. It is methodologically strict, but the MRI file has 21 subjects, so sample sizes can be small.

### 2. Cross-Fitted MRI-Complete Analysis

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

