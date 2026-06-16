import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor


IMAGE_FOLDER = Path("/Users/yichun/Downloads/unannotated images")
MASK_FOLDER = Path("/Users/yichun/Downloads/eroded mask")
OUTPUT_CSV = Path("ultrasound_radiomics_features.csv")

SETTINGS = {
    "binWidth": 25,
    "resampledPixelSpacing": None,
    "interpolator": sitk.sitkBSpline,
    "force2D": True,
    "force2Ddimension": 0,
    "label": 255,
}


def to_single_channel(image):
    if image.GetNumberOfComponentsPerPixel() > 1:
        return sitk.VectorIndexSelectionCast(image, 0)
    return image


def main():
    extractor = featureextractor.RadiomicsFeatureExtractor(**SETTINGS)
    extractor.enableAllFeatures()

    feature_list = []
    print(f"Scanning mask folder: {MASK_FOLDER}")

    mask_files = [f for f in os.listdir(MASK_FOLDER) if f.startswith("mask_bw_")]
    if not mask_files:
        print("Warning: no mask files beginning with 'mask_bw_' were found.")

    for mask_filename in mask_files:
        try:
            mask_name_no_ext = os.path.splitext(mask_filename)[0]
            target_id = mask_name_no_ext.replace("mask_bw_", "")
            search_pattern = str(IMAGE_FOLDER / f"{target_id}.*")
            potential_images = glob.glob(search_pattern)

            if not potential_images:
                print(f"Skipping {mask_filename}: no source image found for ID '{target_id}'")
                continue

            image_path = potential_images[0]
            mask_path = MASK_FOLDER / mask_filename
            print(f"Processing {target_id}: {os.path.basename(image_path)} + {mask_filename}")

            image = to_single_channel(sitk.ReadImage(image_path))
            mask = to_single_channel(sitk.ReadImage(str(mask_path)))
            result = extractor.execute(image, mask)

            patient_features = {"PatientID": target_id}
            for key, value in result.items():
                if key.startswith("original_"):
                    patient_features[key] = float(value) if isinstance(value, np.ndarray) else value

            feature_list.append(patient_features)

        except Exception as exc:
            print(f"Error processing {mask_filename}: {exc}")

    if not feature_list:
        print("No radiomics features were extracted. Check paths, file names, and mask labels.")
        return

    df = pd.DataFrame(feature_list)
    df.to_csv(OUTPUT_CSV, index=False)
    print("-" * 30)
    print(f"Extracted features for {len(df)} cases.")
    print(f"Saved: {OUTPUT_CSV}")
    print(df.iloc[:, :3].head())


if __name__ == "__main__":
    main()
