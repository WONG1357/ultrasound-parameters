import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor


SETTINGS = {
    "binWidth": 25,
    "resampledPixelSpacing": None,
    "interpolator": sitk.sitkBSpline,
    "force2D": True,
    "force2Ddimension": 0,
    "label": 255,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract 2D ultrasound radiomics features from image/mask pairs."
    )
    parser.add_argument(
        "--image-folder",
        default="data/raw/ultrasound_images",
        help="Folder containing source ultrasound images.",
    )
    parser.add_argument(
        "--mask-folder",
        default="data/raw/eroded_masks",
        help="Folder containing binary ROI masks.",
    )
    parser.add_argument(
        "--output-csv",
        default="data/processed/ultrasound_radiomics_features.csv",
        help="Output CSV path for extracted radiomics features.",
    )
    parser.add_argument(
        "--mask-prefix",
        default="mask_bw_",
        help="Prefix used in mask filenames before the image/case ID.",
    )
    parser.add_argument(
        "--label",
        type=int,
        default=255,
        help="Mask label value to extract. Use 255 for white binary masks.",
    )
    return parser.parse_args()


def to_single_channel(image):
    if image.GetNumberOfComponentsPerPixel() > 1:
        return sitk.VectorIndexSelectionCast(image, 0)
    return image


def main():
    args = parse_args()
    image_folder = Path(args.image_folder)
    mask_folder = Path(args.mask_folder)
    output_csv = Path(args.output_csv)

    settings = dict(SETTINGS)
    settings["label"] = args.label

    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.enableAllFeatures()

    feature_list = []
    if not image_folder.exists():
        raise FileNotFoundError(f"Image folder does not exist: {image_folder}")
    if not mask_folder.exists():
        raise FileNotFoundError(f"Mask folder does not exist: {mask_folder}")

    print(f"Scanning mask folder: {mask_folder}")

    mask_files = [f for f in os.listdir(mask_folder) if f.startswith(args.mask_prefix)]
    if not mask_files:
        print(f"Warning: no mask files beginning with '{args.mask_prefix}' were found.")

    for mask_filename in mask_files:
        try:
            mask_name_no_ext = os.path.splitext(mask_filename)[0]
            target_id = mask_name_no_ext.replace(args.mask_prefix, "", 1)
            search_pattern = str(image_folder / f"{target_id}.*")
            potential_images = glob.glob(search_pattern)

            if not potential_images:
                print(f"Skipping {mask_filename}: no source image found for ID '{target_id}'")
                continue

            image_path = potential_images[0]
            mask_path = mask_folder / mask_filename
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
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print("-" * 30)
    print(f"Extracted features for {len(df)} cases.")
    print(f"Saved: {output_csv}")
    print(df.iloc[:, :3].head())


if __name__ == "__main__":
    main()
