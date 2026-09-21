"""
build_augmented_labels.py

Each augmented file (e.g. "03-01-01-01-01-01-05_webmopus.wav") is
derived from an original RAVDESS file ("03-01-01-01-01-01-05.wav")
whose gender we already know (parse_ravdess_labels.py, Step 2's
odd/even actor-ID convention). This script strips the augmentation
suffix to recover the original stem, looks up its gender from the
already-generated ravdess_labels.csv, and writes a matching labels
file for the augmented folder -- so augmented data can be loaded by
gender_dataset.py exactly like any other labeled split.
"""

import os
import glob
import pandas as pd

ORIGINAL_LABELS = "data/splits/ravdess_labels.csv"
AUGMENTED_DIR = "data/processed/ravdess_augmented"
OUT_CSV = "data/splits/ravdess_augmented_labels.csv"

AUGMENT_SUFFIXES = ["_webmopus_noisy", "_webmopus"]  # longer suffix first, so it's stripped correctly


def original_stem(augmented_filename: str) -> str:
    stem = os.path.splitext(os.path.basename(augmented_filename))[0]
    for suffix in AUGMENT_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def main():
    orig_df = pd.read_csv(ORIGINAL_LABELS)
    orig_lookup = {
        os.path.splitext(os.path.basename(row["filename"]))[0]: row["gender"]
        for _, row in orig_df.iterrows()
    }

    aug_files = glob.glob(os.path.join(AUGMENTED_DIR, "*.wav"))
    rows = []
    missing = 0
    for path in aug_files:
        orig_stem = original_stem(path)
        gender = orig_lookup.get(orig_stem)
        if gender is None:
            missing += 1
            continue
        rows.append({"filename": path, "gender": gender})

    if missing:
        print(f"[warn] {missing} augmented files had no matching original label")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(df)} augmented labels -> {OUT_CSV}")
    print("Gender balance:", df["gender"].value_counts().to_dict())


if __name__ == "__main__":
    main()
