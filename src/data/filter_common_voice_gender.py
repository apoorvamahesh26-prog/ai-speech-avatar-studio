"""
filter_common_voice_gender.py

Common Voice's validated.tsv has a 'gender' column, but:
  - Most rows have it BLANK (not every contributor sets it)
  - Raw male/female counts are usually imbalanced (more male
    contributors historically)

This script:
  1. Keeps only rows where gender is exactly 'male' or 'female'
     (drops blank / 'other' / 'non-binary' -- our Gender model
     is a binary male/female classifier, matching the avatar
     selection use case; this is a scoping choice worth stating
     explicitly in your report's limitations section)
  2. Downsamples the majority class so male/female counts match
     exactly -- prevents the model from learning a lazy "always
     predict male" shortcut that would still score well on
     accuracy but be useless in practice.
  3. Writes a clean CSV: data/splits/common_voice_gender.csv
     with columns [filename, gender]
"""

import os
import pandas as pd

TSV_PATH = "data/raw/common_voice/validated.tsv"
OUT_CSV = "data/splits/common_voice_gender.csv"


def main():
    df = pd.read_csv(TSV_PATH, sep="\t")

    df = df[df["gender"].isin(["male", "female"])].copy()
    print(f"Rows with valid gender label: {len(df)}")
    print(df["gender"].value_counts())

    # Balance classes
    min_count = df["gender"].value_counts().min()
    df_balanced = (
        df.groupby("gender", group_keys=False)
        .apply(lambda x: x.sample(n=min_count, random_state=42))
        .reset_index(drop=True)
    )
    print(f"Balanced to {min_count} per class, total {len(df_balanced)} rows")

    out = df_balanced[["path", "gender"]].rename(columns={"path": "filename"})
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"Saved -> {OUT_CSV}")


if __name__ == "__main__":
    main()
