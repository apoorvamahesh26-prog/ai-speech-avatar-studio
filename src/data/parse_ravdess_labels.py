"""
parse_ravdess_labels.py

RAVDESS filenames encode everything in their name, e.g.:
    03-01-06-01-02-01-12.wav
     |  |  |  |  |  |  |
     |  |  |  |  |  |  Actor ID (12 -> even -> female)
     |  |  |  |  |  Repetition
     |  |  |  |  Statement
     |  |  |  Intensity (01=normal, 02=strong)
     |  |  Emotion (01-08, see EMOTION_MAP)
     |  Vocal channel (01=speech, 02=song) -- we only use 01
     Modality (03=audio-only)

Actor ID parity gives gender for free: odd = male, even = female.
This script walks the extracted RAVDESS folder and builds one CSV
with [filename, gender, emotion] so downstream training scripts
never have to re-parse filenames.
"""

import os
import glob
import pandas as pd

RAVDESS_DIR = "data/processed/ravdess"
OUT_CSV = "data/splits/ravdess_labels.csv"

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}


def parse_filename(fname: str):
    stem = os.path.splitext(os.path.basename(fname))[0]
    parts = stem.split("-")
    if len(parts) != 7:
        return None  # not a standard RAVDESS speech filename

    modality, vocal_channel, emotion_code, intensity, statement, rep, actor_id = parts

    if vocal_channel != "01":
        return None  # skip song files, keep speech only

    gender = "male" if int(actor_id) % 2 == 1 else "female"
    emotion = EMOTION_MAP.get(emotion_code, "unknown")

    return {
        "filename": fname,
        "gender": gender,
        "emotion": emotion,
        "intensity": "normal" if intensity == "01" else "strong",
        "actor_id": actor_id,
    }


def main():
    files = glob.glob(os.path.join(RAVDESS_DIR, "**", "*.wav"), recursive=True)
    rows = [r for r in (parse_filename(f) for f in files) if r is not None]
    df = pd.DataFrame(rows)

    print(f"Parsed {len(df)} speech files")
    print("Gender distribution:\n", df["gender"].value_counts())
    print("Emotion distribution:\n", df["emotion"].value_counts())

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved -> {OUT_CSV}")


if __name__ == "__main__":
    main()
