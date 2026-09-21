"""
prepare_mfa_input.py

RAVDESS's 5th filename field is the 'statement' code:
  01 -> "Kids are talking by the door"
  02 -> "Dogs are sitting by the door"
(see parse_ravdess_labels.py's filename breakdown from Step 2)

MFA needs a .txt transcript next to each .wav with the SAME base
filename. This script copies/symlinks the cleaned wavs into
data/mfa_input/ and writes the matching transcript for each.
"""

import os
import glob
import shutil

RAVDESS_PROCESSED = "data/processed/ravdess"
MFA_INPUT_DIR = "data/mfa_input"

STATEMENT_TEXT = {
    "01": "Kids are talking by the door",
    "02": "Dogs are sitting by the door",
}


def main():
    os.makedirs(MFA_INPUT_DIR, exist_ok=True)
    files = glob.glob(os.path.join(RAVDESS_PROCESSED, "*.wav"))

    written = 0
    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]
        parts = stem.split("-")
        if len(parts) != 7:
            continue
        statement_code = parts[4]
        text = STATEMENT_TEXT.get(statement_code)
        if text is None:
            continue

        dest_wav = os.path.join(MFA_INPUT_DIR, f"{stem}.wav")
        dest_txt = os.path.join(MFA_INPUT_DIR, f"{stem}.txt")

        shutil.copyfile(path, dest_wav)
        with open(dest_txt, "w") as f:
            f.write(text)

        written += 1

    print(f"Prepared {written} wav/txt pairs in {MFA_INPUT_DIR}/")
    print("Next, run MFA (in the mfa_env conda environment):")
    print(f"  mfa align {MFA_INPUT_DIR} english_us_arpa english_us_arpa data/mfa_output")


if __name__ == "__main__":
    main()
