"""
download_datasets.py

Handles RAVDESS (fully scriptable) and gives explicit manual steps
for Common Voice (Mozilla requires a click-through license agreement
on their site, so it cannot be scripted -- any tool that claims to
auto-download it without you visiting the site is violating their
terms).

Run: python src/data/download_datasets.py
"""

import os
import zipfile
import urllib.request

RAVDESS_URL = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
RAW_DIR = "data/raw"


def download_ravdess():
    os.makedirs(os.path.join(RAW_DIR, "ravdess"), exist_ok=True)
    zip_path = os.path.join(RAW_DIR, "ravdess_speech.zip")

    if not os.path.exists(zip_path):
        print("Downloading RAVDESS speech audio (~200MB)...")
        urllib.request.urlretrieve(RAVDESS_URL, zip_path)
    else:
        print("RAVDESS zip already downloaded, skipping.")

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(os.path.join(RAW_DIR, "ravdess"))
    print("RAVDESS ready at data/raw/ravdess/")


def common_voice_instructions():
    print(
        """
    -----------------------------------------------------------
    Common Voice cannot be auto-downloaded (Mozilla requires you
    to accept their license on-site). Manual steps:

    1. Go to https://commonvoice.mozilla.org/en/datasets
    2. Select Language: English
    3. Download the smallest/latest release (~few GB compressed)
    4. Extract it into: data/raw/common_voice/
       You should end up with:
         data/raw/common_voice/clips/*.mp3
         data/raw/common_voice/validated.tsv   <- has the 'gender' column

    Once done, run: python src/data/filter_common_voice_gender.py
    -----------------------------------------------------------
    """
    )


if __name__ == "__main__":
    download_ravdess()
    common_voice_instructions()
