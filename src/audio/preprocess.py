"""
preprocess.py

Uniform audio preprocessing applied to every dataset before feature
extraction. Doing this in ONE place (rather than per-dataset) matters:
if RAVDESS and Common Voice were normalized differently, Wav2Vec2
features downstream would carry a dataset-specific bias, and your
Gender/Emotion models could accidentally learn to distinguish
"which dataset this came from" instead of the actual signal.

Steps, and why each is here:
  1. Resample to 16kHz  -> Wav2Vec2 was pretrained on 16kHz audio.
     Feeding it anything else silently degrades feature quality.
  2. Convert to mono     -> stereo doubles data with no extra info
     for speech; Wav2Vec2 expects single-channel input.
  3. Loudness normalize  -> RAVDESS (studio) and Common Voice
     (crowdsourced mics) have very different recording levels.
     Peak-normalizing puts them on comparable footing.
  4. Trim leading/trailing silence -> long silence padding wastes
     compute and can bias viseme label 0 (silence) to be
     over-represented.
"""

import os
import glob
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

TARGET_SR = 16000


def load_and_clean(path: str, top_db: int = 30):
    """
    Load an audio file and apply the four steps above.
    top_db: silence threshold for trimming (librosa.effects.trim).
            30dB is a moderate threshold -- aggressive enough to drop
            true silence, gentle enough not to clip soft speech onsets.
    """
    # librosa.load with sr=TARGET_SR resamples AND converts to mono
    # in one call.
    y, sr = librosa.load(path, sr=TARGET_SR, mono=True)

    # Peak normalization: scale so max absolute sample = 1.0.
    # We guard against silent/empty files (max=0) to avoid div-by-zero.
    peak = np.max(np.abs(y)) if len(y) > 0 else 0.0
    if peak > 0:
        y = y / peak

    # Trim leading/trailing silence only (not internal pauses --
    # internal pauses are real speech content we want to keep,
    # they'll just map to viseme class 0 at those frames).
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)

    return y_trimmed, TARGET_SR


def process_directory(input_dir: str, output_dir: str, ext: str = "*.wav"):
    """
    Batch-process every audio file in input_dir, writing cleaned
    versions to output_dir with the same filename. Returns a list
    of (input_path, output_path) that failed, so you can inspect
    corrupt/unreadable files instead of them silently vanishing.
    """
    os.makedirs(output_dir, exist_ok=True)
    files = glob.glob(os.path.join(input_dir, "**", ext), recursive=True)
    failures = []

    for path in tqdm(files, desc=f"Preprocessing {input_dir}"):
        try:
            y, sr = load_and_clean(path)
            if len(y) == 0:
                failures.append((path, "empty after trim"))
                continue
            out_name = os.path.splitext(os.path.basename(path))[0] + ".wav"
            out_path = os.path.join(output_dir, out_name)
            sf.write(out_path, y, sr)
        except Exception as e:
            failures.append((path, str(e)))

    return failures


if __name__ == "__main__":
    # Example usage -- adjust paths once datasets are downloaded (below).
    fails = process_directory(
        input_dir="data/raw/ravdess",
        output_dir="data/processed/ravdess",
    )
    print(f"RAVDESS: {len(fails)} failures")
    for f in fails:
        print(" ", f)
