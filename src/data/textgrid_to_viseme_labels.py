"""
textgrid_to_viseme_labels.py

Converts MFA's .TextGrid output (phoneme intervals in seconds) into
a per-frame viseme ID sequence at 30fps -- matching the fps we
already aligned Wav2Vec2 features to in Step 3's
resample_features_to_fps(). This alignment is what lets us pair
features[t] with viseme_label[t] directly, frame for frame, when
training the Viseme model.

Uses textgrid.py's tiny parser via the `textgrid` pip package
(pip install textgrid) rather than hand-rolling Praat TextGrid
parsing, which is fiddly and easy to get subtly wrong.
"""

import os
import glob
import numpy as np
import textgrid  # pip install textgrid

import sys
sys.path.append(os.getcwd())
from src.audio.phoneme_to_viseme import phoneme_to_viseme

MFA_OUTPUT_DIR = "data/mfa_output"
OUT_DIR = "data/processed/viseme_labels"
TARGET_FPS = 30


def textgrid_to_viseme_sequence(tg_path: str, fps: int = TARGET_FPS) -> np.ndarray:
    """
    Reads a single TextGrid, finds the phone tier, and produces a
    (num_frames,) int array of viseme IDs -- one per 1/fps second,
    covering the full duration of the TextGrid.
    """
    tg = textgrid.TextGrid.fromFile(tg_path)

    # MFA's phone tier is usually named "phones" -- find it robustly
    # rather than assuming tier index, since tier order can vary.
    phone_tier = None
    for tier in tg.tiers:
        if "phone" in tier.name.lower():
            phone_tier = tier
            break
    if phone_tier is None:
        raise ValueError(f"No phone tier found in {tg_path}")

    duration = tg.maxTime
    num_frames = int(np.ceil(duration * fps))
    labels = np.zeros(num_frames, dtype=np.int64)  # default: silence (0)

    for interval in phone_tier:
        phoneme = interval.mark.strip()
        if phoneme == "":
            continue  # empty interval = MFA's own silence marker, already 0
        viseme_id = phoneme_to_viseme(phoneme)

        start_frame = int(interval.minTime * fps)
        end_frame = int(np.ceil(interval.maxTime * fps))
        end_frame = min(end_frame, num_frames)
        labels[start_frame:end_frame] = viseme_id

    return labels


def batch_convert():
    os.makedirs(OUT_DIR, exist_ok=True)
    tg_files = glob.glob(os.path.join(MFA_OUTPUT_DIR, "*.TextGrid"))
    failures = []

    for tg_path in tg_files:
        try:
            labels = textgrid_to_viseme_sequence(tg_path)
            stem = os.path.splitext(os.path.basename(tg_path))[0]
            np.save(os.path.join(OUT_DIR, f"{stem}.npy"), labels)
        except Exception as e:
            failures.append((tg_path, str(e)))

    print(f"Converted {len(tg_files) - len(failures)} / {len(tg_files)} TextGrids")
    for f in failures:
        print(" FAILED:", f)


if __name__ == "__main__":
    batch_convert()
