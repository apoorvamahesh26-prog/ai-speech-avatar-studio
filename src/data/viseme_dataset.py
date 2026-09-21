"""
viseme_dataset.py

Pairs (features[t], viseme_label[t]) frame-by-frame. Feature extraction
(Step 3) and label generation (this step) round frame counts slightly
differently (ceil vs based on actual audio duration), so we trim both
to their shared minimum length per clip rather than assuming exact
equality -- an off-by-one here would silently misalign every single
training pair, which is a much worse bug than a slightly short sequence.
"""

import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split

FEATS_DIR = "data/processed/features/ravdess"
LABELS_DIR = "data/processed/viseme_labels"

MAX_FRAMES = 150  # same horizon as emotion_dataset.py, ~5s at 30fps


def _pad_or_truncate(arr, max_len, pad_value=0):
    T = arr.shape[0]
    if T >= max_len:
        return arr[:max_len], max_len
    pad_shape = (max_len - T,) + arr.shape[1:]
    pad = np.full(pad_shape, pad_value, dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=0), T


def load_viseme_dataset(test_size=0.15, val_size=0.15, seed=42):
    label_files = glob.glob(os.path.join(LABELS_DIR, "*.npy"))

    X_list, y_list, len_list = [], [], []
    skipped = 0

    for label_path in label_files:
        stem = os.path.splitext(os.path.basename(label_path))[0]
        feat_path = os.path.join(FEATS_DIR, f"{stem}.npy")
        if not os.path.exists(feat_path):
            skipped += 1
            continue

        feats = np.load(feat_path)     # (T_feat, 768)
        labels = np.load(label_path)   # (T_label,)

        shared_len = min(feats.shape[0], labels.shape[0])
        feats = feats[:shared_len]
        labels = labels[:shared_len]

        feats_padded, real_len = _pad_or_truncate(feats, MAX_FRAMES)
        labels_padded, _ = _pad_or_truncate(labels, MAX_FRAMES, pad_value=0)

        X_list.append(feats_padded)
        y_list.append(labels_padded)
        len_list.append(real_len)

    if skipped:
        print(f"[warn] skipped {skipped} clips with no matching feature file")

    X = np.array(X_list, dtype=np.float32)     # (N, MAX_FRAMES, 768)
    y = np.array(y_list, dtype=np.int64)       # (N, MAX_FRAMES)
    lengths = np.array(len_list, dtype=np.int64)

    print(f"Loaded {len(X)} viseme-labeled clips")

    idx = np.arange(len(X))
    idx_train, idx_test = train_test_split(idx, test_size=test_size, random_state=seed)
    val_ratio = val_size / (1 - test_size)
    idx_train, idx_val = train_test_split(idx_train, test_size=val_ratio, random_state=seed)

    def subset(idx_):
        return {"X": X[idx_], "y": y[idx_], "lengths": lengths[idx_]}

    print(f"Split -> train: {len(idx_train)}, val: {len(idx_val)}, test: {len(idx_test)}")
    return {"train": subset(idx_train), "val": subset(idx_val), "test": subset(idx_test)}


if __name__ == "__main__":
    data = load_viseme_dataset()
    for split, d in data.items():
        print(split, d["X"].shape, d["y"].shape)
