"""
emotion_dataset.py

Unlike gender_dataset.py, this KEEPS the temporal sequence rather than
pooling immediately -- emotion is a sequential pattern, not a static
property (see Step 5 explanation). We still provide a pooled version
too, purely for the Random Forest baseline comparison.

Sequences have different lengths (clips are different durations), so
we pad/truncate to a fixed MAX_FRAMES for batching. Padding uses zeros
plus an explicit attention/length mask so the BiLSTM (Step 5 model)
can ignore padded positions rather than being confused by them.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAVDESS_LABELS = "data/splits/ravdess_labels.csv"
RAVDESS_FEATS = "data/processed/features/ravdess"

EMOTION_LIST = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
EMOTION_TO_ID = {e: i for i, e in enumerate(EMOTION_LIST)}

MAX_FRAMES = 150  # ~5 seconds at 30fps -- covers nearly all RAVDESS clips;
                   # longer clips get truncated, shorter get zero-padded


def _pad_or_truncate(feats: np.ndarray, max_len: int = MAX_FRAMES):
    T = feats.shape[0]
    if T >= max_len:
        return feats[:max_len], max_len  # truncate, full length mask
    pad = np.zeros((max_len - T, feats.shape[1]), dtype=feats.dtype)
    padded = np.concatenate([feats, pad], axis=0)
    return padded, T  # real length before padding, for the mask


def load_emotion_dataset(test_size=0.15, val_size=0.15, seed=42):
    df = pd.read_csv(RAVDESS_LABELS)

    X_seq, lengths, X_pooled, y = [], [], [], []
    missing = 0

    for _, row in df.iterrows():
        stem = os.path.splitext(os.path.basename(row["filename"]))[0]
        feat_path = os.path.join(RAVDESS_FEATS, f"{stem}.npy")
        if not os.path.exists(feat_path):
            missing += 1
            continue

        feats = np.load(feat_path)  # (T, 768)
        padded, real_len = _pad_or_truncate(feats)

        X_seq.append(padded)
        lengths.append(real_len)
        X_pooled.append(feats.mean(axis=0))  # pooled version, for RF baseline
        y.append(EMOTION_TO_ID[row["emotion"]])

    if missing:
        print(f"[warn] {missing} rows had no matching feature file")

    X_seq = np.array(X_seq, dtype=np.float32)      # (N, MAX_FRAMES, 768)
    lengths = np.array(lengths, dtype=np.int64)      # (N,)
    X_pooled = np.array(X_pooled, dtype=np.float32)  # (N, 768)
    y = np.array(y, dtype=np.int64)

    print(f"Loaded {len(y)} samples, class distribution:",
          {EMOTION_LIST[i]: c for i, c in zip(*np.unique(y, return_counts=True))})

    idx = np.arange(len(y))
    idx_train, idx_test = train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)
    val_ratio = val_size / (1 - test_size)
    idx_train, idx_val = train_test_split(idx_train, test_size=val_ratio,
                                            stratify=y[idx_train], random_state=seed)

    def subset(idx_):
        return {
            "X_seq": X_seq[idx_], "lengths": lengths[idx_],
            "X_pooled": X_pooled[idx_], "y": y[idx_],
        }

    print(f"Split sizes -> train: {len(idx_train)}, val: {len(idx_val)}, test: {len(idx_test)}")

    return {"train": subset(idx_train), "val": subset(idx_val), "test": subset(idx_test)}


if __name__ == "__main__":
    data = load_emotion_dataset()
    for split, d in data.items():
        print(split, "X_seq:", d["X_seq"].shape, "y:", d["y"].shape)
