"""
gender_dataset.py

Loads cached Wav2Vec2 .npy features (from Step 3) and pairs them with
gender labels from ravdess_labels.csv and common_voice_gender.csv
(from Step 2).

POOLING FIX (post-deployment diagnosis): naive mean-pooling over the
ENTIRE feature sequence -- including long silent pauses -- distorts
the averaged voice signature on real-world recordings, which tend to
have far more internal silence than RAVDESS's short, speech-dense
scripted clips. Diagnosed directly from a misclassified real
recording (median pitch 220Hz, unambiguously female by acoustic
analysis, but 61% near-silent -- a silence ratio RAVDESS training
clips never exhibit). Fix: pool only over VOICE-ACTIVE frames,
selected via a per-clip relative energy threshold on the Wav2Vec2
feature vectors themselves (frame L2 norm) -- this requires no new
data collection, self-calibrates per clip (robust to absolute
recording loudness differences), and directly targets the measured
failure mode. Retraining with THIS pooling function (not new data)
is what fixes the observed bias -- train and inference must use the
identical pooling method, which is why this function is now shared
via voiced_frame_pool() rather than duplicated inline.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAVDESS_LABELS = "data/splits/ravdess_labels.csv"
RAVDESS_FEATS = "data/processed/features/ravdess"

CV_LABELS = "data/splits/common_voice_gender.csv"
CV_FEATS = "data/processed/features/common_voice"

# Domain-augmented split (see augment_for_mic_domain.py) -- WebM/Opus-
# compressed + noise-augmented copies of RAVDESS, giving the model
# actual exposure to mic-realistic audio during training rather than
# only clean studio recordings. Included automatically if present;
# training works fine without it too (falls back to RAVDESS + Common
# Voice only), so this is additive, not a hard requirement.
AUG_LABELS = "data/splits/ravdess_augmented_labels.csv"
AUG_FEATS = "data/processed/features/ravdess_augmented"

GENDER_TO_ID = {"male": 0, "female": 1}


def voiced_frame_pool(feats: np.ndarray, keep_fraction: float = 0.5) -> np.ndarray:
    """
    Pools a (T, 768) Wav2Vec2 feature sequence into a (768,) vector,
    using only the most voice-active frames rather than the whole
    clip.

    Method: compute the L2 norm of each frame's 768-dim feature
    vector as a voice-activity proxy (silent/near-silent audio tends
    to produce lower or more uniform-magnitude activations than
    voiced speech). Keep the top `keep_fraction` of frames by norm
    WITHIN THIS CLIP (a per-clip percentile threshold, not a fixed
    absolute cutoff) -- this makes it robust to different recording
    loudness levels across clips, unlike a hardcoded energy
    threshold which would need re-tuning per microphone/environment.

    Falls back to full-sequence mean if the clip is too short to
    subsample meaningfully (avoids pooling over 0 frames on edge
    cases).
    """
    if feats.shape[0] < 4:
        return feats.mean(axis=0)

    frame_norms = np.linalg.norm(feats, axis=1)  # (T,)
    threshold = np.percentile(frame_norms, (1 - keep_fraction) * 100)
    mask = frame_norms >= threshold
    if mask.sum() == 0:
        return feats.mean(axis=0)
    return feats[mask].mean(axis=0)


def _stem_group_key(filename: str) -> str:
    """
    Maps any filename (original or augmented) to a shared 'group' key
    so that all variants of the SAME underlying recording (clean +
    WebM/Opus + WebM/Opus+noise) are forced into the SAME split.
    Without this, an augmented copy of a clip could land in train
    while another copy of the same clip lands in test -- leaking
    speaker/content identity across the split and inflating reported
    test accuracy in a way that wouldn't hold on genuinely unseen
    voices.
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    for suffix in ("_webmopus_noisy", "_webmopus"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _load_split(labels_csv: str, feats_dir: str):
    df = pd.read_csv(labels_csv)
    X, y, groups = [], [], []
    missing = 0

    for _, row in df.iterrows():
        stem = os.path.splitext(os.path.basename(row["filename"]))[0]
        feat_path = os.path.join(feats_dir, f"{stem}.npy")
        if not os.path.exists(feat_path):
            missing += 1
            continue
        feats = np.load(feat_path)          # (T, 768)
        pooled = voiced_frame_pool(feats)     # (768,) -- voice-active pooling, see above
        X.append(pooled)
        y.append(GENDER_TO_ID[row["gender"]])
        groups.append(_stem_group_key(row["filename"]))

    if missing:
        print(f"[warn] {missing} rows in {labels_csv} had no matching feature file "
              f"(run Step 3 feature extraction on this directory first)")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), np.array(groups)


def load_gender_dataset(test_size: float = 0.15, val_size: float = 0.15, seed: int = 42):
    """
    Merges RAVDESS + Common Voice + mic-domain-augmented gender-labeled
    features, then does a GROUP-AWARE stratified split: all variants
    of the same original recording (clean, WebM/Opus, WebM/Opus+noise)
    are kept together in the same split, so augmentation never leaks
    the same underlying speaker/content across train and test (see
    _stem_group_key's docstring for why this matters).
    Returns dict with 'train', 'val', 'test' -> (X, y) tuples.
    """
    from sklearn.model_selection import GroupShuffleSplit

    X_parts, y_parts, group_parts = [], [], []

    for labels_csv, feats_dir, name in [
        (RAVDESS_LABELS, RAVDESS_FEATS, "RAVDESS"),
        (CV_LABELS, CV_FEATS, "Common Voice"),
        (AUG_LABELS, AUG_FEATS, "RAVDESS (mic-domain augmented)"),
    ]:
        if os.path.exists(labels_csv):
            X, y, groups = _load_split(labels_csv, feats_dir)
            print(f"{name}: {len(X)} usable samples")
            X_parts.append(X)
            y_parts.append(y)
            group_parts.append(groups)
        else:
            print(f"[warn] {labels_csv} not found, skipping {name}")

    X_all = np.concatenate(X_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)
    groups_all = np.concatenate(group_parts, axis=0)
    print(f"Total merged dataset: {len(X_all)} samples across {len(set(groups_all))} unique recordings")

    # Group-aware split: GroupShuffleSplit guarantees every sample
    # sharing a group key (i.e. every augmented variant of the same
    # original clip) ends up in the SAME split.
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(gss_test.split(X_all, y_all, groups=groups_all))

    val_ratio_of_remaining = val_size / (1 - test_size)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_ratio_of_remaining, random_state=seed)
    train_idx_rel, val_idx_rel = next(gss_val.split(
        X_all[train_val_idx], y_all[train_val_idx], groups=groups_all[train_val_idx]
    ))
    train_idx = train_val_idx[train_idx_rel]
    val_idx = train_val_idx[val_idx_rel]

    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]
    X_test, y_test = X_all[test_idx], y_all[test_idx]

    print(f"Split sizes -> train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")
    print(f"Class balance -> train: {np.bincount(y_train)}, val: {np.bincount(y_val)}, test: {np.bincount(y_test)}")

    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }


if __name__ == "__main__":
    data = load_gender_dataset()
    for split_name, (X, y) in data.items():
        print(split_name, X.shape, y.shape, "class balance:", np.bincount(y))
