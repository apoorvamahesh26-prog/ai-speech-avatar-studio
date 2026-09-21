"""
smoothing_dataset.py

Self-supervised setup: ground-truth viseme label sequences (from
Step 6's MFA-derived labels) are treated as the "clean" target.
We synthetically corrupt a copy by randomly flipping some frames to
a random other viseme class -- simulating the kind of single-frame
misclassifications the Viseme Transformer makes -- and train the
smoothing model to reconstruct the clean sequence from the noisy one.

Why synthetic noise instead of using the Viseme Transformer's actual
raw predictions as the noisy input: at this stage the Viseme model's
error PATTERN is a moving target as it keeps training/changing. A
model trained to fix "random single-frame flips" learns the general
smoothing task (favor local majority/continuity) which transfers
regardless of the specific error distribution -- and is much simpler
to set up than re-running Viseme inference for every smoothing
training example. We revisit real error patterns at Step 11 (full
pipeline integration) if additional fine-tuning is warranted.
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split

import sys
sys.path.append(os.getcwd())
from src.audio.phoneme_to_viseme import NUM_VISEMES
from src.data.viseme_dataset import load_viseme_dataset  # reuses clean labels + lengths


def inject_noise(clean_seq: np.ndarray, length: int, flip_prob: float = 0.12, seed=None):
    """
    clean_seq: (MAX_FRAMES,) int array of viseme IDs (may include padding
               past `length` -- we only corrupt REAL frames).
    flip_prob: probability each real frame gets randomly reassigned to
               a different viseme -- 12% is a deliberately aggressive
               noise level (higher than we'd expect from the real
               Viseme model) so the smoothing model learns a robust
               general correction behavior rather than overfitting to
               a narrow noise regime.
    """
    rng = np.random.default_rng(seed)
    noisy = clean_seq.copy()

    for t in range(length):
        if rng.random() < flip_prob:
            # pick any viseme different from the current one
            choices = [v for v in range(NUM_VISEMES) if v != clean_seq[t]]
            noisy[t] = rng.choice(choices)

    return noisy


def build_smoothing_dataset(flip_prob=0.12, seed=42):
    """
    Reuses the SAME clean viseme sequences + splits as viseme_dataset.py
    (train/val/test), so smoothing model evaluation is on the same
    held-out clips as the Viseme model -- keeps Step 12's comparisons
    consistent across the whole pipeline.
    """
    viseme_data = load_viseme_dataset()  # {'train': {...}, 'val': {...}, 'test': {...}}

    out = {}
    for split_name, split in viseme_data.items():
        clean = split["y"]          # (N, MAX_FRAMES)
        lengths = split["lengths"]  # (N,)

        noisy = np.stack([
            inject_noise(clean[i], lengths[i], flip_prob=flip_prob, seed=seed + i)
            for i in range(len(clean))
        ])

        out[split_name] = {"noisy": noisy, "clean": clean, "lengths": lengths}
        print(f"{split_name}: {len(clean)} sequences, "
              f"noise flip_prob={flip_prob}")

    return out


if __name__ == "__main__":
    data = build_smoothing_dataset()
    # Quick sanity check: print how much a sample sequence actually changed
    sample_clean = data["train"]["clean"][0]
    sample_noisy = data["train"]["noisy"][0]
    length = data["train"]["lengths"][0]
    diff_frac = np.mean(sample_clean[:length] != sample_noisy[:length])
    print(f"Sample sequence: {diff_frac:.2%} of frames altered by noise injection")
