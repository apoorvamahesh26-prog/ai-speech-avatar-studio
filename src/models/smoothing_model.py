"""
smoothing_model.py

Classical baselines (EMA, median filter) operate directly on a
sequence of viseme IDs/probabilities -- no training, pure signal
processing. The learned SmoothingCNN is your trained model for this
step, per the project's "own model" requirement.
"""

import numpy as np
import torch
import torch.nn as nn

import sys, os
sys.path.append(os.getcwd())
from src.audio.phoneme_to_viseme import NUM_VISEMES


# ---------------------------------------------------------------------
# Classical baselines
# ---------------------------------------------------------------------
def ema_smooth(viseme_probs: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    viseme_probs: (T, NUM_VISEMES) softmax probabilities per frame
                  (NOT hard argmax labels -- smoothing probabilities
                  before taking the final class is more principled
                  than smoothing discrete IDs, since ID smoothing has
                  no natural notion of "distance" between classes).
    alpha: weight on the current frame; lower alpha = more smoothing,
           more lag. 0.4 is a moderate default -- tune based on val set.
    Returns smoothed probabilities, same shape.
    """
    smoothed = np.zeros_like(viseme_probs)
    smoothed[0] = viseme_probs[0]
    for t in range(1, len(viseme_probs)):
        smoothed[t] = alpha * viseme_probs[t] + (1 - alpha) * smoothed[t - 1]
    return smoothed


def median_filter_smooth(viseme_ids: np.ndarray, window: int = 5) -> np.ndarray:
    """
    viseme_ids: (T,) hard argmax viseme sequence.
    window: odd number, frames on each side to consider (window=5 ->
            looks at t-2..t+2). Operates on discrete IDs directly via
            majority vote -- simpler and more intuitive than EMA for
            catching single-frame outlier spikes specifically, though
            it can't use probability confidence the way EMA does.
    """
    T = len(viseme_ids)
    half = window // 2
    smoothed = viseme_ids.copy()

    for t in range(T):
        lo, hi = max(0, t - half), min(T, t + half + 1)
        window_vals = viseme_ids[lo:hi]
        # majority vote
        vals, counts = np.unique(window_vals, return_counts=True)
        smoothed[t] = vals[np.argmax(counts)]

    return smoothed


# ---------------------------------------------------------------------
# Learned model
# ---------------------------------------------------------------------
class SmoothingCNN(nn.Module):
    """
    Input: (batch, T, NUM_VISEMES) softmax probability sequence
           (the noisy/raw distribution -- see smoothing_dataset.py for
           how training pairs are constructed).
    Output: (batch, T, NUM_VISEMES) refined probability sequence.

    Architecture: small stack of 1D convolutions along the TIME axis
    (kernel slides across frames, not across viseme classes) -- this
    gives each output frame a receptive field spanning several
    neighboring frames, letting the model learn local temporal
    consistency patterns (e.g. "a single-frame spike surrounded by
    the same viseme on both sides is probably noise") directly from
    data, rather than a fixed hand-set window/alpha like the
    classical baselines.

    Residual connection: output = input + learned_correction. This is
    a deliberate design choice -- the model only needs to learn the
    CORRECTION (small deltas to fix flicker), not reconstruct the
    entire distribution from scratch, which is an easier learning
    problem and also guarantees good behavior early in training
    (before the correction branch has learned anything useful, the
    residual identity path still passes through a reasonable output).
    """
    def __init__(self, num_visemes=NUM_VISEMES, hidden=32, kernel_size=5):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(num_visemes, hidden, kernel_size, padding=pad),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),

            nn.Conv1d(hidden, hidden, kernel_size, padding=pad),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),

            nn.Conv1d(hidden, num_visemes, kernel_size, padding=pad),
        )

    def forward(self, x):
        # x: (batch, T, num_visemes) -> conv1d wants (batch, channels, T)
        x_t = x.transpose(1, 2)             # (batch, num_visemes, T)
        correction = self.net(x_t)           # (batch, num_visemes, T)
        out = x_t + correction               # residual connection
        out = out.transpose(1, 2)            # back to (batch, T, num_visemes)
        return out
