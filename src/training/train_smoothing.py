"""
train_smoothing.py

Trains SmoothingCNN and evaluates it against the EMA and median
filter baselines on the SAME noisy sequences, using two metrics:
  1. Frame accuracy vs. clean ground truth (did we recover the
     original sequence?)
  2. Flicker rate: fraction of consecutive-frame transitions in the
     OUTPUT sequence (lower = smoother animation). This matters
     because a smoothing method could match ground truth accuracy
     while still producing choppier transitions, or vice versa
     over-smooth and hide legitimate rapid mouth movements -- we
     want both numbers, not just accuracy, to make an honest
     comparison for the report.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score

sys.path.append(os.getcwd())
from src.data.smoothing_dataset import build_smoothing_dataset
from src.models.smoothing_model import SmoothingCNN, ema_smooth, median_filter_smooth
from src.audio.phoneme_to_viseme import NUM_VISEMES

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def ids_to_onehot(ids: np.ndarray, num_classes=NUM_VISEMES) -> np.ndarray:
    """ (..., T) int ids -> (..., T, num_classes) one-hot float """
    return np.eye(num_classes, dtype=np.float32)[ids]


def flicker_rate(id_seq: np.ndarray, length: int) -> float:
    """ Fraction of adjacent real frames where the viseme ID changes. """
    if length <= 1:
        return 0.0
    seq = id_seq[:length]
    changes = np.sum(seq[1:] != seq[:-1])
    return changes / (length - 1)


def evaluate_sequence_method(method_fn, noisy_ids, clean_ids, lengths, is_prob_based: bool):
    """
    method_fn: function that takes a single sequence and returns a
               smoothed sequence (either probs, for EMA, or ids, for
               median filter) -- is_prob_based tells us which and how
               to convert back to ids for scoring.
    """
    accs, flickers = [], []
    for i in range(len(noisy_ids)):
        L = lengths[i]
        if is_prob_based:
            probs = ids_to_onehot(noisy_ids[i])  # start from one-hot "confidence"
            smoothed_probs = method_fn(probs[:L])
            pred_ids = smoothed_probs.argmax(axis=-1)
        else:
            pred_ids = method_fn(noisy_ids[i][:L])

        accs.append(accuracy_score(clean_ids[i][:L], pred_ids))
        flickers.append(flicker_rate(pred_ids, L))

    return np.mean(accs), np.mean(flickers)


def train_smoothing_cnn(data, epochs=40, batch_size=16, lr=1e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[SmoothingCNN] training on {device}")

    def to_tensors(split):
        noisy_onehot = ids_to_onehot(split["noisy"])   # (N, T, C)
        clean_ids = split["clean"]                      # (N, T) -- CE target
        lengths = split["lengths"]
        return (torch.tensor(noisy_onehot), torch.tensor(clean_ids), torch.tensor(lengths))

    def to_loader(split, shuffle):
        ds = TensorDataset(*to_tensors(split))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = to_loader(data["train"], True)
    val_loader = to_loader(data["val"], False)

    model = SmoothingCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    def masked_ce(logits, targets, lengths):
        B, T, C = logits.shape
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, C), targets.reshape(-1), reduction="none"
        ).reshape(B, T)
        mask = (torch.arange(T, device=logits.device)[None, :] < lengths[:, None].to(logits.device)).float()
        return (loss * mask).sum() / mask.sum().clamp(min=1)

    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for noisy, clean, lengths in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)

            optimizer.zero_grad()
            out_logits = model(noisy)  # (B, T, C) -- treated as logits for CE
            loss = masked_ce(out_logits, clean, lengths)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * noisy.size(0)

        avg_loss = total_loss / len(train_loader.dataset)

        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for noisy, clean, lengths in val_loader:
                noisy = noisy.to(device)
                out_logits = model(noisy)
                preds = out_logits.argmax(dim=-1).cpu().numpy()
                for b in range(noisy.size(0)):
                    L = lengths[b].item()
                    all_preds.extend(preds[b, :L])
                    all_true.extend(clean[b, :L].numpy())

        val_acc = accuracy_score(all_true, all_preds)
        print(f"Epoch {epoch+1}/{epochs} - train_loss: {avg_loss:.4f} - val_acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "smoothing_cnn_best.pt"))

    return model


def main():
    data = build_smoothing_dataset()
    test = data["test"]

    # Baseline: no smoothing at all (sanity floor)
    no_smooth_acc, no_smooth_flicker = evaluate_sequence_method(
        lambda ids: ids, test["noisy"], test["clean"], test["lengths"], is_prob_based=False
    )

    # EMA baseline
    ema_acc, ema_flicker = evaluate_sequence_method(
        lambda probs: ema_smooth(probs, alpha=0.4),
        test["noisy"], test["clean"], test["lengths"], is_prob_based=True
    )

    # Median filter baseline
    median_acc, median_flicker = evaluate_sequence_method(
        lambda ids: median_filter_smooth(ids, window=5),
        test["noisy"], test["clean"], test["lengths"], is_prob_based=False
    )

    # Trained CNN
    model = train_smoothing_cnn(data)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "smoothing_cnn_best.pt")))
    model.eval()

    cnn_accs, cnn_flickers = [], []
    with torch.no_grad():
        for i in range(len(test["noisy"])):
            L = test["lengths"][i]
            probs = ids_to_onehot(test["noisy"][i])
            x = torch.tensor(probs).unsqueeze(0).to(device)
            out = model(x).squeeze(0).cpu().numpy()
            pred_ids = out[:L].argmax(axis=-1)
            cnn_accs.append(accuracy_score(test["clean"][i][:L], pred_ids))
            cnn_flickers.append(flicker_rate(pred_ids, L))

    print("\n=== Temporal Smoothing Comparison (Test Set) ===")
    print(f"{'Method':<20} {'Accuracy':<12} {'Flicker Rate':<12}")
    print(f"{'No smoothing':<20} {no_smooth_acc:<12.4f} {no_smooth_flicker:<12.4f}")
    print(f"{'EMA':<20} {ema_acc:<12.4f} {ema_flicker:<12.4f}")
    print(f"{'Median filter':<20} {median_acc:<12.4f} {median_flicker:<12.4f}")
    print(f"{'SmoothingCNN (ours)':<20} {np.mean(cnn_accs):<12.4f} {np.mean(cnn_flickers):<12.4f}")


if __name__ == "__main__":
    main()
