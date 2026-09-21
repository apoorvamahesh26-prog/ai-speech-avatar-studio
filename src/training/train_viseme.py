"""
train_viseme.py

Trains VisemeTransformer with per-TIMESTEP cross-entropy loss (not
per-clip like Emotion) -- every frame gets its own loss term, masked
so padded frames don't contribute.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report

sys.path.append(os.getcwd())
from src.data.viseme_dataset import load_viseme_dataset, MAX_FRAMES
from src.models.viseme_model import VisemeTransformer
from src.audio.phoneme_to_viseme import VISEME_NAMES, NUM_VISEMES

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def make_key_padding_mask(lengths, max_len):
    # True at PADDED positions (matches PyTorch Transformer convention)
    idx = torch.arange(max_len)[None, :]
    return idx >= lengths[:, None]


def masked_cross_entropy(logits, targets, lengths, max_len):
    """
    logits: (batch, T, num_visemes), targets: (batch, T)
    Computes cross-entropy per frame, then masks out padded frames
    before averaging -- otherwise the loss would be diluted by
    (and the model would be partly trained to predict) the
    meaningless zero-padding we added for batching.
    """
    B, T, C = logits.shape
    loss_per_frame = nn.functional.cross_entropy(
        logits.reshape(-1, C), targets.reshape(-1), reduction="none"
    ).reshape(B, T)

    mask = (torch.arange(T, device=logits.device)[None, :] < lengths[:, None].to(logits.device)).float()
    masked_loss = (loss_per_frame * mask).sum() / mask.sum().clamp(min=1)
    return masked_loss


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training VisemeTransformer on {device}")

    data = load_viseme_dataset()

    def to_loader(split, shuffle, batch_size=16):
        ds = TensorDataset(
            torch.tensor(split["X"]), torch.tensor(split["y"]), torch.tensor(split["lengths"])
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = to_loader(data["train"], True)
    val_loader = to_loader(data["val"], False)
    test_loader = to_loader(data["test"], False)

    model = VisemeTransformer(max_len=MAX_FRAMES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_acc = 0.0
    epochs = 60

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for X, y, lengths in train_loader:
            X, y = X.to(device), y.to(device)
            mask = make_key_padding_mask(lengths, X.size(1)).to(device)

            optimizer.zero_grad()
            logits = model(X, key_padding_mask=mask)
            loss = masked_cross_entropy(logits, y, lengths, X.size(1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item() * X.size(0)

        avg_train_loss = total_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss_total = 0.0
        all_preds, all_true = [], []
        with torch.no_grad():
            for X, y, lengths in val_loader:
                X, y = X.to(device), y.to(device)
                mask = make_key_padding_mask(lengths, X.size(1)).to(device)
                logits = model(X, key_padding_mask=mask)
                val_loss_total += masked_cross_entropy(logits, y, lengths, X.size(1)).item() * X.size(0)

                preds = logits.argmax(dim=-1).cpu().numpy()
                y_np = y.cpu().numpy()
                lens_np = lengths.numpy()
                # Only count real (non-padded) frames toward accuracy
                for b in range(X.size(0)):
                    L = lens_np[b]
                    all_preds.extend(preds[b, :L])
                    all_true.extend(y_np[b, :L])

        avg_val_loss = val_loss_total / len(val_loader.dataset)
        val_acc = accuracy_score(all_true, all_preds)
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs} - train_loss: {avg_train_loss:.4f} "
              f"- val_loss: {avg_val_loss:.4f} - val_frame_acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "viseme_transformer_best.pt"))

    # Final test evaluation
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "viseme_transformer_best.pt")))
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for X, y, lengths in test_loader:
            X = X.to(device)
            mask = make_key_padding_mask(lengths, X.size(1)).to(device)
            logits = model(X, key_padding_mask=mask)
            preds = logits.argmax(dim=-1).cpu().numpy()
            y_np = y.numpy()
            lens_np = lengths.numpy()
            for b in range(X.size(0)):
                L = lens_np[b]
                all_preds.extend(preds[b, :L])
                all_true.extend(y_np[b, :L])

    print("\n[VisemeTransformer - Test, per-frame]")
    present_labels = sorted(set(all_true) | set(all_preds))
    print(classification_report(
        all_true, all_preds,
        labels=present_labels,
        target_names=[VISEME_NAMES[i] for i in present_labels],
    ))


if __name__ == "__main__":
    train()
