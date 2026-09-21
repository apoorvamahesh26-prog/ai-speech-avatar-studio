"""
train_emotion.py

Trains RandomForest baseline + EmotionBiLSTM, reports per-class
metrics (important for an 8-class problem -- overall accuracy can
hide the fact that some emotions, e.g. 'calm' vs 'neutral', are
much harder to distinguish than others).
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.getcwd())
from src.data.emotion_dataset import load_emotion_dataset, EMOTION_LIST
from src.models.emotion_model import RandomForestEmotionClassifier, EmotionBiLSTM

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def plot_confusion(y_true, y_pred, title, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(EMOTION_LIST))))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=EMOTION_LIST, yticklabels=EMOTION_LIST, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved confusion matrix -> {out_path}")


def train_random_forest(data):
    clf = RandomForestEmotionClassifier()
    clf.fit(data["train"]["X_pooled"], data["train"]["y"])

    preds = clf.predict(data["test"]["X_pooled"])
    y_true = data["test"]["y"]

    print("\n[Random Forest - Emotion]")
    print(classification_report(y_true, preds, target_names=EMOTION_LIST))
    plot_confusion(y_true, preds, "RandomForest Emotion Confusion Matrix",
                    "checkpoints/emotion_rf_confusion.png")

    clf.save(os.path.join(CHECKPOINT_DIR, "emotion_rf.joblib"))
    return accuracy_score(y_true, preds)


def train_bilstm(data, epochs=40, batch_size=16, lr=1e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[EmotionBiLSTM] training on {device}")

    def to_loader(split, shuffle):
        ds = TensorDataset(
            torch.tensor(split["X_seq"]),
            torch.tensor(split["lengths"]),
            torch.tensor(split["y"]),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = to_loader(data["train"], True)
    val_loader = to_loader(data["val"], False)
    test_loader = to_loader(data["test"], False)

    model = EmotionBiLSTM().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # ReduceLROnPlateau: halves the LR if val loss stalls -- helps
    # squeeze out extra convergence on a modest-sized dataset where
    # a fixed LR schedule might overshoot as training progresses.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                             factor=0.5, patience=4)

    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for X_batch, len_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch, len_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            # Gradient clipping: LSTMs are prone to occasional gradient
            # spikes; clipping the norm prevents one bad batch from
            # destabilizing training.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)

        avg_train_loss = total_loss / len(train_loader.dataset)

        model.eval()
        val_loss_total = 0.0
        val_preds, val_true = [], []
        with torch.no_grad():
            for X_batch, len_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch, len_batch)
                val_loss_total += criterion(logits, y_batch).item() * X_batch.size(0)
                val_preds.extend(logits.argmax(dim=1).cpu().numpy())
                val_true.extend(y_batch.cpu().numpy())

        avg_val_loss = val_loss_total / len(val_loader.dataset)
        val_acc = accuracy_score(val_true, val_preds)
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs} - train_loss: {avg_train_loss:.4f} "
              f"- val_loss: {avg_val_loss:.4f} - val_acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "emotion_bilstm_best.pt"))

    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "emotion_bilstm_best.pt")))
    model.eval()
    test_preds, test_true = [], []
    with torch.no_grad():
        for X_batch, len_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch, len_batch)
            test_preds.extend(logits.argmax(dim=1).cpu().numpy())
            test_true.extend(y_batch.numpy())

    print("\n[EmotionBiLSTM - Test]")
    print(classification_report(test_true, test_preds, target_names=EMOTION_LIST))
    plot_confusion(test_true, test_preds, "BiLSTM Emotion Confusion Matrix",
                    "checkpoints/emotion_bilstm_confusion.png")

    return accuracy_score(test_true, test_preds)


if __name__ == "__main__":
    data = load_emotion_dataset()

    rf_acc = train_random_forest(data)
    bilstm_acc = train_bilstm(data)

    print("\n=== Final Comparison (Test Accuracy) ===")
    print(f"RandomForest (pooled):  {rf_acc:.4f}")
    print(f"BiLSTM (sequence):      {bilstm_acc:.4f}")
