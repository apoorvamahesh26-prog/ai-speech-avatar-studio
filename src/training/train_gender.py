"""
train_gender.py

Trains BOTH gender models on the same data split and reports metrics
side by side. This is the script that produces the comparison table
for your report's Step 4 section.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

sys.path.append(os.getcwd())
from src.data.gender_dataset import load_gender_dataset
from src.models.gender_model import RandomForestGenderClassifier, GenderCNN

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def evaluate(y_true, y_pred, name=""):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n--- {name} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"Confusion matrix:\n{cm}")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def train_random_forest(data):
    X_train, y_train = data["train"]
    X_val, y_val = data["val"]
    X_test, y_test = data["test"]

    clf = RandomForestGenderClassifier()
    clf.fit(X_train, y_train)

    print("\n[Random Forest]")
    evaluate(y_val, clf.predict(X_val), name="Validation")
    test_metrics = evaluate(y_test, clf.predict(X_test), name="Test")

    clf.save(os.path.join(CHECKPOINT_DIR, "gender_rf.joblib"))
    return test_metrics


def train_cnn(data, epochs=30, batch_size=32, lr=1e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[GenderCNN] training on {device}")

    def to_loader(X, y, shuffle):
        ds = TensorDataset(torch.tensor(X), torch.tensor(y))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = to_loader(*data["train"], shuffle=True)
    val_loader = to_loader(*data["val"], shuffle=False)
    test_loader = to_loader(*data["test"], shuffle=False)

    model = GenderCNN().to(device)

    # CrossEntropyLoss: standard choice for multi-class (here binary)
    # classification with logits output -- combines log-softmax +
    # negative log-likelihood in one numerically stable step.
    criterion = nn.CrossEntropyLoss()

    # Adam: adaptive learning rate per parameter, converges reliably
    # without much tuning -- a sensible default for a small model
    # like this rather than plain SGD, which would need a hand-tuned
    # learning rate schedule to converge as smoothly.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # weight_decay: L2 regularization -- again, guarding against
    # overfitting on a comparatively small dataset.

    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)

        avg_loss = total_loss / len(train_loader.dataset)

        # Validation pass
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                logits = model(X_batch)
                preds = logits.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(y_batch.numpy())

        val_acc = accuracy_score(val_true, val_preds)
        print(f"Epoch {epoch+1}/{epochs} - train_loss: {avg_loss:.4f} - val_acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "gender_cnn_best.pt"))

    # Load best checkpoint and evaluate on test set
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "gender_cnn_best.pt")))
    model.eval()
    test_preds, test_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            preds = logits.argmax(dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_true.extend(y_batch.numpy())

    test_metrics = evaluate(test_true, test_preds, name="Test (best val checkpoint)")
    return test_metrics


if __name__ == "__main__":
    data = load_gender_dataset()

    rf_metrics = train_random_forest(data)
    cnn_metrics = train_cnn(data)

    print("\n=== Final Comparison (Test Set) ===")
    print(f"{'Model':<15} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8}")
    print(f"{'RandomForest':<15} {rf_metrics['accuracy']:<8.4f} {rf_metrics['precision']:<8.4f} "
          f"{rf_metrics['recall']:<8.4f} {rf_metrics['f1']:<8.4f}")
    print(f"{'GenderCNN':<15} {cnn_metrics['accuracy']:<8.4f} {cnn_metrics['precision']:<8.4f} "
          f"{cnn_metrics['recall']:<8.4f} {cnn_metrics['f1']:<8.4f}")
