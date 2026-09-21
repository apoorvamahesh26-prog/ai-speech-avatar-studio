"""
calibrate_gender_threshold.py

Addresses a REAL, reportable issue: the GenderCNN was trained with a
naive argmax (implicit 0.5 probability threshold) decision rule. If
the model is systematically more confident on male voices than female
voices -- which can happen from residual class imbalance, feature
distribution differences, or (very plausibly here) a domain shift
between clean training audio (RAVDESS studio recordings, Common Voice
read speech) and noisier real-world microphone input compressed as
WebM/Opus -- a fixed 0.5 threshold on the SOFTMAX PROBABILITY can
still favor one class even though accuracy on the training/val
distribution looked balanced.

This script does NOT retrain the model. It re-uses the frozen,
already-trained GenderCNN and finds a better decision threshold on
the held-out validation set -- the standard, honest ML technique for
correcting a biased operating point without touching model weights.
If you want a deeper fix later (recommended, see the printed note),
retrain gender_model.py with augmented data that includes WebM/Opus-
compressed and noise-augmented clips so the TRAINING distribution
better matches real microphone input -- that addresses the root cause
(domain shift) rather than just recalibrating around it.

Run: python src/training/calibrate_gender_threshold.py
Output: checkpoints/gender_threshold.json
"""

import os
import sys
import json
import numpy as np
import torch
from sklearn.metrics import roc_curve, accuracy_score, confusion_matrix

sys.path.append(os.getcwd())
from src.data.gender_dataset import load_gender_dataset
from src.models.gender_model import GenderCNN

CHECKPOINT_DIR = "checkpoints"
OUT_PATH = os.path.join(CHECKPOINT_DIR, "gender_threshold.json")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_gender_dataset()
    X_val, y_val = data["val"]  # calibrate on VAL, never on test -- test stays held out for Step 12's honest reporting

    model = GenderCNN().to(device)
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "gender_cnn_best.pt"), map_location=device))
    model.eval()

    with torch.no_grad():
        logits = model(torch.tensor(X_val).to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()  # (N, 2) -> [:,0]=male, [:,1]=female

    female_probs = probs[:, 1]

    # Default (uncalibrated) behavior: argmax == threshold 0.5 on female_probs
    default_preds = (female_probs >= 0.5).astype(int)
    default_acc = accuracy_score(y_val, default_preds)
    default_cm = confusion_matrix(y_val, default_preds)
    print("Default (0.5 threshold) confusion matrix [rows=true, cols=pred; 0=male,1=female]:")
    print(default_cm)
    print(f"Default accuracy: {default_acc:.4f}")

    # Find the threshold that makes the male-error-rate and female-error-rate
    # as EQUAL as possible (this is what "male dominates" means concretely:
    # the female recall is lower than the male recall). This is the
    # standard fix for a biased operating point -- balance per-class error
    # rather than optimizing raw accuracy, which a majority-class bias can
    # inflate even while the minority class suffers.
    fpr, tpr, thresholds = roc_curve(y_val, female_probs)  # treats female=1 as positive class
    # fpr = male-classified-as-female rate at each threshold
    # 1 - tpr = female-classified-as-male rate at each threshold
    # We want the threshold where these two error rates are closest.
    balance_gap = np.abs(fpr - (1 - tpr))
    best_idx = np.argmin(balance_gap)
    calibrated_threshold = float(thresholds[best_idx])
    # roc_curve can return threshold > 1 or < 0 at the boundary points; clip
    # to a sane probability range.
    calibrated_threshold = float(np.clip(calibrated_threshold, 0.05, 0.95))

    calibrated_preds = (female_probs >= calibrated_threshold).astype(int)
    calibrated_acc = accuracy_score(y_val, calibrated_preds)
    calibrated_cm = confusion_matrix(y_val, calibrated_preds)
    print(f"\nCalibrated threshold (female_prob >=): {calibrated_threshold:.4f}")
    print("Calibrated confusion matrix:")
    print(calibrated_cm)
    print(f"Calibrated accuracy: {calibrated_acc:.4f}")

    male_recall_default = default_cm[0, 0] / max(default_cm[0].sum(), 1)
    female_recall_default = default_cm[1, 1] / max(default_cm[1].sum(), 1)
    male_recall_cal = calibrated_cm[0, 0] / max(calibrated_cm[0].sum(), 1)
    female_recall_cal = calibrated_cm[1, 1] / max(calibrated_cm[1].sum(), 1)
    print(f"\nPer-class recall BEFORE calibration: male={male_recall_default:.3f}, female={female_recall_default:.3f}")
    print(f"Per-class recall AFTER  calibration: male={male_recall_cal:.3f}, female={female_recall_cal:.3f}")

    with open(OUT_PATH, "w") as f:
        json.dump({
            "female_probability_threshold": calibrated_threshold,
            "note": "female_prob >= this value -> predict female, else male. "
                    "Replaces naive argmax (0.5) to correct class-recall imbalance "
                    "measured on the validation set.",
        }, f, indent=2)
    print(f"\nSaved calibrated threshold -> {OUT_PATH}")
    print(
        "\nIMPORTANT CAVEAT for your report: this corrects the DECISION "
        "THRESHOLD, not the underlying feature representation. If the bias "
        "is caused by a genuine domain shift (e.g. clean training audio vs. "
        "noisy/WebM-compressed microphone input), the deeper fix is "
        "retraining gender_model.py on data augmented with WebM/Opus "
        "compression and background noise so the training distribution "
        "matches real microphone conditions. This script is the fast, "
        "honest correction available without new data collection; note "
        "both in your report as separate, complementary fixes."
    )


if __name__ == "__main__":
    main()
