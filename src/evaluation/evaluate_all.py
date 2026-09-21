"""
evaluate_all.py

Consolidates Steps 4, 5, 6, 7's individual evaluations into ONE
report. Loads already-trained checkpoints (does NOT retrain anything)
-- this script should be fast to run repeatedly as you iterate on
your report's wording, since the expensive part (training) already
happened.

Run: python src/evaluation/evaluate_all.py
Output: checkpoints/evaluation_report.md (paste-ready for your report)
        plus confusion matrix PNGs in checkpoints/
"""

import os
import sys
import numpy as np
import torch

sys.path.append(os.getcwd())
from src.evaluation.metrics import (
    compute_classification_metrics, plot_confusion_matrix, count_parameters,
    model_size_mb, measure_inference_latency, print_metrics_table
)

from src.data.gender_dataset import load_gender_dataset
from src.models.gender_model import RandomForestGenderClassifier, GenderCNN

from src.data.emotion_dataset import load_emotion_dataset, EMOTION_LIST
from src.models.emotion_model import RandomForestEmotionClassifier, EmotionBiLSTM

from src.data.viseme_dataset import load_viseme_dataset, MAX_FRAMES
from src.models.viseme_model import VisemeTransformer
from src.audio.phoneme_to_viseme import VISEME_NAMES, NUM_VISEMES

from src.data.smoothing_dataset import build_smoothing_dataset
from src.models.smoothing_model import SmoothingCNN, ema_smooth, median_filter_smooth
from src.training.train_smoothing import ids_to_onehot, evaluate_sequence_method, flicker_rate

CHECKPOINT_DIR = "checkpoints"
REPORT_PATH = os.path.join(CHECKPOINT_DIR, "evaluation_report.md")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

report_lines = []


def log(line=""):
    print(line)
    report_lines.append(line)


# =====================================================================
# 1. Gender Classification
# =====================================================================
def evaluate_gender():
    log("\n## 1. Gender Classification\n")
    data = load_gender_dataset()
    X_test, y_test = data["test"]

    rf = RandomForestGenderClassifier()
    rf.load(os.path.join(CHECKPOINT_DIR, "gender_rf.joblib"))
    rf_preds = rf.predict(X_test)
    rf_metrics = compute_classification_metrics(y_test, rf_preds, average="binary")

    cnn = GenderCNN().to(DEVICE)
    cnn.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "gender_cnn_best.pt"), map_location=DEVICE))
    cnn.eval()
    with torch.no_grad():
        logits = cnn(torch.tensor(X_test).to(DEVICE))
        cnn_preds = logits.argmax(dim=1).cpu().numpy()
    cnn_metrics = compute_classification_metrics(y_test, cnn_preds, average="binary")

    plot_confusion_matrix(y_test, cnn_preds, ["male", "female"],
                           "Gender Confusion Matrix (CNN, Test Set)",
                           os.path.join(CHECKPOINT_DIR, "eval_gender_confusion.png"))

    sample = torch.tensor(X_test[:1]).to(DEVICE)
    with torch.no_grad():
        cnn_latency = measure_inference_latency(lambda: cnn(sample))

    rows = [
        ["RandomForest", f"{rf_metrics['accuracy']:.4f}", f"{rf_metrics['f1']:.4f}", "-", "-"],
        ["GenderCNN", f"{cnn_metrics['accuracy']:.4f}", f"{cnn_metrics['f1']:.4f}",
         f"{cnn_latency['mean_ms']}ms", f"{count_parameters(cnn):,}"],
    ]
    print_metrics_table(rows, ["Model", "Accuracy", "F1", "Latency", "Params"])
    log("\n| Model | Accuracy | Precision | Recall | F1 |")
    log("|---|---|---|---|---|")
    log(f"| RandomForest | {rf_metrics['accuracy']:.4f} | {rf_metrics['precision']:.4f} | {rf_metrics['recall']:.4f} | {rf_metrics['f1']:.4f} |")
    log(f"| GenderCNN | {cnn_metrics['accuracy']:.4f} | {cnn_metrics['precision']:.4f} | {cnn_metrics['recall']:.4f} | {cnn_metrics['f1']:.4f} |")
    log(f"\nGenderCNN: {count_parameters(cnn):,} params, {model_size_mb(cnn):.3f} MB, "
        f"{cnn_latency['mean_ms']}ms/inference ({cnn_latency['fps']} predictions/sec)")

    return {"rf": rf_metrics, "cnn": cnn_metrics, "cnn_latency": cnn_latency}


# =====================================================================
# 2. Emotion Recognition
# =====================================================================
def evaluate_emotion():
    log("\n## 2. Emotion Recognition\n")
    data = load_emotion_dataset()
    test = data["test"]

    rf = RandomForestEmotionClassifier()
    rf.load(os.path.join(CHECKPOINT_DIR, "emotion_rf.joblib"))
    rf_preds = rf.predict(test["X_pooled"])
    rf_metrics = compute_classification_metrics(test["y"], rf_preds, average="macro")

    bilstm = EmotionBiLSTM().to(DEVICE)
    bilstm.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "emotion_bilstm_best.pt"), map_location=DEVICE))
    bilstm.eval()
    with torch.no_grad():
        logits = bilstm(torch.tensor(test["X_seq"]).to(DEVICE), torch.tensor(test["lengths"]))
        bilstm_preds = logits.argmax(dim=1).cpu().numpy()
    bilstm_metrics = compute_classification_metrics(test["y"], bilstm_preds, average="macro")

    plot_confusion_matrix(test["y"], bilstm_preds, EMOTION_LIST,
                           "Emotion Confusion Matrix (BiLSTM, Test Set)",
                           os.path.join(CHECKPOINT_DIR, "eval_emotion_confusion.png"))

    sample_seq = torch.tensor(test["X_seq"][:1]).to(DEVICE)
    sample_len = torch.tensor(test["lengths"][:1])
    with torch.no_grad():
        bilstm_latency = measure_inference_latency(lambda: bilstm(sample_seq, sample_len))

    log("\n| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |")
    log("|---|---|---|---|---|")
    log(f"| RandomForest (pooled) | {rf_metrics['accuracy']:.4f} | {rf_metrics['precision']:.4f} | {rf_metrics['recall']:.4f} | {rf_metrics['f1']:.4f} |")
    log(f"| BiLSTM+Attention | {bilstm_metrics['accuracy']:.4f} | {bilstm_metrics['precision']:.4f} | {bilstm_metrics['recall']:.4f} | {bilstm_metrics['f1']:.4f} |")
    log(f"\nEmotionBiLSTM: {count_parameters(bilstm):,} params, {model_size_mb(bilstm):.3f} MB, "
        f"{bilstm_latency['mean_ms']}ms/inference ({bilstm_latency['fps']} predictions/sec)")

    return {"rf": rf_metrics, "bilstm": bilstm_metrics, "bilstm_latency": bilstm_latency}


# =====================================================================
# 3. Viseme Prediction
# =====================================================================
def evaluate_viseme():
    log("\n## 3. Viseme Prediction\n")
    data = load_viseme_dataset()
    test = data["test"]

    model = VisemeTransformer(max_len=MAX_FRAMES).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "viseme_transformer_best.pt"), map_location=DEVICE))
    model.eval()

    all_preds, all_true = [], []
    with torch.no_grad():
        for i in range(len(test["X"])):
            L = test["lengths"][i]
            x = torch.tensor(test["X"][i:i+1]).to(DEVICE)
            logits = model(x, key_padding_mask=None)
            preds = logits.argmax(dim=-1).squeeze(0).cpu().numpy()
            all_preds.extend(preds[:L])
            all_true.extend(test["y"][i][:L])

    metrics = compute_classification_metrics(all_true, all_preds, average="macro")
    plot_confusion_matrix(all_true, all_preds, [VISEME_NAMES[i] for i in range(NUM_VISEMES)],
                           "Viseme Confusion Matrix (Test Set, per-frame)",
                           os.path.join(CHECKPOINT_DIR, "eval_viseme_confusion.png"))

    sample = torch.tensor(test["X"][:1]).to(DEVICE)
    with torch.no_grad():
        latency = measure_inference_latency(lambda: model(sample, key_padding_mask=None))
    frames_per_call = test["X"].shape[1]
    frame_fps = frames_per_call / (latency["mean_ms"] / 1000.0)

    log(f"\n| Metric | Value |")
    log("|---|---|")
    log(f"| Per-frame Accuracy | {metrics['accuracy']:.4f} |")
    log(f"| Precision (macro) | {metrics['precision']:.4f} |")
    log(f"| Recall (macro) | {metrics['recall']:.4f} |")
    log(f"| F1 (macro) | {metrics['f1']:.4f} |")
    log(f"\nVisemeTransformer: {count_parameters(model):,} params, {model_size_mb(model):.3f} MB, "
        f"{latency['mean_ms']}ms per {frames_per_call}-frame chunk ({frame_fps:.1f} frames/sec)")
    log("\n**Note:** macro-averaged metrics used deliberately (see metrics.py) because the "
        "viseme class distribution is heavily silence/limited-phoneme dominated (Step 6's "
        "known 2-sentence RAVDESS limitation) -- accuracy alone would overstate real performance.")

    return {"metrics": metrics, "latency": latency, "frame_fps": frame_fps}


# =====================================================================
# 4. Temporal Smoothing
# =====================================================================
def evaluate_smoothing():
    log("\n## 4. Temporal Smoothing\n")
    data = build_smoothing_dataset()
    test = data["test"]

    no_smooth_acc, no_smooth_flicker = evaluate_sequence_method(
        lambda ids: ids, test["noisy"], test["clean"], test["lengths"], is_prob_based=False)
    ema_acc, ema_flicker = evaluate_sequence_method(
        lambda probs: ema_smooth(probs, alpha=0.4), test["noisy"], test["clean"], test["lengths"], is_prob_based=True)
    median_acc, median_flicker = evaluate_sequence_method(
        lambda ids: median_filter_smooth(ids, window=5), test["noisy"], test["clean"], test["lengths"], is_prob_based=False)

    model = SmoothingCNN().to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "smoothing_cnn_best.pt"), map_location=DEVICE))
    model.eval()

    cnn_accs, cnn_flickers = [], []
    with torch.no_grad():
        for i in range(len(test["noisy"])):
            L = test["lengths"][i]
            probs = ids_to_onehot(test["noisy"][i])
            x = torch.tensor(probs).unsqueeze(0).to(DEVICE)
            out = model(x).squeeze(0).cpu().numpy()
            pred_ids = out[:L].argmax(axis=-1)
            cnn_accs.append((pred_ids == test["clean"][i][:L]).mean())
            cnn_flickers.append(flicker_rate(pred_ids, L))

    sample = torch.tensor(ids_to_onehot(test["noisy"][:1])).to(DEVICE)
    with torch.no_grad():
        latency = measure_inference_latency(lambda: model(sample))

    log("\n| Method | Accuracy | Flicker Rate | Latency |")
    log("|---|---|---|---|")
    log(f"| No smoothing | {no_smooth_acc:.4f} | {no_smooth_flicker:.4f} | - |")
    log(f"| EMA | {ema_acc:.4f} | {ema_flicker:.4f} | - |")
    log(f"| Median filter | {median_acc:.4f} | {median_flicker:.4f} | - |")
    log(f"| SmoothingCNN (ours) | {np.mean(cnn_accs):.4f} | {np.mean(cnn_flickers):.4f} | {latency['mean_ms']}ms |")
    log(f"\nSmoothingCNN: {count_parameters(model):,} params, {model_size_mb(model):.3f} MB")

    return {"no_smooth": (no_smooth_acc, no_smooth_flicker), "ema": (ema_acc, ema_flicker),
            "median": (median_acc, median_flicker), "cnn": (np.mean(cnn_accs), np.mean(cnn_flickers))}


# =====================================================================
# Main
# =====================================================================
if __name__ == "__main__":
    log("# Step 12: Consolidated Evaluation Report")
    log(f"\nDevice used for evaluation: {DEVICE}")

    gender_results = evaluate_gender()
    emotion_results = evaluate_emotion()
    viseme_results = evaluate_viseme()
    smoothing_results = evaluate_smoothing()

    log("\n## Important Note on Comparison to Published Literature\n")
    log(
        "The results above are NOT directly comparable to GaussianSpeech, CodeTalker, "
        "Imitator, or MemoryTalker's reported numbers. Those systems are evaluated on "
        "continuous 3D mesh vertex error (LVE, FVE, LSE-D) using VOCASET/BIWI/multi-view "
        "capture data -- a fundamentally different task and dataset from the discrete "
        "viseme/emotion/gender classification evaluated here on RAVDESS/Common Voice. "
        "This project's contribution is a lightweight, gender-aware, emotion-integrated "
        "pipeline achieving real-time performance on commodity hardware (see Step 11's "
        "RTF results), evaluated against classical machine-learning baselines trained "
        "under identical conditions -- not a claim of outperforming prior published work "
        "on their own benchmarks."
    )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\n\nFull report written to {REPORT_PATH}")
