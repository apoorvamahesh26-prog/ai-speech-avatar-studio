"""
metrics.py

Reusable evaluation utilities so every model (Gender, Emotion, Viseme,
Smoothing) is measured the same way, with the same metric definitions
-- this consistency is what makes evaluate_all.py's final comparison
table meaningful rather than apples-to-oranges.
"""

import os
import time
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


def compute_classification_metrics(y_true, y_pred, average="macro"):
    """
    average='macro': computes precision/recall/F1 per class, then
    averages UNWEIGHTED across classes. This matters for Emotion
    (8 classes, some rarer than others) and Viseme (13 classes,
    heavily silence-dominated per Step 6's note) -- macro averaging
    prevents a model that only does well on the majority class from
    looking artificially good, which a 'weighted' or accuracy-only
    metric would hide.
    """
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def plot_confusion_matrix(y_true, y_pred, labels, title, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    plt.figure(figsize=(max(6, len(labels) * 0.8), max(5, len(labels) * 0.7)))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model: torch.nn.Module) -> float:
    """
    Actual serialized size on disk (what a checkpoint file would take),
    computed directly from parameter dtypes rather than assuming
    float32 -- correct even if a model has been quantized (Step 11).
    """
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    total_bytes += sum(b.numel() * b.element_size() for b in model.buffers())
    return total_bytes / (1024 ** 2)


def measure_inference_latency(forward_fn, n_warmup=3, n_runs=20):
    """
    forward_fn: a zero-argument callable that runs ONE forward pass
                (caller closes over whatever inputs are needed).
    Returns: dict with mean/std latency in milliseconds, and FPS
             (runs per second) -- FPS here means "predictions per
             second" (relevant for per-clip models like Gender/
             Emotion), not video frame rate; Viseme's per-FRAME
             throughput is reported separately using its own T.

    Warmup runs are excluded from the timing average -- the first
    few calls to any model include one-time costs (lazy CUDA kernel
    compilation, memory allocation) that don't reflect steady-state
    performance, exactly the effect you already observed in Step 11's
    9.9x RTF first-call outlier.
    """
    for _ in range(n_warmup):
        forward_fn()

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        forward_fn()
        times.append(time.perf_counter() - t0)

    times = np.array(times)
    mean_ms = float(times.mean() * 1000)
    std_ms = float(times.std() * 1000)
    fps = 1000.0 / mean_ms if mean_ms > 0 else float("inf")

    return {"mean_ms": round(mean_ms, 3), "std_ms": round(std_ms, 3), "fps": round(fps, 2)}


def print_metrics_table(rows: list, headers: list):
    """ Minimal dependency-free table printer for console reports. """
    col_widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0)) + 2
                  for i, h in enumerate(headers)]
    header_line = "".join(str(h).ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        print("".join(str(v).ljust(w) for v, w in zip(row, col_widths)))
