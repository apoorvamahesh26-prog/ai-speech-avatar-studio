# Step 12: Consolidated Evaluation Report

Device used for evaluation: cpu

## 1. Gender Classification


| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| RandomForest | 0.7083 | 0.6891 | 0.7593 | 0.7225 |
| GenderCNN | 0.7130 | 0.6620 | 0.8704 | 0.7520 |

GenderCNN: 9,250 params, 0.036 MB, 1.364ms/inference (733.36 predictions/sec)

## 2. Emotion Recognition


| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| RandomForest (pooled) | 0.3657 | 0.3945 | 0.3701 | 0.3716 |
| BiLSTM+Attention | 0.4630 | 0.5105 | 0.4518 | 0.4608 |

EmotionBiLSTM: 1,005,961 params, 3.837 MB, 4.31ms/inference (232.02 predictions/sec)

## 3. Viseme Prediction


| Metric | Value |
|---|---|
| Per-frame Accuracy | 0.8530 |
| Precision (macro) | 0.8461 |
| Recall (macro) | 0.8354 |
| F1 (macro) | 0.8394 |

VisemeTransformer: 3,359,245 params, 12.961 MB, 12.895ms per 150-frame chunk (11632.4 frames/sec)

**Note:** macro-averaged metrics used deliberately (see metrics.py) because the viseme class distribution is heavily silence/limited-phoneme dominated (Step 6's known 2-sentence RAVDESS limitation) -- accuracy alone would overstate real performance.

## 4. Temporal Smoothing


| Method | Accuracy | Flicker Rate | Latency |
|---|---|---|---|
| No smoothing | 0.8785 | 0.4621 | - |
| EMA | 0.7882 | 0.3675 | - |
| Median filter | 0.7940 | 0.2098 | - |
| SmoothingCNN (ours) | 0.9664 | 0.3149 | 0.344ms |

SmoothingCNN: 9,485 params, 0.037 MB

## Important Note on Comparison to Published Literature

The results above are NOT directly comparable to GaussianSpeech, CodeTalker, Imitator, or MemoryTalker's reported numbers. Those systems are evaluated on continuous 3D mesh vertex error (LVE, FVE, LSE-D) using VOCASET/BIWI/multi-view capture data -- a fundamentally different task and dataset from the discrete viseme/emotion/gender classification evaluated here on RAVDESS/Common Voice. This project's contribution is a lightweight, gender-aware, emotion-integrated pipeline achieving real-time performance on commodity hardware (see Step 11's RTF results), evaluated against classical machine-learning baselines trained under identical conditions -- not a claim of outperforming prior published work on their own benchmarks.