"""
gender_model.py

Two models, as planned in Step 4's comparison table:
  1. RandomForestGenderClassifier -- classical ML baseline (sklearn)
  2. GenderCNN -- lightweight 1D-CNN deep learning model (PyTorch)

Both consume the SAME mean-pooled 768-dim Wav2Vec2 features from
gender_dataset.py, so their results are directly comparable in
Step 12's evaluation -- same input, different algorithm, apples-to-apples.
"""

import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
import joblib


# ---------------------------------------------------------------------
# 1. Classical ML baseline
# ---------------------------------------------------------------------
class RandomForestGenderClassifier:
    """
    Why Random Forest specifically (over SVM/LogReg):
    - Handles the 768-dim feature space without needing feature
      scaling/normalization (unlike SVM, which is sensitive to scale).
    - Gives free feature-importance scores -- useful if you want to
      discuss in your report *which* dimensions of the Wav2Vec2
      embedding matter most for gender (an interesting analysis you
      can't get from a plain CNN).
    - Robust to a modest dataset size and unlikely to badly overfit
      compared to deep nets, given n_estimators/max_depth kept modest.
    """
    def __init__(self, n_estimators=200, max_depth=20, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,  # use all CPU cores -- this model trains on CPU regardless of GPU availability
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)


# ---------------------------------------------------------------------
# 2. Lightweight deep learning model
# ---------------------------------------------------------------------
class GenderCNN(nn.Module):
    """
    Input: (batch, 768) pooled Wav2Vec2 feature vector.
    We treat the 768-dim vector as a 1D "signal" and apply small
    1D convolutions -- this lets the model learn local combinations
    of feature dimensions (which plain fully-connected layers can
    also do, but conv layers do it with far fewer parameters thanks
    to weight sharing, which matters given how little labeled data
    we have).

    Kept deliberately small (3 conv layers, <100K params) --
    a large deep model on ~thousands of pooled vectors would
    almost certainly overfit.
    """
    def __init__(self, input_dim=768, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Unflatten(1, (1, input_dim)),  # (batch, 1, 768) -- add channel dim for Conv1d

            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),                   # 768 -> 384

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),                   # 384 -> 192

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),           # global pool -> (batch, 64, 1)

            nn.Flatten(),                       # (batch, 64)
            nn.Dropout(0.3),                    # regularization -- important given small dataset
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)  # logits, shape (batch, num_classes)
