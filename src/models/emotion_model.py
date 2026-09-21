"""
emotion_model.py

RandomForestEmotionClassifier: baseline, pooled features, same
rationale as gender_model.py's RF (no scaling needed, robust to
small data, gives feature importances).

EmotionBiLSTM: primary model. Key design points explained inline.
"""

import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
import joblib


class RandomForestEmotionClassifier:
    def __init__(self, n_estimators=300, max_depth=25, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=random_state, n_jobs=-1,
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)


class EmotionBiLSTM(nn.Module):
    """
    Input: (batch, MAX_FRAMES, 768) padded Wav2Vec2 sequences,
           plus true lengths (to build a pack_padded_sequence so the
           LSTM doesn't waste computation on / get confused by padding).

    Why BiLSTM over plain LSTM: emotional cues can depend on both
    what came before AND after a given frame (e.g. a rising pitch
    contour is only identifiable by seeing where it peaks and where
    it resolves) -- bidirectional processing gives the model access
    to full-utterance context at every timestep, whereas a
    forward-only LSTM only ever "knows" the past at each point.

    Why attention pooling (not just last hidden state) for the final
    classification vector: with variable-length utterances, the
    "important" emotional cue (e.g. a shout, a sigh) can occur
    anywhere in the clip, not necessarily at the end. A learned
    attention pooling lets the model decide which frames matter most,
    rather than being forced to summarize everything into the last
    timestep's hidden state (a known LSTM bottleneck for longer
    sequences).
    """
    def __init__(self, input_dim=768, hidden_dim=128, num_layers=2,
                 num_classes=8, dropout=0.3):
        super().__init__()

        # Project 768-dim Wav2Vec2 features down to a smaller size first
        # -- reduces LSTM parameter count substantially (LSTM params
        # scale with input_dim * hidden_dim), important given data size.
        self.input_proj = nn.Linear(input_dim, 256)

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out_dim = hidden_dim * 2  # bidirectional -> concat both directions

        # Simple additive attention: learns a scalar "importance" score
        # per timestep, softmax-normalized, then used to weight-sum
        # the LSTM outputs into one vector.
        self.attn = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, num_classes),
        )

    def forward(self, x, lengths):
        # x: (batch, MAX_FRAMES, 768), lengths: (batch,) real lengths
        x = self.input_proj(x)  # (batch, MAX_FRAMES, 256)

        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        # lstm_out: (batch, T_max_in_batch, hidden_dim*2)

        # Build a mask so attention doesn't attend to padded positions
        max_len = lstm_out.size(1)
        mask = torch.arange(max_len, device=x.device)[None, :] < lengths[:, None].to(x.device)
        # mask: (batch, T_max_in_batch), True where real data

        attn_scores = self.attn(lstm_out).squeeze(-1)          # (batch, T)
        attn_scores = attn_scores.masked_fill(~mask, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)  # (batch, T, 1)

        pooled = (lstm_out * attn_weights).sum(dim=1)  # (batch, hidden_dim*2)

        logits = self.classifier(pooled)
        return logits
