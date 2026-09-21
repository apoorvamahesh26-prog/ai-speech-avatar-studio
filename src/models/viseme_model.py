"""
viseme_model.py

VisemeTransformer: predicts a viseme class at every frame from the
Wav2Vec2 feature sequence.

Design choice -- Transformer ENCODER (not full encoder-decoder like
FaceFormer/CodeTalker's autoregressive decoder): those papers predict
continuous, high-dimensional 3D mesh vertex offsets, where
autoregressive generation (conditioning each frame on previously
GENERATED frames) meaningfully helps temporal smoothness. We're
predicting a per-frame CLASSIFICATION label (13 discrete viseme
classes) directly from audio, which is a simpler mapping: viseme at
time t is overwhelmingly determined by the audio around time t, not
by which viseme we predicted at t-1. A causal-masked Transformer
ENCODER (self-attention only over audio, no autoregressive viseme
feedback loop) gets us most of the benefit with much simpler,
faster, more stable training -- and any residual temporal
inconsistency is exactly what Step 7's Temporal Smoothing model
cleans up afterward. This is a deliberate simplification decision,
worth stating explicitly in your report's design-choices section.
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding (Vaswani et al., 2017 --
    same technique FaceFormer and CodeTalker both cite for injecting
    frame-order information, since self-attention itself has no
    inherent notion of sequence order).
    """
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class VisemeTransformer(nn.Module):
    def __init__(self, input_dim=768, d_model=256, nhead=4, num_layers=4,
                 num_visemes=13, dropout=0.2, max_len=150):
        super().__init__()

        # Project 768-dim Wav2Vec2 features down to d_model -- keeps
        # the Transformer itself compact (params scale with d_model^2
        # in the attention/FFN layers), important given dataset size.
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Linear(d_model, num_visemes)

    def forward(self, x, key_padding_mask=None):
        """
        x: (batch, T, 768) Wav2Vec2 features
        key_padding_mask: (batch, T) bool, True at PADDED positions
                           (PyTorch's convention: True = ignore).
        Returns: (batch, T, num_visemes) per-frame logits.

        Note: we do NOT apply a causal mask here -- viseme prediction
        at frame t legitimately benefits from a little future audio
        context too (coarticulation: the mouth shape for a phoneme is
        influenced by the phoneme that follows it, not just the one
        before -- this is well documented in speech science and is
        exactly why lip movement often slightly PRECEDES the sound,
        e.g. lips start rounding for 'oo' before you hear it). Since
        we're not doing live/streaming inference (we process a full
        utterance at once), there's no reason to artificially restrict
        the model to causal-only attention.
        """
        h = self.input_proj(x)
        h = self.pos_encoding(h)
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        logits = self.classifier(h)  # (batch, T, num_visemes)
        return logits
