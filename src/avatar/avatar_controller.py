"""
avatar_controller.py

Design note on why this is a CONTROLLER, not a trained model, despite
being listed alongside Gender/Emotion/Viseme/Smoothing in the project
spec: the viseme->blendshape and emotion->blendshape mappings (Step 8's
blendshape_map.py) are a fixed, well-specified lookup with no genuine
ambiguity to learn from data -- a bilabial closure always means
mouthClose=1.0, regardless of speaker. There's nothing for a network
to discover here that isn't already fully determined by the phoneme-
to-viseme table itself. What DOES benefit from careful, hand-designed
logic (not a black-box model) is temporal interpolation between
blendshape targets, implemented below -- and that logic is exactly
what "avatar controller" refers to in this pipeline. This is a
deliberate scope decision worth stating explicitly in your report:
not every named pipeline component needs to be a neural network,
and the report is stronger for arguing why.
"""

import numpy as np

import sys, os
sys.path.append(os.getcwd())
from src.avatar.blendshape_map import get_frame_blendshapes


def _collect_all_blendshape_names():
    from src.avatar.blendshape_map import VISEME_TO_BLENDSHAPE, EMOTION_TO_BLENDSHAPE
    names = set()
    for d in VISEME_TO_BLENDSHAPE.values():
        names.update(d.keys())
    for d in EMOTION_TO_BLENDSHAPE.values():
        names.update(d.keys())
    return sorted(names)


ALL_BLENDSHAPES = _collect_all_blendshape_names()
BLENDSHAPE_INDEX = {name: i for i, name in enumerate(ALL_BLENDSHAPES)}


def visemes_to_target_weights(viseme_sequence: np.ndarray, emotion: str) -> np.ndarray:
    """
    viseme_sequence: (T,) smoothed viseme IDs (output of Step 7's
                      SmoothingCNN, argmax'd).
    emotion: single emotion label for the whole clip (Step 5's
             EmotionBiLSTM predicts one label per clip -- if per-
             segment emotion is added later, this becomes a
             per-frame array instead of a single string, and the
             loop below indexes into it the same way it does
             viseme_sequence).
    Returns: (T, num_blendshapes) raw TARGET weights per frame,
             before temporal interpolation.
    """
    T = len(viseme_sequence)
    targets = np.zeros((T, len(ALL_BLENDSHAPES)), dtype=np.float32)

    for t in range(T):
        weights = get_frame_blendshapes(int(viseme_sequence[t]), emotion)
        for name, w in weights.items():
            targets[t, BLENDSHAPE_INDEX[name]] = w

    return targets


def interpolate_blendshapes(target_weights: np.ndarray, smoothing_frames: int = 3) -> np.ndarray:
    """
    Applies a short exponential-moving-average style temporal
    interpolation ACROSS BLENDSHAPE WEIGHTS (continuous [0,1] values,
    unlike Step 7's smoothing which operates on discrete viseme
    CLASSES). Even a perfectly class-smoothed viseme sequence can
    still look like a hard "snap" when converted to blendshape
    weights frame-by-frame -- e.g. jawOpen jumping from 0.0 to 0.7 in
    a single frame -- because discrete-class smoothing only prevents
    flicker BETWEEN classes, it doesn't make the resulting continuous
    weight curve itself smooth. This is a second, complementary
    smoothing pass at the continuous-signal level, applied only here
    at the rendering stage.

    smoothing_frames: larger = smoother but more lag; 3 frames at
    30fps is a 100ms time constant, short enough to preserve snappy
    consonant closures (important for bilabial accuracy) while still
    removing visible jumps in slower-moving shapes like jaw/brow.
    """
    alpha = 2.0 / (smoothing_frames + 1)  # standard EMA-span relationship
    smoothed = np.zeros_like(target_weights)
    smoothed[0] = target_weights[0]
    for t in range(1, len(target_weights)):
        smoothed[t] = alpha * target_weights[t] + (1 - alpha) * smoothed[t - 1]
    return smoothed


def build_animation(viseme_sequence: np.ndarray, emotion: str, smoothing_frames: int = 3):
    """
    Full controller entry point: viseme sequence + emotion -> ready-
    to-render per-frame blendshape animation.
    Returns: dict with 'blendshape_names' (list[str]) and 'weights'
    ((T, num_blendshapes) array) -- this is the payload the FastAPI
    backend (Step 9) will serialize to JSON for the frontend renderer.
    """
    raw_targets = visemes_to_target_weights(viseme_sequence, emotion)
    smoothed = interpolate_blendshapes(raw_targets, smoothing_frames=smoothing_frames)

    return {
        "blendshape_names": ALL_BLENDSHAPES,
        "weights": smoothed,  # (T, num_blendshapes)
    }


if __name__ == "__main__":
    # Quick smoke test with a synthetic viseme sequence
    demo_sequence = np.array([0, 0, 7, 7, 7, 1, 1, 0, 9, 9, 0], dtype=np.int64)
    anim = build_animation(demo_sequence, emotion="happy")
    print("Blendshapes controlled:", anim["blendshape_names"])
    print("Weight sequence shape:", anim["weights"].shape)
    print("Frame 2 (open_wide + happy) weights:",
          {name: round(float(w), 2) for name, w in zip(anim["blendshape_names"], anim["weights"][2]) if w > 0.01})
