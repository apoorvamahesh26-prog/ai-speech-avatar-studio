"""
blendshape_map.py

Two mapping tables:
  1. VISEME_TO_BLENDSHAPE -- viseme ID -> {blendshape_name: target_weight}
     Only mouth/jaw blendshapes are touched here; upper-face blendshapes
     are left to the emotion mapping so the two systems can be combined
     additively without conflicting over the same shape.
  2. EMOTION_TO_BLENDSHAPE -- emotion label -> {blendshape_name: target_weight}
     Drives brows/eyes/cheeks; combined with the (dominant) mouth shape
     from the current viseme in avatar/blendshape_mapper.py.

Blendshape names follow the ARKit 52-shape standard used by Ready
Player Me exports, so this file assumes that avatar source (see
Step 8 discussion). If a different avatar rig is used later, only
these two dictionaries need to change -- no other code in the
pipeline references specific shape names directly.
"""

import sys, os
sys.path.append(os.getcwd())
from src.audio.phoneme_to_viseme import VISEME_NAMES

# viseme_id -> {blendshape: weight in [0,1]}
VISEME_TO_BLENDSHAPE = {
    0:  {},  # silence -- neutral mouth, no override (rest pose)
    1:  {"mouthClose": 1.0, "mouthPressLeft": 0.3, "mouthPressRight": 0.3},          # bilabial (P B M)
    2:  {"mouthLowerDownLeft": 0.5, "mouthLowerDownRight": 0.5, "mouthUpperUpLeft": 0.2}, # labiodental (F V)
    3:  {"tongueOut": 0.4, "jawOpen": 0.15},                                          # interdental (TH DH)
    4:  {"jawOpen": 0.2, "mouthStretchLeft": 0.2, "mouthStretchRight": 0.2},           # alveolar (T D N L S Z)
    5:  {"mouthFunnel": 0.6, "mouthPucker": 0.3},                                      # postalveolar (SH ZH CH JH)
    6:  {"jawOpen": 0.35},                                                             # velar (K G NG)
    7:  {"jawOpen": 0.7, "mouthStretchLeft": 0.3, "mouthStretchRight": 0.3},           # open_wide (AA AE AH)
    8:  {"jawOpen": 0.45},                                                             # open_mid (EH ER AY)
    9:  {"mouthPucker": 0.7, "mouthFunnel": 0.3},                                      # rounded (OW OY UW W)
    10: {"mouthSmileLeft": 0.25, "mouthSmileRight": 0.25, "jawOpen": 0.1},             # close_spread (IY IH EY Y)
    11: {"mouthPucker": 0.4, "mouthFunnel": 0.2},                                      # r_sound
    12: {"jawOpen": 0.5},                                                              # glottal (HH)
}

# emotion label -> {blendshape: weight}. Applied continuously (not
# per-frame like visemes) -- held for the duration of the recognized
# emotion segment, and blended additively with mouth blendshapes.
EMOTION_TO_BLENDSHAPE = {
    "neutral":   {},
    "calm":      {"eyeSquintLeft": 0.1, "eyeSquintRight": 0.1},
    "happy":     {"mouthSmileLeft": 0.5, "mouthSmileRight": 0.5,
                   "cheekSquintLeft": 0.3, "cheekSquintRight": 0.3},
    "sad":       {"browInnerUp": 0.5, "mouthFrownLeft": 0.4, "mouthFrownRight": 0.4},
    "angry":     {"browDownLeft": 0.6, "browDownRight": 0.6, "noseSneerLeft": 0.3, "noseSneerRight": 0.3},
    "fearful":   {"browInnerUp": 0.6, "eyeWideLeft": 0.5, "eyeWideRight": 0.5},
    "disgust":   {"noseSneerLeft": 0.5, "noseSneerRight": 0.5, "mouthUpperUpLeft": 0.3, "mouthUpperUpRight": 0.3},
    "surprised": {"browInnerUp": 0.7, "browOuterUpLeft": 0.5, "browOuterUpRight": 0.5,
                   "eyeWideLeft": 0.6, "eyeWideRight": 0.6, "jawOpen": 0.3},
}


def get_frame_blendshapes(viseme_id: int, emotion: str) -> dict:
    """
    Combines the mouth-shape contribution of the current viseme with
    the (held, slower-changing) expression contribution of the
    current emotion into one weight dict ready to send to the
    renderer. Emotion contributions are applied first, then viseme
    contributions -- if both touch the same shape name (rare, since
    the tables are largely disjoint by design), the viseme value wins
    since lip-sync accuracy should not be visibly compromised by
    expression.
    """
    weights = {}
    weights.update(EMOTION_TO_BLENDSHAPE.get(emotion, {}))
    weights.update(VISEME_TO_BLENDSHAPE.get(viseme_id, {}))
    return weights
