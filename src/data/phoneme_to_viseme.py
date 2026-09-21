"""
phoneme_to_viseme.py

Maps ARPAbet phonemes (as produced by Montreal Forced Aligner / CMUdict)
to a compact set of viseme classes (mouth-shape groups).

Why a dict and not a function: we want this table to be the single
source of truth. Every other script (label generation, evaluation,
avatar blendshape mapping) imports VISEME_MAP and VISEME_NAMES from
here so there's never a mismatch between training labels and
inference-time mouth shapes.
"""

# Viseme ID -> human-readable name (for confusion matrices, logging, docs)
VISEME_NAMES = {
    0: "silence",
    1: "bilabial",       # P B M
    2: "labiodental",    # F V
    3: "interdental",    # TH DH
    4: "alveolar",       # T D N L S Z
    5: "postalveolar",   # SH ZH CH JH
    6: "velar",          # K G NG
    7: "open_wide",      # AA AE AH
    8: "open_mid",       # EH ER AY
    9: "rounded",        # OW OY UW W
    10: "close_spread",  # IY IH EY Y
    11: "r_sound",       # R
    12: "glottal",       # HH
}

# ARPAbet phoneme -> viseme ID
# NOTE: MFA/CMUdict phonemes carry stress markers as trailing digits
# (e.g. AA0, AA1, AA2). We strip digits before lookup (see
# phoneme_to_viseme() below), so list only the base phoneme here.
PHONEME_TO_VISEME = {
    # Silence / non-speech
    "sil": 0, "sp": 0, "spn": 0,

    # Bilabial
    "P": 1, "B": 1, "M": 1,

    # Labiodental
    "F": 2, "V": 2,

    # Interdental
    "TH": 3, "DH": 3,

    # Alveolar
    "T": 4, "D": 4, "N": 4, "L": 4, "S": 4, "Z": 4,

    # Postalveolar
    "SH": 5, "ZH": 5, "CH": 5, "JH": 5,

    # Velar
    "K": 6, "G": 6, "NG": 6,

    # Open vowels (wide jaw)
    "AA": 7, "AE": 7, "AH": 7,

    # Open-mid vowels
    "EH": 8, "ER": 8, "AY": 8,

    # Rounded vowels / glide
    "OW": 9, "OY": 9, "UW": 9, "UH": 9, "W": 9,

    # Close/spread vowels
    "IY": 10, "IH": 10, "EY": 10, "Y": 10,

    # R
    "R": 11,

    # Glottal
    "HH": 12,
}

NUM_VISEMES = len(VISEME_NAMES)  # 13


def phoneme_to_viseme(phoneme: str) -> int:
    """
    Convert a raw ARPAbet phoneme (possibly with stress digit, e.g. 'AA1')
    to its viseme class ID. Unknown phonemes fall back to silence (0)
    rather than raising, so a single bad alignment doesn't crash a
    training run -- we log these instead (see preprocess.py).
    """
    base = "".join(ch for ch in phoneme if not ch.isdigit()).upper()
    return PHONEME_TO_VISEME.get(base, 0)
