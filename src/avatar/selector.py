"""
selector.py

Takes the Gender model's prediction (Step 4) and returns the file
path of the corresponding avatar mesh. Kept deliberately tiny and
isolated -- if you add more avatar styles later (e.g. multiple
male/female variants, or a neutral option), only this file changes.
"""

import os

AVATAR_DIR = "assets/avatars"

AVATAR_PATHS = {
    "male": os.path.join(AVATAR_DIR, "male_avatar.glb"),
    "female": os.path.join(AVATAR_DIR, "female_avatar.glb"),
}


def select_avatar(gender_label: str) -> str:
    """
    gender_label: 'male' or 'female' (Gender model's predicted class,
                  from gender_model.py's GENDER_TO_ID inverse mapping).
    Returns the avatar file path to load in the frontend viewer.
    Falls back to the male avatar on an unexpected label rather than
    raising, so a rare misclassification never crashes the render
    pipeline -- it just picks a default, which is a safer failure
    mode for a real-time system than an exception.
    """
    path = AVATAR_PATHS.get(gender_label)
    if path is None:
        print(f"[warn] unrecognized gender label '{gender_label}', defaulting to male avatar")
        path = AVATAR_PATHS["male"]

    if not os.path.exists(path):
        print(f"[warn] avatar file not found at {path} -- "
              f"place your Ready Player Me .glb exports in {AVATAR_DIR}/")

    return path
