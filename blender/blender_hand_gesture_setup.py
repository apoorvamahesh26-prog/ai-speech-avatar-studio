"""
Blender helper for AI Speech Avatar Studio
------------------------------------------
Imports the two Ready Player Me / Mixamo-style GLBs, verifies their 67-bone
skeleton, creates a subtle conversational hand-gesture animation around the
existing standing/rest pose, and exports new GLBs.

Run from Blender:
    blender --background --python blender_hand_gesture_setup.py

Edit INPUT_DIR if the two source GLBs are in another folder.
"""

import bpy
import math
import os
from mathutils import Euler

INPUTS = {
    "female": "female_avatar(3).glb",
    "male": "male_avatar(1).glb",
}

OUTPUTS = {
    "female": "female_avatar_hand_gesture.glb",
    "male": "male_avatar_hand_gesture.glb",
}

# Small, presentation-friendly movement.
# Angles are degrees and are deliberately conservative.
KEYFRAMES = [1, 24, 48, 72, 96, 120]

POSES = {
    1:   {"LFA": (0, 0, 0),    "RFA": (0, 0, 0),    "LH": (0, 0, 0),    "RH": (0, 0, 0)},
    24:  {"LFA": (-3, 4, 5),   "RFA": (1, -2, -3), "LH": (-2, 2, 2),   "RH": (1, -2, -2)},
    48:  {"LFA": (-5, 7, 8),   "RFA": (2, -3, -4), "LH": (-3, 3, 3),   "RH": (1, -2, -2)},
    72:  {"LFA": (-1, 2, 3),   "RFA": (-3, 5, 6),  "LH": (-1, 1, 1),   "RH": (-2, 2, 2)},
    96:  {"LFA": (2, -4, -5),  "RFA": (-5, 7, 8),  "LH": (1, -2, -2),  "RH": (-3, 3, 3)},
    120: {"LFA": (0, 0, 0),    "RFA": (0, 0, 0),    "LH": (0, 0, 0),    "RH": (0, 0, 0)},
}

BONE_NAMES = {
    "LFA": "LeftForeArm",
    "RFA": "RightForeArm",
    "LH": "LeftHand",
    "RH": "RightHand",
}


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # Remove orphan data that can otherwise make repeated runs confusing.
    for datablocks in (
        bpy.data.actions,
        bpy.data.armatures,
        bpy.data.meshes,
        bpy.data.materials,
    ):
        for block in list(datablocks):
            if block.users == 0:
                datablocks.remove(block)


def find_armature():
    arms = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    if not arms:
        raise RuntimeError("No armature found after importing GLB.")
    return arms[0]


def make_gesture_action(armature):
    # Remove imported actions from the armature so the authored clip starts
    # from the GLB's actual standing/rest pose.
    armature.animation_data_clear()
    action = bpy.data.actions.new("SubtleHandGesture")
    armature.animation_data_create()
    armature.animation_data.action = action

    pose_bones = armature.pose.bones

    required = []
    for key, name in BONE_NAMES.items():
        if name not in pose_bones:
            raise RuntimeError(f"Required bone missing: {name}")
        required.append(name)

    print("Verified gesture bones:", required)

    # Use quaternion mode to avoid Euler-order surprises on imported rigs.
    for name in required:
        pb = pose_bones[name]
        pb.rotation_mode = "XYZ"

    # Frame 1 is the avatar's natural imported standing pose.
    bpy.context.scene.frame_set(1)

    for frame in KEYFRAMES:
        bpy.context.scene.frame_set(frame)
        pose = POSES[frame]

        for key, bone_name in BONE_NAMES.items():
            pb = pose_bones[bone_name]
            x, y, z = pose[key]
            pb.rotation_euler = Euler((
                math.radians(x),
                math.radians(y),
                math.radians(z),
            ), "XYZ")
            pb.keyframe_insert(data_path="rotation_euler", frame=frame)

    # Smooth interpolation for a human-like transition.
    for fc in action.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = "BEZIER"

    action.frame_range = (1, 120)
    return action


def export_glb(filepath):
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.context.scene.objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = find_armature()

    # Blender 3.6/4.x-compatible core export options.
    bpy.ops.export_scene.gltf(
        filepath=filepath,
        export_format="GLB",
        use_selection=True,
        export_animations=True,
    )


def process_one(input_dir, key):
    clear_scene()

    src = os.path.join(input_dir, INPUTS[key])
    dst = os.path.join(input_dir, OUTPUTS[key])

    if not os.path.exists(src):
        raise FileNotFoundError(src)

    print("\n=== Processing", key, "===")
    print("Import:", src)
    bpy.ops.import_scene.gltf(filepath=src)

    armature = find_armature()
    print("Armature:", armature.name)
    print("Bone count:", len(armature.data.bones))

    # The uploaded avatars have a 67-joint skin; this is a useful safety check.
    if len(armature.data.bones) < 60:
        raise RuntimeError(
            f"Unexpectedly small skeleton ({len(armature.data.bones)} bones). "
            "Use the full-body avatar export."
        )

    action = make_gesture_action(armature)
    print("Created action:", action.name, "frames:", action.frame_range)

    export_glb(dst)
    print("Exported:", dst)


def main():
    input_dir = os.path.dirname(os.path.abspath(__file__))

    for key in ("female", "male"):
        process_one(input_dir, key)

    print("\nDONE.")
    print("The generated files are:")
    for name in OUTPUTS.values():
        print("  ", os.path.join(input_dir, name))


if __name__ == "__main__":
    main()
