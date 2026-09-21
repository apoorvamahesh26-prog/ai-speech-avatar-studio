"""
backend/inference_pipeline.py

Complete inference pipeline for the speech-driven 3D avatar system.

Pipeline:

Audio
  ↓
Preprocessing
  ↓
Wav2Vec2 feature extraction
  ↓
Gender classification
  ├── Gender CNN
  ├── Pitch/F0 acoustic evidence
  └── Stable fusion
  ↓
Emotion recognition
  ↓
Viseme prediction
  ↓
Temporal smoothing
  ↓
Avatar selection
  ↓
Animation payload

Gender improvements:
- Uses the same voiced-frame pooling as gender training.
- Uses calibrated female probability threshold when available.
- Uses pitch as supporting acoustic evidence.
- Handles weak/uncertain CNN predictions more robustly.
- Prevents a strong female pitch signal from being completely
  overridden by a weak CNN prediction.
- Does NOT use pitch as an absolute male/female rule.
- Keeps avatar selection based on final gender decision.
"""

import os
import sys
import time
import json

import numpy as np
import torch


# ============================================================================
# Project path
# ============================================================================

sys.path.append(os.getcwd())


# ============================================================================
# Project imports
# ============================================================================

from src.audio.preprocess import load_and_clean

from src.audio.features import (
    Wav2VecFeatureExtractor,
    resample_features_to_fps,
)

from src.models.gender_model import GenderCNN
from src.models.emotion_model import EmotionBiLSTM
from src.models.viseme_model import VisemeTransformer
from src.models.smoothing_model import SmoothingCNN

from src.avatar.selector import select_avatar
from src.avatar.avatar_controller import build_animation

from src.data.gender_dataset import voiced_frame_pool


# ============================================================================
# Configuration
# ============================================================================

TARGET_FPS = 30

GENDER_LABELS = {
    0: "male",
    1: "female",
}

EMOTION_LABELS = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised",
]

CHECKPOINT_DIR = "checkpoints"

# Maximum sequence length expected by the models.
MAX_MODEL_FRAMES = 150


# ============================================================================
# Gender configuration
# ============================================================================

# Pitch is supporting evidence.
#
# We intentionally give the CNN more importance than pitch,
# but not so much that a clearly female-pitched voice gets
# incorrectly classified because of microphone/domain variation.

MODEL_WEIGHT = 0.60
PITCH_WEIGHT = 0.40

# When the CNN is very uncertain, pitch is allowed to have
# slightly more influence.
UNCERTAIN_MODEL_LOW = 0.35
UNCERTAIN_MODEL_HIGH = 0.65

# Strong pitch evidence.
STRONG_FEMALE_PITCH = 0.78
STRONG_MALE_PITCH = 0.22

# Very high female pitch evidence.
VERY_STRONG_FEMALE_PITCH = 0.88

# Very low female pitch evidence.
VERY_STRONG_MALE_PITCH = 0.12


# ============================================================================
# F0 / pitch estimation
# ============================================================================

def _estimate_median_f0(y, sr):
    """
    Estimate median voiced fundamental frequency using autocorrelation.

    Pitch is used only as secondary acoustic evidence.

    Returns:
        float or None
    """

    try:

        y = np.asarray(
            y,
            dtype=np.float32
        ).reshape(-1)

        # Need enough audio for reliable estimation.
        if len(y) < int(sr * 0.25):
            return None

        # Remove DC offset.
        y = y - np.mean(y)

        # Normalize amplitude.
        peak = float(
            np.max(
                np.abs(y)
            )
        )

        if peak < 1e-5:
            return None

        y = y / peak

        # ------------------------------------------------------------
        # Analysis parameters
        # ------------------------------------------------------------

        # 40 ms frame.
        frame_len = int(
            sr * 0.040
        )

        # 20 ms hop.
        hop = int(
            sr * 0.020
        )

        # Human speech F0 range.
        min_f0 = 70.0
        max_f0 = 350.0

        min_lag = max(
            1,
            int(sr / max_f0)
        )

        max_lag = int(
            sr / min_f0
        )

        f0_values = []

        # ------------------------------------------------------------
        # Frame analysis
        # ------------------------------------------------------------

        for start in range(
            0,
            len(y) - frame_len + 1,
            hop
        ):

            frame = y[
                start:start + frame_len
            ]

            # RMS energy.
            rms = float(
                np.sqrt(
                    np.mean(
                        frame * frame
                    )
                )
            )

            # Ignore very quiet frames.
            if rms < 0.025:
                continue

            # Remove frame DC.
            frame = (
                frame -
                np.mean(frame)
            )

            energy = float(
                np.dot(
                    frame,
                    frame
                )
            )

            if energy < 1e-6:
                continue

            # --------------------------------------------------------
            # Autocorrelation
            # --------------------------------------------------------

            corr = np.correlate(
                frame,
                frame,
                mode="full"
            )

            corr = corr[
                frame_len - 1:
            ]

            upper = min(
                max_lag,
                len(corr) - 1
            )

            if upper <= min_lag:
                continue

            region = corr[
                min_lag:upper + 1
            ]

            lag = (
                int(
                    np.argmax(region)
                )
                + min_lag
            )

            strength = float(
                corr[lag] /
                (corr[0] + 1e-8)
            )

            # Reject weak periodicity.
            if strength < 0.30:
                continue

            f0 = (
                sr /
                float(lag)
            )

            if (
                min_f0 <= f0 <= max_f0
            ):
                f0_values.append(
                    f0
                )

        # Need enough voiced frames.
        if len(f0_values) < 5:
            return None

        # Remove extreme outliers before median.
        values = np.asarray(
            f0_values,
            dtype=np.float32
        )

        q1 = np.percentile(
            values,
            10
        )

        q9 = np.percentile(
            values,
            90
        )

        filtered = values[
            (values >= q1) &
            (values <= q9)
        ]

        if len(filtered) < 3:
            filtered = values

        return float(
            np.median(filtered)
        )

    except Exception as e:

        print(
            f"[F0] estimation failed: {e}"
        )

        return None


# ============================================================================
# Pitch -> soft female probability
# ============================================================================

def _pitch_female_probability(median_f0):
    """
    Convert median F0 to a soft female probability.

    This is deliberately a broad transition because male and
    female pitch ranges overlap.

    Approximate interpretation:

        ~100 Hz  -> strongly male
        ~150 Hz  -> mostly male
        ~180 Hz  -> uncertain
        ~200 Hz  -> leaning female
        ~220 Hz  -> female leaning
        ~250 Hz  -> strongly female
        ~300 Hz  -> very strongly female

    Pitch is NEVER used as a hard gender rule.
    """

    if median_f0 is None:
        return None

    # Logistic transition.
    probability = (
        1.0 /
        (
            1.0 +
            np.exp(
                -(median_f0 - 165.0)
                / 22.0
            )
        )
    )

    return float(
        np.clip(
            probability,
            0.0,
            1.0
        )
    )


# ============================================================================
# Gender fusion
# ============================================================================

def _fuse_gender_probabilities(
    model_female_prob,
    pitch_female_prob
):
    """
    Combine CNN and pitch evidence.

    The CNN remains the primary signal.

    However, microphone recordings can cause domain shift.
    In that situation the CNN can become unreliable while
    pitch remains a useful supporting signal.

    Returns:
        final_female_probability,
        effective_model_weight,
        effective_pitch_weight,
        fusion_reason
    """

    model_female_prob = float(
        np.clip(
            model_female_prob,
            0.0,
            1.0
        )
    )

    # No usable pitch.
    if pitch_female_prob is None:

        return (
            model_female_prob,
            1.0,
            0.0,
            "CNN only - pitch unavailable"
        )

    pitch_female_prob = float(
        np.clip(
            pitch_female_prob,
            0.0,
            1.0
        )
    )

    # ------------------------------------------------------------
    # Default fusion
    # ------------------------------------------------------------

    model_weight = MODEL_WEIGHT
    pitch_weight = PITCH_WEIGHT

    reason = (
        "CNN + pitch weighted fusion"
    )

    # ------------------------------------------------------------
    # Case 1:
    # CNN is uncertain.
    #
    # Give pitch slightly more influence.
    # ------------------------------------------------------------

    if (
        UNCERTAIN_MODEL_LOW
        <= model_female_prob
        <= UNCERTAIN_MODEL_HIGH
    ):

        model_weight = 0.50
        pitch_weight = 0.50

        reason = (
            "CNN uncertain - balanced CNN/pitch fusion"
        )

    # ------------------------------------------------------------
    # Case 2:
    # Very strong female pitch.
    #
    # If CNN says male but is not overwhelmingly confident,
    # do not let it completely suppress the acoustic evidence.
    # ------------------------------------------------------------

    elif (
        pitch_female_prob
        >= VERY_STRONG_FEMALE_PITCH
        and
        model_female_prob < 0.35
    ):

        model_weight = 0.45
        pitch_weight = 0.55

        reason = (
            "Very strong female pitch - increased pitch support"
        )

    # ------------------------------------------------------------
    # Case 3:
    # Very strong male pitch.
    #
    # Keep CNN primary, but allow pitch to support male decision.
    # ------------------------------------------------------------

    elif (
        pitch_female_prob
        <= VERY_STRONG_MALE_PITCH
        and
        model_female_prob > 0.65
    ):

        model_weight = 0.60
        pitch_weight = 0.40

        reason = (
            "Very strong male pitch - CNN retained as primary signal"
        )

    # ------------------------------------------------------------
    # Weighted fusion.
    # ------------------------------------------------------------

    female_prob = (
        model_weight *
        model_female_prob
        +
        pitch_weight *
        pitch_female_prob
    )

    # ------------------------------------------------------------
    # Additional consistency adjustment.
    #
    # If pitch is strongly female and CNN is moderately male,
    # prevent an extreme male result caused by domain shift.
    #
    # Example:
    #
    # CNN female = 0.10
    # pitch female = 0.93
    #
    # Normal fusion:
    # 0.45*0.10 + 0.55*0.93 = 0.5565
    #
    # This appropriately moves the result toward female.
    # ------------------------------------------------------------

    if (
        pitch_female_prob
        >= STRONG_FEMALE_PITCH
        and
        model_female_prob
        < 0.20
    ):

        # Move the probability moderately toward pitch.
        female_prob = (
            0.40 *
            female_prob
            +
            0.60 *
            pitch_female_prob
        )

        reason += (
            " + strong female acoustic consistency"
        )

    elif (
        pitch_female_prob
        <= STRONG_MALE_PITCH
        and
        model_female_prob
        > 0.80
    ):

        # Keep CNN strong but acknowledge male pitch.
        female_prob = (
            0.70 *
            female_prob
            +
            0.30 *
            pitch_female_prob
        )

        reason += (
            " + strong male acoustic consistency"
        )

    female_prob = float(
        np.clip(
            female_prob,
            0.0,
            1.0
        )
    )

    return (
        female_prob,
        model_weight,
        pitch_weight,
        reason
    )


# ============================================================================
# Inference Pipeline
# ============================================================================

class InferencePipeline:

    def __init__(
        self,
        device: str = None
    ):

        # ------------------------------------------------------------
        # Device
        # ------------------------------------------------------------

        self.device = (
            device
            or
            (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        print(
            "[InferencePipeline] "
            f"loading all models on "
            f"{self.device}..."
        )

        # ------------------------------------------------------------
        # CPU optimization
        # ------------------------------------------------------------

        if self.device == "cpu":

            try:

                cpu_count = os.cpu_count()

                if cpu_count is not None:
                    torch.set_num_threads(
                        max(
                            1,
                            cpu_count
                        )
                    )

            except Exception:
                pass

        # ------------------------------------------------------------
        # Wav2Vec2 feature extractor
        # ------------------------------------------------------------

        self.feature_extractor = (
            Wav2VecFeatureExtractor(
                device=self.device,
                quantize=False
            )
        )

        # ------------------------------------------------------------
        # Gender threshold
        # ------------------------------------------------------------

        self.gender_female_threshold = 0.5

        threshold_path = os.path.join(
            CHECKPOINT_DIR,
            "gender_threshold.json"
        )

        if os.path.exists(
            threshold_path
        ):

            try:

                with open(
                    threshold_path,
                    "r",
                    encoding="utf-8"
                ) as f:

                    calibration = json.load(f)

                threshold = calibration.get(
                    "female_probability_threshold",
                    0.5
                )

                self.gender_female_threshold = float(
                    threshold
                )

                if not (
                    0.0
                    <
                    self.gender_female_threshold
                    <
                    1.0
                ):

                    print(
                        "[Warning] Invalid gender "
                        "threshold. Using 0.5."
                    )

                    self.gender_female_threshold = 0.5

                print(
                    "[InferencePipeline] "
                    "loaded calibrated gender "
                    "threshold: "
                    f"female_prob >= "
                    f"{self.gender_female_threshold:.4f}"
                )

            except Exception as e:

                print(
                    "[Warning] Could not load "
                    "gender_threshold.json: "
                    f"{e}"
                )

                self.gender_female_threshold = 0.5

        else:

            print(
                "[InferencePipeline] "
                "gender_threshold.json not found. "
                "Using default threshold 0.5."
            )

        # ------------------------------------------------------------
        # Gender CNN
        # ------------------------------------------------------------

        self.gender_model = (
            GenderCNN().to(
                self.device
            )
        )

        gender_checkpoint = os.path.join(
            CHECKPOINT_DIR,
            "gender_cnn_best.pt"
        )

        if not os.path.exists(
            gender_checkpoint
        ):

            raise FileNotFoundError(
                "Gender checkpoint not found: "
                f"{gender_checkpoint}"
            )

        gender_state = torch.load(
            gender_checkpoint,
            map_location=self.device
        )

        # Support both plain state_dict and checkpoint dictionaries.
        if isinstance(
            gender_state,
            dict
        ) and "state_dict" in gender_state:

            gender_state = (
                gender_state[
                    "state_dict"
                ]
            )

        self.gender_model.load_state_dict(
            gender_state
        )

        self.gender_model.eval()

        # ------------------------------------------------------------
        # Emotion model
        # ------------------------------------------------------------

        self.emotion_model = (
            EmotionBiLSTM().to(
                self.device
            )
        )

        emotion_checkpoint = os.path.join(
            CHECKPOINT_DIR,
            "emotion_bilstm_best.pt"
        )

        if not os.path.exists(
            emotion_checkpoint
        ):

            raise FileNotFoundError(
                "Emotion checkpoint not found: "
                f"{emotion_checkpoint}"
            )

        emotion_state = torch.load(
            emotion_checkpoint,
            map_location=self.device
        )

        if isinstance(
            emotion_state,
            dict
        ) and "state_dict" in emotion_state:

            emotion_state = (
                emotion_state[
                    "state_dict"
                ]
            )

        self.emotion_model.load_state_dict(
            emotion_state
        )

        self.emotion_model.eval()

        # ------------------------------------------------------------
        # Viseme model
        # ------------------------------------------------------------

        self.viseme_model = (
            VisemeTransformer().to(
                self.device
            )
        )

        viseme_checkpoint = os.path.join(
            CHECKPOINT_DIR,
            "viseme_transformer_best.pt"
        )

        if not os.path.exists(
            viseme_checkpoint
        ):

            raise FileNotFoundError(
                "Viseme checkpoint not found: "
                f"{viseme_checkpoint}"
            )

        viseme_state = torch.load(
            viseme_checkpoint,
            map_location=self.device
        )

        if isinstance(
            viseme_state,
            dict
        ) and "state_dict" in viseme_state:

            viseme_state = (
                viseme_state[
                    "state_dict"
                ]
            )

        self.viseme_model.load_state_dict(
            viseme_state
        )

        self.viseme_model.eval()

        # ------------------------------------------------------------
        # Smoothing model
        # ------------------------------------------------------------

        self.smoothing_model = (
            SmoothingCNN().to(
                self.device
            )
        )

        smoothing_checkpoint = os.path.join(
            CHECKPOINT_DIR,
            "smoothing_cnn_best.pt"
        )

        if not os.path.exists(
            smoothing_checkpoint
        ):

            raise FileNotFoundError(
                "Smoothing checkpoint not found: "
                f"{smoothing_checkpoint}"
            )

        smoothing_state = torch.load(
            smoothing_checkpoint,
            map_location=self.device
        )

        if isinstance(
            smoothing_state,
            dict
        ) and "state_dict" in smoothing_state:

            smoothing_state = (
                smoothing_state[
                    "state_dict"
                ]
            )

        self.smoothing_model.load_state_dict(
            smoothing_state
        )

        self.smoothing_model.eval()

        # ------------------------------------------------------------
        # Ready
        # ------------------------------------------------------------

        print(
            "[InferencePipeline] "
            "all models loaded, ready to "
            "serve requests."
        )

    # =========================================================================
    # Full inference
    # =========================================================================

    @torch.no_grad()
    def run(
        self,
        audio_path: str,
        profile: bool = False
    ) -> dict:

        """
        Run complete inference.

        Returns the animation payload expected by the backend.
        """

        timings = {}

        t_start = time.time()

        # ---------------------------------------------------------------------
        # Timing helper
        # ---------------------------------------------------------------------

        def mark(
            stage_name,
            t_stage_start
        ):

            timings[stage_name] = round(
                time.time()
                -
                t_stage_start,
                4
            )

        # =====================================================================
        # 1. AUDIO PREPROCESSING
        # =====================================================================

        t0 = time.time()

        y, sr = load_and_clean(
            audio_path
        )

        y = np.asarray(
            y,
            dtype=np.float32
        ).reshape(-1)

        if len(y) == 0:

            raise ValueError(
                "Audio preprocessing returned "
                "an empty audio signal."
            )

        duration = (
            len(y)
            /
            float(sr)
        )

        mark(
            "preprocess",
            t0
        )

        # =====================================================================
        # 2. WAV2VEC2 FEATURE EXTRACTION
        # =====================================================================

        t0 = time.time()

        raw_feats = (
            self.feature_extractor.extract(
                y
            )
        )

        feats = (
            resample_features_to_fps(
                raw_feats,
                duration,
                TARGET_FPS
            )
        )

        feats = np.asarray(
            feats,
            dtype=np.float32
        )

        if feats.ndim != 2:

            raise ValueError(
                "Expected Wav2Vec2 features with "
                "shape (T, 768), but received "
                f"{feats.shape}"
            )

        T = feats.shape[0]

        if T <= 0:

            raise ValueError(
                "Feature extraction produced "
                "zero frames."
            )

        feats_t = torch.tensor(
            feats,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        # Shape:
        # (1, T, 768)

        mark(
            "feature_extraction",
            t0
        )

        # =====================================================================
        # 3. GENDER CLASSIFICATION
        # =====================================================================

        t0 = time.time()

        # ---------------------------------------------------------------------
        # IMPORTANT:
        #
        # Training used:
        #
        # voiced_frame_pool(
        #     feats,
        #     keep_fraction=0.50
        # )
        #
        # Therefore inference MUST use the same representation.
        # ---------------------------------------------------------------------

        pooled_np = voiced_frame_pool(
            feats,
            keep_fraction=0.50
        )

        pooled_np = np.asarray(
            pooled_np,
            dtype=np.float32
        ).reshape(-1)

        if pooled_np.shape[0] != 768:

            raise ValueError(
                "Gender pooled feature must contain "
                f"768 values, received "
                f"{pooled_np.shape[0]}."
            )

        pooled = torch.tensor(
            pooled_np,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        # Shape:
        # (1, 768)

        # ---------------------------------------------------------------------
        # CNN prediction
        # ---------------------------------------------------------------------

        gender_logits = (
            self.gender_model(
                pooled
            )
        )

        gender_probs = torch.softmax(
            gender_logits,
            dim=1
        ).squeeze(0)

        model_male_prob = float(
            gender_probs[0].item()
        )

        model_female_prob = float(
            gender_probs[1].item()
        )

        # ---------------------------------------------------------------------
        # Pitch analysis
        # ---------------------------------------------------------------------

        median_f0 = (
            _estimate_median_f0(
                y,
                sr
            )
        )

        pitch_female_prob = (
            _pitch_female_probability(
                median_f0
            )
        )

        # ---------------------------------------------------------------------
        # CNN + pitch fusion
        # ---------------------------------------------------------------------

        (
            female_prob,
            effective_model_weight,
            effective_pitch_weight,
            fusion_reason
        ) = _fuse_gender_probabilities(
            model_female_prob,
            pitch_female_prob
        )

        # ---------------------------------------------------------------------
        # Final gender decision
        # ---------------------------------------------------------------------

        if (
            female_prob
            >=
            self.gender_female_threshold
        ):

            gender_id = 1

        else:

            gender_id = 0

        gender_label = (
            GENDER_LABELS[
                gender_id
            ]
        )

        # ---------------------------------------------------------------------
        # Confidence
        # ---------------------------------------------------------------------

        if gender_id == 1:

            gender_confidence = (
                female_prob
            )

        else:

            gender_confidence = (
                1.0
                -
                female_prob
            )

        gender_confidence = round(
            float(
                np.clip(
                    gender_confidence,
                    0.0,
                    1.0
                )
            ),
            4
        )

        mark(
            "gender",
            t0
        )

        # =====================================================================
        # GENDER DEBUG INFORMATION
        # =====================================================================

        pitch_text = (
            f"{pitch_female_prob:.4f}"
            if pitch_female_prob is not None
            else "n/a"
        )

        f0_text = (
            f"{median_f0:.2f}"
            if median_f0 is not None
            else "n/a"
        )

        print(
            "\n"
            "============================================================\n"
            "[Gender Debug]\n"
            "------------------------------------------------------------\n"
            f"  model_male_prob       = "
            f"{model_male_prob:.4f}\n"
            f"  model_female_prob     = "
            f"{model_female_prob:.4f}\n"
            f"  median_f0              = "
            f"{f0_text}\n"
            f"  pitch_female_prob     = "
            f"{pitch_text}\n"
            f"  model_weight          = "
            f"{effective_model_weight:.2f}\n"
            f"  pitch_weight          = "
            f"{effective_pitch_weight:.2f}\n"
            f"  final_female_prob     = "
            f"{female_prob:.4f}\n"
            f"  calibrated_threshold  = "
            f"{self.gender_female_threshold:.4f}\n"
            f"  final_gender          = "
            f"{gender_label}\n"
            f"  gender_confidence     = "
            f"{gender_confidence:.4f}\n"
            f"  fusion_reason         = "
            f"{fusion_reason}\n"
            "============================================================"
        )

        print(
            f"[Pipeline] "
            f"audio={os.path.basename(audio_path)} "
            f"-> gender={gender_label} "
            f"(confidence={gender_confidence}, "
            f"model_female={model_female_prob:.3f}, "
            f"pitch_female={pitch_text}, "
            f"median_f0={f0_text})"
        )

        # =====================================================================
        # 4. EMOTION RECOGNITION
        # =====================================================================

        t0 = time.time()

        length_t = torch.tensor(
            [T],
            dtype=torch.int64
        )

        emotion_logits = (
            self.emotion_model(
                feats_t,
                length_t
            )
        )

        emotion_probs = torch.softmax(
            emotion_logits,
            dim=1
        ).squeeze(0)

        emotion_id = int(
            emotion_logits
            .argmax(
                dim=1
            )
            .item()
        )

        emotion_label = (
            EMOTION_LABELS[
                emotion_id
            ]
        )

        emotion_confidence = round(
            float(
                emotion_probs[
                    emotion_id
                ].item()
            ),
            4
        )

        mark(
            "emotion",
            t0
        )

        print(
            f"[Pipeline] "
            f"audio={os.path.basename(audio_path)} "
            f"-> emotion={emotion_label} "
            f"(confidence={emotion_confidence})"
        )

        # =====================================================================
        # 5. VISEME PREDICTION
        # =====================================================================

        t0 = time.time()

        viseme_probs_chunks = []

        for start in range(
            0,
            T,
            MAX_MODEL_FRAMES
        ):

            end = min(
                start
                +
                MAX_MODEL_FRAMES,
                T
            )

            chunk = (
                feats_t[
                    :,
                    start:end,
                    :
                ]
            )

            chunk_logits = (
                self.viseme_model(
                    chunk,
                    key_padding_mask=None
                )
            )

            chunk_probs = torch.softmax(
                chunk_logits,
                dim=-1
            )

            viseme_probs_chunks.append(
                chunk_probs
            )

        if len(
            viseme_probs_chunks
        ) == 0:

            raise ValueError(
                "Viseme model received no "
                "feature chunks."
            )

        viseme_probs = torch.cat(
            viseme_probs_chunks,
            dim=1
        )

        mark(
            "viseme",
            t0
        )

        # =====================================================================
        # 6. TEMPORAL SMOOTHING
        # =====================================================================

        t0 = time.time()

        smoothed_logits = (
            self.smoothing_model(
                viseme_probs
            )
        )

        smoothed_ids = (
            smoothed_logits
            .argmax(
                dim=-1
            )
            .squeeze(0)
            .cpu()
            .numpy()
        )

        mark(
            "smoothing",
            t0
        )

        # =====================================================================
        # 7. AVATAR SELECTION
        # =====================================================================

        t0 = time.time()

        avatar_path = select_avatar(
            gender_label
        )

        # =====================================================================
        # 8. AVATAR ANIMATION
        # =====================================================================

        animation = build_animation(
            smoothed_ids,
            emotion=emotion_label
        )

        mark(
            "avatar_controller",
            t0
        )

        print(
            f"[Pipeline] "
            f"audio={os.path.basename(audio_path)} "
            f"-> avatar_path={avatar_path} "
            f"(selected FROM gender={gender_label})"
        )

        # =====================================================================
        # 9. TIMING
        # =====================================================================

        timings["total"] = round(
            time.time()
            -
            t_start,
            4
        )

        timings[
            "audio_duration_sec"
        ] = round(
            duration,
            4
        )

        timings["rtf"] = round(
            timings["total"]
            /
            max(
                duration,
                1e-6
            ),
            4
        )

        # =====================================================================
        # 10. FINAL RESULT
        # =====================================================================

        result = {

            "gender":
                gender_label,

            "gender_confidence":
                gender_confidence,

            "avatar_path":
                avatar_path,

            "emotion":
                emotion_label,

            "emotion_confidence":
                emotion_confidence,

            "audio_duration":
                round(
                    duration,
                    3
                ),

            "fps":
                TARGET_FPS,

            "blendshape_names":
                animation[
                    "blendshape_names"
                ],

            "weights":
                animation[
                    "weights"
                ].tolist(),

            "visemes":
                smoothed_ids.tolist(),
        }

        # ---------------------------------------------------------------------
        # Profiling information
        # ---------------------------------------------------------------------

        if profile:

            result[
                "timings"
            ] = timings

        return result