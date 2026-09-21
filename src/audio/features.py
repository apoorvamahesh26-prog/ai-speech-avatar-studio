"""
features.py

Frozen Wav2Vec2 feature extractor used by the speech-avatar project.

Pipeline:

    Raw audio
        ↓
    Wav2Vec2 processor
        ↓
    Frozen Wav2Vec2 model
        ↓
    (T, 768) feature sequence
        ↓
    30 FPS interpolation
        ↓
    Gender / Emotion / Viseme models

The Wav2Vec2 model is NOT trained in this file.

This module supports:

1. Loading an already-cached Hugging Face model.
2. Downloading the model when it is not cached.
3. Clear error messages when the Hugging Face cache is incomplete.
4. CPU/GPU execution.
5. Optional dynamic INT8 quantization on CPU.
"""

import os
import glob
import traceback

import numpy as np
import torch
import librosa
from tqdm import tqdm

from transformers import (
    Wav2Vec2Model,
    Wav2Vec2FeatureExtractor,
)


# ============================================================================
# Configuration
# ============================================================================

MODEL_NAME = "facebook/wav2vec2-base-960h"

TARGET_SR = 16000

EXPECTED_HIDDEN_SIZE = 768


# ============================================================================
# Utility: determine device
# ============================================================================

def _get_device(device=None):
    """
    Determine the torch device.

    If device is explicitly supplied, use it.

    Otherwise:
        CUDA → GPU
        otherwise → CPU
    """

    if device is not None:
        return device

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


# ============================================================================
# Utility: load Hugging Face processor
# ============================================================================

def _load_processor():
    """
    Load the Wav2Vec2 feature extractor.

    Strategy:

    1. Try local cache first.
       This avoids unnecessary network access when the model
       has already been downloaded.

    2. If local loading fails, try normal Hugging Face loading.

    3. If both fail, raise a detailed error explaining what happened.
    """

    print(
        f"[Wav2Vec2] Loading processor: {MODEL_NAME}"
    )

    local_error = None

    # ------------------------------------------------------------------------
    # First attempt: local cache
    # ------------------------------------------------------------------------

    try:

        processor = Wav2Vec2FeatureExtractor.from_pretrained(
            MODEL_NAME,
            local_files_only=True,
        )

        print(
            "[Wav2Vec2] Processor loaded from local cache."
        )

        return processor

    except Exception as e:

        local_error = e

        print(
            "[Wav2Vec2] Processor not available completely "
            "in local cache."
        )

    # ------------------------------------------------------------------------
    # Second attempt: Hugging Face download
    # ------------------------------------------------------------------------

    try:

        print(
            "[Wav2Vec2] Attempting Hugging Face download..."
        )

        processor = Wav2Vec2FeatureExtractor.from_pretrained(
            MODEL_NAME
        )

        print(
            "[Wav2Vec2] Processor downloaded successfully."
        )

        return processor

    except Exception as download_error:

        print(
            "\n"
            "=" * 78
        )

        print(
            "[Wav2Vec2 ERROR] Unable to load processor."
        )

        print(
            f"Model: {MODEL_NAME}"
        )

        print(
            "\nLocal-cache error:"
        )

        print(
            f"{local_error}"
        )

        print(
            "\nDownload error:"
        )

        print(
            f"{download_error}"
        )

        print(
            "\nPossible causes:"
        )

        print(
            "  1. Hugging Face model download was interrupted."
        )

        print(
            "  2. Internet connection was temporarily unavailable."
        )

        print(
            "  3. Hugging Face cache contains incomplete files."
        )

        print(
            "  4. The model files are being accessed by another "
            "process."
        )

        print(
            "=" * 78
        )

        raise RuntimeError(
            "Could not load the Wav2Vec2 feature extractor. "
            "The model must be available in the Hugging Face "
            "cache or downloadable from Hugging Face."
        ) from download_error


# ============================================================================
# Utility: load Hugging Face model
# ============================================================================

def _load_model(device):
    """
    Load Wav2Vec2Model.

    Strategy:

    1. Try local cache first.
    2. If local loading fails, download normally.
    3. Move the model to the requested device.
    """

    print(
        f"[Wav2Vec2] Loading model: {MODEL_NAME}"
    )

    local_error = None

    # ------------------------------------------------------------------------
    # First attempt: local cache
    # ------------------------------------------------------------------------

    try:

        model = Wav2Vec2Model.from_pretrained(
            MODEL_NAME,
            local_files_only=True,
        )

        print(
            "[Wav2Vec2] Model loaded from local cache."
        )

    except Exception as e:

        local_error = e

        print(
            "[Wav2Vec2] Model not available completely "
            "in local cache."
        )

        # --------------------------------------------------------------------
        # Second attempt: Hugging Face download
        # --------------------------------------------------------------------

        try:

            print(
                "[Wav2Vec2] Attempting Hugging Face model download..."
            )

            model = Wav2Vec2Model.from_pretrained(
                MODEL_NAME
            )

            print(
                "[Wav2Vec2] Model downloaded successfully."
            )

        except Exception as download_error:

            print(
                "\n"
                "=" * 78
            )

            print(
                "[Wav2Vec2 ERROR] Unable to load model."
            )

            print(
                f"Model: {MODEL_NAME}"
            )

            print(
                "\nLocal-cache error:"
            )

            print(
                f"{local_error}"
            )

            print(
                "\nDownload error:"
            )

            print(
                f"{download_error}"
            )

            print(
                "\n"
                "The Hugging Face cache may contain an incomplete "
                "download."
            )

            print(
                "Delete the incomplete cached model and download "
                "it again."
            )

            print(
                "=" * 78
            )

            raise RuntimeError(
                "Could not load Wav2Vec2Model."
            ) from download_error

    # ------------------------------------------------------------------------
    # Move to requested device
    # ------------------------------------------------------------------------

    model = model.to(device)

    return model


# ============================================================================
# Wav2Vec2 Feature Extractor
# ============================================================================

class Wav2VecFeatureExtractor:

    def __init__(
        self,
        device: str = None,
        quantize: bool = False,
    ):
        """
        Create a frozen Wav2Vec2 feature extractor.

        Parameters
        ----------
        device:
            "cpu", "cuda", etc.
            If None, automatically selects CUDA when available,
            otherwise CPU.

        quantize:
            If True and running on CPU, dynamically quantize Linear
            layers to INT8.
        """

        # --------------------------------------------------------------------
        # Device
        # --------------------------------------------------------------------

        self.device = _get_device(
            device
        )

        print(
            "\n"
            "[Wav2Vec2] Initializing feature extractor"
        )

        print(
            f"[Wav2Vec2] Model       : {MODEL_NAME}"
        )

        print(
            f"[Wav2Vec2] Device      : {self.device}"
        )

        print(
            f"[Wav2Vec2] Target SR   : {TARGET_SR}"
        )

        print(
            f"[Wav2Vec2] Quantize    : {quantize}"
        )

        # --------------------------------------------------------------------
        # CPU optimization
        # --------------------------------------------------------------------

        if self.device == "cpu":

            try:

                cpu_count = os.cpu_count()

                if cpu_count is not None:

                    # Avoid unnecessarily huge thread counts.
                    thread_count = max(
                        1,
                        min(
                            cpu_count,
                            8
                        )
                    )

                    torch.set_num_threads(
                        thread_count
                    )

                    print(
                        "[Wav2Vec2] CPU threads:"
                        f" {thread_count}"
                    )

            except Exception as e:

                print(
                    "[Wav2Vec2] Could not configure "
                    f"CPU threads: {e}"
                )

        # --------------------------------------------------------------------
        # Load processor
        # --------------------------------------------------------------------

        self.processor = _load_processor()

        # --------------------------------------------------------------------
        # Load Wav2Vec2 model
        # --------------------------------------------------------------------

        self.model = _load_model(
            self.device
        )

        # --------------------------------------------------------------------
        # Freeze model
        # --------------------------------------------------------------------

        self.model.eval()

        for parameter in self.model.parameters():

            parameter.requires_grad = False

        # --------------------------------------------------------------------
        # Verify model hidden size
        # --------------------------------------------------------------------

        hidden_size = getattr(
            self.model.config,
            "hidden_size",
            None
        )

        if hidden_size is not None:

            print(
                f"[Wav2Vec2] Hidden size : {hidden_size}"
            )

            if hidden_size != EXPECTED_HIDDEN_SIZE:

                print(
                    "[Wav2Vec2 WARNING] Expected "
                    f"{EXPECTED_HIDDEN_SIZE} hidden dimensions "
                    f"but model reports {hidden_size}."
                )

        # --------------------------------------------------------------------
        # Dynamic INT8 quantization
        # --------------------------------------------------------------------

        if (
            quantize
            and
            self.device == "cpu"
        ):

            print(
                "[Wav2Vec2] Applying dynamic INT8 "
                "quantization to Linear layers..."
            )

            try:

                self.model = torch.quantization.quantize_dynamic(
                    self.model,
                    {
                        torch.nn.Linear
                    },
                    dtype=torch.qint8,
                )

                print(
                    "[Wav2Vec2] INT8 quantization enabled."
                )

            except Exception as e:

                print(
                    "[Wav2Vec2 WARNING] INT8 quantization "
                    f"failed: {e}"
                )

                print(
                    "[Wav2Vec2] Continuing without quantization."
                )

        elif (
            quantize
            and
            self.device != "cpu"
        ):

            print(
                "[Wav2Vec2 WARNING] quantize=True was requested "
                "but dynamic INT8 quantization is intended for CPU."
            )

            print(
                "[Wav2Vec2] Continuing without INT8 quantization."
            )

        # --------------------------------------------------------------------
        # Final confirmation
        # --------------------------------------------------------------------

        print(
            "[Wav2Vec2] Feature extractor ready."
        )

    # =========================================================================
    # Extract features from waveform
    # =========================================================================

    @torch.no_grad()
    def extract(
        self,
        waveform: np.ndarray,
    ) -> np.ndarray:
        """
        Extract frozen Wav2Vec2 features.

        Parameters
        ----------
        waveform:
            1-D numpy array containing mono audio at 16 kHz.

        Returns
        -------
        np.ndarray
            Shape:

                (T, 768)

            where T is the temporal feature length.
        """

        # --------------------------------------------------------------------
        # Validate waveform
        # --------------------------------------------------------------------

        if waveform is None:

            raise ValueError(
                "waveform is None."
            )

        waveform = np.asarray(
            waveform,
            dtype=np.float32
        )

        # Flatten possible (N, 1) input.
        waveform = waveform.reshape(-1)

        if waveform.size == 0:

            raise ValueError(
                "waveform is empty."
            )

        # --------------------------------------------------------------------
        # Remove NaN / Inf
        # --------------------------------------------------------------------

        if not np.all(
            np.isfinite(waveform)
        ):

            print(
                "[Wav2Vec2 WARNING] waveform contains "
                "NaN/Inf. Cleaning."
            )

            waveform = np.nan_to_num(
                waveform,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

        # --------------------------------------------------------------------
        # Processor
        # --------------------------------------------------------------------

        inputs = self.processor(
            waveform,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=False,
        )

        input_values = (
            inputs.input_values
            .to(self.device)
        )

        # --------------------------------------------------------------------
        # Model forward
        # --------------------------------------------------------------------

        outputs = self.model(
            input_values
        )

        features = (
            outputs
            .last_hidden_state
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False
            )
        )

        # --------------------------------------------------------------------
        # Safety check
        # --------------------------------------------------------------------

        if features.ndim != 2:

            raise RuntimeError(
                "Unexpected Wav2Vec2 feature shape: "
                f"{features.shape}"
            )

        if features.shape[1] != EXPECTED_HIDDEN_SIZE:

            raise RuntimeError(
                "Unexpected Wav2Vec2 hidden dimension. "
                f"Expected {EXPECTED_HIDDEN_SIZE}, "
                f"received {features.shape[1]}."
            )

        return features

    # =========================================================================
    # Extract features directly from audio file
    # =========================================================================

    def extract_from_file(
        self,
        path: str,
    ) -> np.ndarray:
        """
        Load an audio file, resample to 16 kHz, and extract features.
        """

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"Audio file not found: {path}"
            )

        y, sr = librosa.load(
            path,
            sr=TARGET_SR,
            mono=True,
        )

        return self.extract(
            y
        )


# ============================================================================
# Feature resampling
# ============================================================================

def resample_features_to_fps(
    features: np.ndarray,
    audio_duration_sec: float,
    target_fps: int,
) -> np.ndarray:
    """
    Resample Wav2Vec2 features to the target animation FPS.

    Wav2Vec2 produces a feature sequence at approximately 49 Hz.

    The avatar animation uses 30 FPS.

    Therefore:

        Wav2Vec2 features
              ↓
        Linear interpolation
              ↓
        30 FPS features

    Parameters
    ----------
    features:
        Array of shape (T, D).

    audio_duration_sec:
        Duration of the original audio in seconds.

    target_fps:
        Desired output FPS, normally 30.

    Returns
    -------
    np.ndarray
        Shape:

            (target_frames, D)
    """

    features = np.asarray(
        features,
        dtype=np.float32
    )

    # ------------------------------------------------------------------------
    # Validate input
    # ------------------------------------------------------------------------

    if features.ndim != 2:

        raise ValueError(
            "features must be a 2-D array "
            "(T, D). "
            f"Received shape: {features.shape}"
        )

    src_len = features.shape[0]

    feature_dim = features.shape[1]

    if src_len <= 0:

        raise ValueError(
            "features contains zero time frames."
        )

    if feature_dim <= 0:

        raise ValueError(
            "features contains zero feature dimensions."
        )

    if audio_duration_sec <= 0:

        raise ValueError(
            "audio_duration_sec must be greater than zero."
        )

    if target_fps <= 0:

        raise ValueError(
            "target_fps must be greater than zero."
        )

    # ------------------------------------------------------------------------
    # Number of target frames
    # ------------------------------------------------------------------------

    target_len = max(
        1,
        int(
            round(
                audio_duration_sec *
                target_fps
            )
        )
    )

    # ------------------------------------------------------------------------
    # If lengths already match, no interpolation is required.
    # ------------------------------------------------------------------------

    if target_len == src_len:

        return features.astype(
            np.float32,
            copy=False
        )

    # ------------------------------------------------------------------------
    # Source and target temporal indices
    # ------------------------------------------------------------------------

    src_idx = np.linspace(
        0.0,
        float(src_len - 1),
        src_len,
    )

    tgt_idx = np.linspace(
        0.0,
        float(src_len - 1),
        target_len,
    )

    # ------------------------------------------------------------------------
    # Interpolate every feature channel
    # ------------------------------------------------------------------------

    resampled = np.stack(
        [
            np.interp(
                tgt_idx,
                src_idx,
                features[:, channel],
            )
            for channel
            in range(feature_dim)
        ],
        axis=1,
    )

    return resampled.astype(
        np.float32,
        copy=False
    )


# ============================================================================
# Batch feature extraction
# ============================================================================

def batch_extract_directory(
    input_dir: str,
    output_dir: str,
    target_fps: int = 30,
    ext: str = "*.wav",
):
    """
    Extract and FPS-align features for every matching audio file.

    Output:

        output_dir/<same_stem>.npy

    Each .npy file contains:

        (T, 768)

    Returns
    -------
    list
        Failed files as:

            [(path, error_string), ...]
    """

    # ------------------------------------------------------------------------
    # Validate input directory
    # ------------------------------------------------------------------------

    if not os.path.isdir(
        input_dir
    ):

        raise FileNotFoundError(
            f"Input directory not found: {input_dir}"
        )

    # ------------------------------------------------------------------------
    # Create output directory
    # ------------------------------------------------------------------------

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    # ------------------------------------------------------------------------
    # Create extractor once.
    #
    # IMPORTANT:
    #
    # Do NOT create a new Wav2Vec2 model for every audio file.
    # ------------------------------------------------------------------------

    extractor = Wav2VecFeatureExtractor()

    # ------------------------------------------------------------------------
    # Find files recursively
    # ------------------------------------------------------------------------

    files = glob.glob(
        os.path.join(
            input_dir,
            "**",
            ext,
        ),
        recursive=True,
    )

    files.sort()

    print(
        f"[Feature Extraction] Found "
        f"{len(files)} files in {input_dir}"
    )

    failures = []

    # ------------------------------------------------------------------------
    # Process files
    # ------------------------------------------------------------------------

    for path in tqdm(
        files,
        desc=f"Extracting features: {input_dir}",
    ):

        try:

            # ---------------------------------------------------------------
            # Load audio
            # ---------------------------------------------------------------

            y, sr = librosa.load(
                path,
                sr=TARGET_SR,
                mono=True,
            )

            if y is None or len(y) == 0:

                raise ValueError(
                    "Audio file contains no samples."
                )

            duration = (
                len(y) /
                float(sr)
            )

            # ---------------------------------------------------------------
            # Wav2Vec2 features
            # ---------------------------------------------------------------

            raw_feats = extractor.extract(
                y
            )

            # ---------------------------------------------------------------
            # Align to target FPS
            # ---------------------------------------------------------------

            aligned_feats = (
                resample_features_to_fps(
                    raw_feats,
                    duration,
                    target_fps,
                )
            )

            # ---------------------------------------------------------------
            # Output filename
            # ---------------------------------------------------------------

            stem = os.path.splitext(
                os.path.basename(path)
            )[0]

            output_path = os.path.join(
                output_dir,
                f"{stem}.npy",
            )

            # ---------------------------------------------------------------
            # Save
            # ---------------------------------------------------------------

            np.save(
                output_path,
                aligned_feats,
            )

        except Exception as e:

            failures.append(
                (
                    path,
                    str(e),
                )
            )

            print(
                "\n[Feature Extraction ERROR]"
            )

            print(
                f"File: {path}"
            )

            print(
                f"Error: {e}"
            )

    # ------------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------------

    print(
        "\n"
        "[Feature Extraction] Complete."
    )

    print(
        f"  Total files : {len(files)}"
    )

    print(
        f"  Successful  : {len(files) - len(failures)}"
    )

    print(
        f"  Failed      : {len(failures)}"
    )

    return failures


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":

    fails = batch_extract_directory(
        input_dir="data/processed/ravdess",
        output_dir="data/processed/features/ravdess",
        target_fps=30,
    )

    print(
        "\n"
        f"RAVDESS feature extraction: "
        f"{len(fails)} failures"
    )

    if fails:

        print(
            "\nFailed files:"
        )

        for path, error in fails:

            print(
                f"\n  {path}"
            )

            print(
                f"    {error}"
            )