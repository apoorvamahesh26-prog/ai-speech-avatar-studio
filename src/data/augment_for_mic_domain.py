"""
augment_for_mic_domain.py

Creates browser-microphone-like training audio by:

1. Converting RAVDESS WAV -> WebM/Opus
2. Decoding WebM/Opus back to WAV
3. Creating an additional WebM/Opus + noise variant

This reproduces the compression characteristics of browser
MediaRecorder audio and helps reduce train/inference domain shift.

Run:
    python src/data/augment_for_mic_domain.py

Then:
    python src/data/build_augmented_labels.py

Then extract features and retrain the gender model.
"""

import os
import glob
import subprocess

import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

SOURCE_DIRS = [
    (
        "data/processed/ravdess",
        "data/processed/ravdess_augmented",
    ),
]

TARGET_SR = 16000


# ============================================================
# WEBM / OPUS ROUNDTRIP
# ============================================================

def webm_opus_roundtrip(wav_path: str, out_path: str):
    """
    Convert:

        WAV -> WebM/Opus -> WAV

    Encoding settings:

        Codec       : Opus
        Bitrate     : 32 kbps
        Sample rate : 48 kHz
        Channels    : Mono
        Container   : WebM

    The -strict -2 option is included because the installed
    FFmpeg build requires experimental Opus/WebM muxing support.
    """

    # Temporary WebM filename
    temp_webm = os.path.splitext(out_path)[0] + ".tmp.webm"

    try:

        # ----------------------------------------------------
        # STEP 1: WAV -> WebM/Opus
        # ----------------------------------------------------

        encode_cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",

            # Input
            "-i",
            wav_path,

            # Audio only
            "-vn",

            # Select first audio stream
            "-map",
            "0:a:0",

            # Opus codec
            "-c:a",
            "libopus",

            # Browser-like bitrate
            "-b:a",
            "32k",

            # Browser microphone sample rate
            "-ar",
            "48000",

            # Mono
            "-ac",
            "1",

            # Required by this FFmpeg build
            # for Opus inside WebM
            "-strict",
            "-2",

            # Explicit WebM container
            "-f",
            "webm",

            # Temporary WebM output
            temp_webm,
        ]

        subprocess.run(
            encode_cmd,
            check=True,
        )

        # ----------------------------------------------------
        # STEP 2: WebM/Opus -> WAV
        # ----------------------------------------------------

        decode_cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",

            # Input WebM
            "-i",
            temp_webm,

            # Audio only
            "-vn",

            # Select first audio stream
            "-map",
            "0:a:0",

            # Target sample rate
            "-ar",
            str(TARGET_SR),

            # Mono
            "-ac",
            "1",

            # WAV output
            out_path,
        ]

        subprocess.run(
            decode_cmd,
            check=True,
        )

    finally:

        # Always remove temporary WebM file,
        # even if FFmpeg fails.
        if os.path.exists(temp_webm):
            try:
                os.remove(temp_webm)
            except OSError:
                pass


# ============================================================
# NOISE AUGMENTATION
# ============================================================

def add_noise(
    y: np.ndarray,
    snr_db: float,
) -> np.ndarray:
    """
    Add Gaussian noise at the requested SNR.

    snr_db=25 represents mild realistic
    microphone/background noise.
    """

    # Calculate signal power
    signal_power = np.mean(y ** 2)

    if signal_power <= 0:
        return y

    # Calculate required noise power
    noise_power = signal_power / (
        10 ** (snr_db / 10)
    )

    # Generate Gaussian noise
    noise = np.random.normal(
        0,
        np.sqrt(noise_power),
        size=y.shape,
    )

    # Add noise
    noisy = y + noise

    # Prevent clipping
    max_abs = np.max(np.abs(noisy))

    if max_abs > 1.0:
        noisy = noisy / max_abs

    return noisy


# ============================================================
# DIRECTORY AUGMENTATION
# ============================================================

def augment_directory(
    input_dir: str,
    output_dir: str,
):
    """
    Generate two augmented versions of every WAV file:

    Variant 1:
        WebM/Opus compressed audio

    Variant 2:
        WebM/Opus compressed audio + mild noise
    """

    # Create output directory
    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    # Find WAV files
    files = glob.glob(
        os.path.join(
            input_dir,
            "*.wav",
        )
    )

    print()
    print("=" * 70)
    print("MIC DOMAIN AUGMENTATION")
    print("=" * 70)
    print(
        f"Input directory : {input_dir}"
    )
    print(
        f"Output directory: {output_dir}"
    )
    print(
        f"Input WAV files : {len(files)}"
    )
    print("=" * 70)
    print()

    successful_webm = 0
    successful_noisy = 0
    failed = 0

    # --------------------------------------------------------
    # Process every WAV file
    # --------------------------------------------------------

    for path in tqdm(
        files,
        desc="Creating mic-domain audio",
    ):

        stem = os.path.splitext(
            os.path.basename(path)
        )[0]

        # ----------------------------------------------------
        # Variant 1:
        # WebM/Opus compressed audio
        # ----------------------------------------------------

        out1 = os.path.join(
            output_dir,
            f"{stem}_webmopus.wav",
        )

        try:

            webm_opus_roundtrip(
                path,
                out1,
            )

            successful_webm += 1

        except subprocess.CalledProcessError as e:

            failed += 1

            print()
            print(
                "[WARNING] FFmpeg conversion failed:"
            )
            print(
                f"          Input: {path}"
            )
            print(
                f"          Error: {e}"
            )

            continue

        except Exception as e:

            failed += 1

            print()
            print(
                "[WARNING] Unexpected conversion error:"
            )
            print(
                f"          Input: {path}"
            )
            print(
                f"          Error: {e}"
            )

            continue

        # ----------------------------------------------------
        # Variant 2:
        # WebM/Opus + noise
        # ----------------------------------------------------

        try:

            # Load compressed/decompressed audio
            y, sr = librosa.load(
                out1,
                sr=TARGET_SR,
                mono=True,
            )

            # Add mild noise
            y_noisy = add_noise(
                y,
                snr_db=25,
            )

            # Output filename
            out2 = os.path.join(
                output_dir,
                f"{stem}_webmopus_noisy.wav",
            )

            # Save noisy version
            sf.write(
                out2,
                y_noisy,
                TARGET_SR,
            )

            successful_noisy += 1

        except Exception as e:

            print()
            print(
                "[WARNING] Noise augmentation failed:"
            )
            print(
                f"          Input: {out1}"
            )
            print(
                f"          Error: {e}"
            )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("AUGMENTATION COMPLETE")
    print("=" * 70)

    print(
        f"Original WAV files       : {len(files)}"
    )

    print(
        f"WebM/Opus variants       : {successful_webm}"
    )

    print(
        f"WebM/Opus + noise        : {successful_noisy}"
    )

    print(
        f"Failed conversions       : {failed}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Check whether augmentation succeeded
    # --------------------------------------------------------

    if successful_webm == 0:

        print()
        print(
            "[ERROR] No WebM/Opus files were created."
        )
        print()
        print(
            "Please verify that FFmpeg supports:"
        )
        print(
            "    libopus"
        )
        print()
        return

    # ========================================================
    # NEXT STEPS
    # ========================================================

    print()
    print("Next steps:")
    print()

    # Step 1
    print(
        "1. Build labels:"
    )

    print(
        "   python src/data/build_augmented_labels.py"
    )

    print()

    # Step 2
    print(
        "2. Extract Wav2Vec2 features:"
    )

    print(
        '   python -c "from src.audio.features import '
        'batch_extract_directory; '
        'batch_extract_directory('
        "'data/processed/ravdess_augmented', "
        "'data/processed/features/ravdess_augmented', "
        'target_fps=30)"'
    )

    print()

    # Step 3
    print(
        "3. Retrain gender model:"
    )

    print(
        "   python src/training/train_gender.py"
    )

    print()

    # Step 4
    print(
        "4. Recalibrate gender threshold:"
    )

    print(
        "   python src/training/calibrate_gender_threshold.py"
    )

    print()

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    for input_dir, output_dir in SOURCE_DIRS:

        augment_directory(
            input_dir,
            output_dir,
        )