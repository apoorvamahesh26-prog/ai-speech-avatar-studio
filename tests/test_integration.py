"""
test_integration.py

Two things, per Step 11:
  1. Correctness checks -- run the FULL pipeline on real audio and
     assert every output is well-formed (right types, right value
     ranges, right array shapes). This catches integration bugs that
     unit-testing each model in isolation (Steps 4-7) cannot: e.g. a
     shape mismatch between the Viseme model's output and the
     Smoothing model's expected input only shows up when they're
     actually chained together.
  2. Performance profiling -- aggregate per-stage timings across
     multiple clips into a report, so optimization effort (Step 11's
     later half) targets the ACTUAL bottleneck instead of a guess.

Run: python -m pytest tests/test_integration.py -v -s
(the -s flag is needed to see the printed profiling report, since
pytest captures stdout by default)
"""

import os
import sys
import glob
import numpy as np
import pytest

sys.path.append(os.getcwd())
from backend.inference_pipeline import InferencePipeline
from src.audio.phoneme_to_viseme import NUM_VISEMES

TEST_AUDIO_DIR = "data/processed/ravdess"  # reuse a few already-preprocessed clips
NUM_TEST_CLIPS = 5
VALID_GENDERS = {"male", "female"}
VALID_EMOTIONS = {"neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"}


@pytest.fixture(scope="module")
def pipeline():
    """
    scope="module": load models ONCE for the whole test file, not once
    per test function -- loading Wav2Vec2 + 4 checkpoints repeatedly
    would make the test suite itself slow enough to discourage running
    it often, defeating the purpose of having fast integration tests.
    """
    return InferencePipeline()


def _get_test_clips(n=NUM_TEST_CLIPS):
    files = sorted(glob.glob(os.path.join(TEST_AUDIO_DIR, "*.wav")))
    if len(files) == 0:
        pytest.skip(f"No test audio found in {TEST_AUDIO_DIR} -- run Step 2's "
                     f"preprocessing first.")
    return files[:n]


def test_pipeline_output_schema(pipeline):
    """Runs one clip and checks every field of the response is well-formed."""
    clips = _get_test_clips(1)
    result = pipeline.run(clips[0], profile=True)

    assert result["gender"] in VALID_GENDERS
    assert result["emotion"] in VALID_EMOTIONS
    assert result["fps"] == 30
    assert isinstance(result["blendshape_names"], list) and len(result["blendshape_names"]) > 0

    weights = np.array(result["weights"])
    assert weights.ndim == 2, "weights should be (T, num_blendshapes)"
    assert weights.shape[1] == len(result["blendshape_names"])
    assert np.all(weights >= -0.01) and np.all(weights <= 1.01), \
        "blendshape weights should stay within [0, 1] (small float tolerance)"

    assert "timings" in result
    print(f"\n[Schema check] OK -- gender={result['gender']}, emotion={result['emotion']}, "
          f"frames={weights.shape[0]}, blendshapes={weights.shape[1]}")


def test_pipeline_handles_multiple_clips_without_state_leak(pipeline):
    """
    Runs several DIFFERENT clips back-to-back through the SAME
    pipeline instance (mirroring how the FastAPI server actually
    reuses one InferencePipeline across many requests) and checks
    results don't leak state between calls -- e.g. two clips with
    different genders should actually produce different gender
    predictions, not the first call's cached result repeated.
    """
    clips = _get_test_clips(NUM_TEST_CLIPS)
    results = [pipeline.run(c, profile=True) for c in clips]

    genders = [r["gender"] for r in results]
    print(f"\n[State-leak check] genders across {len(clips)} clips: {genders}")
    # Not asserting variety here (a real batch of test clips might
    # legitimately share a gender), but printing lets you visually
    # confirm the predictions vary sensibly rather than being frozen.

    for clip, r in zip(clips, results):
        weights = np.array(r["weights"])
        assert weights.shape[0] > 0, f"empty animation for {clip}"


def test_performance_report(pipeline):
    """
    Aggregates per-stage timings across several clips and prints a
    report -- THIS is what tells you where to spend optimization
    effort, rather than guessing "Wav2Vec2 is probably slow."
    """
    clips = _get_test_clips(NUM_TEST_CLIPS)
    all_timings = []

    for clip in clips:
        result = pipeline.run(clip, profile=True)
        all_timings.append(result["timings"])

    stages = [k for k in all_timings[0].keys() if k not in ("total", "audio_duration_sec", "rtf")]

    print("\n=== Step 11: Performance Profiling Report ===")
    print(f"{'Stage':<20} {'Mean (s)':<12} {'% of total':<12}")
    mean_total = np.mean([t["total"] for t in all_timings])
    for stage in stages:
        mean_stage = np.mean([t[stage] for t in all_timings])
        pct = 100 * mean_stage / mean_total
        print(f"{stage:<20} {mean_stage:<12.4f} {pct:<12.1f}")

    mean_rtf = np.mean([t["rtf"] for t in all_timings])
    print(f"\nMean total latency: {mean_total:.4f}s")
    print(f"Mean Real-Time Factor (RTF): {mean_rtf:.4f} "
          f"({'faster' if mean_rtf < 1 else 'slower'} than real-time)")

    # Not a hard pass/fail assertion -- profiling is informational,
    # not correctness. We just sanity check nothing took a suspicious
    # eternity (e.g. a silently-retrying network call).
    assert mean_total < 60, "Pipeline took unexpectedly long -- check for a hung stage."


if __name__ == "__main__":
    # Allows running this file directly (python tests/test_integration.py)
    # as a quick smoke test, without needing pytest installed.
    p = InferencePipeline()
    clips = _get_test_clips()
    for c in clips:
        r = p.run(c, profile=True)
        print(c, "->", r["gender"], r["emotion"], "| RTF:", r["timings"]["rtf"])
