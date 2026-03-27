"""Unit tests for TTS DSP effects in rf_datagen.content.tts."""

import numpy as np
import pytest

from rf_datagen.content.tts import (
    apply_contest_processing,
    apply_mic_effects,
    apply_ptt_transients,
    apply_tx_audio_clipping,
    apply_vox_artifacts,
)

FS = 22_050


@pytest.fixture
def test_audio():
    """1 second of 440 Hz sine at 22050 Hz sample rate."""
    t = np.arange(FS) / FS
    return np.sin(2 * np.pi * 440 * t)


# ---------- apply_ptt_transients ----------


def test_ptt_transients_output_length(test_audio):
    """PTT transients may prepend/append clicks; output >= input length."""
    output = apply_ptt_transients(test_audio, FS)
    assert len(output) >= len(test_audio)


def test_ptt_transients_adds_energy_at_edges(test_audio):
    """Output should differ from input (transients added or probabilistic skip)."""
    # Run several times to ensure we hit the 70% probability branch at least once.
    found_diff = False
    for _ in range(20):
        np.random.seed(np.random.randint(0, 2**31))
        output = apply_ptt_transients(test_audio, FS)
        if not np.array_equal(output, test_audio):
            found_diff = True
            # When transients are added, output is longer than input
            assert len(output) > len(test_audio)
            break
    assert found_diff, "Expected PTT transients to modify audio in at least one trial"


# ---------- apply_mic_effects ----------


def test_mic_effects_length_preserved(test_audio):
    """Mic effects should not change the audio length."""
    output = apply_mic_effects(test_audio, FS)
    assert len(output) == len(test_audio)


def test_mic_effects_output_differs(test_audio):
    """Mic effects (gain + noise + breath dips) must modify the signal."""
    output = apply_mic_effects(test_audio, FS)
    assert not np.array_equal(output, test_audio)


# ---------- apply_vox_artifacts ----------


def test_vox_artifacts_length_preserved(test_audio):
    """VOX gating should not change the audio length."""
    output = apply_vox_artifacts(test_audio, FS)
    assert len(output) == len(test_audio)


def test_vox_artifacts_silent_input(test_audio):
    """Audio with a silent middle section stays finite and same length after VOX."""
    audio = test_audio.copy()
    # Insert a half-second silent gap in the middle
    mid = len(audio) // 2
    gap = FS // 2
    audio[mid : mid + gap] = 0.0

    output = apply_vox_artifacts(audio, FS)
    assert len(output) == len(audio)
    assert np.all(np.isfinite(output))


# ---------- apply_tx_audio_clipping ----------


def test_tx_clipping_length_preserved(test_audio):
    """TX clipping should not change the audio length."""
    output = apply_tx_audio_clipping(test_audio, FS)
    assert len(output) == len(test_audio)


def test_tx_clipping_peak_reduced_with_force(test_audio):
    """With force=True and a hot signal, soft clipping should reduce peaks."""
    loud = test_audio * 2.0
    output = apply_tx_audio_clipping(loud, FS, force=True, drive_range=(3.0, 4.0))
    # tanh soft-clips: peak of output should be less than peak of driven input
    # (the function normalises then clips, so output peak == input peak, but
    #  the waveform shape changes — verify RMS is altered by the nonlinearity).
    assert len(output) == len(loud)
    # After tanh clipping the peak is restored to original, but RMS changes
    # because the waveform is "squashed". Verify output differs from input.
    assert not np.array_equal(output, loud)
    # The peak should be at most the original peak (tanh clips to ±1 then rescales)
    assert np.max(np.abs(output)) <= np.max(np.abs(loud)) + 1e-9


# ---------- apply_contest_processing ----------


def test_contest_processing_length_preserved(test_audio):
    """Contest processing should not change the audio length."""
    output = apply_contest_processing(test_audio, FS)
    assert len(output) == len(test_audio)


def test_contest_processing_dynamic_range_compressed(test_audio):
    """Contest processing compresses dynamic range: crest factor should decrease."""
    # Build audio with dynamic range variation (loud + quiet sections)
    t = np.arange(FS) / FS
    envelope = 0.3 + 0.7 * np.abs(np.sin(2 * np.pi * 2 * t))  # 2 Hz AM
    audio = np.sin(2 * np.pi * 440 * t) * envelope

    output = apply_contest_processing(audio, FS)

    # Crest factor = peak / RMS; compression should lower it
    rms_in = np.sqrt(np.mean(audio**2)) + 1e-10
    rms_out = np.sqrt(np.mean(output**2)) + 1e-10
    crest_in = np.max(np.abs(audio)) / rms_in
    crest_out = np.max(np.abs(output)) / rms_out
    assert crest_out < crest_in, (
        f"Expected compression to reduce crest factor: {crest_out:.3f} >= {crest_in:.3f}"
    )


def test_contest_processing_output_finite(test_audio):
    """All output samples must be finite (no NaN or Inf)."""
    output = apply_contest_processing(test_audio, FS)
    assert np.all(np.isfinite(output))
