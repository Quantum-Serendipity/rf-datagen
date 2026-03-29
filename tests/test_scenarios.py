"""Tests for scenario selection and weighting logic."""

import numpy as np
import pytest

from rf_datagen.constants import FS
from rf_datagen.config import ImpairmentConfig
from rf_datagen.impairments.scenarios import (
    _build_weights, _SCENARIO_FUNCS, SCENARIO_NAMES,
    apply_scenario, apply_scenario_continuous, configure,
)


@pytest.fixture
def tone_signal():
    """2048-sample complex tone at 1 kHz."""
    n = 2048
    t = np.arange(n) / FS
    return np.exp(2j * np.pi * 1000 * t)


# ---- Tests ----


def test_build_weights_normalizes():
    """_build_weights() returns an array that sums to 1.0."""
    weights = _build_weights()
    assert weights.shape == (len(_SCENARIO_FUNCS),)
    assert abs(weights.sum() - 1.0) < 1e-9


def test_build_weights_uniform_fallback():
    """When all scenario weights are 0, _build_weights() falls back to uniform."""
    original_cfg = ImpairmentConfig()
    try:
        configure(ImpairmentConfig(scenario_weights={}))
        weights = _build_weights()
        expected = 1.0 / len(_SCENARIO_FUNCS)
        np.testing.assert_allclose(weights, expected, atol=1e-12)
    finally:
        configure(original_cfg)


def test_apply_scenario_returns_tuple(tone_signal):
    """apply_scenario() returns (complex_array, scenario_name_string)."""
    snr_db = 20
    result = apply_scenario(tone_signal, snr_db, FS)
    sig_out, name = result
    assert isinstance(sig_out, np.ndarray)
    assert np.iscomplexobj(sig_out)
    assert len(sig_out) == len(tone_signal)
    assert isinstance(name, str)
    assert name in SCENARIO_NAMES


def test_apply_scenario_all_19_scenarios_execute(tone_signal):
    """Each of the 19 scenarios executes without error on a synthetic signal."""
    snr_db = 15
    for name, func in _SCENARIO_FUNCS.items():
        np.random.seed(12345)
        result = func(tone_signal.copy(), snr_db, FS)
        assert isinstance(result, np.ndarray), f"Scenario {name!r} did not return ndarray"
        assert np.iscomplexobj(result), f"Scenario {name!r} output is not complex"
        assert len(result) == len(tone_signal), (
            f"Scenario {name!r} changed signal length"
        )


def test_all_scenario_names_in_registry():
    """SCENARIO_NAMES contains exactly 19 entries matching _SCENARIO_FUNCS keys."""
    assert len(SCENARIO_NAMES) == 19
    assert set(SCENARIO_NAMES) == set(_SCENARIO_FUNCS.keys())


def test_apply_scenario_continuous_explicit_name(tone_signal):
    """apply_scenario_continuous with explicit scenario returns that scenario name."""
    snr_db = 20
    sig_out, name = apply_scenario_continuous(tone_signal, snr_db, FS, scenario="hf_clean")
    assert name == "hf_clean"
    assert isinstance(sig_out, np.ndarray)
    assert len(sig_out) == len(tone_signal)


def test_apply_scenario_continuous_invalid_name_raises(tone_signal):
    """apply_scenario_continuous raises ValueError for unknown scenario."""
    with pytest.raises(ValueError, match="Unknown scenario"):
        apply_scenario_continuous(tone_signal, 20, FS, scenario="nonexistent")


def test_new_scenarios_nonzero_weight():
    """All 7 new scenario names have nonzero weight in the default config."""
    cfg = ImpairmentConfig()
    new_names = [
        "indoor_multipath", "leo_satellite", "automotive",
        "urban_cellular", "radar_clutter", "maritime", "ism_congested",
    ]
    for name in new_names:
        assert name in cfg.scenario_weights, f"{name!r} missing from scenario_weights"
        assert cfg.scenario_weights[name] > 0, f"{name!r} has zero weight"
