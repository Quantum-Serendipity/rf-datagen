"""Shared fixtures for rf-datagen unit tests."""

import numpy as np
import pytest

FS = 12_000


@pytest.fixture(autouse=True)
def seed_rng():
    """Deterministic seed for every test function."""
    np.random.seed(12345)


@pytest.fixture
def tone_1k():
    """2048-sample complex tone at 1000 Hz, fs=12000."""
    n = 2048
    t = np.arange(n) / FS
    return np.exp(2j * np.pi * 1000 * t)


@pytest.fixture
def white_noise_iq():
    """4096-sample complex white noise (seeded via seed_rng)."""
    n = 4096
    return (np.random.randn(n) + 1j * np.random.randn(n)) / np.sqrt(2)


@pytest.fixture
def sample_bits():
    """200-element uint8 array of random bits."""
    return np.random.randint(0, 2, size=200).astype(np.uint8)


@pytest.fixture
def sample_dibits():
    """100-element uint8 array of random dibits {0,1,2,3}."""
    return np.random.randint(0, 4, size=100).astype(np.uint8)
