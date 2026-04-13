"""Unit tests for hardening changes: atomic writes, config hashing,
checkpoint validation, shutdown state, structured logging."""

import csv
import json
import os
import tempfile
import shutil

import numpy as np
import pytest

from rf_datagen import _state
from rf_datagen.config import (
    Config, ConfigError, GeneratorConfig, ImpairmentConfig,
    checkpoint_config_hash, load_config,
)
from rf_datagen.generators.base import BaseGenerator
from rf_datagen.output import (
    atomic_save_npy, atomic_write_csv, atomic_write_json,
    assemble_parts,
)
from rf_datagen.logging_config import get_logger


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp(tmp_path):
    """Return a temporary directory path as a string."""
    return str(tmp_path)


@pytest.fixture
def parts_dir(tmp):
    """Create and return a parts/ subdirectory."""
    d = os.path.join(tmp, "parts")
    os.makedirs(d)
    return d


# ---------------------------------------------------------------------------
# Atomic writes
# ---------------------------------------------------------------------------

class TestAtomicSaveNpy:
    def test_basic(self, tmp):
        path = os.path.join(tmp, "data.npy")
        arr = np.array([1.0, 2.0, 3.0])
        atomic_save_npy(path, arr)
        loaded = np.load(path)
        np.testing.assert_array_equal(loaded, arr)

    def test_overwrite(self, tmp):
        path = os.path.join(tmp, "data.npy")
        atomic_save_npy(path, np.array([1.0]))
        atomic_save_npy(path, np.array([2.0, 3.0]))
        loaded = np.load(path)
        np.testing.assert_array_equal(loaded, np.array([2.0, 3.0]))

    def test_no_temp_files_left(self, tmp):
        path = os.path.join(tmp, "data.npy")
        atomic_save_npy(path, np.zeros(10))
        files = os.listdir(tmp)
        assert files == ["data.npy"]


class TestAtomicWriteCsv:
    def test_basic(self, tmp):
        path = os.path.join(tmp, "meta.csv")
        atomic_write_csv(path, ["a", "b"], [["1", "2"], ["3", "4"]])
        with open(path) as f:
            reader = csv.reader(f)
            rows = list(reader)
        assert rows[0] == ["a", "b"]
        assert rows[1] == ["1", "2"]
        assert rows[2] == ["3", "4"]

    def test_no_temp_files_left(self, tmp):
        path = os.path.join(tmp, "meta.csv")
        atomic_write_csv(path, ["x"], [["y"]])
        files = os.listdir(tmp)
        assert files == ["meta.csv"]


class TestAtomicWriteJson:
    def test_basic(self, tmp):
        path = os.path.join(tmp, "report.json")
        data = {"key": "value", "num": 42, "nested": [1, 2]}
        atomic_write_json(path, data)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded == data

    def test_overwrite(self, tmp):
        path = os.path.join(tmp, "report.json")
        atomic_write_json(path, {"v": 1})
        atomic_write_json(path, {"v": 2})
        with open(path) as f:
            assert json.load(f) == {"v": 2}

    def test_no_temp_files_left(self, tmp):
        path = os.path.join(tmp, "report.json")
        atomic_write_json(path, {})
        files = os.listdir(tmp)
        assert files == ["report.json"]


# ---------------------------------------------------------------------------
# Config hashing
# ---------------------------------------------------------------------------

class TestCheckpointConfigHash:
    def test_deterministic(self):
        gc = GeneratorConfig()
        ic = ImpairmentConfig()
        h1 = checkpoint_config_hash(gc, ic, "FT8", 6000)
        h2 = checkpoint_config_hash(gc, ic, "FT8", 6000)
        assert h1 == h2

    def test_length(self):
        h = checkpoint_config_hash(GeneratorConfig(), ImpairmentConfig(),
                                   "FT8", 6000)
        assert len(h) == 12
        assert all(c in "0123456789abcdef" for c in h)

    def test_differs_on_class(self):
        gc, ic = GeneratorConfig(), ImpairmentConfig()
        assert (checkpoint_config_hash(gc, ic, "FT8", 6000) !=
                checkpoint_config_hash(gc, ic, "CW", 6000))

    def test_differs_on_n_samples(self):
        gc, ic = GeneratorConfig(), ImpairmentConfig()
        assert (checkpoint_config_hash(gc, ic, "FT8", 6000) !=
                checkpoint_config_hash(gc, ic, "FT8", 9000))

    def test_differs_on_impairments(self):
        gc = GeneratorConfig()
        ic1 = ImpairmentConfig(max_freq_offset=100)
        ic2 = ImpairmentConfig(max_freq_offset=200)
        assert (checkpoint_config_hash(gc, ic1, "FT8", 6000) !=
                checkpoint_config_hash(gc, ic2, "FT8", 6000))

    def test_differs_on_generator_rsid(self):
        ic = ImpairmentConfig()
        gc1 = GeneratorConfig(rsid_probability=0.35)
        gc2 = GeneratorConfig(rsid_probability=0.50)
        assert (checkpoint_config_hash(gc1, ic, "FT8", 6000) !=
                checkpoint_config_hash(gc2, ic, "FT8", 6000))

    def test_differs_on_generator_codec2(self):
        ic = ImpairmentConfig()
        gc1 = GeneratorConfig(codec2_mode="3200")
        gc2 = GeneratorConfig(codec2_mode="1600")
        assert (checkpoint_config_hash(gc1, ic, "DMR", 6000) !=
                checkpoint_config_hash(gc2, ic, "DMR", 6000))

    def test_differs_on_generator_cw_wpm(self):
        ic = ImpairmentConfig()
        gc1 = GeneratorConfig(cw_wpm_range=[10, 30])
        gc2 = GeneratorConfig(cw_wpm_range=[15, 35])
        assert (checkpoint_config_hash(gc1, ic, "CW", 6000) !=
                checkpoint_config_hash(gc2, ic, "CW", 6000))

    def test_differs_on_generator_freedv_modes(self):
        ic = ImpairmentConfig()
        gc1 = GeneratorConfig(freedv_modes=["1600", "700C"])
        gc2 = GeneratorConfig(freedv_modes=["1600", "700D"])
        assert (checkpoint_config_hash(gc1, ic, "FREEDV", 6000) !=
                checkpoint_config_hash(gc2, ic, "FREEDV", 6000))

    def test_differs_on_fs(self):
        gc, ic = GeneratorConfig(), ImpairmentConfig()
        assert (checkpoint_config_hash(gc, ic, "FT8", 6000, fs=48000) !=
                checkpoint_config_hash(gc, ic, "FT8", 6000, fs=24000))


class TestGeneratorConfigHashDict:
    def test_keys(self):
        d = GeneratorConfig()._hash_dict()
        expected = {"utterances_per_class", "rsid_probability",
                    "cw_wpm_range", "messages_per_mode", "images_per_mode",
                    "packets_per_baud", "codec2_mode", "freedv_modes",
                    "cw_tone_range", "minimodem_modes", "ardop_speeds"}
        assert set(d.keys()) == expected

    def test_excludes_non_content_fields(self):
        d = GeneratorConfig()._hash_dict()
        # These should NOT be in the hash
        for key in ("enabled", "workers", "voice_cache", "classes",
                    "samples_per_class", "boost"):
            assert key not in d


class TestImpairmentConfigHashDict:
    def test_keys(self):
        d = ImpairmentConfig()._hash_dict()
        expected = {"snr_levels", "max_freq_offset", "scenario_weights",
                    "watterson_model", "window_stride", "window_power_threshold"}
        assert set(d.keys()) == expected

    def test_sorted_snr(self):
        ic = ImpairmentConfig(snr_levels=[20, 5, 10])
        d = ic._hash_dict()
        assert d["snr_levels"] == [5, 10, 20]


# ---------------------------------------------------------------------------
# Checkpoint validation
# ---------------------------------------------------------------------------

class TestCheckCheckpoint:
    def _make_checkpoint(self, parts_dir, name, n_samples, cfg_hash):
        """Create a valid checkpoint triplet."""
        npy_path = os.path.join(parts_dir, f"{name}.npy")
        meta_path = os.path.join(parts_dir, f"{name}_meta.csv")
        hash_path = os.path.join(parts_dir, f"{name}.hash")
        arr = np.zeros((n_samples, 2048), dtype=np.complex128)
        np.save(npy_path, arr)
        with open(meta_path, "w") as f:
            f.write("scenario\n" + "hf_clean\n" * n_samples)
        with open(hash_path, "w") as f:
            f.write(cfg_hash)
        return npy_path, meta_path, hash_path

    def test_valid_checkpoint(self, parts_dir):
        npy, meta, hsh = self._make_checkpoint(parts_dir, "FT8", 100, "abc123")
        assert BaseGenerator._check_checkpoint(npy, meta, hsh, 100, "abc123")

    def test_missing_npy(self, parts_dir):
        _, meta, hsh = self._make_checkpoint(parts_dir, "FT8", 100, "abc123")
        os.remove(os.path.join(parts_dir, "FT8.npy"))
        assert not BaseGenerator._check_checkpoint(
            os.path.join(parts_dir, "FT8.npy"), meta, hsh, 100, "abc123")

    def test_missing_meta(self, parts_dir):
        npy, meta, hsh = self._make_checkpoint(parts_dir, "FT8", 100, "abc123")
        os.remove(meta)
        assert not BaseGenerator._check_checkpoint(npy, meta, hsh, 100, "abc123")

    def test_missing_hash_file_invalidates(self, parts_dir):
        """Missing .hash file must invalidate — fail-safe."""
        npy, meta, hsh = self._make_checkpoint(parts_dir, "FT8", 100, "abc123")
        os.remove(hsh)
        assert not BaseGenerator._check_checkpoint(npy, meta, hsh, 100, "abc123")

    def test_wrong_hash_invalidates(self, parts_dir):
        npy, meta, hsh = self._make_checkpoint(parts_dir, "FT8", 100, "abc123")
        assert not BaseGenerator._check_checkpoint(npy, meta, hsh, 100, "different")

    def test_wrong_sample_count_invalidates(self, parts_dir):
        npy, meta, hsh = self._make_checkpoint(parts_dir, "FT8", 100, "abc123")
        assert not BaseGenerator._check_checkpoint(npy, meta, hsh, 200, "abc123")

    def test_corrupt_npy_invalidates(self, parts_dir):
        npy, meta, hsh = self._make_checkpoint(parts_dir, "FT8", 100, "abc123")
        with open(npy, "wb") as f:
            f.write(b"not a numpy file")
        assert not BaseGenerator._check_checkpoint(npy, meta, hsh, 100, "abc123")


# ---------------------------------------------------------------------------
# Shutdown state
# ---------------------------------------------------------------------------

class TestShutdownState:
    def setup_method(self):
        _state.reset_shutdown()

    def test_initial_state(self):
        assert not _state.shutdown_requested()

    def test_request_shutdown(self):
        _state.request_shutdown()
        assert _state.shutdown_requested()

    def test_reset_shutdown(self):
        _state.request_shutdown()
        _state.reset_shutdown()
        assert not _state.shutdown_requested()

    def test_no_cli_import_needed(self):
        """shutdown_requested is importable from _state, not cli."""
        from rf_datagen._state import shutdown_requested
        assert callable(shutdown_requested)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

class TestConfigValidation:
    def test_default_valid(self):
        Config().validate()  # should not raise

    def test_bad_sample_rate(self):
        cfg = Config()
        cfg.dataset.sample_rate = -1
        with pytest.raises(ConfigError, match="sample_rate"):
            cfg.validate()

    def test_bad_window_length(self):
        cfg = Config()
        cfg.dataset.window_length = 0
        with pytest.raises(ConfigError, match="window_length"):
            cfg.validate()

    def test_empty_snr_levels(self):
        cfg = Config()
        cfg.impairments.snr_levels = []
        with pytest.raises(ConfigError, match="snr_levels"):
            cfg.validate()

    def test_negative_stride(self):
        cfg = Config()
        cfg.impairments.window_stride = -1
        with pytest.raises(ConfigError, match="window_stride"):
            cfg.validate()

    def test_bad_samples_per_class(self):
        cfg = Config()
        cfg.generators["synthetic"].samples_per_class = 0
        with pytest.raises(ConfigError, match="samples_per_class"):
            cfg.validate()

    def test_disabled_generator_skips_validation(self):
        cfg = Config()
        cfg.generators["synthetic"].enabled = False
        cfg.generators["synthetic"].samples_per_class = 0
        cfg.validate()  # should not raise


class TestConfigUnknownKeys:
    def test_unknown_top_level(self, tmp, capsys):
        import tomllib
        path = os.path.join(tmp, "test.toml")
        with open(path, "w") as f:
            f.write('[typo_section]\nkey = 1\n')
        load_config(path)
        assert "typo_section" in capsys.readouterr().out

    def test_unknown_dataset_key(self, tmp, capsys):
        path = os.path.join(tmp, "test.toml")
        with open(path, "w") as f:
            f.write('[dataset]\ntypo_key = 1\n')
        load_config(path)
        assert "typo_key" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Assembly validation
# ---------------------------------------------------------------------------

class TestAssembleParts:
    def test_valid_parts(self, tmp):
        parts_dir = os.path.join(tmp, "parts")
        os.makedirs(parts_dir)
        # Use a real signal label
        from rf_datagen.constants import SIGNAL_LABELS, WINDOW_LEN
        label = SIGNAL_LABELS[0]
        arr = np.ones((10, WINDOW_LEN), dtype=np.complex128)
        np.save(os.path.join(parts_dir, f"{label}.npy"), arr)
        iq, tags, scenarios, _snrs = assemble_parts(tmp)
        assert iq.shape == (10, WINDOW_LEN)
        assert tags == [label] * 10

    def test_skips_wrong_shape(self, tmp, capsys):
        parts_dir = os.path.join(tmp, "parts")
        os.makedirs(parts_dir)
        from rf_datagen.constants import SIGNAL_LABELS
        label = SIGNAL_LABELS[0]
        arr = np.ones((10, 512), dtype=np.complex128)  # wrong width
        np.save(os.path.join(parts_dir, f"{label}.npy"), arr)
        iq, tags, *_ = assemble_parts(tmp)
        assert len(iq) == 0
        assert "wrong shape" in capsys.readouterr().out

    def test_skips_non_complex(self, tmp, capsys):
        parts_dir = os.path.join(tmp, "parts")
        os.makedirs(parts_dir)
        from rf_datagen.constants import SIGNAL_LABELS, WINDOW_LEN
        label = SIGNAL_LABELS[0]
        arr = np.ones((10, WINDOW_LEN), dtype=np.float64)  # not complex
        np.save(os.path.join(parts_dir, f"{label}.npy"), arr)
        iq, tags, *_ = assemble_parts(tmp)
        assert len(iq) == 0
        assert "complex dtype" in capsys.readouterr().out

    def test_skips_nan(self, tmp, capsys):
        parts_dir = os.path.join(tmp, "parts")
        os.makedirs(parts_dir)
        from rf_datagen.constants import SIGNAL_LABELS, WINDOW_LEN
        label = SIGNAL_LABELS[0]
        arr = np.ones((10, WINDOW_LEN), dtype=np.complex128)
        arr[0, 0] = np.nan
        np.save(os.path.join(parts_dir, f"{label}.npy"), arr)
        iq, tags, *_ = assemble_parts(tmp)
        assert len(iq) == 0
        assert "NaN" in capsys.readouterr().out

    def test_empty_parts_dir(self, tmp):
        iq, tags, scenarios, _snrs = assemble_parts(tmp)
        assert len(iq) == 0
        assert tags == []


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class TestLogging:
    def test_get_logger_namespace(self):
        log = get_logger("test_module")
        # structlog lazy proxy stores factory args with the logger name
        assert log._logger_factory_args == ("rf_datagen.test_module",)

    def test_loggers_independent(self):
        log1 = get_logger("a")
        log2 = get_logger("b")
        assert log1._logger_factory_args != log2._logger_factory_args
