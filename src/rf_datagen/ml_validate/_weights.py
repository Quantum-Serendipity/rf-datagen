"""Model weight download and cache management."""

import hashlib
import os
import sys
import urllib.request


# Default cache directory
_DEFAULT_CACHE = os.path.join(
    os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
    "rf-datagen", "ml-models")

CACHE_DIR = os.environ.get("RF_DATAGEN_MODEL_CACHE", _DEFAULT_CACHE)

# Known model weights: name -> (url, sha256, filename)
KNOWN_MODELS = {
    "torchsig-xcit-tiny12": {
        "url": "https://github.com/TorchDSP/torchsig/releases/download/v0.4.0/xcit_tiny_12_p16_narrowband.ckpt",
        "sha256": None,  # Populated when official hash published
        "filename": "xcit_tiny_12_p16_narrowband.ckpt",
    },
    "cgdnn-radioml2016": {
        "url": None,  # User must provide or download separately
        "sha256": None,
        "filename": "cgdnn_radioml2016.h5",
    },
}


def cache_dir():
    """Return (and create) the model cache directory."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    return CACHE_DIR


def model_path(name):
    """Return the local path for a cached model. Does not check existence."""
    info = KNOWN_MODELS.get(name)
    if info is None:
        raise ValueError(f"Unknown model: {name}. "
                         f"Available: {list(KNOWN_MODELS.keys())}")
    return os.path.join(cache_dir(), info["filename"])


def is_cached(name):
    """Check if a model is already downloaded."""
    path = model_path(name)
    return os.path.exists(path) and os.path.getsize(path) > 0


def _sha256_file(path):
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_model(name, force=False):
    """Download a model to the cache. Returns local path.

    Raises RuntimeError if download fails or hash mismatch.
    """
    info = KNOWN_MODELS.get(name)
    if info is None:
        raise ValueError(f"Unknown model: {name}")

    path = model_path(name)
    if not force and is_cached(name):
        return path

    url = info["url"]
    if url is None:
        raise RuntimeError(
            f"Model '{name}' must be downloaded manually. "
            f"Place it at: {path}")

    print(f"Downloading {name} from {url}...", file=sys.stderr)
    try:
        urllib.request.urlretrieve(url, path)
    except Exception as e:
        raise RuntimeError(f"Download failed for {name}: {e}") from e

    # Verify hash if known
    expected_hash = info.get("sha256")
    if expected_hash is not None:
        actual_hash = _sha256_file(path)
        if actual_hash != expected_hash:
            os.remove(path)
            raise RuntimeError(
                f"SHA-256 mismatch for {name}: "
                f"expected {expected_hash}, got {actual_hash}")

    return path


def ensure_model(name):
    """Ensure model is cached, downloading if necessary. Returns path.

    Raises RuntimeError with user-friendly message if unavailable.
    """
    if is_cached(name):
        return model_path(name)
    return download_model(name)
