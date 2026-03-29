"""Signal-to-model class mapping registry."""

import json
import os

from ..domains import ALL_SIGNAL_LABELS

_MAPPING_FILE = os.path.join(os.path.dirname(__file__), "mapping.json")

_CACHE = {}


def load_mapping():
    """Load and validate the signal-to-model mapping.

    Returns dict: signal_name -> {model_family -> expected_class}.
    Every key must be in ALL_SIGNAL_LABELS.
    """
    if "data" in _CACHE:
        return _CACHE["data"]

    with open(_MAPPING_FILE) as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"mapping.yaml: expected dict, got {type(raw)}")

    # Validate all keys are known signal labels
    unknown = set(raw.keys()) - set(ALL_SIGNAL_LABELS)
    if unknown:
        raise ValueError(
            f"mapping.yaml: unknown signal labels: {sorted(unknown)}")

    missing = set(ALL_SIGNAL_LABELS) - set(raw.keys())
    if missing:
        raise ValueError(
            f"mapping.yaml: missing signal labels: {sorted(missing)}")

    _CACHE["data"] = raw
    return raw


def get_expected_class(signal_name, model_family):
    """Get expected model class for a signal under a model family.

    Returns class name string, or None if unmappable.
    """
    mapping = load_mapping()
    entry = mapping.get(signal_name, {})
    return entry.get(model_family)


def mappable_signals(model_family):
    """Return list of signals that have a mapping for the given model family."""
    mapping = load_mapping()
    return [s for s in ALL_SIGNAL_LABELS
            if mapping.get(s, {}).get(model_family) is not None]
