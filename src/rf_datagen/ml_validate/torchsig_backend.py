"""TorchSig XCiT-Tiny12 backend for narrowband modulation classification."""

import numpy as np

from ._runner import register_backend


# TorchSig narrowband class names (53 classes)
TORCHSIG_CLASSES = [
    "ook", "bpsk", "4ask", "4pam",
    "2fsk", "2gfsk", "2msk", "2gmsk",
    "qpsk", "oqpsk", "pi4dqpsk",
    "8psk", "16psk", "32psk",
    "16qam", "32qam", "64qam", "128qam", "256qam",
    "am-ssb-sc", "am-ssb-wc", "am-dsb-sc", "am-dsb-wc",
    "fm",
    "4fsk", "4gfsk", "4msk", "4gmsk",
    "8fsk", "8gfsk",
    "16fsk",
    "ofdm-64", "ofdm-72", "ofdm-128", "ofdm-180",
    "ofdm-256", "ofdm-300", "ofdm-512", "ofdm-600",
    "ofdm-900", "ofdm-1024", "ofdm-1200", "ofdm-2048",
]


@register_backend("torchsig")
class TorchSigValidator:
    """TorchSig XCiT-Tiny12 narrowband classifier."""

    name = "torchsig"
    class_names = TORCHSIG_CLASSES

    def __init__(self):
        self._model = None

    def load(self):
        """Load the TorchSig XCiT model (lazy)."""
        from . import _require_torch
        torch = _require_torch()

        from ._weights import ensure_model

        try:
            model_path = ensure_model("torchsig-xcit-tiny12")
        except RuntimeError as e:
            print(f"WARNING: {e}")
            print("TorchSig backend will use random predictions for testing.")
            self._model = None
            return

        try:
            # Try to load as a TorchSig checkpoint
            import torchsig
            from torchsig.models.iq_models.xcit import XCiT1D

            model = XCiT1D(
                model_name="xcit_tiny_12_p16",
                num_classes=len(TORCHSIG_CLASSES),
                in_chans=2,
            )

            checkpoint = torch.load(model_path, map_location="cpu",
                                    weights_only=False)
            if "state_dict" in checkpoint:
                state = checkpoint["state_dict"]
                # Strip "model." prefix if present (Lightning convention)
                state = {k.replace("model.", "", 1): v
                         for k, v in state.items()}
                model.load_state_dict(state, strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)

            model.eval()
            self._model = model
        except Exception as e:
            print(f"WARNING: Failed to load TorchSig model: {e}")
            self._model = None

    def predict(self, iq_batch):
        """Run inference on a batch of adapted IQ data.

        iq_batch: (N, 2, 4096) float32 array
        Returns: list of (class_name, confidence) tuples
        """
        if self._model is None:
            # Fallback: random predictions for testing
            results = []
            for _ in range(len(iq_batch)):
                idx = np.random.randint(len(TORCHSIG_CLASSES))
                results.append((TORCHSIG_CLASSES[idx], 0.0))
            return results

        from . import _require_torch
        torch = _require_torch()

        with torch.no_grad():
            x = torch.from_numpy(iq_batch).float()
            logits = self._model(x)
            probs = torch.softmax(logits, dim=-1)
            confs, indices = probs.max(dim=-1)

        results = []
        for conf, idx in zip(confs.numpy(), indices.numpy()):
            cls_name = TORCHSIG_CLASSES[idx]
            results.append((cls_name, float(conf)))
        return results
