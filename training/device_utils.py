"""Device helper. Pick best GPU backend."""

import torch


def get_best_device() -> str:
    """Return cuda, directml, mps, or cpu."""
    if torch.cuda.is_available():
        return "cuda"
    try:
        import torch_directml
        if torch_directml.is_available():
            return "privateuseone"  # directml uses this device name
    except Exception:
        pass
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_module():
    """Return torch module for special backends."""
    dev = get_best_device()
    if dev == "privateuseone":
        try:
            import torch_directml
            return torch_directml
        except Exception:
            pass
    return torch
