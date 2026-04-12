"""
Device detection utilities.

Supports CUDA (NVIDIA GPU), MPS (Apple Silicon), and CPU fallback.
All training code should call get_device() once and pass the result around,
rather than calling torch.cuda.is_available() inline.
"""

import torch


def get_device() -> torch.device:
    """Return the best available compute device.

    Priority: CUDA > MPS (Apple Silicon) > CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def is_amp_supported(device: torch.device) -> bool:
    """Return True only if Automatic Mixed Precision is supported.

    AMP requires CUDA. It is NOT supported on MPS or CPU.
    """
    return device.type == "cuda"


def get_dataloader_kwargs(device: torch.device) -> dict:
    """Return safe DataLoader keyword arguments for the given device.

    - num_workers > 0 is only safe on CUDA (multiprocessing issues on Mac/CPU).
    - pin_memory only helps on CUDA.
    """
    if device.type == "cuda":
        return {"num_workers": 2, "pin_memory": True}
    return {"num_workers": 0, "pin_memory": False}


def print_device_info(device: torch.device) -> None:
    """Print device info at startup for quick sanity check."""
    print(f"[Device] Using: {device}")
    if device.type == "cuda":
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[Device] AMP: enabled")
    elif device.type == "mps":
        print(f"[Device] Apple Silicon MPS backend")
        print(f"[Device] AMP: disabled (not supported on MPS)")
    else:
        print(f"[Device] CPU only")
        print(f"[Device] AMP: disabled")
