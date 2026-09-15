"""BaP v5: Backdoors as Probes.

The public path uses an output-low-energy semantic target for implantation
and a bias-free target-direction response for detection.  Datasets shipped in
``data/`` are benign only; model checkpoints, implanted artifacts and attack
outputs are runtime inputs/outputs and are intentionally not bundled.
"""

from __future__ import annotations

import os


def configure_gpu_visibility(gpu_id: str | int | None = None) -> str | None:
    """Optionally select visible CUDA devices without imposing lab-local IDs.

    ``gpu_id`` may be a CUDA device index, a comma-separated device list, or
    ``None``.  When omitted, an existing ``CUDA_VISIBLE_DEVICES`` setting is
    preserved.  CPU jobs can use ``gpu_id="cpu"`` or simply omit the option.
    """

    selected = gpu_id
    if selected is None:
        selected = os.environ.get("BAP_GPU_ID")
    if selected is None:
        selected = os.environ.get("GPU_ID")
    if selected is None:
        selected = os.environ.get("CUDA_VISIBLE_DEVICES")
    if selected is None:
        return None

    selected_text = str(selected).strip()
    if not selected_text or selected_text.lower() in {"cpu", "none"}:
        return selected_text or None
    os.environ["BAP_GPU_ID"] = selected_text
    os.environ["CUDA_VISIBLE_DEVICES"] = selected_text
    return selected_text


def enforce_gpu_policy(device: str | object) -> str | None:
    """Validate that the requested logical CUDA device is visible.

    The launcher should set ``CUDA_VISIBLE_DEVICES`` before importing/running
    a CUDA workload.  Logical ``cuda:0``/``cuda:1`` then map to the selected
    physical IDs in order.  CPU execution is allowed for smoke checks.
    """

    import torch

    parsed = torch.device(device)
    if parsed.type != "cuda":
        return None
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible.strip():
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES before starting a CUDA workload.")
    physical = [item.strip() for item in visible.split(",") if item.strip()]
    if not physical:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must contain at least one device id.")
    logical = 0 if parsed.index is None else int(parsed.index)
    if logical < 0 or logical >= len(physical):
        raise RuntimeError(
            f"Logical device {parsed} is not available in CUDA_VISIBLE_DEVICES={visible!r}."
        )
    return physical[logical]


__version__ = "5.0.0"
