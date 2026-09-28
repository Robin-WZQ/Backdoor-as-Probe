"""CLI compatibility entry point for the output-low-energy implant.

Run from the BaP root, for example::

    CUDA_VISIBLE_DEVICES=6 \
      python -m BaP.probe.implant \
      --model_id openai/clip-vit-base-patch16 \
      --calibration_dir external_data/train_data \
      --output_dir artifacts/output_low_energy

An external ``--trigger_path`` may be supplied when an attack-aligned input
trigger has been prepared elsewhere.  If it is omitted, the implementation
derives a deterministic clean-only fallback trigger; no attack data is read
from this repository.
"""

from .output_low_energy import main


if __name__ == "__main__":
    main()
