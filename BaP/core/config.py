"""Shared constants and optional test-time correction settings for BaP."""

from dataclasses import dataclass


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
# Canonical single-path BaP correction. Keep these values in one lightweight
# module so the optimizer, runners, and generated reports cannot drift apart.
CORRECTION_EPSILON = 4 / 255
CORRECTION_STEP_SIZE = 2 / 255
CORRECTION_ESCAPE_STEPS = 2
CORRECTION_REPAIR_STEPS = 1
CORRECTION_RANDOM_STARTS = 1
CORRECTION_SEED = 123
CORRECTION_RESIDUAL_WEIGHT = 1.0
CORRECTION_BENIGN_DIRECTION_WEIGHT = 0.05
CORRECTION_DIRECTION_POLICY = "benign_only"
CORRECTION_OUTPUT_RULE = "fixed_repair_endpoint"
CORRECTION_CANDIDATE_SELECTION = "none"
CORRECTION_METHOD_ID = (
    f"single_start_ttc_escape{CORRECTION_ESCAPE_STEPS}_"
    f"probe_repair{CORRECTION_REPAIR_STEPS}_no_attack_direction"
)


@dataclass
class TTCConfig:
    """Hyperparameters for the optional TTC baseline/evaluation wrapper."""

    eps: float = 4 / 255
    alpha: float = 2 / 255
    steps: int = 3
    tau: float = 0.3
    beta: float = 2.0
