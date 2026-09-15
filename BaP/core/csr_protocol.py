"""Constants required by the CSR benchmark protocol."""

CSR_OFFICIAL_REPOSITORY = "https://github.com/Summu77/CSR"
CSR_OFFICIAL_COMMIT = "a5009cc02f1c7ec8a554a12e77cd1f171a0d60ce"
CSR_ATTACK_GENERATOR = "scripts/generate_adv.py"
ZERO_SHOT_PROMPT = "a photo of a {}"
TEXT_BATCH_SIZE = 100
IMAGE_SERIALIZATION = "torchvision_to_pil_uint8_truncation"

ATTACK_IMPLEMENTATIONS = {
    "pgd": "PGD-Linf",
    "autoattack_apgd": "APGD-CE",
}
