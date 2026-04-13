import sys
from pathlib import Path

ROOT = Path("/public/ATIQA")
BIOGPT_ROOT = ROOT / "BioGPTLLM"
NVAE_ROOT = ROOT / "NVAEmaster"

for p in (BIOGPT_ROOT, NVAE_ROOT):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

__all__ = []
