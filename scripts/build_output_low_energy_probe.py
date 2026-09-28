#!/usr/bin/env python3
"""Convenience wrapper for ``python -m BaP.probe.implant``."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from BaP.probe.output_low_energy import main


if __name__ == "__main__":
    main()
