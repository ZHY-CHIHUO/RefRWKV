#!/usr/bin/env python3
"""Alias for the standalone YAML-driven test entry point."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.test.run import main


if __name__ == "__main__":
    main()
