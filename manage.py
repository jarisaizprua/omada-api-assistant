#!/usr/bin/env python3
"""Project entry point.  Run ``python manage.py --help`` to see the commands."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from omada_agent.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
