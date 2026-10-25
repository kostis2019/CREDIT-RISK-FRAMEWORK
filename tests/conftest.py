"""
Pytest configuration.

Adds the project root to sys.path so that tests can import modules from
the src package, matching the project-root setup used by the notebooks.
This allows pytest to be run directly with `pytest` from the repository root.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))