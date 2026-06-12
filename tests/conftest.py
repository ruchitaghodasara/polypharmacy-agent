"""
Shared pytest configuration.

Inserts the project root on sys.path so every test file can import
project modules without installing the package.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
