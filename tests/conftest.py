# conftest.py tells pytest that this `tests/` folder is part of the project.
# It also adds the project root to sys.path so test files can import
# modules like `validate_prices` and `validate_ohlcv` without installing them.

import os
import sys

# Add the project root (one level up from this file) to the import path.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
