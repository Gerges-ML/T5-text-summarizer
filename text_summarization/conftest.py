"""
conftest.py
-----------
Pytest configuration: ensures the project root is on sys.path so that all
package imports (utils, models, services, interface) resolve correctly when
running ``pytest`` from any working directory.
"""

import sys
from pathlib import Path

# Insert project root at the front of sys.path
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
