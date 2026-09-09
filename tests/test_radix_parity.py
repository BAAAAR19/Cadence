"""Week 3: differential fuzz of the C++17 core against the Python reference.

The extension does not exist yet, so this file is a placeholder that skips
rather than a passing test that checks nothing -- a green tick for work that has
not been done is worse than a visible gap.
"""

from __future__ import annotations

import pytest

pytest.importorskip("cadence._core", reason="Week 3: the C++ extension is not built yet")
