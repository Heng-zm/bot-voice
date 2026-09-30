"""Canonical aliasing to app.features.ai.chat."""

from __future__ import annotations

import sys
import app.features.ai.chat as _mod

sys.modules[__name__] = _mod
