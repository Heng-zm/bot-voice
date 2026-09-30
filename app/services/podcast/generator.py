"""Canonical aliasing to app.features.podcast.service."""

from __future__ import annotations

import sys
import app.features.podcast.service as _mod

sys.modules[__name__] = _mod
