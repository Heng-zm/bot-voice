"""Canonical aliasing to app.features.ocr.extractor."""

from __future__ import annotations

import sys
import app.features.ocr.extractor as _mod

sys.modules[__name__] = _mod
