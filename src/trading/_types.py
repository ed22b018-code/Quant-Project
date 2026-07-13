"""Shared type aliases for the src.trading package."""
from __future__ import annotations

from typing import Literal

Direction = Literal["LONG", "SHORT"]

# Superset of Direction — includes the model's "no opinion" signal.
PredictionDirection = Literal["LONG", "SHORT", "NO_TRADE"]
