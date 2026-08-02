"""
src
---
Core package for the AI Finger Counter System.

Modules:
    hand_tracker      -- MediaPipe-based hand detection and landmark extraction
    finger_counter    -- Geometric logic for counting extended fingers
    gesture_detector  -- Temporal smoothing and named-gesture classification
    fps               -- Lightweight, smoothed frames-per-second counter
    utils             -- Reusable OpenCV drawing/UI helper functions
"""

from .fps import FPSCounter
from .finger_counter import FingerCounter
from .gesture_detector import GestureDetector, GestureStabilizer
from .hand_tracker import HandData, HandTracker

__all__ = [
    "HandTracker",
    "HandData",
    "FingerCounter",
    "GestureDetector",
    "GestureStabilizer",
    "FPSCounter",
]

__version__ = "1.0.0"
