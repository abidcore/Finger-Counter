"""
config/settings.py
-------------------
Centralized, single-source-of-truth configuration for the AI Finger
Counter System.

Keeping every tunable constant in one dedicated module (instead of
scattering "magic numbers" throughout the codebase) is a standard
production-software practice. It makes the application easy to tune,
review, and extend without touching business logic.
"""

import cv2

# --------------------------------------------------------------------------
# Camera / Capture Settings
# --------------------------------------------------------------------------
CAMERA_INDEX: int = 0                 # Default system webcam
FRAME_WIDTH: int = 1280
FRAME_HEIGHT: int = 720
FLIP_CAMERA_HORIZONTALLY: bool = True  # Mirror view feels natural to users
CAMERA_WARMUP_FRAMES: int = 5          # Frames to discard while sensor stabilizes
CAMERA_RECONNECT_ATTEMPTS: int = 3     # Attempts before declaring camera lost

# --------------------------------------------------------------------------
# MediaPipe Hands Model Settings
# --------------------------------------------------------------------------
MAX_NUM_HANDS: int = 2
MIN_DETECTION_CONFIDENCE: float = 0.75
MIN_TRACKING_CONFIDENCE: float = 0.70
BBOX_PADDING: int = 20                 # Pixels of padding around hand bounding box

# --------------------------------------------------------------------------
# Finger Counting Settings
# --------------------------------------------------------------------------
THUMB_EXTENSION_RATIO: float = 0.15    # Relative-distance threshold for thumb logic

# --------------------------------------------------------------------------
# Gesture Stabilization Settings
# --------------------------------------------------------------------------
STABILIZATION_BUFFER_SIZE: int = 7     # Frames considered for majority-vote smoothing
STABILIZATION_MIN_AGREEMENT: int = 4   # Minimum votes required to accept a new count

# --------------------------------------------------------------------------
# FPS Counter Settings
# --------------------------------------------------------------------------
FPS_SMOOTHING_FACTOR: float = 0.9      # Exponential moving average weight

# --------------------------------------------------------------------------
# Window / Display Settings
# --------------------------------------------------------------------------
WINDOW_NAME: str = "AI Finger Counter System | Abid Ali"
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_LARGE: float = 1.4
FONT_SCALE_MEDIUM: float = 0.75
FONT_SCALE_SMALL: float = 0.55
FONT_THICKNESS: int = 2

# --------------------------------------------------------------------------
# Color Palette (BGR format, as used by OpenCV)
# --------------------------------------------------------------------------
COLOR_PRIMARY = (255, 130, 0)          # Vivid azure-blue accent
COLOR_SECONDARY = (0, 210, 255)        # Amber highlight
COLOR_SUCCESS = (80, 220, 100)         # Green — connected / detected
COLOR_DANGER = (60, 60, 235)           # Red — error / disconnected
COLOR_WARNING = (0, 165, 255)          # Orange — warnings
COLOR_TEXT_LIGHT = (245, 245, 245)     # Near-white text
COLOR_TEXT_DARK = (25, 25, 25)         # Near-black text
COLOR_PANEL_BG = (35, 35, 35)          # Dark translucent panel background
COLOR_LEFT_HAND = (255, 130, 0)        # Accent color for left-hand overlay
COLOR_RIGHT_HAND = (0, 210, 255)       # Accent color for right-hand overlay

# --------------------------------------------------------------------------
# UI Layout Settings
# --------------------------------------------------------------------------
PANEL_OPACITY: float = 0.55            # Alpha for translucent overlay panels
TOP_BAR_HEIGHT: int = 90
HAND_PANEL_WIDTH: int = 230
HAND_PANEL_HEIGHT: int = 150
CORNER_RADIUS: int = 12

# --------------------------------------------------------------------------
# Keyboard Shortcuts
# --------------------------------------------------------------------------
EXIT_KEYS = {ord("q"), ord("Q"), 27}   # 'q', 'Q', and ESC key code
SNAPSHOT_KEY = ord("s")                # Save a snapshot of the current frame
