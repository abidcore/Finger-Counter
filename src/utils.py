"""
utils.py
--------
Reusable OpenCV drawing and UI helper functions shared across the
application. Isolating these here keeps main.py focused on the
high-level control flow rather than low-level pixel manipulation.
"""

from typing import Tuple

import cv2
import numpy as np

from config import settings


def draw_translucent_rect(
    frame: np.ndarray,
    top_left: Tuple[int, int],
    bottom_right: Tuple[int, int],
    color: Tuple[int, int, int],
    alpha: float = settings.PANEL_OPACITY,
) -> None:
    """Draw a semi-transparent filled rectangle directly onto `frame`."""
    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, thickness=cv2.FILLED)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)


def draw_rounded_panel(
    frame: np.ndarray,
    top_left: Tuple[int, int],
    bottom_right: Tuple[int, int],
    color: Tuple[int, int, int],
    alpha: float = settings.PANEL_OPACITY,
    radius: int = settings.CORNER_RADIUS,
    border_color: Tuple[int, int, int] = None,
    border_thickness: int = 2,
) -> None:
    """
    Draw a translucent panel with rounded corners -- a small visual
    touch that gives the overlay a noticeably more polished, modern
    look compared to plain rectangular boxes.
    """
    x1, y1 = top_left
    x2, y2 = bottom_right
    overlay = frame.copy()

    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, cv2.FILLED)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, cv2.FILLED)
    cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, cv2.FILLED)
    cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, cv2.FILLED)
    cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, cv2.FILLED)
    cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, cv2.FILLED)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)

    if border_color is not None:
        cv2.rectangle(frame, (x1 + radius, y1), (x2 - radius, y2), border_color, border_thickness)
        cv2.rectangle(frame, (x1, y1 + radius), (x2, y2 - radius), border_color, border_thickness)


def draw_text(
    frame: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    scale: float = settings.FONT_SCALE_MEDIUM,
    color: Tuple[int, int, int] = settings.COLOR_TEXT_LIGHT,
    thickness: int = settings.FONT_THICKNESS,
    font=settings.FONT,
    shadow: bool = True,
) -> None:
    """Draw text with an optional soft drop-shadow for readability over video."""
    if shadow:
        shadow_origin = (origin[0] + 2, origin[1] + 2)
        cv2.putText(frame, text, shadow_origin, font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, origin, font, scale, color, thickness, cv2.LINE_AA)


def get_text_size(
    text: str, scale: float = settings.FONT_SCALE_MEDIUM, thickness: int = settings.FONT_THICKNESS, font=settings.FONT
) -> Tuple[int, int]:
    """Return (width, height) in pixels that `text` will occupy when rendered."""
    (width, height), _ = cv2.getTextSize(text, font, scale, thickness)
    return width, height


def draw_confidence_bar(
    frame: np.ndarray,
    top_left: Tuple[int, int],
    width: int,
    height: int,
    confidence: float,
    color: Tuple[int, int, int] = settings.COLOR_SUCCESS,
) -> None:
    """Draw a horizontal confidence/progress bar with a filled proportion."""
    x, y = top_left
    cv2.rectangle(frame, (x, y), (x + width, y + height), (90, 90, 90), 1)
    filled_width = int(width * max(0.0, min(confidence, 1.0)))
    if filled_width > 0:
        cv2.rectangle(frame, (x, y), (x + filled_width, y + height), color, cv2.FILLED)


def draw_bounding_box(
    frame: np.ndarray,
    box: Tuple[int, int, int, int],
    label: str,
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> None:
    """Draw a labeled bounding box around a detected hand region."""
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    label_w, label_h = get_text_size(label, scale=settings.FONT_SCALE_SMALL, thickness=1)
    label_bg_top_left = (x1, y1 - label_h - 12)
    label_bg_bottom_right = (x1 + label_w + 14, y1)

    draw_translucent_rect(frame, label_bg_top_left, label_bg_bottom_right, color, alpha=0.75)
    draw_text(
        frame,
        label,
        (x1 + 7, y1 - 8),
        scale=settings.FONT_SCALE_SMALL,
        color=settings.COLOR_TEXT_LIGHT,
        thickness=1,
        shadow=False,
    )


def clamp(value: float, minimum: float, maximum: float) -> float:
    """Constrain `value` to the inclusive range [minimum, maximum]."""
    return max(minimum, min(value, maximum))
