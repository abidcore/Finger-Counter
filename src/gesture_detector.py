"""
gesture_detector.py
--------------------
Provides two complementary capabilities on top of raw per-frame finger
readings:

    1. GestureStabilizer -- temporal smoothing via a sliding-window
       majority vote, which eliminates the frame-to-frame flicker that
       naturally occurs with any landmark-based detector (a finger that
       is genuinely borderline between "up" and "down" would otherwise
       cause the displayed count to jitter distractingly).

    2. GestureDetector -- maps a stable 5-element finger-state pattern
       onto a human-readable named gesture (e.g. "Fist", "Peace",
       "Thumbs Up", "Open Palm"), adding an extra layer of interpretive
       intelligence beyond a raw numeric count.
"""

from collections import Counter, deque
from typing import Deque, List, Tuple

from config import settings

# Maps an exact (thumb, index, middle, ring, pinky) boolean pattern to a
# friendly gesture name. Patterns not present here fall back to a
# generic "<N> Finger(s)" label generated dynamically.
_GESTURE_PATTERNS = {
    (False, False, False, False, False): "Fist",
    (True, False, False, False, False): "Thumbs Up",
    (False, True, False, False, False): "Pointing",
    (False, True, True, False, False): "Peace / Victory",
    (True, True, False, False, False): "Gun Sign",
    (False, False, False, False, True): "Pinky Promise",
    (True, False, False, False, True): "Call Me",
    (True, True, True, True, True): "Open Palm",
    (False, True, True, True, False): "Three",
    (False, True, True, True, True): "Four",
}


class GestureStabilizer:
    """
    Smooths a noisy per-frame integer signal (finger count) using a
    fixed-size sliding window and majority-vote resolution.

    This is a general-purpose temporal debouncer; each tracked hand
    should own its own independent instance so that the left and right
    hands never influence each other's smoothing window.
    """

    def __init__(
        self,
        buffer_size: int = settings.STABILIZATION_BUFFER_SIZE,
        min_agreement: int = settings.STABILIZATION_MIN_AGREEMENT,
    ) -> None:
        self._buffer_size = buffer_size
        self._min_agreement = min_agreement
        self._history: Deque[int] = deque(maxlen=buffer_size)
        self._stable_value: int = 0

    def update(self, raw_value: int) -> int:
        """
        Feed in the latest raw reading and receive the current stable
        (debounced) value back.

        Args:
            raw_value: The finger count observed in the current frame.

        Returns:
            The stabilized finger count.
        """
        self._history.append(raw_value)

        if len(self._history) < self._buffer_size:
            # Not enough history yet -- trust the raw value directly so
            # the UI feels responsive during the first few frames.
            self._stable_value = raw_value
            return self._stable_value

        counts = Counter(self._history)
        most_common_value, frequency = counts.most_common(1)[0]

        if frequency >= self._min_agreement:
            self._stable_value = most_common_value

        return self._stable_value

    def reset(self) -> None:
        """Clear all accumulated history (e.g. when a hand disappears)."""
        self._history.clear()
        self._stable_value = 0


class GestureDetector:
    """
    Classifies a 5-element finger-state pattern into a descriptive,
    human-friendly gesture name.
    """

    @staticmethod
    def classify(finger_states: List[bool]) -> str:
        """
        Args:
            finger_states: [thumb, index, middle, ring, pinky] booleans.

        Returns:
            A human-readable gesture label.
        """
        pattern: Tuple[bool, ...] = tuple(finger_states)

        if pattern in _GESTURE_PATTERNS:
            return _GESTURE_PATTERNS[pattern]

        finger_count = sum(finger_states)
        if finger_count == 0:
            return "Fist"
        if finger_count == 5:
            return "Open Palm"
        return f"{finger_count} Finger{'s' if finger_count != 1 else ''}"
