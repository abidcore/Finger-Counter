"""
finger_counter.py
------------------
Pure-logic module responsible for counting extended fingers from a set
of 21 hand landmarks produced by MediaPipe. This module contains no
OpenCV or UI code so it can be unit tested independently of the video
pipeline.

MediaPipe Hand Landmark Reference (21 points per hand):
    0  WRIST
    1  THUMB_CMC     2  THUMB_MCP    3  THUMB_IP     4  THUMB_TIP
    5  INDEX_MCP     6  INDEX_PIP    7  INDEX_DIP    8  INDEX_TIP
    9  MIDDLE_MCP   10  MIDDLE_PIP  11  MIDDLE_DIP  12  MIDDLE_TIP
    13 RING_MCP     14  RING_PIP    15  RING_DIP    16  RING_TIP
    17 PINKY_MCP    18  PINKY_PIP   19  PINKY_DIP   20  PINKY_TIP
"""

from typing import List, Tuple

from config import settings

# Named landmark indices for readability throughout this module.
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

Landmark = Tuple[float, float, float]

# Finger names in the fixed order used across the whole application.
FINGER_NAMES = ("Thumb", "Index", "Middle", "Ring", "Pinky")


class FingerCounter:
    """
    Determines which fingers are extended and produces a total count.

    The algorithm combines two complementary geometric heuristics:

      1. Vertical comparison (tip vs. PIP vs. MCP joint) for the four
         non-thumb fingers — reliable when the hand is held roughly
         upright, which is the natural pose for a webcam demo.

      2. Distance-based comparison for the thumb (tip-to-pinky-MCP vs.
         IP-to-pinky-MCP, normalized by palm width) — this approach is
         orientation-independent and correctly handles both the left
         and right hand without fragile handedness-based branching.
    """

    def __init__(
        self,
        thumb_extension_ratio: float = settings.THUMB_EXTENSION_RATIO,
    ) -> None:
        self._thumb_ratio_threshold = thumb_extension_ratio

    @staticmethod
    def _distance(point_a: Landmark, point_b: Landmark) -> float:
        """Euclidean distance between two landmarks in the XY plane."""
        return ((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2) ** 0.5

    def _is_thumb_extended(self, landmarks: List[Landmark]) -> bool:
        tip_to_pinky = self._distance(landmarks[THUMB_TIP], landmarks[PINKY_MCP])
        ip_to_pinky = self._distance(landmarks[THUMB_IP], landmarks[PINKY_MCP])

        # Normalize against palm width so the threshold is scale-invariant
        # (works consistently whether the hand is close to or far from
        # the camera).
        palm_width = self._distance(landmarks[INDEX_MCP], landmarks[PINKY_MCP])
        palm_width = palm_width if palm_width > 1e-6 else 1e-6

        relative_extension = (tip_to_pinky - ip_to_pinky) / palm_width
        return relative_extension > self._thumb_ratio_threshold

    @staticmethod
    def _is_finger_extended(landmarks: List[Landmark], tip_idx: int, pip_idx: int, mcp_idx: int) -> bool:
        """
        A non-thumb finger is considered extended when its tip sits
        clearly above (smaller y value) both its PIP and MCP joints,
        i.e. the finger is straightened rather than curled into the palm.
        """
        tip_y = landmarks[tip_idx][1]
        pip_y = landmarks[pip_idx][1]
        mcp_y = landmarks[mcp_idx][1]
        return tip_y < pip_y and tip_y < mcp_y

    def get_finger_states(self, landmarks: List[Landmark]) -> List[bool]:
        """
        Args:
            landmarks: Exactly 21 (x, y, z) landmark tuples for one hand.

        Returns:
            A list of 5 booleans, ordered [thumb, index, middle, ring, pinky],
            where True means the finger is extended (raised).
        """
        if len(landmarks) != 21:
            raise ValueError(f"Expected 21 hand landmarks, received {len(landmarks)}.")

        return [
            self._is_thumb_extended(landmarks),
            self._is_finger_extended(landmarks, INDEX_TIP, INDEX_PIP, INDEX_MCP),
            self._is_finger_extended(landmarks, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP),
            self._is_finger_extended(landmarks, RING_TIP, RING_PIP, RING_MCP),
            self._is_finger_extended(landmarks, PINKY_TIP, PINKY_PIP, PINKY_MCP),
        ]

    def count_fingers(self, landmarks: List[Landmark]) -> Tuple[int, List[bool]]:
        """
        Convenience method returning both the total extended-finger
        count and the underlying per-finger boolean states.

        Returns:
            Tuple of (total_count, finger_states).
        """
        finger_states = self.get_finger_states(landmarks)
        return sum(finger_states), finger_states
