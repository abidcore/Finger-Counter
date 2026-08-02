#!/usr/bin/env python3
"""
AI Finger Counter System
=========================
Main application entry point.

A real-time computer vision application that detects one or two hands
via a webcam feed, tracks 21 hand landmarks per hand using MediaPipe,
counts extended fingers (0-5) per hand, distinguishes left from right
hands, stabilizes the detected gesture against frame-to-frame noise,
and renders a professional on-screen UI including FPS, detection
confidence, and webcam status.

Author:  Abid Ali
Project: AI & Machine Learning Diploma Portfolio
License: MIT (see LICENSE file)

Usage:
    python main.py

Keyboard Shortcuts:
    Q / ESC  -- Exit the application
    S        -- Save a timestamped snapshot of the current frame
"""

import datetime
import os
import sys
import time
from typing import Dict

import cv2

from config import settings
from src import (
    FingerCounter,
    FPSCounter,
    GestureDetector,
    GestureStabilizer,
    HandTracker,
)
from src import utils


class CameraNotAvailableError(Exception):
    """Raised when the webcam cannot be opened after all retry attempts."""


class AIFingerCounterApp:
    """
    Top-level application controller.

    Owns the video capture device and coordinates the hand-tracking,
    finger-counting, gesture-stabilization, and UI-rendering pipeline
    on every frame of the main loop.
    """

    def __init__(self) -> None:
        self._capture = self._open_camera()
        self._hand_tracker = HandTracker()
        self._finger_counter = FingerCounter()
        self._fps_counter = FPSCounter()

        # Each hand label ("Left" / "Right") owns an independent
        # stabilizer instance so their smoothing windows never mix.
        self._stabilizers: Dict[str, GestureStabilizer] = {
            "Left": GestureStabilizer(),
            "Right": GestureStabilizer(),
        }

        self._webcam_connected = True
        self._snapshot_dir = os.path.join(os.getcwd(), "snapshots")

    # ------------------------------------------------------------------
    # Camera lifecycle
    # ------------------------------------------------------------------
    @staticmethod
    def _open_camera() -> cv2.VideoCapture:
        """
        Attempt to open the configured webcam, retrying a bounded
        number of times before raising a descriptive, catchable error.
        """
        last_error = None
        for attempt in range(1, settings.CAMERA_RECONNECT_ATTEMPTS + 1):
            try:
                capture = cv2.VideoCapture(settings.CAMERA_INDEX)
                if capture.isOpened():
                    capture.set(cv2.CAP_PROP_FRAME_WIDTH, settings.FRAME_WIDTH)
                    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.FRAME_HEIGHT)

                    # Discard the first few frames while the sensor's
                    # auto-exposure/white-balance settles.
                    for _ in range(settings.CAMERA_WARMUP_FRAMES):
                        capture.read()

                    return capture

                capture.release()
            except Exception as exc:  # pragma: no cover - hardware dependent
                last_error = exc

            time.sleep(0.5)

        raise CameraNotAvailableError(
            f"Could not access webcam at index {settings.CAMERA_INDEX} "
            f"after {settings.CAMERA_RECONNECT_ATTEMPTS} attempts. "
            f"Ensure no other application is using the camera and that "
            f"OS-level camera permissions are granted."
            + (f" Last error: {last_error}" if last_error else "")
        )

    # ------------------------------------------------------------------
    # Per-frame UI rendering
    # ------------------------------------------------------------------
    def _draw_top_bar(self, frame, total_fingers: int, num_hands: int) -> None:
        height, width = frame.shape[:2]
        utils.draw_translucent_rect(
            frame, (0, 0), (width, settings.TOP_BAR_HEIGHT), settings.COLOR_PANEL_BG, alpha=0.65
        )

        utils.draw_text(
            frame,
            "AI FINGER COUNTER SYSTEM",
            (20, 35),
            scale=settings.FONT_SCALE_MEDIUM,
            color=settings.COLOR_SECONDARY,
            thickness=2,
        )
        utils.draw_text(
            frame,
            f"Hands Detected: {num_hands}   |   Total Fingers: {total_fingers}",
            (20, 68),
            scale=settings.FONT_SCALE_SMALL,
            color=settings.COLOR_TEXT_LIGHT,
            thickness=1,
        )

        # FPS readout, right-aligned.
        fps_text = f"FPS: {self._fps_counter.fps:5.1f}"
        text_w, _ = utils.get_text_size(fps_text, scale=settings.FONT_SCALE_MEDIUM, thickness=2)
        utils.draw_text(
            frame,
            fps_text,
            (width - text_w - 25, 35),
            scale=settings.FONT_SCALE_MEDIUM,
            color=settings.COLOR_SUCCESS,
            thickness=2,
        )

        # Webcam status indicator, right-aligned beneath FPS.
        status_text = "WEBCAM: LIVE" if self._webcam_connected else "WEBCAM: LOST"
        status_color = settings.COLOR_SUCCESS if self._webcam_connected else settings.COLOR_DANGER
        status_w, _ = utils.get_text_size(status_text, scale=settings.FONT_SCALE_SMALL, thickness=1)
        utils.draw_text(
            frame,
            status_text,
            (width - status_w - 25, 68),
            scale=settings.FONT_SCALE_SMALL,
            color=status_color,
            thickness=1,
        )
        cv2.circle(frame, (width - status_w - 40, 62), 5, status_color, cv2.FILLED)

    def _draw_hand_panel(self, frame, hand_data, stable_count: int, gesture_label: str, panel_index: int) -> None:
        height, width = frame.shape[:2]
        accent = (
            settings.COLOR_LEFT_HAND if hand_data.handedness_label == "Left" else settings.COLOR_RIGHT_HAND
        )

        panel_x2 = width - 20 - panel_index * (settings.HAND_PANEL_WIDTH + 15)
        panel_x1 = panel_x2 - settings.HAND_PANEL_WIDTH
        panel_y1 = settings.TOP_BAR_HEIGHT + 20
        panel_y2 = panel_y1 + settings.HAND_PANEL_HEIGHT

        utils.draw_rounded_panel(
            frame,
            (panel_x1, panel_y1),
            (panel_x2, panel_y2),
            settings.COLOR_PANEL_BG,
            alpha=0.68,
            border_color=accent,
            border_thickness=2,
        )

        utils.draw_text(
            frame,
            f"{hand_data.handedness_label} Hand",
            (panel_x1 + 15, panel_y1 + 30),
            scale=settings.FONT_SCALE_MEDIUM,
            color=accent,
            thickness=2,
        )

        utils.draw_text(
            frame,
            str(stable_count),
            (panel_x1 + 15, panel_y1 + 85),
            scale=settings.FONT_SCALE_LARGE + 0.6,
            color=settings.COLOR_TEXT_LIGHT,
            thickness=3,
        )

        utils.draw_text(
            frame,
            gesture_label,
            (panel_x1 + 90, panel_y1 + 60),
            scale=settings.FONT_SCALE_SMALL,
            color=settings.COLOR_SECONDARY,
            thickness=1,
        )

        conf_label = f"Confidence: {hand_data.handedness_score * 100:.0f}%"
        utils.draw_text(
            frame,
            conf_label,
            (panel_x1 + 15, panel_y1 + 112),
            scale=settings.FONT_SCALE_SMALL,
            color=settings.COLOR_TEXT_LIGHT,
            thickness=1,
        )
        utils.draw_confidence_bar(
            frame,
            (panel_x1 + 15, panel_y1 + 122),
            width=settings.HAND_PANEL_WIDTH - 30,
            height=10,
            confidence=hand_data.handedness_score,
            color=accent,
        )

        utils.draw_bounding_box(
            frame,
            hand_data.bounding_box,
            f"{hand_data.handedness_label}: {stable_count}",
            accent,
        )

    def _draw_footer(self, frame) -> None:
        height, width = frame.shape[:2]
        footer_text = "Press [Q] or [ESC] to Exit   |   Press [S] to Save Snapshot"
        text_w, text_h = utils.get_text_size(footer_text, scale=settings.FONT_SCALE_SMALL, thickness=1)
        utils.draw_translucent_rect(
            frame,
            (0, height - text_h - 24),
            (width, height),
            settings.COLOR_PANEL_BG,
            alpha=0.55,
        )
        utils.draw_text(
            frame,
            footer_text,
            ((width - text_w) // 2, height - 15),
            scale=settings.FONT_SCALE_SMALL,
            color=settings.COLOR_TEXT_LIGHT,
            thickness=1,
            shadow=False,
        )

    def _save_snapshot(self, frame) -> None:
        os.makedirs(self._snapshot_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self._snapshot_dir, f"snapshot_{timestamp}.png")
        cv2.imwrite(filepath, frame)
        print(f"[INFO] Snapshot saved to: {filepath}")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        print("[INFO] AI Finger Counter System starting...")
        print("[INFO] Press 'Q' or 'ESC' to exit, 'S' to save a snapshot.")

        cv2.namedWindow(settings.WINDOW_NAME, cv2.WINDOW_NORMAL)

        try:
            while True:
                success, frame = self._capture.read()

                if not success or frame is None:
                    self._webcam_connected = False
                    print("[WARNING] Failed to read frame from webcam. Retrying...")
                    time.sleep(0.5)
                    continue

                self._webcam_connected = True

                if settings.FLIP_CAMERA_HORIZONTALLY:
                    frame = cv2.flip(frame, 1)

                frame, detected_hands = self._hand_tracker.find_hands(frame, draw=True)

                total_fingers = 0
                seen_labels = set()

                for panel_index, hand_data in enumerate(detected_hands):
                    raw_count, finger_states = self._finger_counter.count_fingers(
                        hand_data.landmarks_norm
                    )
                    label = hand_data.handedness_label
                    seen_labels.add(label)

                    stabilizer = self._stabilizers.setdefault(label, GestureStabilizer())
                    stable_count = stabilizer.update(raw_count)
                    gesture_label = GestureDetector.classify(finger_states)

                    total_fingers += stable_count
                    self._draw_hand_panel(frame, hand_data, stable_count, gesture_label, panel_index)

                # Reset stabilizers for hands that left the frame so
                # stale history doesn't bleed into a future re-entry.
                for label, stabilizer in self._stabilizers.items():
                    if label not in seen_labels:
                        stabilizer.reset()

                self._fps_counter.tick()
                self._draw_top_bar(frame, total_fingers, len(detected_hands))
                self._draw_footer(frame)

                cv2.imshow(settings.WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in settings.EXIT_KEYS:
                    print("[INFO] Exit key pressed. Shutting down...")
                    break
                if key == settings.SNAPSHOT_KEY:
                    self._save_snapshot(frame)

                # Allow graceful exit if the user closes the window
                # via the OS window-manager "X" button.
                if cv2.getWindowProperty(settings.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    print("[INFO] Window closed by user. Shutting down...")
                    break

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user (Ctrl+C). Shutting down...")
        finally:
            self._cleanup()

    def _cleanup(self) -> None:
        """Release all hardware and library resources deterministically."""
        self._hand_tracker.close()
        if self._capture is not None:
            self._capture.release()
        cv2.destroyAllWindows()
        print("[INFO] Resources released. Goodbye!")


def main() -> int:
    """Application bootstrap with top-level error handling."""
    try:
        app = AIFingerCounterApp()
        app.run()
        return 0
    except CameraNotAvailableError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - top-level safety net
        print(f"[FATAL] An unexpected error occurred: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
