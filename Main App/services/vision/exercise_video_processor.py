import os
import cv2
import av
import numpy as np
import mediapipe as mp
import threading

from streamlit_webrtc import VideoProcessorBase
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from detectors.squat import SquatDetector
from detectors.pushup import PushUpDetector
from detectors.biceps_curl import BicepsCurlDetector
from detectors.shoulder_press import ShoulderPressDetector
from detectors.lunges import LungesDetector

from services.config.workout_config import POSE_CONNECTIONS


class VideoProcessorClass(VideoProcessorBase):

    def __init__(self):
        self._lock = threading.Lock()

        self._latest_metrics = None
        self._exercise_type = "Squats"

        # ---------------------------------------------------------
        # FIND PROJECT ROOT
        # ---------------------------------------------------------
        # Current file:
        # Main App/services/vision/exercise_video_processor.py
        #
        # We need:
        # Main App/ml_models/pose_landmarker_full.task
        # ---------------------------------------------------------

        current_file = os.path.abspath(__file__)

        project_root = os.path.abspath(
            os.path.join(
                os.path.dirname(current_file),
                "..",
                "..",
            )
        )

        model_path = os.path.join(
            project_root,
            "ml_models",
            "pose_landmarker_full.task",
        )

        # ---------------------------------------------------------
        # CHECK MODEL
        # ---------------------------------------------------------

        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"MediaPipe model not found.\n\n"
                f"Expected path:\n{model_path}\n\n"
                f"Make sure pose_landmarker_full.task exists inside:\n"
                f"{os.path.join(project_root, 'ml_models')}"
            )

        # ---------------------------------------------------------
        # MEDIAPIPE POSE LANDMARKER
        # ---------------------------------------------------------

        base_options = python.BaseOptions(
            model_asset_path=model_path
        )

        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.7,
            min_pose_presence_confidence=0.7,
            min_tracking_confidence=0.7,
            output_segmentation_masks=False,
        )

        self._landmarker = vision.PoseLandmarker.create_from_options(
            options
        )

        # ---------------------------------------------------------
        # EXERCISE DETECTORS
        # ---------------------------------------------------------

        self._detectors = {
            "Squats": SquatDetector(),
            "Push-ups": PushUpDetector(),
            "Biceps Curls (Dumbbell)": BicepsCurlDetector(),
            "Shoulder Press": ShoulderPressDetector(),
            "Lunges": LungesDetector(),
        }

        # MediaPipe VIDEO mode requires increasing timestamps
        self._frame_timestamps_ms = 0

    # =========================================================
    # METRICS
    # =========================================================

    def set_latest_metrics(self, metrics):
        with self._lock:
            if metrics is None:
                self._latest_metrics = None
            else:
                self._latest_metrics = metrics.copy()

    def get_latest_metrics(self):
        with self._lock:
            if self._latest_metrics is None:
                return None

            return self._latest_metrics.copy()

    # =========================================================
    # EXERCISE
    # =========================================================

    def set_exercise(self, exercise_type):
        with self._lock:
            self._exercise_type = exercise_type

    def get_exercise(self):
        with self._lock:
            return self._exercise_type

    # =========================================================
    # DRAW SKELETON
    # =========================================================

    def _draw_skeleton(self, img, landmarks):

        h, w = img.shape[:2]

        # Draw connections
        for start_idx, end_idx in POSE_CONNECTIONS:

            if start_idx >= len(landmarks):
                continue

            if end_idx >= len(landmarks):
                continue

            p1 = landmarks[start_idx]
            p2 = landmarks[end_idx]

            if (
                p1.visibility > 0.7
                and p2.visibility > 0.7
            ):
                cv2.line(
                    img,
                    (
                        int(p1.x * w),
                        int(p1.y * h),
                    ),
                    (
                        int(p2.x * w),
                        int(p2.y * h),
                    ),
                    (0, 255, 0),
                    4,
                )

        # Draw landmarks
        for lm in landmarks:

            if lm.visibility > 0.7:

                cv2.circle(
                    img,
                    (
                        int(lm.x * w),
                        int(lm.y * h),
                    ),
                    6,
                    (255, 0, 0),
                    -1,
                )

    # =========================================================
    # NO POSE WARNING
    # =========================================================

    def _draw_no_pose_warnings(self, img):

        cv2.putText(
            img,
            "NO POSE DETECTED",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            img,
            "PLEASE FACE THE CAMERA",
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # =========================================================
    # EXERCISE OVERLAYS
    # =========================================================

    def _draw_overlays(self, img, metrics, ex_type):

        if ex_type == "Squats":
            self._draw_squats_overlays(img, metrics)

        elif ex_type == "Push-ups":
            self._draw_pushup_overlays(img, metrics)

        elif ex_type == "Biceps Curls (Dumbbell)":
            self._draw_curl_overlays(img, metrics)

        elif ex_type == "Shoulder Press":
            self._draw_press_overlays(img, metrics)

        elif ex_type == "Lunges":
            self._draw_lunge_overlays(img, metrics)

    # =========================================================
    # SQUAT OVERLAY
    # =========================================================

    def _draw_squats_overlays(self, img, metrics):

        h, _ = img.shape[:2]

        depth_status = metrics.get(
            "depth_status",
            "Unknown"
        )

        cv2.putText(
            img,
            f"DEPTH: {depth_status}",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # =========================================================
    # PUSH-UP OVERLAY
    # =========================================================

    def _draw_pushup_overlays(self, img, metrics):

        h, _ = img.shape[:2]

        body_alignment = metrics.get(
            "body_alignment",
            "Unknown"
        )

        hip_status = metrics.get(
            "hip_status",
            "Unknown"
        )

        cv2.putText(
            img,
            f"BODY: {body_alignment} | HIP: {hip_status}",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # =========================================================
    # CURL OVERLAY
    # =========================================================

    def _draw_curl_overlays(self, img, metrics):

        h, _ = img.shape[:2]

        swing_status = metrics.get(
            "swing_status",
            "Unknown"
        )

        cv2.putText(
            img,
            f"SWING: {swing_status}",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # =========================================================
    # SHOULDER PRESS OVERLAY
    # =========================================================

    def _draw_press_overlays(self, img, metrics):

        h, _ = img.shape[:2]

        extension_status = metrics.get(
            "extension_status",
            "Unknown"
        )

        back_arch_status = metrics.get(
            "back_arch_status",
            "Unknown"
        )

        cv2.putText(
            img,
            f"EXT: {extension_status} | BACK: {back_arch_status}",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # =========================================================
    # LUNGE OVERLAY
    # =========================================================

    def _draw_lunge_overlays(self, img, metrics):

        h, _ = img.shape[:2]

        balance_status = metrics.get(
            "balance_status",
            "Unknown"
        )

        cv2.putText(
            img,
            f"BALANCE: {balance_status}",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # =========================================================
    # RECEIVE VIDEO FRAME
    # =========================================================

    def recv(self, frame):

        # -----------------------------------------------------
        # Convert WebRTC frame -> OpenCV BGR image
        # -----------------------------------------------------

        image = frame.to_ndarray(format="bgr24")

        # Mirror camera
        image = cv2.flip(image, 1)

        # -----------------------------------------------------
        # MediaPipe expects RGB
        # -----------------------------------------------------

        rgb_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_image
        )

        # -----------------------------------------------------
        # VIDEO TIMESTAMP
        # -----------------------------------------------------

        self._frame_timestamps_ms += 33

        # -----------------------------------------------------
        # RUN POSE DETECTION
        # -----------------------------------------------------

        result = self._landmarker.detect_for_video(
            mp_image,
            self._frame_timestamps_ms
        )

        # -----------------------------------------------------
        # POSE FOUND
        # -----------------------------------------------------

        if result.pose_landmarks:

            landmarks = result.pose_landmarks[0]

            # Draw skeleton
            self._draw_skeleton(
                image,
                landmarks
            )

            # Current exercise
            ex_type = self.get_exercise()

            # Get detector
            detector = self._detectors.get(
                ex_type
            )

            if detector:

                try:

                    # Process exercise
                    metrics = detector.process(
                        landmarks
                    )

                    if metrics is None:
                        metrics = {}

                    metrics["pose_detected"] = True

                    # Draw exercise information
                    self._draw_overlays(
                        image,
                        metrics,
                        ex_type
                    )

                    # Save metrics
                    self.set_latest_metrics(
                        metrics
                    )

                except Exception as e:

                    # Don't kill WebRTC if detector has an error
                    error_metrics = {
                        "pose_detected": True,
                        "detector_error": str(e),
                    }

                    self.set_latest_metrics(
                        error_metrics
                    )

        # -----------------------------------------------------
        # NO POSE
        # -----------------------------------------------------

        else:

            self._draw_no_pose_warnings(
                image
            )

            with self._lock:

                if self._latest_metrics is not None:

                    self._latest_metrics[
                        "pose_detected"
                    ] = False

                else:

                    self._latest_metrics = {
                        "pose_detected": False
                    }

        # -----------------------------------------------------
        # RETURN FRAME
        # -----------------------------------------------------

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )
