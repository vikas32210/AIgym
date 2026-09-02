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
        self._frame_timestamps_ms = 0

        # -----------------------------
        # MediaPipe model
        # -----------------------------
        model_path = os.path.join(
            os.getcwd(),
            "ml_models",
            "pose_landmarker_full.task"
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"MediaPipe model not found: {model_path}"
            )

        base_options = python.BaseOptions(
            model_asset_path=model_path
        )

        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.7,
            min_pose_presence_confidence=0.7,
            min_tracking_confidence=0.7,
            output_segmentation_masks=False
        )

        self._landmarker = (
            vision.PoseLandmarker.create_from_options(options)
        )

        # -----------------------------
        # Exercise detectors
        # -----------------------------
        self._detectors = {
            "Squats": SquatDetector(),
            "Push-ups": PushUpDetector(),
            "Biceps Curls (Dumbbell)": BicepsCurlDetector(),
            "Shoulder Press": ShoulderPressDetector(),
            "Lunges": LungesDetector(),
        }

    # =========================================================
    # Metrics
    # =========================================================

    def set_latest_metrics(self, metrics):
        with self._lock:
            self._latest_metrics = metrics.copy()

    def get_latest_metrics(self):
        with self._lock:
            if self._latest_metrics is None:
                return None

            return self._latest_metrics.copy()

    # =========================================================
    # Exercise
    # =========================================================

    def set_exercise(self, exercise_type):
        with self._lock:
            self._exercise_type = exercise_type

    def get_exercise(self):
        with self._lock:
            return self._exercise_type

    # =========================================================
    # Draw skeleton
    # =========================================================

    def _draw_skeleton(self, img, landmarks):
        height, width = img.shape[:2]

        for start_idx, end_idx in POSE_CONNECTIONS:

            p1 = landmarks[start_idx]
            p2 = landmarks[end_idx]

            if (
                getattr(p1, "visibility", 1.0) > 0.7
                and getattr(p2, "visibility", 1.0) > 0.7
            ):
                cv2.line(
                    img,
                    (
                        int(p1.x * width),
                        int(p1.y * height)
                    ),
                    (
                        int(p2.x * width),
                        int(p2.y * height)
                    ),
                    (0, 255, 0),
                    4
                )

        for landmark in landmarks:

            if getattr(landmark, "visibility", 1.0) > 0.7:

                cv2.circle(
                    img,
                    (
                        int(landmark.x * width),
                        int(landmark.y * height)
                    ),
                    6,
                    (255, 0, 0),
                    -1
                )

    # =========================================================
    # No pose warning
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
            cv2.LINE_AA
        )

        cv2.putText(
            img,
            "PLEASE FACE THE CAMERA",
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    # =========================================================
    # Overlay dispatcher
    # =========================================================

    def _draw_overlays(self, img, metrics, exercise_type):

        if exercise_type == "Squats":
            self._draw_squats_overlays(img, metrics)

        elif exercise_type == "Push-ups":
            self._draw_pushup_overlays(img, metrics)

        elif exercise_type == "Biceps Curls (Dumbbell)":
            self._draw_curl_overlays(img, metrics)

        elif exercise_type == "Shoulder Press":
            self._draw_press_overlays(img, metrics)

        elif exercise_type == "Lunges":
            self._draw_lunge_overlays(img, metrics)

    # =========================================================
    # Squat overlay
    # =========================================================

    def _draw_squats_overlays(self, img, metrics):

        height = img.shape[0]

        depth_status = metrics.get(
            "depth_status",
            "N/A"
        )

        cv2.putText(
            img,
            f"DEPTH: {depth_status}",
            (20, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    # =========================================================
    # Push-up overlay
    # =========================================================

    def _draw_pushup_overlays(self, img, metrics):

        height = img.shape[0]

        body_alignment = metrics.get(
            "body_alignment",
            "N/A"
        )

        hip_status = metrics.get(
            "hip_status",
            "N/A"
        )

        cv2.putText(
            img,
            f"BODY: {body_alignment} | HIP: {hip_status}",
            (20, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    # =========================================================
    # Biceps curl overlay
    # =========================================================

    def _draw_curl_overlays(self, img, metrics):

        height = img.shape[0]

        swing_status = metrics.get(
            "swing_status",
            "N/A"
        )

        cv2.putText(
            img,
            f"SWING: {swing_status}",
            (20, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    # =========================================================
    # Shoulder press overlay
    # =========================================================

    def _draw_press_overlays(self, img, metrics):

        height = img.shape[0]

        extension_status = metrics.get(
            "extension_status",
            "N/A"
        )

        back_arch_status = metrics.get(
            "back_arch_status",
            "N/A"
        )

        cv2.putText(
            img,
            f"EXT: {extension_status} | BACK: {back_arch_status}",
            (20, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    # =========================================================
    # Lunge overlay
    # =========================================================

    def _draw_lunge_overlays(self, img, metrics):

        height = img.shape[0]

        balance_status = metrics.get(
            "balance_status",
            "N/A"
        )

        cv2.putText(
            img,
            f"BALANCE: {balance_status}",
            (20, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    # =========================================================
    # Main video processing
    # =========================================================

    def recv(self, frame):

        # Convert WebRTC frame → OpenCV image
        image = frame.to_ndarray(format="bgr24")

        # Mirror camera
        image = cv2.flip(image, 1)

        image = np.asarray(
            image,
            dtype=np.uint8
        )

        # -----------------------------------------------------
        # OpenCV BGR → RGB for MediaPipe
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
        # Timestamp
        # -----------------------------------------------------

        self._frame_timestamps_ms += 30

        # -----------------------------------------------------
        # MediaPipe pose detection
        # -----------------------------------------------------

        result = self._landmarker.detect_for_video(
            mp_image,
            self._frame_timestamps_ms
        )

        # -----------------------------------------------------
        # Pose detected
        # -----------------------------------------------------

        if result.pose_landmarks:

            landmarks = result.pose_landmarks[0]

            self._draw_skeleton(
                image,
                landmarks
            )

            exercise_type = self.get_exercise()

            detector = self._detectors.get(
                exercise_type
            )

            if detector:

                try:

                    metrics = detector.process(
                        landmarks
                    )

                    if metrics is None:
                        metrics = {}

                    metrics["pose_detected"] = True

                    self._draw_overlays(
                        image,
                        metrics,
                        exercise_type
                    )

                    self.set_latest_metrics(
                        metrics
                    )

                except Exception as detector_error:

                    self.set_latest_metrics({
                        "pose_detected": True,
                        "detector_error": str(
                            detector_error
                        )
                    })

        # -----------------------------------------------------
        # No pose detected
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
        # Return frame
        # -----------------------------------------------------

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )
