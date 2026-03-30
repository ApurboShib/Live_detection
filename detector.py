"""
detector.py — Core detection engine.
Wraps OpenCV Haar Cascades (face) and YOLOv8 (objects) into one unified pipeline.
"""

import cv2
import time
import numpy as np
from typing import Tuple, List, Optional
from models import Detection, BoundingBox, FrameResult, DetectionMode, InputSource


# ──────────────────────────────────────────────
# Color palette for YOLO class bounding boxes
# ──────────────────────────────────────────────
COLORS = [
    (0, 255, 100),   # green
    (0, 180, 255),   # orange-blue
    (255, 100, 0),   # blue
    (255, 0, 150),   # pink
    (100, 255, 255), # yellow
    (200, 0, 255),   # purple
]


def get_color(class_id: int) -> Tuple[int, int, int]:
    return COLORS[class_id % len(COLORS)]


# ──────────────────────────────────────────────
# DetectionEngine
# ──────────────────────────────────────────────

class DetectionEngine:
    """
    Unified face + object detection engine.
    - Face detection: OpenCV Haar Cascade
    - Object detection: YOLOv8 nano (via Ultralytics)
    """

    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self._haar = None
        self._yolo = None
        self._yolo_loaded = False
        self._frame_count = 0
        self._fps = 0.0
        self._prev_time = time.time()

        # Load Haar Cascade immediately (lightweight)
        self._load_haar()

    # ── Loaders ───────────────────────────────

    def _load_haar(self):
        """Load OpenCV's built-in frontal face Haar Cascade."""
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._haar = cv2.CascadeClassifier(cascade_path)
        if self._haar.empty():
            raise RuntimeError("Failed to load Haar Cascade XML. Check OpenCV installation.")

    def load_yolo(self, model_name: str = "yolov8n.pt") -> str:
        """
        Load YOLOv8 model. Called lazily so Streamlit doesn't block at startup.
        Returns a status message.
        """
        try:
            from ultralytics import YOLO
            self._yolo = YOLO(model_name)  # downloads automatically on first run
            self._yolo_loaded = True
            return f"✅ YOLOv8 model '{model_name}' loaded successfully."
        except ImportError:
            return "❌ Ultralytics not installed. Run: pip install ultralytics"
        except Exception as e:
            return f"❌ Failed to load YOLO: {str(e)}"

    @property
    def yolo_ready(self) -> bool:
        return self._yolo_loaded and self._yolo is not None

    # ── FPS Tracking ──────────────────────────

    def _update_fps(self) -> float:
        now = time.time()
        elapsed = now - self._prev_time
        self._prev_time = now
        if elapsed > 0:
            self._fps = 1.0 / elapsed
        return round(self._fps, 1)

    # ── Face Detection ────────────────────────

    def detect_faces(self, frame: np.ndarray) -> List[Detection]:
        """Run Haar Cascade face detection on a BGR frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # histogram equalization for better accuracy

        faces = self._haar.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        detections = []
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                detections.append(
                    Detection(
                        label="face",
                        confidence=1.0,  # Haar doesn't give confidence, use 1.0
                        bounding_box=BoundingBox(x=int(x), y=int(y), width=int(w), height=int(h)),
                        detection_type="face",
                    )
                )
        return detections

    # ── Object Detection ──────────────────────

    def detect_objects(self, frame: np.ndarray) -> List[Detection]:
        """Run YOLOv8 object detection on a BGR frame."""
        if not self.yolo_ready:
            return []

        results = self._yolo(frame, conf=self.confidence_threshold, verbose=False)
        detections = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = self._yolo.names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w = x2 - x1
                h = y2 - y1

                detections.append(
                    Detection(
                        label=label,
                        confidence=round(conf, 3),
                        bounding_box=BoundingBox(x=x1, y=y1, width=w, height=h),
                        detection_type="object",
                    )
                )
        return detections

    # ── Main Process Frame ─────────────────────

    def process_frame(
        self,
        frame: np.ndarray,
        mode: DetectionMode = DetectionMode.BOTH,
        source: InputSource = InputSource.WEBCAM,
        annotate: bool = True,
    ) -> Tuple[np.ndarray, FrameResult]:
        """
        Run detection on a single frame.

        Parameters
        ----------
        annotate : bool
            If True  → return frame with OpenCV bounding-box annotations drawn.
            If False → return the raw frame unmodified (used by OpenGL renderer,
                       which draws boxes itself via GL_LINE_LOOP primitives).
        """
        self._frame_count += 1
        fps = self._update_fps()
        all_detections: List[Detection] = []

        # ── Run detectors based on mode ──
        if mode in (DetectionMode.FACE, DetectionMode.BOTH):
            all_detections.extend(self.detect_faces(frame))

        if mode in (DetectionMode.OBJECT, DetectionMode.BOTH) and self.yolo_ready:
            all_detections.extend(self.detect_objects(frame))

        # ── Optionally draw OpenCV annotations (skipped when OpenGL renders) ──
        if annotate:
            output_frame = self._draw_annotations(frame.copy(), all_detections, fps)
        else:
            output_frame = frame  # OpenGL renderer takes the raw frame

        result = FrameResult(
            frame_id=self._frame_count,
            fps=fps,
            detections=all_detections,
            source=source,
            mode=mode,
        )
        return output_frame, result

    # ── Drawing ───────────────────────────────

    def _draw_annotations(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        fps: float,
    ) -> np.ndarray:
        """Draw bounding boxes, labels, and FPS on the frame."""

        for i, det in enumerate(detections):
            bb = det.bounding_box
            color = (0, 255, 80) if det.detection_type == "face" else get_color(i)

            # Bounding box
            cv2.rectangle(frame, (bb.x, bb.y), (bb.x + bb.width, bb.y + bb.height), color, 2)

            # Label background
            label_text = f"{det.label} {det.confidence:.0%}" if det.detection_type == "object" else "Face"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (bb.x, bb.y - th - 8), (bb.x + tw + 6, bb.y), color, -1)

            # Label text
            cv2.putText(
                frame, label_text,
                (bb.x + 3, bb.y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA,
            )

        # FPS counter (top-right)
        fps_text = f"FPS: {fps:.1f}"
        (fw, fh), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (w - fw - 16, 8), (w - 4, fh + 16), (0, 0, 0), -1)
        cv2.putText(
            frame, fps_text,
            (w - fw - 10, fh + 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 150), 2, cv2.LINE_AA,
        )

        # Detection count (top-left)
        count_text = f"Detected: {len(detections)}"
        cv2.putText(
            frame, count_text,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
        )

        return frame

    # ── Static Image Processing ───────────────

    def process_image_file(
        self,
        image_path: str,
        mode: DetectionMode = DetectionMode.BOTH,
    ) -> Tuple[Optional[np.ndarray], Optional[FrameResult]]:
        """Process a single image file."""
        frame = cv2.imread(image_path)
        if frame is None:
            return None, None
        return self.process_frame(frame, mode=mode, source=InputSource.IMAGE)
