"""
logger.py — CSV logging utility for detection results.
Logs each detection event with timestamp, label, confidence, bbox, and FPS.
"""

import csv
import os
from datetime import datetime
from typing import List
from models import FrameResult, LogEntry


LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)


class DetectionLogger:
    """Handles writing detection results to a CSV file."""

    CSV_HEADERS = [
        "timestamp", "frame_id", "label", "confidence",
        "x", "y", "width", "height", "detection_type", "fps"
    ]

    def __init__(self, session_name: str = None):
        if session_name is None:
            session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(LOG_DIR, f"detections_{session_name}.csv")
        self._init_csv()

    def _init_csv(self):
        """Create the CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.CSV_HEADERS)
                writer.writeheader()

    def log_frame(self, frame_result: FrameResult):
        """Append all detections from a frame to the CSV log."""
        if not frame_result.detections:
            return
        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_HEADERS)
            for det in frame_result.detections:
                entry = LogEntry(
                    timestamp=det.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f"),
                    frame_id=frame_result.frame_id,
                    label=det.label,
                    confidence=round(det.confidence, 4),
                    x=det.bounding_box.x,
                    y=det.bounding_box.y,
                    width=det.bounding_box.width,
                    height=det.bounding_box.height,
                    detection_type=det.detection_type,
                    fps=round(frame_result.fps, 2),
                )
                writer.writerow(entry.model_dump())

    def get_log_path(self) -> str:
        return self.log_path

    def get_all_entries(self) -> List[dict]:
        """Read all log entries for display."""
        if not os.path.exists(self.log_path):
            return []
        with open(self.log_path, "r") as f:
            reader = csv.DictReader(f)
            return list(reader)
