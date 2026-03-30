
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from enum import Enum



# Enums


class DetectionMode(str, Enum):
    FACE = "face"
    OBJECT = "object"
    BOTH = "both"


class InputSource(str, Enum):
    WEBCAM = "webcam"
    IMAGE = "image"
    VIDEO = "video"



# Core Detection Models


class BoundingBox(BaseModel):
    """Bounding box coordinates (pixels)."""
    x: int = Field(..., description="Top-left X coordinate")
    y: int = Field(..., description="Top-left Y coordinate")
    width: int = Field(..., description="Box width in pixels")
    height: int = Field(..., description="Box height in pixels")

class Keypoint(BaseModel):
    """Skeleton keypoint coordinate."""
    x: float
    y: float
    confidence: float

class Detection(BaseModel):
    """A single detected object or face."""
    label: str = Field(..., description="Class label, e.g. 'person', 'car', 'face'")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0.0–1.0")
    bounding_box: BoundingBox
    detection_type: str = Field(..., description="'face' or 'object' or 'pose'")
    keypoints: Optional[List[Keypoint]] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class FrameResult(BaseModel):
    """All detections from a single video frame."""
    frame_id: int = Field(..., description="Frame index in the stream")
    fps: float = Field(..., description="Current frames per second")
    detections: List[Detection] = Field(default_factory=list)
    total_detections: int = Field(0)
    source: InputSource = InputSource.WEBCAM
    mode: DetectionMode = DetectionMode.BOTH
    processed_at: datetime = Field(default_factory=datetime.now)

    def model_post_init(self, __context):
        self.total_detections = len(self.detections)



# API Request / Response Models


class DetectionRequest(BaseModel):
    """Request body for image-based detection via API."""
    mode: DetectionMode = DetectionMode.BOTH
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0)
    source: InputSource = InputSource.IMAGE


class DetectionResponse(BaseModel):
    """Standard API response wrapping FrameResult."""
    success: bool = True
    message: str = "Detection completed"
    data: Optional[FrameResult] = None
    error: Optional[str] = None


class SessionStats(BaseModel):
    """Stats for the current detection session."""
    total_frames: int = 0
    total_detections: int = 0
    average_fps: float = 0.0
    session_start: datetime = Field(default_factory=datetime.now)
    detection_breakdown: dict = Field(default_factory=dict)


class LogEntry(BaseModel):
    """One row in the CSV detection log."""
    timestamp: str
    frame_id: int
    label: str
    confidence: float
    x: int
    y: int
    width: int
    height: int
    detection_type: str
    fps: float
