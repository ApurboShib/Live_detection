"""
api.py — FastAPI backend.
Provides REST endpoints for image-based detection and session stats.
Run with: uvicorn api:app --reload --port 8000
"""

import cv2
import numpy as np
import base64
from datetime import datetime
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from models import (
    DetectionMode,
    DetectionRequest,
    DetectionResponse,
    FrameResult,
    SessionStats,
    InputSource,
)
from detector import DetectionEngine
from logger import DetectionLogger


# ──────────────────────────────────────────────
# App Setup
# ──────────────────────────────────────────────

app = FastAPI(
    title="Face & Object Detection API",
    description="Real-time face & object detection powered by OpenCV + YOLOv8",
    version="1.0.0",
    contact={"name": "Joy Shib", "email": "joysb@example.com"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared state ──────────────────────────────
engine = DetectionEngine(confidence_threshold=0.5)
logger = DetectionLogger()
session_stats = SessionStats()


# ──────────────────────────────────────────────
# Startup
# ──────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Load YOLO model when API starts."""
    msg = engine.load_yolo("yolov8n.pt")
    print(f"[API Startup] {msg}")


# ──────────────────────────────────────────────
# Health & Info
# ──────────────────────────────────────────────

@app.get("/", tags=["Info"])
def root():
    return {
        "message": "Face & Object Detection API is running 🚀",
        "docs": "/docs",
        "yolo_ready": engine.yolo_ready,
    }


@app.get("/health", tags=["Info"])
def health():
    return {
        "status": "ok",
        "yolo_loaded": engine.yolo_ready,
        "timestamp": datetime.now().isoformat(),
    }


# ──────────────────────────────────────────────
# Detection Endpoints
# ──────────────────────────────────────────────

@app.post("/detect/image", response_model=DetectionResponse, tags=["Detection"])
async def detect_image(
    file: UploadFile = File(..., description="Upload an image (jpg/png)"),
    mode: DetectionMode = Query(DetectionMode.BOTH, description="Detection mode"),
    confidence: float = Query(0.5, ge=0.0, le=1.0, description="Confidence threshold"),
):
    """
    Upload an image and receive face/object detection results.
    Returns bounding boxes, labels, and confidence scores.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (jpg/png).")

    # Decode image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=422, detail="Could not decode image.")

    engine.confidence_threshold = confidence
    annotated_frame, result = engine.process_frame(frame, mode=mode, source=InputSource.IMAGE)

    # Update session stats
    session_stats.total_frames += 1
    session_stats.total_detections += result.total_detections
    for det in result.detections:
        session_stats.detection_breakdown[det.label] = (
            session_stats.detection_breakdown.get(det.label, 0) + 1
        )

    logger.log_frame(result)

    return DetectionResponse(
        success=True,
        message=f"Detected {result.total_detections} object(s).",
        data=result,
    )


@app.post("/detect/image/annotated", tags=["Detection"])
async def detect_image_annotated(
    file: UploadFile = File(...),
    mode: DetectionMode = Query(DetectionMode.BOTH),
    confidence: float = Query(0.5, ge=0.0, le=1.0),
):
    """
    Same as /detect/image but returns the annotated image as base64.
    Useful for previewing results in a browser or Streamlit app.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=422, detail="Could not decode image.")

    engine.confidence_threshold = confidence
    annotated_frame, result = engine.process_frame(frame, mode=mode, source=InputSource.IMAGE)

    _, buffer = cv2.imencode(".jpg", annotated_frame)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "success": True,
        "detections": result.total_detections,
        "image_base64": img_base64,
        "results": result.model_dump(),
    }


# ──────────────────────────────────────────────
# Session & Logs
# ──────────────────────────────────────────────

@app.get("/session/stats", response_model=SessionStats, tags=["Session"])
def get_session_stats():
    """Get cumulative stats for the current API session."""
    return session_stats


@app.get("/session/log", tags=["Session"])
def get_detection_log(limit: int = Query(50, ge=1, le=500)):
    """Get the last N rows from the detection CSV log."""
    entries = logger.get_all_entries()
    return {
        "total_entries": len(entries),
        "entries": entries[-limit:],
        "log_file": logger.get_log_path(),
    }


@app.delete("/session/reset", tags=["Session"])
def reset_session():
    """Reset session statistics."""
    global session_stats
    session_stats = SessionStats()
    return {"message": "Session stats reset.", "new_session": session_stats}
