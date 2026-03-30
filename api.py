"""
api.py — FastAPI backend.
Provides REST endpoints for image-based detection and session stats.
Run with: uvicorn api:app --reload --port 8000
"""

import cv2
import numpy as np
import base64
from datetime import datetime

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
    title="Quantum Vision API",
    description="Real-time face & full-body tracking REST API",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = DetectionEngine(confidence_threshold=0.5)
logger = DetectionLogger()
session_stats = SessionStats()


# ──────────────────────────────────────────────
# Startup
# ──────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """YOLO model auto-loads in detector.__init__ now"""
    print(f"[API Startup] Engine initialized with YOLOv8s-pose")


# ──────────────────────────────────────────────
# Health & Info
# ──────────────────────────────────────────────

@app.get("/", tags=["Info"])
def root():
    return {
        "message": "Quantum Vision Face & Body Tracking API is running 🚀",
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
# Classic Detection Endpoints
# ──────────────────────────────────────────────

@app.post("/detect/image", response_model=DetectionResponse, tags=["Detection"])
async def detect_image(
    file: UploadFile = File(..., description="Upload an image (jpg/png)"),
    mode: DetectionMode = Query(DetectionMode.BOTH, description="Detection mode"),
    confidence: float = Query(0.5, ge=0.0, le=1.0, description="Confidence threshold"),
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (jpg/png).")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        raise HTTPException(status_code=422, detail="Could not decode image.")

    engine.confidence_threshold = confidence
    annotated_frame, result = engine.process_frame(frame, mode=mode, source=InputSource.IMAGE)

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
    return session_stats


@app.get("/session/log", tags=["Session"])
def get_detection_log(limit: int = Query(50, ge=1, le=500)):
    entries = logger.get_all_entries()
    return {
        "total_entries": len(entries),
        "entries": entries[-limit:],
        "log_file": logger.get_log_path(),
    }


@app.delete("/session/reset", tags=["Session"])
def reset_session():
    global session_stats
    session_stats = SessionStats()
    return {"message": "Session stats reset.", "new_session": session_stats}
