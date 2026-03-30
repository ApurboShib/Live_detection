
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import os
from PIL import Image

from models import DetectionMode, InputSource
from detector import DetectionEngine
from logger import DetectionLogger


st.set_page_config(
    page_title="Face & Object Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

─────────
# Custom CSS
─────────

st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0d1117; color: #e6edf3; }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }

    /* Cards */
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        margin: 4px 0;
    }
    .metric-number { font-size: 2rem; font-weight: 700; color: #00ff87; }
    .metric-label  { font-size: 0.8rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }

    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .status-running { background: #0d3321; color: #00ff87; border: 1px solid #00ff87; }
    .status-stopped { background: #3d1212; color: #ff6b6b; border: 1px solid #ff6b6b; }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #0d1117 0%, #1a2332 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px 28px;
        margin-bottom: 20px;
    }
    .header-title { font-size: 1.6rem; font-weight: 700; color: #e6edf3; margin: 0; }
    .header-sub   { color: #8b949e; font-size: 0.9rem; margin: 4px 0 0 0; }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s;
    }

    /* Tables */
    .dataframe { font-size: 0.8rem !important; }
</style>
""", unsafe_allow_html=True)


def init_state():
    defaults = {
        "engine": None,
        "logger": None,
        "running": False,
        "frame_count": 0,
        "total_detections": 0,
        "fps_history": [],
        "detection_history": [],
        "yolo_loaded": False,
        "yolo_status": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


@st.cache_resource(show_spinner="Loading detection engine...")
def get_engine(confidence: float) -> DetectionEngine:
    return DetectionEngine(confidence_threshold=confidence)

with st.sidebar:
    st.markdown("## ⚙️ Controls")
    st.markdown("---")

    # Detection mode
    mode_label = st.selectbox(
        "Detection Mode",
        ["Face + Object (Both)", "Face Only", "Object Only"],
    )
    mode_map = {
        "Face + Object (Both)": DetectionMode.BOTH,
        "Face Only": DetectionMode.FACE,
        "Object Only": DetectionMode.OBJECT,
    }
    selected_mode = mode_map[mode_label]

    # Input source
    source_label = st.selectbox("Input Source", ["Webcam", "Upload Image", "Upload Video"])

    # Confidence
    confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)

    st.markdown("---")

    # YOLO loader
    st.markdown("### 🤖 YOLO Model")
    yolo_model = st.selectbox("Model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
                               help="nano=fastest, small=balanced, medium=accurate")

    if st.button("⬇️ Load YOLO Model", use_container_width=True):
        engine_obj = get_engine(confidence)
        with st.spinner("Downloading & loading YOLOv8..."):
            status = engine_obj.load_yolo(yolo_model)
        st.session_state.yolo_loaded = engine_obj.yolo_ready
        st.session_state.yolo_status = status
        st.session_state.engine = engine_obj

    if st.session_state.yolo_status:
        st.info(st.session_state.yolo_status)

    st.markdown("---")
    st.markdown("### 📡 FastAPI Backend")
    st.code("uvicorn api:app --reload --port 8000", language="bash")
    st.markdown("[Open API Docs →](http://localhost:8000/docs)")


st.markdown("""
<div class="main-header">
    <p class="header-title">🔍 Real-Time Face & Object Detection</p>
    <p class="header-sub">Joy Shib · 222-115-111 · Computer Graphics & Image Processing · March 2026</p>
</div>
""", unsafe_allow_html=True)


tab_live, tab_image, tab_logs, tab_api = st.tabs(["📹 Live Detection", "🖼️ Image Upload", "📊 Detection Log", "🔌 API Info"])



# Tab 1: Live Webcam


with tab_live:
    col_vid, col_stats = st.columns([3, 1])

    with col_stats:
        st.markdown("### 📈 Live Stats")
        fps_placeholder     = st.empty()
        frames_placeholder  = st.empty()
        detect_placeholder  = st.empty()
        status_placeholder  = st.empty()

    with col_vid:
        frame_placeholder = st.empty()

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            start_btn = st.button("▶️ Start Detection", use_container_width=True, type="primary")
        with btn_col2:
            stop_btn  = st.button("⏹️ Stop", use_container_width=True)

    if start_btn:
        st.session_state.running = True
        st.session_state.frame_count = 0
        st.session_state.total_detections = 0
        st.session_state.fps_history = []
        if st.session_state.engine is None:
            st.session_state.engine = get_engine(confidence)
        if st.session_state.logger is None:
            st.session_state.logger = DetectionLogger()

    if stop_btn:
        st.session_state.running = False

    # Status badge
    if st.session_state.running:
        status_placeholder.markdown('<span class="status-badge status-running">● RUNNING</span>', unsafe_allow_html=True)
    else:
        status_placeholder.markdown('<span class="status-badge status-stopped">■ STOPPED</span>', unsafe_allow_html=True)

    # Live loop
    if st.session_state.running:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("❌ Cannot open webcam. Check camera permissions or try a different index.")
            st.session_state.running = False
        else:
            engine_obj = st.session_state.engine
            engine_obj.confidence_threshold = confidence
            log_obj = st.session_state.logger

            while st.session_state.running:
                ret, frame = cap.read()
                if not ret:
                    st.warning("⚠️ Frame capture failed.")
                    break

                annotated, result = engine_obj.process_frame(frame, mode=selected_mode)

                # Update state
                st.session_state.frame_count += 1
                st.session_state.total_detections += result.total_detections
                st.session_state.fps_history.append(result.fps)
                if len(st.session_state.fps_history) > 60:
                    st.session_state.fps_history.pop(0)

                # Log
                log_obj.log_frame(result)

                # Display frame
                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(rgb, channels="RGB", use_container_width=True)

                # Stats
                avg_fps = sum(st.session_state.fps_history) / max(len(st.session_state.fps_history), 1)
                fps_placeholder.metric("FPS", f"{result.fps:.1f}", f"avg {avg_fps:.1f}")
                frames_placeholder.metric("Frames", st.session_state.frame_count)
                detect_placeholder.metric("Total Detections", st.session_state.total_detections)

                time.sleep(0.01)  # yield to Streamlit

            cap.release()



# Tab 2: Image Upload


with tab_image:
    st.markdown("### Upload an image for detection")

    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp"])

    if uploaded:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        col_orig, col_det = st.columns(2)

        with col_orig:
            st.markdown("**Original**")
            rgb_orig = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(rgb_orig, use_container_width=True)

        engine_obj = st.session_state.engine or get_engine(confidence)
        engine_obj.confidence_threshold = confidence

        if not engine_obj.yolo_ready and selected_mode != DetectionMode.FACE:
            st.warning("⚠️ YOLO not loaded — showing face detection only. Load YOLO from the sidebar.")

        with st.spinner("Running detection..."):
            annotated, result = engine_obj.process_frame(
                frame, mode=selected_mode, source=InputSource.IMAGE
            )

        with col_det:
            st.markdown("**Detected**")
            rgb_ann = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(rgb_ann, use_container_width=True)

        # Results table
        st.markdown(f"#### 🎯 {result.total_detections} detection(s) found")
        if result.detections:
            rows = [
                {
                    "Label": d.label,
                    "Type": d.detection_type,
                    "Confidence": f"{d.confidence:.1%}",
                    "X": d.bounding_box.x,
                    "Y": d.bounding_box.y,
                    "W": d.bounding_box.width,
                    "H": d.bounding_box.height,
                }
                for d in result.detections
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # Log it
        log_obj = st.session_state.logger or DetectionLogger()
        log_obj.log_frame(result)
        st.session_state.logger = log_obj



# Tab 3: Detection Log


with tab_logs:
    st.markdown("### 📋 Detection Log (CSV)")

    log_obj = st.session_state.logger

    if log_obj is None:
        st.info("No log yet — run some detections first.")
    else:
        entries = log_obj.get_all_entries()

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Log Entries", len(entries))
        if entries:
            labels = [e["label"] for e in entries]
            unique = len(set(labels))
            col_b.metric("Unique Classes", unique)
            col_c.metric("Log File", os.path.basename(log_obj.get_log_path()))

        if entries:
            df = pd.DataFrame(entries)
            st.dataframe(df, use_container_width=True, height=350)

            # Download button
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download CSV",
                data=csv_data,
                file_name=os.path.basename(log_obj.get_log_path()),
                mime="text/csv",
            )

            # Bar chart — detections per class
            if "label" in df.columns:
                st.markdown("#### Detections per Class")
                counts = df["label"].value_counts().reset_index()
                counts.columns = ["Class", "Count"]
                st.bar_chart(counts.set_index("Class"))



# Tab 4: API Info


with tab_api:
    st.markdown("### 🔌 FastAPI Backend")
    st.markdown("""
The FastAPI server (`api.py`) runs independently and exposes REST endpoints
you can call from any HTTP client, Postman, or another Python script.
    """)

    st.markdown("#### Start the API server")
    st.code("uvicorn api:app --reload --port 8000", language="bash")

    st.markdown("#### Available Endpoints")
    endpoints = [
        ("GET",    "/",                      "Health check + YOLO status"),
        ("GET",    "/health",                "Detailed health info"),
        ("POST",   "/detect/image",          "Upload image → detection JSON"),
        ("POST",   "/detect/image/annotated","Upload image → annotated base64 image"),
        ("GET",    "/session/stats",         "Cumulative session statistics"),
        ("GET",    "/session/log",           "View recent detection log entries"),
        ("DELETE", "/session/reset",         "Reset session stats"),
    ]
    df_ep = pd.DataFrame(endpoints, columns=["Method", "Endpoint", "Description"])
    st.dataframe(df_ep, use_container_width=True, hide_index=True)

    st.markdown("#### Example: Call /detect/image with Python requests")
    st.code("""
import requests

url = "http://localhost:8000/detect/image?mode=both&confidence=0.5"
with open("photo.jpg", "rb") as f:
    response = requests.post(url, files={"file": f})

data = response.json()
print(f"Found {data['data']['total_detections']} objects")
for det in data['data']['detections']:
    print(f"  {det['label']} ({det['confidence']:.0%}) at {det['bounding_box']}")
""", language="python")

    st.markdown("#### Swagger UI")
    st.markdown("Once the API is running, visit → **[http://localhost:8000/docs](http://localhost:8000/docs)**")
