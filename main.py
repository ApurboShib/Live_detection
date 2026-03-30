
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

# ─────────
# Custom CSS
# ─────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

    /* Global Font */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif !important;
    }

    /* Animated Gradient Background for the App */
    .stApp {
        background: linear-gradient(-45deg, #050b14, #0b1121, #081a28, #110821);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        color: #f1f5f9;
        font-family: 'Outfit', sans-serif !important;
    }

    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Glassmorphic Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(11, 17, 33, 0.4) !important;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Native Metrics override - Glassmorphism */
    [data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-top: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 20px;
        padding: 20px 25px;
        box-shadow: 0 10px 40px 0 rgba(0, 0, 0, 0.4);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-8px);
        box-shadow: 0 15px 50px 0 rgba(0, 255, 136, 0.25);
        border-color: rgba(0, 255, 136, 0.5);
    }
    
    [data-testid="stMetricValue"] > div {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        background: linear-gradient(90deg, #00ff87, #60efff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.95rem !important;
        color: #94a3b8 !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 600;
    }

    /* Premium Header */
    .main-header {
        background: rgba(15, 23, 42, 0.5);
        backdrop-filter: blur(25px);
        -webkit-backdrop-filter: blur(25px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 40px;
        margin-bottom: 35px;
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: "";
        position: absolute;
        top: -100%; left: -100%; width: 300%; height: 300%;
        background: radial-gradient(circle, rgba(96, 239, 255, 0.1) 0%, transparent 50%);
        animation: rotate 25s linear infinite;
        z-index: 0;
        pointer-events: none;
    }
    
    @keyframes rotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    .header-title { 
        font-size: 2.8rem; 
        font-weight: 800; 
        color: #ffffff; 
        margin: 0; 
        z-index: 1; 
        position: relative; 
        letter-spacing: -1px;
        background: linear-gradient(90deg, #fff, #a5b4fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .header-sub { 
        color: #00ff87; 
        font-size: 1.1rem; 
        margin: 12px 0 0 0; 
        z-index: 1; 
        position: relative; 
        font-weight: 600; 
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }

    /* Buttons */
    .stButton > button {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        color: #f1f5f9;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        padding: 0.5rem 1.5rem;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(96, 239, 255, 0.3);
        border-color: rgba(96, 239, 255, 0.5);
        color: #60efff;
        background: rgba(30, 41, 59, 1.0);
    }
    
    /* Primary buttons specifically (Start Detection) */
    .stButton > button[data-testid="baseButton-primary"] {
        background: linear-gradient(90deg, #00ff87, #60efff) !important;
        border: none !important;
        color: #050b14 !important;
        box-shadow: 0 8px 25px rgba(0, 255, 135, 0.4) !important;
        font-weight: 800 !important;
    }
    .stButton > button[data-testid="baseButton-primary"]:hover {
        background: linear-gradient(90deg, #60efff, #00ff87) !important;
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(0, 255, 135, 0.6) !important;
    }

    /* File uploader */
    [data-testid="stFileUploadDropzone"] {
        background: rgba(30, 41, 59, 0.3) !important;
        border: 2px dashed rgba(255, 255, 255, 0.2) !important;
        border-radius: 20px !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #00ff87 !important;
        background: rgba(0, 255, 135, 0.05) !important;
    }

    /* Status badge animations */
    .status-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        margin-top: 10px;
    }
    .status-running { 
        background: rgba(0, 255, 135, 0.1); 
        color: #00ff87; 
        border: 1px solid rgba(0, 255, 135, 0.5); 
        animation: pulse-green 2s infinite;
    }
    .status-stopped { 
        background: rgba(255, 107, 107, 0.1); 
        color: #ff6b6b; 
        border: 1px solid rgba(255, 107, 107, 0.5); 
    }
    
    @keyframes pulse-green {
        0% { box-shadow: 0 0 0 0 rgba(0, 255, 135, 0.4); }
        70% { box-shadow: 0 0 0 15px rgba(0, 255, 135, 0); }
        100% { box-shadow: 0 0 0 0 rgba(0, 255, 135, 0); }
    }

    /* Customizing Tabs */
    button[role="tab"] {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        color: #94a3b8 !important;
        transition: 0.3s !important;
    }
    button[role="tab"][aria-selected="true"] {
        color: #60efff !important;
    }
    button[role="tab"][aria-selected="true"]::after {
        background: linear-gradient(90deg, #00ff87, #60efff) !important;
        height: 3px !important;
        border-radius: 3px !important;
    }

    /* Dataframes */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.1);
    }
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
    yolo_model = st.selectbox("Model", ["yolov8n-pose.pt", "yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
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
