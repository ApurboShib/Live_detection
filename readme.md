# Real-Time Face & Object Detection System


---

## Project Structure

```
face_object_detection/
├── gl_renderer.py   ★ NEW — OpenGL 3.3 renderer (texture + bounding boxes)
├── gl_window.py     ★ NEW — Standalone GLFW window (run this for OpenGL mode)
├── detector.py          Detection engine (Haar Cascade + YOLOv8)
├── models.py            Pydantic data models
├── logger.py            CSV logging utility
├── api.py               FastAPI REST backend
├── main.py              Streamlit UI
└── requirements.txt
```

---

## OpenGL Concepts Used (for your report)

| Concept | Where |
|---|---|
| OpenGL 3.3 Core Profile context | `gl_renderer.py` → `_init_glfw()` |
| Vertex + Fragment shaders (GLSL 330) | `_VERT_QUAD`, `_FRAG_QUAD`, `_VERT_FLAT`, `_FRAG_FLAT` |
| VAO / VBO setup | `_init_gl()` |
| Frame-as-texture (`glTexSubImage2D`) | `_upload_frame()` |
| Fullscreen quad (2 triangles, 6 indices) | `_draw_quad()` |
| `GL_LINE_LOOP` bounding boxes | `_draw_boxes()` |
| Pixel → NDC coordinate conversion | `_pixel_to_ndc()` |
| `GL_LINES` corner accents | `_draw_boxes()` accents section |
| `GL_DYNAMIC_DRAW` VBO updates | per-frame buffer upload |
| Alpha blending | `glEnable(GL_BLEND)` |

---

## Setup & Run

### 1. Install all dependencies
```bash
pip install -r requirements.txt
```

### 2a. OpenGL Window (main deliverable — recommended for demo)
```bash
python gl_window.py
# Options:
python gl_window.py --source 0 --mode both --confidence 0.5
python gl_window.py --source video.mp4 --mode object
python gl_window.py --no-yolo   # face-only, no YOLO needed
```

### Keyboard Controls (OpenGL window)
| Key | Action |
|-----|--------|
| `ESC` | Quit |
| `F` | Face-only mode |
| `O` | Object-only mode |
| `B` | Both (default) |
| `SPACE` | Pause / Resume |
| `S` | Save current frame as PNG |

### 2b. Streamlit UI (optional — shows image upload + log viewer)
```bash
streamlit run main.py
```

### 2c. FastAPI Backend (optional — REST API)
```bash
uvicorn api:app --reload --port 8000
# Docs: http://localhost:8000/docs
```

---

## Expected Output
- GLFW window showing live camera feed rendered as an OpenGL texture
- Green GL_LINE_LOOP rectangles + corner accents around detected faces
- Colour-coded GL_LINE_LOOP rectangles around detected objects
- FPS + detection count overlay (via OpenCV text blit onto texture)
- CSV log saved to `logs/detections_<timestamp>.csv`
