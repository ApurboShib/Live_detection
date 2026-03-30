import cv2
import argparse
import glfw
import time
import os
import numpy as np
from datetime import datetime

from gl_renderer import GLRenderer
from detector import DetectionEngine
from models import DetectionMode
from logger import DetectionLogger

def main():
    parser = argparse.ArgumentParser(description="Live Face & Object Detection (OpenGL)")
    parser.add_argument("--source", default="0", help="Camera index or video file (default: 0)")
    parser.add_argument("--mode", choices=["face", "object", "both"], default="both", help="Detection mode")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--no-yolo", action="store_true", help="Disable YOLO (face only)")
    args = parser.parse_args()

    # Determine mode
    mode_map = {
        "face": DetectionMode.FACE,
        "object": DetectionMode.OBJECT,
        "both": DetectionMode.BOTH
    }
    initial_mode = mode_map[args.mode]
    if args.no_yolo:
        initial_mode = DetectionMode.FACE

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"Error: Could not open source {args.source}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if width <= 0 or height <= 0:
        width, height = 640, 480

    renderer = GLRenderer(width, height, "Live Detection - OpenGL")
    engine = DetectionEngine(confidence_threshold=args.confidence)
    logger = DetectionLogger()
    
    if args.no_yolo and initial_mode != DetectionMode.FACE:
        print("Warning: --no-yolo specified but mode is not face. Forcing face mode.")
        initial_mode = DetectionMode.FACE

    if initial_mode in (DetectionMode.OBJECT, DetectionMode.BOTH) and not args.no_yolo:
        print("Loading YOLO model...")
        engine.load_yolo("yolov8n-pose.pt")

    print("\nControls:")
    print("  ESC: Quit")
    print("  F: Face-only mode")
    print("  O: Object-only mode")
    print("  B: Both (Face + Object) mode")
    print("  SPACE: Pause / Resume")
    print("  S: Save current frame as PNG\n")

    state = {
        "mode": initial_mode,
        "paused": False,
        "frame": np.zeros((height, width, 3), dtype=np.uint8)
    }

    def key_callback(window, key, scancode, action, mods):
        if action == glfw.PRESS:
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)
            elif key == glfw.KEY_F:
                state["mode"] = DetectionMode.FACE
                print("Switched to FACE mode")
            elif key == glfw.KEY_O:
                state["mode"] = DetectionMode.OBJECT
                print("Switched to OBJECT mode")
            elif key == glfw.KEY_B:
                state["mode"] = DetectionMode.BOTH
                print("Switched to BOTH mode")
            elif key == glfw.KEY_SPACE:
                state["paused"] = not state["paused"]
                print("Paused" if state["paused"] else "Resumed")
            elif key == glfw.KEY_S:
                os.makedirs("saved_frames", exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = f"saved_frames/frame_{ts}.png"
                cv2.imwrite(fname, state["frame"])
                print(f"Saved: {fname}")

    glfw.set_key_callback(renderer.window, key_callback)

    while not glfw.window_should_close(renderer.window):
        glfw.poll_events()
        
        if not state["paused"]:
            ret, frame = cap.read()
            if not ret:
                print("End of stream.")
                break
                
            frame = cv2.resize(frame, (width, height))
            
            output_frame, result = engine.process_frame(frame, mode=state["mode"], annotate=True)
            
            # OpenCV blit for FPS and Detection count
            fps_text = f"FPS: {result.fps:.1f}"
            count_text = f"Detected: {result.total_detections}"
            
            cv2.putText(output_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 150), 2, cv2.LINE_AA)
            cv2.putText(output_frame, count_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            state["frame"] = output_frame.copy()
            logger.log_frame(result)
            
            renderer.render(output_frame, result.detections)
        else:
            time.sleep(0.01)

    cap.release()
    glfw.terminate()

if __name__ == "__main__":
    main()
