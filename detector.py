import cv2
import time
import numpy as np
from typing import Tuple, List, Optional
from models import Detection, BoundingBox, FrameResult, DetectionMode, InputSource, Keypoint

COLORS = [
    (0, 255, 100),   (0, 180, 255),   (255, 100, 0),
    (255, 0, 150),   (100, 255, 255), (200, 0, 255),
]

def get_color(class_id: int) -> Tuple[int, int, int]:
    return COLORS[class_id % len(COLORS)]

class DetectionEngine:
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self._yolo = None
        self._yolo_loaded = False
        self._frame_count = 0
        self._fps = 0.0
        self._prev_time = time.time()
        
        # Load highly accurate pose model automatically
        self.load_yolo("yolov8s-pose.pt")

    def load_yolo(self, model_name: str = "yolov8s-pose.pt") -> str:
        try:
            from ultralytics import YOLO
            self._yolo = YOLO(model_name)
            self._yolo_loaded = True
            return f"✅ YOLOv8 model '{model_name}' loaded successfully."
        except Exception as e:
            return f"❌ Failed to load YOLO: {str(e)}"

    @property
    def yolo_ready(self) -> bool:
        return self._yolo_loaded and self._yolo is not None

    def _update_fps(self) -> float:
        now = time.time()
        elapsed = now - self._prev_time
        self._prev_time = now
        if elapsed > 0:
            self._fps = 1.0 / elapsed
        return round(self._fps, 1)

    def process_frame(
        self,
        frame: np.ndarray,
        mode: DetectionMode = DetectionMode.BOTH,
        source: InputSource = InputSource.WEBCAM,
        annotate: bool = True,
    ) -> Tuple[np.ndarray, FrameResult]:
        
        self._frame_count += 1
        fps = self._update_fps()
        all_detections = []
        
        if self.yolo_ready:
            results = self._yolo(frame, conf=self.confidence_threshold, verbose=False)
            for result in results:
                if result.boxes is not None:
                    for i, box in enumerate(result.boxes):
                        cls_id = int(box.cls[0])
                        label = self._yolo.names[cls_id]
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        w = x2 - x1
                        h = y2 - y1

                        kpts = None
                        if hasattr(result, 'keypoints') and result.keypoints is not None and result.keypoints.xy is not None:
                            if len(result.keypoints.xy) > i:
                                xy = result.keypoints.xy[i].cpu().numpy()
                                confs = result.keypoints.conf[i].cpu().numpy() if result.keypoints.conf is not None else None
                                kpts = []
                                for j in range(len(xy)):
                                    kconf = float(confs[j]) if confs is not None else 1.0
                                    kpts.append(Keypoint(x=float(xy[j][0]), y=float(xy[j][1]), confidence=kconf))

                        if label != "person":
                            label = "unknown object"
                            all_detections.append(Detection(
                                label=label,
                                confidence=round(conf, 3),
                                bounding_box=BoundingBox(x=x1, y=y1, width=w, height=h),
                                detection_type="object",
                                keypoints=None
                            ))
                        else:
                            all_detections.append(Detection(
                                label="Body",
                                confidence=round(conf, 3),
                                bounding_box=BoundingBox(x=x1, y=y1, width=w, height=h),
                                detection_type="pose" if kpts else "object",
                                keypoints=kpts
                            ))
                            
                            if kpts:
                                face_pts = [kpts[j] for j in range(5) if j < len(kpts) and kpts[j].confidence > 0.5]
                                if face_pts:
                                    fx_min = min(p.x for p in face_pts)
                                    fx_max = max(p.x for p in face_pts)
                                    fy_min = min(p.y for p in face_pts)
                                    fy_max = max(p.y for p in face_pts)
                                    
                                    fw = max(fx_max - fx_min, 20)
                                    fh = max(fy_max - fy_min, 20)
                                    margin_w = fw * 0.5
                                    margin_h = fh * 0.5
                                    bx1, by1 = max(0, int(fx_min - margin_w)), max(0, int(fy_min - margin_h))
                                    bx2, by2 = int(fx_max + margin_w), int(fy_max + margin_h)
                                    
                                    all_detections.append(Detection(
                                        label="Face",
                                        confidence=min(round(sum(p.confidence for p in face_pts)/len(face_pts), 3), 1.0),
                                        bounding_box=BoundingBox(x=bx1, y=by1, width=bx2-bx1, height=by2-by1),
                                        detection_type="face",
                                        keypoints=None
                                    ))
                                
                                if len(kpts) > 9 and kpts[9].confidence > 0.3:
                                    kx, ky = kpts[9].x, kpts[9].y
                                    hw = w * 0.15
                                    hh = h * 0.1
                                    bx1, by1 = max(0, int(kx - hw/2)), max(0, int(ky - hh/2))
                                    all_detections.append(Detection(
                                        label="Left Hand",
                                        confidence=round(kpts[9].confidence, 3),
                                        bounding_box=BoundingBox(x=bx1, y=by1, width=int(hw), height=int(hh)),
                                        detection_type="object",
                                        keypoints=None
                                    ))

                                if len(kpts) > 10 and kpts[10].confidence > 0.3:
                                    kx, ky = kpts[10].x, kpts[10].y
                                    hw = w * 0.15
                                    hh = h * 0.1
                                    bx1, by1 = max(0, int(kx - hw/2)), max(0, int(ky - hh/2))
                                    all_detections.append(Detection(
                                        label="Right Hand",
                                        confidence=round(kpts[10].confidence, 3),
                                        bounding_box=BoundingBox(x=bx1, y=by1, width=int(hw), height=int(hh)),
                                        detection_type="object",
                                        keypoints=None
                                    ))

        output_frame = frame.copy() if annotate else frame
        if annotate:
            self._draw_annotations(output_frame, all_detections)

        result = FrameResult(
            frame_id=self._frame_count,
            fps=fps,
            detections=all_detections,
            source=source,
            mode=mode,
        )
        return output_frame, result

    def _draw_annotations(self, frame: np.ndarray, detections: List[Detection]):
        for i, det in enumerate(detections):
            bb = det.bounding_box
            color = (0, 255, 0) if det.detection_type == "face" else get_color(i)

            cv2.rectangle(frame, (bb.x, bb.y), (bb.x + bb.width, bb.y + bb.height), color, 2)

            if det.keypoints:
                SKELETON_EDGES = [
                    (15, 13), (13, 11), (11, 12), (12, 14), (14, 16),
                    (11, 5), (12, 6), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)
                ]
                for kp in det.keypoints:
                    if kp.confidence > 0.5:
                        cv2.circle(frame, (int(kp.x), int(kp.y)), 4, color, -1)
                
                for pt1, pt2 in SKELETON_EDGES:
                    if pt1 < len(det.keypoints) and pt2 < len(det.keypoints):
                        k1 = det.keypoints[pt1]
                        k2 = det.keypoints[pt2]
                        if k1.confidence > 0.5 and k2.confidence > 0.5:
                            cv2.line(frame, (int(k1.x), int(k1.y)), (int(k2.x), int(k2.y)), color, 2)

            label_text = f"{det.label} {det.confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (bb.x, bb.y - th - 5), (bb.x + tw, bb.y), color, -1)
            cv2.putText(frame, label_text, (bb.x, bb.y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    def process_image_file(self, image_path: str, mode: DetectionMode = DetectionMode.BOTH):
        frame = cv2.imread(image_path)
        if frame is None:
            return None, None
        return self.process_frame(frame, mode=mode, source=InputSource.IMAGE)
