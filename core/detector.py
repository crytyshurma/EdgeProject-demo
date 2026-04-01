from ultralytics import YOLO
from config import DETECT_CLASSES, CONFIDENCE_THRESH

class Detector:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')

    def detect(self, frame):
        results = self.model(frame, imgsz=416)[0]

        detections = []
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls in DETECT_CLASSES and conf > CONFIDENCE_THRESH:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append([x1, y1, x2, y2, conf])

        return detections