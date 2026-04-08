import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
import cv2

from config import DETECT_CLASSES, CONFIDENCE_THRESH


class Detector:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # smaller + faster model
        self.model = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
        self.model.to(self.device)
        self.model.eval()

        self.transform = T.Compose([
            T.ToTensor()
        ])

    def detect(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = self.transform(frame_rgb).to(self.device)

        with torch.no_grad():
            outputs = self.model([img])[0]

        detections = []

        for box, score, label in zip(
            outputs["boxes"],
            outputs["scores"],
            outputs["labels"]
        ):
            conf = float(score)

            if conf < CONFIDENCE_THRESH:
                continue

            cls = int(label)

            if cls in DETECT_CLASSES:
                x1, y1, x2, y2 = map(int, box.tolist())
                detections.append([x1, y1, x2, y2, conf])

        return detections
