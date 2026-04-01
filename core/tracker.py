import supervision as sv
import numpy as np

class Tracker:
    def __init__(self,cam_id):
        self.tracker = sv.ByteTrack()
        self.cam_id = cam_id
        self.seen_ids = set()

    def update(self, detections):
        if len(detections) == 0:
            return sv.Detections.empty(), []

        xyxy = np.array([d[:4] for d in detections])
        confidence = np.array([d[4] for d in detections])

        tracked = self.tracker.update_with_detections(sv.Detections(xyxy=xyxy, confidence=confidence))
        
        new_ids = []
        if tracked.tracker_id is not None:
            for tid in tracked.tracker_id:
                if tid not in self.seen_ids:
                    self.seen_ids.add(tid)
                    new_ids.append(tid)

        return tracked, new_ids