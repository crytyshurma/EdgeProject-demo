import numpy as np

class Tracker:
    def __init__(self, cam_id):
        self.cam_id = cam_id
        self.next_id = 0
        self.tracks = {}

    def _iou(self, a, b):
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])

        inter = max(0, x2-x1) * max(0, y2-y1)

        area_a = (a[2]-a[0])*(a[3]-a[1])
        area_b = (b[2]-b[0])*(b[3]-b[1])

        union = area_a + area_b - inter

        if union == 0:
            return 0

        return inter / union

    def update(self, detections):
        new_ids = []
        updated = {}

        for det in detections:
            box = det[:4]

            best_id = None
            best_iou = 0

            for tid, tbox in self.tracks.items():
                iou = self._iou(box, tbox)
                if iou > best_iou:
                    best_iou = iou
                    best_id = tid

            if best_iou > 0.3:
                updated[best_id] = box
            else:
                tid = self.next_id
                self.next_id += 1
                updated[tid] = box
                new_ids.append(tid)

        self.tracks = updated

        tracked = [
            (x1, y1, x2, y2, tid)
            for tid, (x1,y1,x2,y2) in updated.items()
        ]

        return tracked, new_ids
