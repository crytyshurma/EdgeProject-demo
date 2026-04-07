import numpy as np

class Tracker:
    def __init__(self, cam_id):
        self.cam_id = cam_id
        self.next_id = 0
        self.objects = {}
        self.seen_ids = set()

    def _get_centroid(self, box):
        x1, y1, x2, y2 = box
        return ((x1+x2)//2, (y1+y2)//2)

    def update(self, detections):
        new_ids = []
        tracked = []

        updated_objects = {}

        for det in detections:
            x1, y1, x2, y2, conf = det
            centroid = self._get_centroid((x1,y1,x2,y2))

            assigned_id = None

            for obj_id, prev_centroid in self.objects.items():
                dist = np.linalg.norm(
                    np.array(centroid) - np.array(prev_centroid)
                )

                if dist < 60:
                    assigned_id = obj_id
                    break

            if assigned_id is not None:
                updated_objects[assigned_id] = centroid
                tracked.append((x1, y1, x2, y2, assigned_id))
            else:
                obj_id = self.next_id
                self.next_id += 1

                updated_objects[obj_id] = centroid
                tracked.append((x1, y1, x2, y2, obj_id))

                if obj_id not in self.seen_ids:
                    self.seen_ids.add(obj_id)
                    new_ids.append(obj_id)

        self.objects = updated_objects

        return tracked, new_ids
