from deep_sort_realtime.deepsort_tracker import DeepSort

class Tracker:
    def __init__(self, cam_id):
        self.cam_id = cam_id

        # ⚠️ Optimized for Jetson Nano
        self.tracker = DeepSort(
            max_age=30,
            n_init=2,
            embedder=None   # 🔥 disables heavy ReID model
        )

        self.seen_ids = set()

    def update(self, detections, frame=None):
        """
        detections: [[x1, y1, x2, y2, conf], ...]
        """

        ds_detections = []

        for det in detections:
            x1, y1, x2, y2, conf = det
            w = x2 - x1
            h = y2 - y1

            ds_detections.append(([x1, y1, w, h], conf, 'object'))

        tracks = self.tracker.update_tracks(ds_detections, frame=frame)

        tracked = []
        new_ids = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, r, b = track.to_ltrb()

            tracked.append((int(l), int(t), int(r), int(b), track_id))

            if track_id not in self.seen_ids:
                self.seen_ids.add(track_id)
                new_ids.append(track_id)

        return tracked, new_ids
    

    #pip install deep-sort-realtime