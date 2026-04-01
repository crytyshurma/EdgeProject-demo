from core.detector import Detector
from core.tracker import Tracker
from core.recorder import SingleFileRecorder
from core.alert import AlertManager

from utils.drawing import *
from utils.helpers import get_label
from utils.logger import setup_logging

from config import *

import cv2
import time
import os

log = setup_logging()

def main():
    num_cams = len(CAMERA_SOURCES)

    detector = Detector()
    trackers = [Tracker(i) for i in range(num_cams)]
    alert_mgr = AlertManager()
    recorder = SingleFileRecorder(num_cams, FPS)

    caps = [cv2.VideoCapture(src) for src in CAMERA_SOURCES]

    while True:
        live_tiles = []
        active_frames = {}
        idle_ids = []

        for i, cap in enumerate(caps):
            ret, frame = cap.read()

            if not ret:
                idle_ids.append(i)
                live_tiles.append(make_idle_tile(i))
                continue

            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            dets = detector.detect(frame)
            tracks, new_ids = trackers[i].update(dets)

            annotated = draw_active_tile(frame.copy(), tracks, i)
            live_tiles.append(annotated)

            if len(tracks) > 0:
                active_frames[i] = annotated
            else:
                idle_ids.append(i)

            if new_ids:
                alert_mgr.trigger(i, get_label(i), new_ids, annotated)

        live_grid = build_grid(live_tiles, LIVE_GRID_COLS)
        live_grid = add_live_banner(live_grid)

        cv2.imshow("Live", live_grid)

        recorder.write(active_frames, idle_ids)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    recorder.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()