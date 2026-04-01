import os
import subprocess
from datetime import datetime
import numpy as np
import cv2

from config import FRAME_WIDTH, FRAME_HEIGHT, RECORD_COLS, RECORD_ROWS, RECORDING_DIR, CAMERA_LABELS
from utils.drawing import make_idle_tile, build_grid, stamp_rec_header
from utils.logger import setup_logging

log = setup_logging()

def _label(i: int) -> str:
    return CAMERA_LABELS[i] if i < len(CAMERA_LABELS) else f"CAM {i}"

def _grid_dims(n: int) -> tuple[int, int]:
    """Return (cols, rows) for a grid that fits n cameras."""
    import math
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return cols, rows

class SingleFileRecorder:
    """
    Writes ONE video file for the entire session with a FIXED canvas size.

    Canvas = RECORD_COLS × RECORD_ROWS tiles  (each FRAME_WIDTH × FRAME_HEIGHT).

    Every frame written to the file contains:
      • Active cameras  → annotated feed (bounding boxes, label, timestamp)
      • Idle cameras    → dark placeholder tile (camera name + "NO DETECTION" + timestamp)
      • Top-right HUD   → ● REC + wall-clock time + active camera list

    Because the canvas size never changes, only one FFmpeg process is ever
    needed for the whole session — no segments, no gaps.
    """

    def __init__(self, num_cams: int, fps: int):
        self.fps      = fps
        self.num_cams = num_cams

        # Derive canvas grid dimensions
        if RECORD_COLS and RECORD_ROWS:
            self.cols, self.rows = RECORD_COLS, RECORD_ROWS
        else:
            self.cols, self.rows = _grid_dims(num_cams)

        self.canvas_w = FRAME_WIDTH  * self.cols
        self.canvas_h = FRAME_HEIGHT * self.rows

        os.makedirs(RECORDING_DIR, exist_ok=True)
        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = f"{RECORDING_DIR}/surveillance_{ts}.mp4"

        # cmd = [
        #     "ffmpeg", "-y",
        #     "-f", "rawvideo", "-vcodec", "rawvideo",
        #     "-pix_fmt", "bgr24",
        #     "-s", f"{self.canvas_w}x{self.canvas_h}",
        #     "-r", str(fps),
        #     "-i", "-",
        #     "-an", "-vcodec", "mpeg4",
        #     self.path,
        # ]

        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.canvas_w}x{self.canvas_h}",
            "-r", str(fps),
            "-i", "-",

            "-an",

            # 🔥 MP4 compatible codec
            "-vcodec", "libx264",
            "-preset", "veryfast",   # speed vs quality
            "-crf", "23",            # lower = better quality (18–28 range)
            "-pix_fmt", "yuv420p",   # REQUIRED for MP4 compatibility

            self.path,
        ]

        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                      stderr=subprocess.PIPE)

        log.info(
            "SingleFileRecorder started | canvas: %dx%d (%dx%d tiles) | -> %s",
            self.canvas_w, self.canvas_h, self.cols, self.rows, self.path
        )

    # ── Public ────────────────────────────────────────────────────────────────

    def write(self,
              all_frames:    dict[int, np.ndarray], 
              idle_cam_ids:  list[int]):               # cams with NO detections this frame
        """
        Build the full recording canvas and push it to FFmpeg.

        all_frames    : {cam_id: frame} for cameras that HAVE detections
        idle_cam_ids  : cam IDs with no detections (get placeholder tile)
        """
        if self._proc is None:
            return

        # Build one tile per camera slot in the canvas grid
        # Slot order: cam 0 → slot 0, cam 1 → slot 1, …  (left→right, top→bottom)
        tiles = []
        active_labels = []
        for cam_id in range(self.num_cams):
            if cam_id in all_frames:
                tiles.append(all_frames[cam_id])
                active_labels.append(_label(cam_id))
            else:
                tiles.append(make_idle_tile(cam_id))

        # Pad remaining slots (if canvas has more slots than cameras)
        while len(tiles) < self.cols * self.rows:
            blank = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
            blank[:] = (10, 10, 10)
            tiles.append(blank)

        # Assemble canvas
        canvas = build_grid(tiles, self.cols)

        # Overlay REC HUD
        canvas = stamp_rec_header(canvas, active_labels)

        # Push to FFmpeg
        try:
            self._proc.stdin.write(canvas.tobytes())
        except BrokenPipeError:
            log.error("FFmpeg pipe broken: %s", self._proc.stderr.read().decode())
            self._proc = None

    def close(self):
        if self._proc is not None:
            try:
                self._proc.stdin.close()
                self._proc.wait(timeout=15)
                log.info("SingleFileRecorder closed -> %s", self.path)
            except Exception as exc:
                log.warning("FFmpeg close warning: %s", exc)
            self._proc = None
