import os
import subprocess
from datetime import datetime
import numpy as np
import cv2
from dotenv import load_dotenv
load_dotenv()
from typing import Tuple,Dict,List
from config import FRAME_WIDTH, FRAME_HEIGHT, RECORD_COLS, RECORD_ROWS, RECORDING_DIR, CAMERA_LABELS
from utils.drawing import make_idle_tile, build_grid, stamp_rec_header
from utils.logger import setup_logging

log = setup_logging()


def _get_rtsp_url() -> str:
    """
    App runs natively; MediaMTX runs in Docker with port 8554 published.
    FFmpeg pushes to localhost:8554 — Docker forwards it into the container.
    Override via RTSP_PUSH_URL env-var for a remote MediaMTX host.
    """
    return os.environ.get("RTSP_PUSH_URL", "rtsp://localhost:8554/mystream").strip()


RTSP_SERVER_URL = _get_rtsp_url()


def _label(i: int) -> str:
    return CAMERA_LABELS[i] if i < len(CAMERA_LABELS) else f"CAM {i}"


def _grid_dims(n: int) -> Tuple[int, int]:
    """Return (cols, rows) for a grid that fits n cameras."""
    import math
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return cols, rows


class SingleFileRecorder:
    """
    Writes ONE video file for the entire session with a FIXED canvas size,
    and simultaneously streams via RTSP → MediaMTX → WebRTC / HLS / RTSP.

    Canvas = RECORD_COLS × RECORD_ROWS tiles  (each FRAME_WIDTH × FRAME_HEIGHT).

    Every frame contains:
      • Active cameras  → annotated feed (bounding boxes, label, timestamp)
      • Idle cameras    → dark placeholder tile (camera name + "NO DETECTION" + ts)
      • Top-right HUD   → ● REC + wall-clock time + active camera List

    One FFmpeg process is spawned for the whole session using the tee muxer:
    it encodes once and writes to both an MP4 file and the RTSP push URL.
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
        self.rtsp_url = RTSP_SERVER_URL

        log.info(
            "RTSP push target: %s  |  file: %s  |  canvas: %dx%d (%dx%d tiles)",
            self.rtsp_url, self.path, self.canvas_w, self.canvas_h, self.cols, self.rows,
        )

        # Tee muxer: encode ONCE → write to MP4 file AND push to RTSP in one pass.
        # movflags=faststart+frag_keyframe makes the MP4 seekable while recording.
        tee_targets = "|".join([
            f"[f=mp4:movflags=faststart+frag_keyframe]{self.path}",
            f"[f=rtsp:rtsp_transport=tcp]{self.rtsp_url}",
        ])

        cmd = [
            "ffmpeg", "-y",

            # ── Input ──────────────────────────────────────────────────────────
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-strict", "experimental",

            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.canvas_w}x{self.canvas_h}",
            "-r", str(fps),
            "-i", "-",

            "-an",   # no audio

            # ── Encoding ───────────────────────────────────────────────────────
            "-c:v", "libx264",
            "-preset", "ultrafast",   # lowest latency for live streaming
            "-tune", "zerolatency",
            "-crf", "28",
            "-pix_fmt", "yuv420p",    # required for broad player compat
            "-g", str(fps),           # keyframe every 1 s → faster stream join
            "-sc_threshold", "0",     # no scene-change keyframes (stable GOP)
            "-bf", "0",               # no B-frames (adds latency)

            # ── Output ─────────────────────────────────────────────────────────
            "-f", "tee",
            "-map", "0:v",
            tee_targets,
        ]

        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

        log.info("SingleFileRecorder started — FFmpeg PID %d", self._proc.pid)

    # ── Public ────────────────────────────────────────────────────────────────

    def write(
        self,
        all_frames:   Dict[int, np.ndarray],
        idle_cam_ids: List[int],
    ) -> None:
        """
        Build the full recording canvas and push it to FFmpeg.

        all_frames    : {cam_id: annotated_frame} for cameras WITH detections
        idle_cam_ids  : cam IDs with no detections (shown as placeholder tiles)
        """
        if self._proc is None:
            return

        tiles: List[np.ndarray] = []
        active_labels: List[str] = []

        for cam_id in range(self.num_cams):
            if cam_id in all_frames:
                tiles.append(all_frames[cam_id])
                active_labels.append(_label(cam_id))
            else:
                tiles.append(make_idle_tile(cam_id))

        # Pad empty slots so the canvas is always full
        while len(tiles) < self.cols * self.rows:
            blank = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
            blank[:] = (10, 10, 10)
            tiles.append(blank)

        canvas = build_grid(tiles, self.cols)
        canvas = stamp_rec_header(canvas, active_labels)

        try:
            self._proc.stdin.write(canvas.tobytes())
        except (BrokenPipeError, OSError) as exc:
            stderr_out = self._proc.stderr.read().decode(errors="replace")
            log.error("FFmpeg pipe error (%s): %s", exc, stderr_out)
            self._proc = None

    def close(self) -> None:
        if self._proc is not None:
            try:
                self._proc.stdin.close()
                self._proc.wait(timeout=15)
                log.info("SingleFileRecorder closed | file: %s", self.path)
            except Exception as exc:
                log.warning("FFmpeg close warning: %s", exc)
            finally:
                self._proc = None
