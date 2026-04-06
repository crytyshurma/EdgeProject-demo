import cv2
import numpy as np
from datetime import datetime
from typing import List
from config import FRAME_WIDTH, FRAME_HEIGHT, CAMERA_LABELS

def _label(i: int) -> str:
    return CAMERA_LABELS[i] if i < len(CAMERA_LABELS) else f"CAM {i}"


def draw_active_tile(frame: np.ndarray, tracks, cam_id: int) -> np.ndarray:
    """
    Annotate a frame that HAS active detections:
      tracks format:
      [
          (x1, y1, x2, y2, id),
          ...
      ]
    """

    for t in tracks:
        x1, y1, x2, y2, tid = map(int, t)

        # bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # track id
        cv2.putText(
            frame,
            f"ID {tid}",
            (x1, max(y1 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

    # camera label + count
    lbl = f"{_label(cam_id)}  |  {len(tracks)} person(s)"
    cv2.putText(
        frame,
        lbl,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    # timestamp
    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(
        frame,
        ts,
        (10, FRAME_HEIGHT - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (210, 210, 210),
        1,
        cv2.LINE_AA
    )

    return frame


def make_idle_tile(cam_id: int) -> np.ndarray:
    """
    Build a black placeholder tile for cameras with NO detections.
    """
    tile = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    tile[:] = (18, 18, 18)

    label = _label(cam_id)
    font = cv2.FONT_HERSHEY_SIMPLEX

    (tw, th), _ = cv2.getTextSize(label, font, 0.9, 2)
    cx = (FRAME_WIDTH - tw) // 2
    cy = (FRAME_HEIGHT - th) // 2 - 20

    cv2.putText(tile, label, (cx, cy), font, 0.9, (80, 80, 80), 2, cv2.LINE_AA)

    badge = "NO DETECTION"
    (bw, bh), _ = cv2.getTextSize(badge, font, 0.55, 1)
    bx = (FRAME_WIDTH - bw) // 2
    by = cy + th + 18

    cv2.putText(tile, badge, (bx, by), font, 0.55, (50, 50, 180), 1, cv2.LINE_AA)

    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(
        tile,
        ts,
        (10, FRAME_HEIGHT - 10),
        font,
        0.48,
        (60, 60, 60),
        1,
        cv2.LINE_AA
    )

    return tile


def build_grid(tiles: List[np.ndarray], cols: int) -> np.ndarray:
    """Tile frames into a cols-wide grid."""
    cells = [cv2.resize(t, (FRAME_WIDTH, FRAME_HEIGHT)) for t in tiles]

    rows = (len(cells) + cols - 1) // cols

    while len(cells) < rows * cols:
        cells.append(np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8))

    return np.vstack([
        np.hstack(cells[r * cols: r * cols + cols])
        for r in range(rows)
    ])


def add_live_banner(grid: np.ndarray) -> np.ndarray:
    """Semi-transparent LIVE banner"""
    h, w = grid.shape[:2]

    roi = grid[h - 28: h]
    overlay = roi.copy()

    cv2.rectangle(overlay, (0, 0), (w, 28), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, roi, 0.4, 0, roi)

    ts = datetime.now().strftime("LIVE  |  %Y-%m-%d  %H:%M:%S")

    cv2.putText(
        grid,
        ts,
        (w // 2 - 185, h - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (180, 220, 255),
        1,
        cv2.LINE_AA
    )

    return grid


def stamp_rec_header(grid: np.ndarray, active_labels: List[str]) -> np.ndarray:
    """
    Add REC header + active cameras
    """
    h, w = grid.shape[:2]
    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

    pill_x1, pill_y1 = w - 318, 5
    pill_x2, pill_y2 = w - 5, 32

    roi = grid[pill_y1:pill_y2, pill_x1:pill_x2].copy()
    overlay = roi.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (pill_x2 - pill_x1, pill_y2 - pill_y1),
        (25, 25, 25),
        -1
    )

    cv2.addWeighted(overlay, 0.55, roi, 0.45, 0, roi)
    grid[pill_y1:pill_y2, pill_x1:pill_x2] = roi

    cv2.circle(grid, (w - 300, 19), 7, (0, 0, 210), -1)

    cv2.putText(
        grid,
        "REC",
        (w - 288, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 0, 210),
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        grid,
        ts,
        (w - 248, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (220, 220, 220),
        1,
        cv2.LINE_AA
    )

    if active_labels:
        line = "Active: " + "  |  ".join(active_labels)

        (lw, _), _ = cv2.getTextSize(
            line,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            1
        )

        lx = w - lw - 8

        cv2.putText(
            grid,
            line,
            (max(lx, 5), 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (100, 220, 100),
            1,
            cv2.LINE_AA
        )

    return grid
