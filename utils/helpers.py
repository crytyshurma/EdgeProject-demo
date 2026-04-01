import math
from config import CAMERA_LABELS

def get_label(i):
    return CAMERA_LABELS[i] if i < len(CAMERA_LABELS) else f"Camera {i+1}"

def grid_dims(n):
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols