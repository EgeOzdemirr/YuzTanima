import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def load_bgr(path: str) -> np.ndarray:
    path_str = str(path)
    img = None
    try:
        data = np.fromfile(path_str, dtype=np.uint8)
        if data.size > 0:
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        img = None
    if img is None:
        img = cv2.imread(path_str)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {path}")
    return img


def save_image(path: str, image: np.ndarray) -> str:
    path_str = str(path)
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(path_str, image)
    if not ok:
        suffix = Path(path_str).suffix or ".jpg"
        success, buf = cv2.imencode(suffix, image)
        if success:
            buf.tofile(path_str)
            return path_str
        raise IOError(f"Failed to write image to {path}")
    return path_str


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
