from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

# Standard ArcFace reference points for 112x112 output
REFERENCE_FACIAL_POINTS = np.array(
    [
        [38.2946, 51.6963],  # left eye
        [73.5318, 51.5014],  # right eye
        [56.0252, 71.7366],  # nose
        [41.5493, 92.3655],  # left mouth
        [70.7299, 92.2041],  # right mouth
    ],
    dtype=np.float32,
)


def _clamp(val: float, low: float, high: float) -> float:
    return max(low, min(val, high))


def expand_bbox(bbox: Tuple[float, float, float, float], margin: float, shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
    """
    Expand a bounding box by a relative margin and clamp to image bounds.
    bbox: (x1, y1, x2, y2)
    shape: (H, W, C)
    """
    h, w = shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    mx, my = bw * margin, bh * margin
    nx1 = _clamp(x1 - mx, 0, w - 1)
    ny1 = _clamp(y1 - my, 0, h - 1)
    nx2 = _clamp(x2 + mx, 0, w - 1)
    ny2 = _clamp(y2 + my, 0, h - 1)
    return int(nx1), int(ny1), int(nx2), int(ny2)


def crop_face(image: np.ndarray, bbox: Tuple[float, float, float, float], margin: float = 0.25, out_size: int = 112) -> np.ndarray:
    """
    Crop and resize a face region.
    """
    x1, y1, x2, y2 = expand_bbox(bbox, margin=margin, shape=image.shape)
    face = image[y1:y2, x1:x2]
    if face.size == 0:
        raise ValueError("Empty crop from bbox")
    return cv2.resize(face, (out_size, out_size))


def pick_largest(detections: List[Dict]) -> Dict:
    if not detections:
        return {}
    return max(detections, key=lambda det: (det.get("bbox", [0, 0, 0, 0])[2] - det.get("bbox", [0, 0, 0, 0])[0]) * (det.get("bbox", [0, 0, 0, 0])[3] - det.get("bbox", [0, 0, 0, 0])[1]))


def blur_score(image: np.ndarray) -> float:
    """Sharpness metric: variance of the Laplacian (higher = sharper)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def align_face(image: np.ndarray, landmarks: Sequence[Sequence[float]], out_size: int = 112) -> np.ndarray:
    """
    Align face using five landmarks and return a normalized crop.
    Landmarks order is expected: left_eye, right_eye, nose, mouth_left, mouth_right.
    """
    pts = np.array(landmarks, dtype=np.float32)
    if pts.shape != (5, 2):
        raise ValueError(f"Expected 5 landmarks with shape (5,2), got {pts.shape}")

    scale = out_size / 112.0
    dst = REFERENCE_FACIAL_POINTS * scale
    tfm, _ = cv2.estimateAffinePartial2D(pts, dst, method=cv2.LMEDS)
    if tfm is None:
        raise RuntimeError("Failed to estimate affine transform for alignment")
    return cv2.warpAffine(image, tfm, (out_size, out_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def extract_face(
    image: np.ndarray,
    detection: Dict,
    margin: float = 0.25,
    out_size: int = 112,
    align: bool = False,
) -> np.ndarray:
    """
    Align face when landmarks are available; otherwise fall back to a simple crop.
    """
    if align and detection.get("kps") is not None:
        try:
            return align_face(image, detection["kps"], out_size=out_size)
        except Exception:
            # Fall back to vanilla crop if alignment fails
            pass
    return crop_face(image, detection["bbox"], margin=margin, out_size=out_size)
