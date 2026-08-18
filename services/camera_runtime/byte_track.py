from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def bbox_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    denom = area_a + area_b - inter
    if denom <= 0:
        return 0.0
    return float(inter / denom)


def tlbr_to_xyah(bbox: Sequence[float]) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    w = max(1e-6, x2 - x1)
    h = max(1e-6, y2 - y1)
    x = x1 + w / 2.0
    y = y1 + h / 2.0
    a = w / h
    return np.array([x, y, a, h], dtype=np.float32)


def xyah_to_tlbr(xyah: Sequence[float]) -> np.ndarray:
    x, y, a, h = [float(v) for v in xyah]
    w = a * h
    return np.array([x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0], dtype=np.float32)


def clamp_bbox_to_frame(bbox: Sequence[float], frame_shape: Sequence[int]) -> Optional[np.ndarray]:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1 = max(0.0, min(x1, w - 1.0))
    y1 = max(0.0, min(y1, h - 1.0))
    x2 = max(0.0, min(x2, w - 1.0))
    y2 = max(0.0, min(y2, h - 1.0))
    if x2 - x1 < 2.0 or y2 - y1 < 2.0:
        return None
    return np.array([x1, y1, x2, y2], dtype=np.float32)


class KalmanFilterXYAH:
    # State: [x, y, a, h, vx, vy, va, vh]
    def __init__(self) -> None:
        ndim = 4
        dt = 1.0
        self._motion_mat = np.eye(2 * ndim, dtype=np.float32)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim, dtype=np.float32)
        self._std_weight_position = 1.0 / 20.0
        self._std_weight_velocity = 1.0 / 160.0

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean = np.zeros(8, dtype=np.float32)
        mean[:4] = measurement
        h = measurement[3]
        std = np.array(
            [
                2 * self._std_weight_position * h,
                2 * self._std_weight_position * h,
                1e-2,
                2 * self._std_weight_position * h,
                10 * self._std_weight_velocity * h,
                10 * self._std_weight_velocity * h,
                1e-5,
                10 * self._std_weight_velocity * h,
            ],
            dtype=np.float32,
        )
        covariance = np.diag(np.square(std)).astype(np.float32)
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        std_pos = np.array(
            [
                self._std_weight_position * h,
                self._std_weight_position * h,
                1e-2,
                self._std_weight_position * h,
            ],
            dtype=np.float32,
        )
        std_vel = np.array(
            [
                self._std_weight_velocity * h,
                self._std_weight_velocity * h,
                1e-5,
                self._std_weight_velocity * h,
            ],
            dtype=np.float32,
        )
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel])).astype(np.float32)
        mean = self._motion_mat @ mean
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return mean.astype(np.float32), covariance.astype(np.float32)

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        std = np.array(
            [
                self._std_weight_position * h,
                self._std_weight_position * h,
                1e-1,
                self._std_weight_position * h,
            ],
            dtype=np.float32,
        )
        innovation_cov = np.diag(np.square(std)).astype(np.float32)
        mean_proj = self._update_mat @ mean
        cov_proj = self._update_mat @ covariance @ self._update_mat.T + innovation_cov
        return mean_proj.astype(np.float32), cov_proj.astype(np.float32)

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        proj_mean, proj_cov = self.project(mean, covariance)
        kalman_gain = covariance @ self._update_mat.T @ np.linalg.inv(proj_cov)
        innovation = measurement - proj_mean
        new_mean = mean + kalman_gain @ innovation
        new_cov = covariance - kalman_gain @ proj_cov @ kalman_gain.T
        return new_mean.astype(np.float32), new_cov.astype(np.float32)


@dataclass
class _Track:
    mean: np.ndarray
    covariance: np.ndarray
    track_id: int
    score: float
    age: int = 1
    hits: int = 1
    time_since_update: int = 0


class MultiFaceByteTracker:
    """Tracks several faces at once, matching detections to existing tracks by IOU
    (greedy, highest-IOU-first) and spawning new tracks for unmatched detections,
    bounded by ``max_tracks``. Tracks that go unmatched for ``max_time_lost`` frames
    are dropped.
    """

    def __init__(self, max_time_lost: int = 12, min_iou: float = 0.2, max_tracks: int = 5) -> None:
        self.kf = KalmanFilterXYAH()
        self.max_time_lost = max(1, int(max_time_lost))
        self.min_iou = float(min_iou)
        self.max_tracks = max(1, int(max_tracks))
        self._tracks: Dict[int, _Track] = {}
        self._next_id = 1

    @property
    def has_tracks(self) -> bool:
        return bool(self._tracks)

    def reset(self) -> None:
        self._tracks.clear()

    def _bbox_of(self, track: _Track, frame_shape: Sequence[int]) -> Optional[np.ndarray]:
        return clamp_bbox_to_frame(xyah_to_tlbr(track.mean[:4]), frame_shape)

    def predict(self, frame_shape: Sequence[int]) -> Dict[int, np.ndarray]:
        """Advance every track by one frame and drop tracks that are lost or off-frame."""
        predicted: Dict[int, np.ndarray] = {}
        stale_ids: List[int] = []
        for track_id, track in self._tracks.items():
            track.mean, track.covariance = self.kf.predict(track.mean, track.covariance)
            track.age += 1
            track.time_since_update += 1
            bbox = self._bbox_of(track, frame_shape)
            if bbox is None or track.time_since_update > self.max_time_lost:
                stale_ids.append(track_id)
                continue
            predicted[track_id] = bbox
        for track_id in stale_ids:
            del self._tracks[track_id]
        return predicted

    def update(self, detections: List[Dict], frame_shape: Sequence[int]) -> Dict[int, Tuple[np.ndarray, Dict]]:
        """Match detections against current tracks and update/create tracks accordingly.

        Returns {track_id: (bbox, detection)} only for tracks touched by a detection
        this call (i.e. newly created or re-matched); tracks predicted-but-unmatched
        this frame are left untouched here and keep coasting via ``predict``.
        """
        results: Dict[int, Tuple[np.ndarray, Dict]] = {}
        if not detections:
            return results

        candidates = [
            (track_id, bbox)
            for track_id, track in self._tracks.items()
            if (bbox := self._bbox_of(track, frame_shape)) is not None
        ]
        det_boxes = [
            (det, bbox)
            for det in detections
            if (bbox := clamp_bbox_to_frame(det["bbox"], frame_shape)) is not None
        ]

        pairs = []
        for ti, (track_id, tbbox) in enumerate(candidates):
            for di, (_det, dbbox) in enumerate(det_boxes):
                iou = bbox_iou(tbbox, dbbox)
                if iou >= self.min_iou:
                    pairs.append((iou, ti, di))
        pairs.sort(key=lambda item: item[0], reverse=True)

        matched_tracks: set = set()
        matched_dets: set = set()
        for _iou, ti, di in pairs:
            track_id = candidates[ti][0]
            if track_id in matched_tracks or di in matched_dets:
                continue
            matched_tracks.add(track_id)
            matched_dets.add(di)
            det, dbbox = det_boxes[di]
            measurement = tlbr_to_xyah(dbbox)
            track = self._tracks[track_id]
            track.mean, track.covariance = self.kf.update(track.mean, track.covariance, measurement)
            track.score = float(det.get("score", 0.0))
            track.hits += 1
            track.time_since_update = 0
            new_bbox = self._bbox_of(track, frame_shape)
            if new_bbox is not None:
                results[track_id] = (new_bbox, det)

        for di, (det, dbbox) in enumerate(det_boxes):
            if di in matched_dets or len(self._tracks) >= self.max_tracks:
                continue
            measurement = tlbr_to_xyah(dbbox)
            mean, covariance = self.kf.initiate(measurement)
            track_id = self._next_id
            self._next_id += 1
            self._tracks[track_id] = _Track(
                mean=mean, covariance=covariance, track_id=track_id, score=float(det.get("score", 0.0))
            )
            results[track_id] = (dbbox, det)

        return results
