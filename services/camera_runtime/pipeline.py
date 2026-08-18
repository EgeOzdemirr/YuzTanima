import logging
import os
import time
import unicodedata
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import json
import numpy as np
import yaml
try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover - runtime fallback
    Image = None
    ImageDraw = None
    ImageFont = None

from services.camera_runtime.byte_track import MultiFaceByteTracker
from services.core.io.capture_store import CaptureStore
from services.core.io.events_store import EventsStore
from services.core.models.detector_retinaface import RetinaFaceDetector
from services.core.models.embed_adaface_onnx import AdaFaceONNXEmbedder
from services.core.models.match_faiss import FaissMatcher
from services.core.utils.identity import resolve_match
from services.core.utils.preprocess import blur_score, extract_face

logger = logging.getLogger("camera_pipeline")


@dataclass
class _TrackMatchState:
    last_det_score: float = 0.0
    candidate_person_id: Optional[str] = None
    candidate_hits: int = 0
    confirmed_person_id: Optional[str] = None
    confirmed_raw_id: Optional[str] = None
    confirmed_similarity: float = 0.0
    contradiction_hits: int = 0
    last_saved_person_id: Optional[str] = None
    emb_window: Optional[Deque[np.ndarray]] = None
    emb_count: int = 0


def can_embed_face(
    has_kps: bool,
    bbox_side: float,
    detector_frames_only: bool,
    min_face_px: int,
) -> bool:
    """Quality gate: only aligned (landmarked) detector crops of sufficient
    size are worth embedding; coasted/tiny crops poison matching."""
    if detector_frames_only and not has_kps:
        return False
    if min_face_px > 0 and bbox_side < min_face_px:
        return False
    return True


def load_yaml(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_gallery_map(gallery_file: Path) -> Dict[str, Dict]:
    if not gallery_file.exists():
        logger.warning("Gallery file missing: %s", gallery_file)
        return {}
    with open(gallery_file, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    return {p["personId"]: p for p in data.get("persons", [])}


def normalize_person_id(value: str) -> str:
    folded = (value or "").casefold()
    normalized = unicodedata.normalize("NFKD", folded)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def rel_web_path(path: Path, root: Path) -> str:
    return "/" + str(path.relative_to(root).as_posix())


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class CameraPipeline:
    _ASCII_FALLBACK_MAP = str.maketrans(
        {
            "ç": "c",
            "Ç": "C",
            "ğ": "g",
            "Ğ": "G",
            "ı": "i",
            "İ": "I",
            "ö": "o",
            "Ö": "O",
            "ş": "s",
            "Ş": "S",
            "ü": "u",
            "Ü": "U",
        }
    )

    def __init__(self, config_path: str) -> None:
        self.cfg = load_yaml(config_path)
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
        self.camera_cfg = self.cfg["camera"]
        self.camera_enabled = self.camera_cfg.get("enabled", True)
        self.preview_cfg = self.camera_cfg.get("preview", {})
        self.preview_enabled = bool(self.preview_cfg.get("enabled", False))
        self.preview_window = self.preview_cfg.get("window_name", "OpenCV Camera")
        self.schedule_cfg = self.cfg.get("schedule", {})
        self.tracking_cfg = self.cfg.get("tracking", {})
        self.ui_cfg = self.cfg.get("ui", {})
        self.preprocess_cfg = self.cfg.get("preprocess", {"margin": 0.25, "out_size": 112})
        self.match_cfg = self.cfg.get("match", {"threshold": 0.45, "topk": 1})
        self.match_threshold = float(self.match_cfg.get("threshold", 0.45))
        self.match_topk = max(1, int(self.match_cfg.get("topk", 1)))
        self.match_margin = max(0.0, float(self.match_cfg.get("margin", 0.0)))
        self.confirmation_frames = max(1, int(self.match_cfg.get("confirmation_frames", 1)))
        self.confirm_on_detector_only = bool(self.match_cfg.get("confirm_on_detector_only", False))
        self.match_detector_frames_only = bool(self.match_cfg.get("match_detector_frames_only", True))
        self.min_face_px = max(0, int(self.match_cfg.get("min_face_px", 0)))
        self.avg_window = max(1, int(self.match_cfg.get("avg_window", 1)))
        self.flip_tta = bool(self.match_cfg.get("flip_tta", False))
        self.relock_after = max(0, int(self.match_cfg.get("relock_after", 0)))
        quality_cfg = self.cfg.get("quality", {}) or {}
        self.blur_var_min = max(0.0, float(quality_cfg.get("blur_var_min", 0.0)))
        self.detect_every_n = max(1, int(self.schedule_cfg.get("detect_every_n", 5)))
        self.match_every_n = max(1, int(self.schedule_cfg.get("match_every_n", 1)))
        self.save_face_every_n = max(1, int(self.schedule_cfg.get("save_face_every_n", 1)))
        self.event_every_n = max(1, int(self.schedule_cfg.get("event_every_n", 1)))
        self.log_every_n = max(1, int(self.schedule_cfg.get("log_every_n", 1)))
        self.max_faces_per_frame = max(1, int(self.schedule_cfg.get("max_faces_per_frame", 5)))
        max_time_lost = int(self.tracking_cfg.get("max_time_lost", max(10, self.detect_every_n * 2)))
        min_iou = float(self.tracking_cfg.get("min_iou", 0.2))
        self.preflip_box_text = bool(self.tracking_cfg.get("preflip_box_text", True))
        self.overlay_font_size = max(10, int(self.tracking_cfg.get("overlay_font_size", 16)))
        self.overlay_font_path = str(self.tracking_cfg.get("overlay_font_path", "")).strip()
        self._overlay_font = self._load_overlay_font()
        self.face_tracker = MultiFaceByteTracker(
            max_time_lost=max_time_lost, min_iou=min_iou, max_tracks=self.max_faces_per_frame
        )
        self._frame_idx = 0
        self._track_states: Dict[int, _TrackMatchState] = {}
        self.suspicious_hold_seconds = max(0.0, float(self.ui_cfg.get("suspicious_hold_seconds", 8)))
        self._last_suspicious_event_at = 0.0

        paths_cfg = self.cfg["paths"]
        self.repo_root = Path(__file__).resolve().parents[2]
        self.public_root = self.repo_root / paths_cfg["public_root"]
        self.persons_root = self.public_root / "gallery" / "persons"
        self.capture_store = CaptureStore(public_root=paths_cfg["public_root"], captures_subdir=paths_cfg["captures_subdir"])
        self.events_store = EventsStore(events_path=paths_cfg["events_file"])

        models_cfg = self.cfg["models"]
        detector_cfg = models_cfg["detector"]
        self.detector = RetinaFaceDetector(
            weights_path=detector_cfg["weights_path"],
            device=detector_cfg.get("device", "cpu"),
            min_score=detector_cfg.get("min_score", 0.8),
            det_size=tuple(detector_cfg.get("det_size", [640, 640])),
            detector_type=detector_cfg.get("type"),
        )
        self.embedder = AdaFaceONNXEmbedder(
            onnx_path=models_cfg["embedder"]["onnx_path"],
            device=models_cfg["embedder"].get("device", "cpu"),
            input_size=models_cfg["embedder"].get("input_size", 112),
            normalize=True,
        )

        gallery_idx_cfg = self.cfg["gallery_index"]
        self.index_dir = Path(gallery_idx_cfg["dir"])
        self.matcher: Optional[FaissMatcher] = None
        index_path = self.index_dir / gallery_idx_cfg["index_file"]
        names_path = self.index_dir / gallery_idx_cfg["names_file"]
        if index_path.exists() and names_path.exists():
            self.matcher = FaissMatcher(str(index_path), str(names_path), normalize=True)
        else:
            logger.warning("FAISS index or names file missing; matching disabled")
        self.gallery_map = load_gallery_map(self.index_dir / gallery_idx_cfg["gallery_file"])
        self.gallery_norm_map = {normalize_person_id(pid): entry for pid, entry in self.gallery_map.items()}

    def _load_overlay_font(self):
        if ImageFont is None:
            logger.warning("Pillow unavailable; OpenCV text renderer will be used for box labels")
            return None
        candidates: List[Path] = []
        if self.overlay_font_path:
            candidates.append(Path(self.overlay_font_path))
        candidates.extend(
            [
                Path("C:/Windows/Fonts/segoeui.ttf"),
                Path("C:/Windows/Fonts/arial.ttf"),
                Path("C:/Windows/Fonts/tahoma.ttf"),
                Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
                Path("/Library/Fonts/Arial Unicode.ttf"),
            ]
        )
        seen = set()
        for font_path in candidates:
            key = str(font_path).lower()
            if key in seen:
                continue
            seen.add(key)
            if not font_path.exists():
                continue
            try:
                font = ImageFont.truetype(str(font_path), self.overlay_font_size, encoding="unic")
                logger.info("Overlay label font loaded: %s", font_path)
                return font
            except Exception:
                continue
        logger.warning("No Unicode-capable TTF font found; falling back to OpenCV text renderer")
        return None

    def _ascii_for_opencv(self, text: str) -> str:
        mapped = text.translate(self._ASCII_FALLBACK_MAP)
        normalized = unicodedata.normalize("NFKD", mapped)
        ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
        return ascii_text or "Bilinmeyen"

    def _camera_source(self):
        ctype = self.camera_cfg["type"]
        if ctype == "webcam":
            return int(self.camera_cfg["webcam"]["device_index"])
        if ctype == "rtsp":
            return self.camera_cfg["rtsp"]["url"]
        if ctype == "http_mjpeg":
            return self.camera_cfg["http_mjpeg"]["url"]
        if ctype == "video_file":
            return self.camera_cfg["video_file"]["path"]
        raise ValueError(f"Unsupported camera type {ctype}")

    def _webcam_backends(self):
        webcam_cfg = self.camera_cfg.get("webcam", {})
        backend = str(webcam_cfg.get("backend", "auto")).strip().lower()
        backends = {
            "dshow": cv2.CAP_DSHOW,
            "msmf": cv2.CAP_MSMF,
            "avfoundation": cv2.CAP_AVFOUNDATION,
        }
        if backend in backends:
            return [(backend, backends[backend]), ("auto", None)]
        if backend != "auto":
            logger.warning("Unknown webcam backend '%s'; falling back to auto", backend)
        if os.name == "nt":
            return [("dshow", cv2.CAP_DSHOW), ("msmf", cv2.CAP_MSMF), ("auto", None)]
        if os.name == "posix":
            return [("avfoundation", cv2.CAP_AVFOUNDATION), ("auto", None)]
        return [("auto", None)]

    def _open_capture(self, source) -> cv2.VideoCapture:
        if not isinstance(source, int):
            return cv2.VideoCapture(source)
        for backend_name, backend_id in self._webcam_backends():
            cap = cv2.VideoCapture(source, backend_id) if backend_id is not None else cv2.VideoCapture(source)
            if cap.isOpened():
                logger.info("Webcam opened with backend %s at index %s", backend_name, source)
                return cap
            cap.release()
        return cv2.VideoCapture(source)

    def _prepare_capture(self, cap: cv2.VideoCapture) -> None:
        decode_cfg = self.camera_cfg.get("decode", {})
        if not decode_cfg.get("enabled", False):
            return
        if decode_cfg.get("width"):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, decode_cfg["width"])
        if decode_cfg.get("height"):
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, decode_cfg["height"])

    def _resize_for_preview(self, frame):
        decode_cfg = self.camera_cfg.get("decode", {})
        preview_cfg = self.preview_cfg or {}
        mirror = bool(preview_cfg.get("mirror", False))
        if not decode_cfg.get("enabled", False):
            return cv2.flip(frame, 1) if mirror else frame
        width = int(decode_cfg.get("width") or 0)
        height = int(decode_cfg.get("height") or 0)
        if width <= 0 or height <= 0:
            return cv2.flip(frame, 1) if mirror else frame
        keep_aspect = preview_cfg.get("keep_aspect")
        if keep_aspect is None:
            keep_aspect = bool(decode_cfg.get("keep_aspect", False))
        if not keep_aspect:
            resized = cv2.resize(frame, (width, height))
            if mirror:
                resized = cv2.flip(resized, 1)
            return resized
        h, w = frame.shape[:2]
        if w == 0 or h == 0:
            return frame
        scale = max(width / w, height / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = cv2.resize(frame, (new_w, new_h))
        x = max(0, (new_w - width) // 2)
        y = max(0, (new_h - height) // 2)
        cropped = resized[y : y + height, x : x + width]
        if mirror:
            cropped = cv2.flip(cropped, 1)
        return cropped

    def _match_embedding(self, embedding: np.ndarray) -> Tuple[Optional[str], float, Optional[str]]:
        """Search the gallery and resolve to a canonical person.

        Returns (canonical_person_id or None, top1_score, raw_person_id or None).
        """
        if not self.matcher:
            return None, 0.0, None
        results = self.matcher.search(embedding, topk=self.match_topk)
        if not results:
            return None, 0.0, None
        person_id, score, raw_id = resolve_match(results, self.match_threshold, self.match_margin)
        if self._frame_idx % self.log_every_n == 0:
            topk_str = "; ".join(f"{name}={s:.3f}" for name, s, _ in results[:5])
            logger.info("match topk [%s] -> resolved=%s score=%.3f", topk_str, person_id, score)
        return person_id, score, raw_id

    def _update_confirmed_match(
        self,
        state: _TrackMatchState,
        candidate_person_id: Optional[str],
        similarity: float,
        eligible_for_confirmation: bool,
        raw_person_id: Optional[str] = None,
    ) -> Optional[str]:
        # Once a track is confirmed, stay locked on that identity for the
        # lifetime of the track (until it leaves the frame and the track is
        # dropped); only the similarity value keeps refreshing.
        if state.confirmed_person_id is not None:
            if candidate_person_id == state.confirmed_person_id and similarity >= self.match_threshold:
                state.confirmed_similarity = similarity
                state.confirmed_raw_id = raw_person_id or state.confirmed_raw_id
                state.contradiction_hits = 0
            elif (
                self.relock_after > 0
                and candidate_person_id
                and similarity >= self.match_threshold + self.match_margin
            ):
                # Escape hatch (off by default): a strong, repeated contradiction
                # unlocks the track and restarts confirmation.
                state.contradiction_hits += 1
                if state.contradiction_hits >= self.relock_after:
                    state.confirmed_person_id = None
                    state.confirmed_raw_id = None
                    state.confirmed_similarity = 0.0
                    state.candidate_person_id = candidate_person_id
                    state.candidate_hits = 1
                    state.contradiction_hits = 0
            return state.confirmed_person_id

        if candidate_person_id and similarity >= self.match_threshold:
            if eligible_for_confirmation:
                if candidate_person_id == state.candidate_person_id:
                    state.candidate_hits += 1
                else:
                    state.candidate_person_id = candidate_person_id
                    state.candidate_hits = 1
                if state.candidate_hits >= self.confirmation_frames:
                    state.confirmed_person_id = candidate_person_id
                    state.confirmed_raw_id = raw_person_id
                    state.confirmed_similarity = similarity
        elif eligible_for_confirmation:
            state.candidate_person_id = None
            state.candidate_hits = 0
        return state.confirmed_person_id

    def _frame_generator(self, stop_event=None):
        source = self._camera_source()
        cap = None
        if self.camera_cfg["type"] == "webcam" and self.camera_cfg["webcam"].get("auto_scan"):
            scan_list = self.camera_cfg["webcam"].get("scan_indices", [source])
            for idx in scan_list:
                cap = self._open_capture(int(idx))
                if cap.isOpened():
                    logger.info("Webcam opened at index %s", idx)
                    break
                cap.release()
                cap = None
        else:
            cap = self._open_capture(source)
            if cap.isOpened():
                logger.info("Webcam opened at fixed index %s", source)

        self._prepare_capture(cap)
        if not cap or not cap.isOpened():
            raise RuntimeError(f"Cannot open camera source {source}")
        decode_cfg = self.camera_cfg.get("decode", {})
        fps_limit = decode_cfg.get("fps_limit")
        try:
            while True:
                if stop_event is not None and getattr(stop_event, "is_set", lambda: False)():
                    break
                ret, frame = cap.read()
                if not ret:
                    if self.camera_cfg["type"] == "video_file" and self.camera_cfg["video_file"].get("loop", False):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break
                yield frame
                if fps_limit:
                    time.sleep(1.0 / max(fps_limit, 1))
        finally:
            cap.release()

    @staticmethod
    def _detection_area(det: Dict) -> float:
        x1, y1, x2, y2 = det.get("bbox", [0, 0, 0, 0])
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def _draw_bbox(
        self,
        frame: np.ndarray,
        bbox: Optional[np.ndarray],
        matched_person_id: Optional[str],
        track_id: Optional[int],
    ) -> None:
        if bbox is None:
            return
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        color = (46, 204, 113) if matched_person_id else (0, 176, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        name = matched_person_id or "Bilinmeyen"
        label = f"ID {track_id if track_id is not None else '-'} | {name}"
        pad_x = 3
        pad_y = 3
        patch = None
        if self._overlay_font is not None and Image is not None and ImageDraw is not None:
            try:
                left, top, right, bottom = self._overlay_font.getbbox(label)
                tw = max(1, int(right - left))
                th = max(1, int(bottom - top))
                patch_w = tw + (pad_x * 2)
                patch_h = th + (pad_y * 2)
                patch = np.zeros((patch_h, patch_w, 3), dtype=np.uint8)
                patch[:, :] = color
                patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(patch_rgb)
                draw = ImageDraw.Draw(pil_img)
                draw.text((pad_x - left, pad_y - top), label, font=self._overlay_font, fill=(0, 0, 0))
                patch = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
            except Exception:
                patch = None
        if patch is None:
            safe_label = self._ascii_for_opencv(label)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.48
            text_thickness = 1
            (tw, th), baseline = cv2.getTextSize(safe_label, font, font_scale, text_thickness)
            patch_w = tw + (pad_x * 2)
            patch_h = th + baseline + (pad_y * 2)
            patch = np.zeros((patch_h, patch_w, 3), dtype=np.uint8)
            patch[:, :] = color
            text_org = (pad_x, patch_h - baseline - pad_y)
            cv2.putText(patch, safe_label, text_org, font, font_scale, (0, 0, 0), text_thickness, cv2.LINE_AA)
        # Output stream/preview is mirrored later; pre-flip text patch so final text stays readable.
        if self.preflip_box_text:
            patch = cv2.flip(patch, 1)
        fh, fw = frame.shape[:2]
        text_top = max(0, y1 - patch_h - 6)
        text_left = max(0, min(x1, fw - 1))
        text_bottom = min(fh, text_top + patch_h)
        text_right = min(fw, text_left + patch_w)
        draw_h = text_bottom - text_top
        draw_w = text_right - text_left
        if draw_h <= 0 or draw_w <= 0:
            return
        frame[text_top:text_bottom, text_left:text_right] = patch[:draw_h, :draw_w]

    def _save_frame_event(self, persons: List[Dict], frame_shape) -> None:
        """Write a single event for the whole frame; ``persons`` holds every track."""
        matched = [p for p in persons if p.get("matchedPersonId")]
        status = "SUSPICIOUS" if matched else "CLEAR"
        if status != "SUSPICIOUS" and self.suspicious_hold_seconds > 0:
            if time.monotonic() - self._last_suspicious_event_at < self.suspicious_hold_seconds:
                return
        best = max(matched, key=lambda p: p.get("similarity", 0.0)) if matched else (persons[0] if persons else {})
        h, w = frame_shape.shape[:2] if hasattr(frame_shape, "shape") else frame_shape[:2]
        payload = {
            "version": 2,
            "timestamp": now_iso(),
            "status": status,
            "captureId": best.get("captureId"),
            "cameraFacePath": best.get("cameraFacePath"),
            "matchedPersonId": best.get("matchedPersonId"),
            "galleryPhotoPath": best.get("galleryPhotoPath"),
            "galleryFace3dPath": best.get("galleryFace3dPath"),
            "similarity": float(best.get("similarity", 0.0)),
            "threshold": self.match_cfg.get("threshold", 0.45),
            "bbox": best.get("bbox"),
            "detectorScore": float(best.get("detectorScore", 0.0)),
            "frameSize": [int(w), int(h)],
            "persons": persons,
        }
        self.events_store.save_event(payload)
        if status == "SUSPICIOUS":
            self._last_suspicious_event_at = time.monotonic()

    def _gallery_entry_for(self, person_id: Optional[str]) -> Dict:
        if not person_id:
            return {}

        entry = dict(self.gallery_map.get(person_id) or self.gallery_norm_map.get(normalize_person_id(person_id)) or {})
        if entry.get("photoPath") and entry.get("face3dPath"):
            return entry

        person_dir = self.persons_root / person_id
        if not person_dir.exists():
            normalized_target = normalize_person_id(person_id)
            try:
                person_dir = next(
                    p for p in self.persons_root.iterdir() if p.is_dir() and normalize_person_id(p.name) == normalized_target
                )
            except StopIteration:
                return entry

        if not entry.get("photoPath"):
            photo = next(iter(sorted(person_dir.glob("photo.*"))), None)
            if photo and photo.is_file():
                entry["photoPath"] = rel_web_path(photo, self.repo_root)
        if not entry.get("face3dPath"):
            glb_path = person_dir / "face.glb"
            if glb_path.exists():
                entry["face3dPath"] = rel_web_path(glb_path, self.repo_root)
        return entry

    def process_frame(self, frame) -> np.ndarray:
        self._frame_idx += 1
        frame_idx = self._frame_idx

        predicted = self.face_tracker.predict(frame.shape)
        run_detector = (not predicted) or (frame_idx % self.detect_every_n == 0)

        # Seed with predicted (tracker-only) boxes; detector matches below overwrite
        # the ones it actually touched this frame.
        tracks: Dict[int, Tuple[np.ndarray, Optional[Dict]]] = {
            track_id: (bbox, None) for track_id, bbox in predicted.items()
        }

        if run_detector:
            detections = self.detector.detect(frame)
            detections.sort(key=self._detection_area, reverse=True)
            detections = detections[: self.max_faces_per_frame]
            matched = self.face_tracker.update(detections, frame.shape)
            tracks.update(matched)

        # Drop match-state for tracks that no longer exist.
        for stale_id in [tid for tid in self._track_states if tid not in tracks]:
            del self._track_states[stale_id]

        if not tracks:
            if frame_idx % self.event_every_n == 0:
                self._save_frame_event([], frame)
            return frame

        frame_persons: List[Dict] = []
        for track_id, (bbox, detection) in tracks.items():
            state = self._track_states.get(track_id)
            if state is None:
                state = self._track_states[track_id] = _TrackMatchState(
                    emb_window=deque(maxlen=self.avg_window)
                )
            updated_by_detector = detection is not None
            det_score = float(detection.get("score", 0.0)) if detection is not None else state.last_det_score
            if updated_by_detector:
                state.last_det_score = det_score

            track_detection = {
                "bbox": [float(v) for v in bbox],
                "score": det_score,
                "kps": detection.get("kps") if detection else None,
            }
            face_crop = extract_face(
                frame,
                track_detection,
                margin=self.preprocess_cfg.get("margin", 0.25),
                out_size=self.preprocess_cfg.get("out_size", 112),
                align=self.preprocess_cfg.get("align", False),
            )

            has_kps = updated_by_detector and track_detection.get("kps") is not None
            x1f, y1f, x2f, y2f = [float(v) for v in bbox]
            bbox_side = min(x2f - x1f, y2f - y1f)
            can_embed = can_embed_face(
                has_kps=has_kps,
                bbox_side=bbox_side,
                detector_frames_only=self.match_detector_frames_only,
                min_face_px=self.min_face_px,
            )
            if can_embed and self.blur_var_min > 0:
                can_embed = blur_score(face_crop) >= self.blur_var_min

            candidate_person_id: Optional[str] = None
            candidate_raw_id: Optional[str] = None
            matched_person_id: Optional[str] = None
            similarity = 0.0
            # New tracks match on their first eligible frame (detections can
            # start off-modulo); afterwards the match_every_n cadence applies.
            match_attempted = can_embed and (
                frame_idx % self.match_every_n == 0 or state.emb_count == 0
            )
            if match_attempted:
                embedding = self.embedder.embed(face_crop)
                if self.flip_tta:
                    embedding = embedding + self.embedder.embed(cv2.flip(face_crop, 1))
                    embedding = embedding / (np.linalg.norm(embedding) + 1e-12)
                state.emb_window.append(embedding)
                state.emb_count += 1
                query = embedding
                if len(state.emb_window) > 1:
                    query = np.mean(np.stack(state.emb_window, axis=0), axis=0)
                    query = query / (np.linalg.norm(query) + 1e-12)
                person_id, score, raw_id = self._match_embedding(query)
                similarity = score
                if person_id:
                    candidate_person_id = person_id
                    candidate_raw_id = raw_id

            if match_attempted:
                eligible_for_confirmation = (not self.confirm_on_detector_only) or updated_by_detector
                matched_person_id = self._update_confirmed_match(
                    state,
                    candidate_person_id=candidate_person_id,
                    similarity=similarity,
                    eligible_for_confirmation=eligible_for_confirmation,
                    raw_person_id=candidate_raw_id,
                )
            else:
                matched_person_id = state.confirmed_person_id
                similarity = state.confirmed_similarity

            camera_face_path = None
            capture_id = None
            if matched_person_id is None:
                # reset so the next valid match gets captured
                state.last_saved_person_id = None
            elif frame_idx % self.save_face_every_n == 0 and matched_person_id != state.last_saved_person_id:
                capture_id, camera_face_path = self.capture_store.save_face(face_crop)
                state.last_saved_person_id = matched_person_id

            status = "SUSPICIOUS" if matched_person_id else "CLEAR"
            self._draw_bbox(
                frame,
                bbox=bbox,
                matched_person_id=matched_person_id,
                track_id=track_id,
            )
            gallery_entry = self._gallery_entry_for(matched_person_id) if matched_person_id else {}
            if (
                matched_person_id
                and not gallery_entry.get("photoPath")
                and state.confirmed_raw_id
                and state.confirmed_raw_id != matched_person_id
            ):
                # Canonical id has no assets on disk (only a "Name (2)" dir
                # exists); fall back to the raw gallery row that matched.
                gallery_entry = self._gallery_entry_for(state.confirmed_raw_id) or gallery_entry
            frame_persons.append(
                {
                    "trackId": track_id,
                    "matchedPersonId": matched_person_id,
                    "similarity": float(similarity),
                    "detectorScore": float(det_score),
                    "bbox": [float(v) for v in bbox],
                    "captureId": capture_id,
                    "cameraFacePath": camera_face_path,
                    "galleryPhotoPath": gallery_entry.get("photoPath"),
                    "galleryFace3dPath": gallery_entry.get("face3dPath"),
                }
            )
            if matched_person_id or frame_idx % self.log_every_n == 0:
                logger.info(
                    "[%s][%s] track=%s det=%.3f sim=%.3f matched=%s",
                    status,
                    "DET" if updated_by_detector else "TRK",
                    track_id,
                    det_score,
                    similarity,
                    matched_person_id,
                )

        any_matched = any(p["matchedPersonId"] for p in frame_persons)
        if any_matched or frame_idx % self.event_every_n == 0:
            self._save_frame_event(frame_persons, frame)
        return frame

    def run(self) -> None:
        if not self.camera_enabled:
            logger.warning("Camera disabled in config; exiting pipeline")
            return

        for frame in self._frame_generator():
            self.process_frame(frame)
