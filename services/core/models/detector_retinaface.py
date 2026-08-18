import logging
from pathlib import Path
from typing import Dict, List, Optional

import cv2

from services.core.models.onnxruntime_helpers import preload_cuda_dlls

logger = logging.getLogger(__name__)


class RetinaFaceDetector:
    """
    SCRFD detector wrapper (InsightFace FaceAnalysis). RetinaFace backends removed.
    """

    def __init__(
        self,
        weights_path: str,
        device: str = "cpu",
        min_score: float = 0.55,
        det_name: str = "buffalo_l",
        det_size=(640, 640),
        detector_type: Optional[str] = None,
    ) -> None:
        self.device = device
        self.min_score = min_score
        self.weights_path = weights_path
        self.backend = None
        self.app = None
        self.det_name = det_name
        self.det_size = det_size
        self.detector_type = (detector_type or "").strip().lower()
        self._init_backend()

    def _init_backend(self) -> None:
        if self.detector_type and self.detector_type not in {"insightface", "scrfd", "insightface_scrfd"}:
            raise ValueError(f"Unsupported detector type: {self.detector_type}")

        if self._init_insightface():
            return
        raise RuntimeError("Failed to initialize insightface SCRFD backend")

    def _init_insightface(self) -> bool:
        try:
            preload_cuda_dlls(self.device)
            from insightface.app import FaceAnalysis

            if self.device == "cpu":
                providers = ["CPUExecutionProvider"]
                ctx_id = -1
            elif self.device == "mps":
                providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
                ctx_id = -1
            else:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                ctx_id = 0

            # Modeller proje içinde paketlenmiş; ~/.insightface'e bağımlılığı kaldır.
            bundled_root = str(Path(__file__).resolve().parent / "weights" / "insightface")

            # Load only detection to avoid landmark model errors in some builds
            self.app = FaceAnalysis(name=self.det_name, root=bundled_root, providers=providers, allowed_modules=["detection"])
            self.app.prepare(ctx_id=ctx_id, det_size=self.det_size)
            self.backend = "insightface_scrfd"
            logger.info("Loaded detector via insightface SCRFD backend (%s)", self.det_name)
            return True
        except Exception as exc:
            logger.error("Insightface SCRFD backend unavailable (%s)", exc)
            return False

    def detect(self, image_bgr) -> List[Dict]:
        if self.backend and self.backend.startswith("insightface"):
            return self._detect_insightface(image_bgr)
        raise RuntimeError("Detector backend not initialized")

    def _detect_insightface(self, image_bgr) -> List[Dict]:
        if image_bgr is None:
            return []
        try:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            faces = self.app.get(image_rgb)
        except Exception as exc:
            logger.warning("Insightface detect failed: %s", exc)
            return []
        detections: List[Dict] = []
        for face in faces:
            score = float(getattr(face, "det_score", 0.0))
            if score < self.min_score:
                continue
            bbox = [float(v) for v in face.bbox]
            kps = getattr(face, "kps", None)
            detections.append(
                {
                    "bbox": bbox,
                    "score": score,
                    "kps": [[float(pt[0]), float(pt[1])] for pt in kps] if kps is not None else None,
                }
            )
        return detections
