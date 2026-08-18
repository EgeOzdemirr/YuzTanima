import time
import uuid
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


class CaptureStore:
    def __init__(self, public_root: str, captures_subdir: str = "captures") -> None:
        self.public_root = Path(public_root)
        self.captures_root = self.public_root / captures_subdir
        self.captures_root.mkdir(parents=True, exist_ok=True)

    def _new_capture_id(self) -> str:
        return f"{int(time.time())}_{uuid.uuid4().hex[:8]}"

    def save_face(self, face_bgr: np.ndarray, capture_id: str = None) -> Tuple[str, str]:
        capture_id = capture_id or self._new_capture_id()
        out_dir = self.captures_root / capture_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "face.jpg"
        ok = cv2.imwrite(str(out_path), face_bgr)
        if not ok:
            raise IOError(f"Failed to save face to {out_path}")
        rel_path = f"/{out_path.relative_to(self.public_root.parent)}"
        return capture_id, rel_path
