import logging
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort

from services.core.models.onnxruntime_helpers import preload_cuda_dlls

logger = logging.getLogger(__name__)


class AdaFaceONNXEmbedder:
    def __init__(self, onnx_path: str, device: str = "cpu", input_size: int = 112, normalize: bool = True) -> None:
        preload_cuda_dlls(device)
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device != "cpu" else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = input_size
        self.normalize = normalize
        logger.info("Loaded AdaFace ONNX model from %s with providers=%s", onnx_path, self.session.get_providers())

    def preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        image_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, (self.input_size, self.input_size))
        image_rgb = image_rgb.astype(np.float32)
        image_rgb = (image_rgb - 127.5) / 128.0
        chw = np.transpose(image_rgb, (2, 0, 1))
        return np.expand_dims(chw, axis=0)

    def embed(self, face_bgr: np.ndarray) -> np.ndarray:
        input_tensor = self.preprocess(face_bgr)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        embedding = outputs[0][0]
        if self.normalize:
            norm = np.linalg.norm(embedding) + 1e-12
            embedding = embedding / norm
        return embedding.astype(np.float32)
