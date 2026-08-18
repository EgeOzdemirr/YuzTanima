import logging
import os
import site
from pathlib import Path

import onnxruntime as ort

logger = logging.getLogger(__name__)
_DLL_HANDLES = []
_PATHS_ADDED = set()


def _add_nvidia_dll_dirs() -> None:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    for site_dir in site.getsitepackages():
        nvidia_dir = Path(site_dir) / "nvidia"
        if not nvidia_dir.exists():
            continue
        for bin_dir in nvidia_dir.glob("*/bin"):
            if not bin_dir.is_dir():
                continue
            bin_dir_str = str(bin_dir)
            if bin_dir_str.lower() not in _PATHS_ADDED:
                os.environ["PATH"] = bin_dir_str + os.pathsep + os.environ.get("PATH", "")
                _PATHS_ADDED.add(bin_dir_str.lower())
            try:
                _DLL_HANDLES.append(os.add_dll_directory(bin_dir_str))
                logger.info("Added NVIDIA DLL directory: %s", bin_dir)
            except OSError as exc:
                logger.warning("Failed to add NVIDIA DLL directory %s: %s", bin_dir, exc)


def preload_cuda_dlls(device: str) -> None:
    if str(device).lower() == "cpu":
        return
    _add_nvidia_dll_dirs()
    preload = getattr(ort, "preload_dlls", None)
    if preload is None:
        return
    try:
        preload(directory="")
        logger.info("ONNX Runtime CUDA DLL preload completed")
    except Exception as exc:
        logger.warning("ONNX Runtime CUDA DLL preload failed: %s", exc)
