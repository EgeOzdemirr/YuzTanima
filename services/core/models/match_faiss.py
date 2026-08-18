import json
import os
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np


def _get_short_path(path: str) -> str:
    if os.name != "nt":
        return ""
    try:
        import ctypes
        from ctypes import wintypes
    except Exception:
        return ""
    try:
        get_short = ctypes.windll.kernel32.GetShortPathNameW
        get_short.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
        get_short.restype = wintypes.DWORD
        buf = ctypes.create_unicode_buffer(260)
        size = get_short(path, buf, len(buf))
        if size == 0:
            return ""
        if size >= len(buf):
            buf = ctypes.create_unicode_buffer(size + 1)
            size = get_short(path, buf, len(buf))
        return buf.value if size > 0 else ""
    except Exception:
        return ""


def _faiss_path(path: Path, for_write: bool) -> str:
    if os.name != "nt":
        return str(path)
    if for_write:
        parent_short = _get_short_path(str(path.parent))
        if parent_short:
            return str(Path(parent_short) / path.name)
        return str(path)
    short_path = _get_short_path(str(path))
    return short_path or str(path)


def build_index(embeddings: np.ndarray, names: List[str], index_path: str, normalize: bool = True) -> None:
    vectors = embeddings.astype(np.float32)
    if normalize:
        faiss.normalize_L2(vectors)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    index_file = Path(index_path)
    index_file.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, _faiss_path(index_file, for_write=True))


class FaissMatcher:
    def __init__(self, index_path: str, names_path: str, normalize: bool = True) -> None:
        self.index_path = index_path
        self.names_path = names_path
        self.normalize = normalize
        self.index = faiss.read_index(_faiss_path(Path(index_path), for_write=False))
        with open(names_path, "r") as f:
            self.names: List[str] = json.load(f)

    def search(self, embedding: np.ndarray, topk: int = 1) -> List[Tuple[str, float, int]]:
        vector = embedding.astype(np.float32)
        if self.normalize:
            faiss.normalize_L2(vector.reshape(1, -1))
        scores, ids = self.index.search(np.expand_dims(vector, axis=0), topk)
        results: List[Tuple[str, float, int]] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            person_id = self.names[idx] if idx < len(self.names) else None
            results.append((person_id, float(score), int(idx)))
        return results
