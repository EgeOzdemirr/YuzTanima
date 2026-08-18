import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure repo root is importable when run as a script
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import yaml

from services.core.io.gallery_store import PersonEntry, write_gallery_json, write_names
from services.core.models.detector_retinaface import RetinaFaceDetector
from services.core.models.embed_adaface_onnx import AdaFaceONNXEmbedder
from services.core.models.match_faiss import build_index
from services.core.utils.identity import canonical_person_id, dedupe_embeddings
from services.core.utils.image import load_bgr
from services.core.utils.preprocess import extract_face, pick_largest

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("build_gallery")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def rel_web_path(path: Path, repo_root: Path) -> str:
    return "/" + str(path.relative_to(repo_root).as_posix())


def find_photo(person_dir: Path, pattern: str) -> Path:
    matches = sorted(person_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No photo matching {pattern} in {person_dir}")
    return matches[0]


def _as_list(value, default: Optional[List[str]] = None) -> List[str]:
    if value is None:
        return list(default or [])
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def should_include_person(person_dir: Path, include_person_ids: List[str]) -> bool:
    if not include_person_ids:
        return True
    wanted = {person_id.casefold() for person_id in include_person_ids}
    return person_dir.name.casefold() in wanted


def find_reference_photos(person_dir: Path, gallery_cfg: Dict) -> List[Path]:
    patterns = _as_list(
        gallery_cfg.get("reference_photo_globs"),
        default=[gallery_cfg.get("photo_glob", "photo.*")],
    )
    photos: List[Path] = []
    seen = set()
    for pattern in patterns:
        for photo_path in sorted(person_dir.glob(pattern)):
            resolved = photo_path.resolve()
            if resolved in seen:
                continue
            if not photo_path.is_file() or photo_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            seen.add(resolved)
            photos.append(photo_path)
    if not photos:
        raise FileNotFoundError(f"No reference photos found in {person_dir}")
    return photos


def embed_reference_photo(
    photo_path: Path,
    detector: RetinaFaceDetector,
    embedder: AdaFaceONNXEmbedder,
    margin: float,
    out_size: int,
    align: bool,
    flip_tta: bool = False,
) -> np.ndarray:
    image = load_bgr(str(photo_path))
    detections = detector.detect(image)
    detection = pick_largest(detections)
    if not detection:
        raise RuntimeError(f"No face detected in {photo_path}")
    face = extract_face(
        image,
        detection,
        margin=margin,
        out_size=out_size,
        align=align,
    )
    emb = embedder.embed(face)
    if flip_tta:
        # Average with the horizontally mirrored view for pose robustness.
        emb = emb + embedder.embed(cv2.flip(face, 1))
        emb = emb / (np.linalg.norm(emb) + 1e-12)
    return emb


def build_person_entry(
    person_dir: Path,
    repo_root: Path,
    photo_glob: str,
    glb_name: str,
) -> PersonEntry:
    photo_path = find_photo(person_dir, pattern=photo_glob)
    glb_path = person_dir / glb_name
    return PersonEntry(
        person_id=person_dir.name,
        photo_path=rel_web_path(photo_path, repo_root),
        face3d_path=rel_web_path(glb_path, repo_root) if glb_path.exists() else None,
        index=-1,
    )


def save_arrays(embeddings: np.ndarray, names: List[str], output_dir: Path, embeddings_file: str, names_file: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / embeddings_file, embeddings)
    write_names(names, output_dir / names_file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build gallery embeddings and index")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    gallery_cfg = cfg["gallery"]
    index_cfg = cfg["index"]
    model_cfg = cfg["models"]
    preprocess_cfg = cfg.get("preprocess", {"margin": 0.25, "out_size": 112})

    repo_root = Path(__file__).resolve().parents[2]
    persons_root = repo_root / gallery_cfg["persons_root"]
    detector = RetinaFaceDetector(
        weights_path=model_cfg["detector"]["weights_path"],
        device=model_cfg["detector"].get("device", "cpu"),
        min_score=model_cfg["detector"].get("min_score", 0.8),
        det_size=tuple(model_cfg["detector"].get("det_size", [640, 640])),
        detector_type=model_cfg["detector"].get("type"),
    )
    embedder = AdaFaceONNXEmbedder(
        onnx_path=model_cfg["embedder"]["onnx_path"],
        device=model_cfg["embedder"].get("device", "cpu"),
        input_size=model_cfg["embedder"].get("input_size", 112),
        normalize=index_cfg.get("normalize", True),
    )

    embeddings: List[np.ndarray] = []
    names: List[str] = []
    entries: List[PersonEntry] = []
    include_person_ids = _as_list(gallery_cfg.get("include_person_ids"))
    photo_glob = gallery_cfg.get("photo_glob", "photo.*")
    glb_name = gallery_cfg.get("glb_name", "face.glb")
    augment_cfg = cfg.get("augment", {}) or {}
    flip_tta = bool(augment_cfg.get("flip_tta", False))
    canonical_dirs: Dict[str, List[str]] = {}

    logger.info("Scanning gallery persons in %s", persons_root)
    for person_dir in sorted(persons_root.iterdir()):
        if not person_dir.is_dir():
            continue
        if not should_include_person(person_dir, include_person_ids):
            continue
        try:
            reference_photos = find_reference_photos(person_dir, gallery_cfg)
            entry = build_person_entry(
                person_dir,
                repo_root=repo_root,
                photo_glob=photo_glob,
                glb_name=glb_name,
            )
            entry.index = len(embeddings)
            embedded_count = 0
            for photo_path in reference_photos:
                try:
                    emb = embed_reference_photo(
                        photo_path,
                        detector=detector,
                        embedder=embedder,
                        margin=preprocess_cfg.get("margin", 0.25),
                        out_size=preprocess_cfg.get("out_size", 112),
                        align=preprocess_cfg.get("align", False),
                        flip_tta=flip_tta,
                    )
                    embeddings.append(emb)
                    names.append(entry.person_id)
                    embedded_count += 1
                except Exception as exc:
                    logger.warning("Skipping reference %s: %s", photo_path, exc)
            if embedded_count == 0:
                raise RuntimeError("No usable reference photos")
            entries.append(entry)
            canonical_dirs.setdefault(canonical_person_id(entry.person_id), []).append(person_dir.name)
            logger.info("Processed %s (%d references)", person_dir.name, embedded_count)
        except Exception as exc:
            logger.warning("Skipping %s: %s", person_dir.name, exc)

    if not embeddings:
        logger.error("No embeddings generated; aborting")
        return

    # Warn about the same real person enrolled under multiple directories
    # ("Ad" + "Ad (2)"); runtime canonicalization keeps them from competing,
    # but an on-disk merge is the real fix.
    for cid, dirs in sorted(canonical_dirs.items()):
        if len(dirs) > 1:
            logger.warning("Duplicate identity dirs for '%s': %s", cid, ", ".join(dirs))

    embeddings_arr = np.stack(embeddings, axis=0)

    dedupe_threshold = float(index_cfg.get("dedupe_threshold", 0.0))
    if dedupe_threshold > 0:
        embeddings_arr, names, dropped = dedupe_embeddings(embeddings_arr, names, dedupe_threshold)
        for dropped_row, kept_row in dropped:
            logger.info("Dropped near-duplicate embedding row %d (kept row %d)", dropped_row, kept_row)
        if dropped:
            logger.info("Dedupe removed %d rows; %d remain", len(dropped), len(names))
        # Row indices shifted; point each entry at its first surviving row.
        first_row = {}
        for row, name in enumerate(names):
            first_row.setdefault(name, row)
        for entry in entries:
            entry.index = first_row.get(entry.person_id, -1)

    output_dir = repo_root / index_cfg["output_dir"]
    save_arrays(
        embeddings_arr,
        names,
        output_dir=output_dir,
        embeddings_file=index_cfg["embeddings_file"],
        names_file=index_cfg["names_file"],
    )

    index_path = output_dir / index_cfg["index_file"]
    build_index(embeddings_arr, names, str(index_path), normalize=index_cfg.get("normalize", True))
    gallery_file = output_dir / index_cfg["gallery_file"]
    write_gallery_json(gallery_file, entries, embedding_dim=embeddings_arr.shape[1])
    logger.info("Gallery build complete: %d persons, %d references", len(entries), len(embeddings))


if __name__ == "__main__":
    main()
