import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List

import yaml

# Ensure repo root is importable when run as a script
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("build_3d_gallery")


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_photo(person_dir: Path, pattern: str) -> Path:
    matches = sorted(person_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No photo matching {pattern} in {person_dir}")
    return matches[0]


def run_cmd(cmd: list) -> None:
    subprocess.run(cmd, check=True)


def convert_obj_to_glb(obj_path: Path, glb_path: Path, blender_bin: str, assimp_bin: str) -> bool:
    try:
        glb_path.parent.mkdir(parents=True, exist_ok=True)
        if blender_bin:
            run_cmd(
                [
                    sys.executable,
                    str(ROOT / "tools/3d/obj_to_glb.py"),
                    "--obj",
                    str(obj_path),
                    "--out",
                    str(glb_path),
                    "--blender",
                    blender_bin,
                ]
            )
        else:
            run_cmd([assimp_bin, "export", str(obj_path), str(glb_path)])
        logger.info("Converted OBJ -> GLB: %s -> %s", obj_path, glb_path)
        return True
    except Exception as exc:
        logger.warning("OBJ -> GLB conversion failed for %s: %s", obj_path, exc)
        return False


def run_tddfa_avatar(
    image_path: Path,
    outdir: Path,
    tddfa_repo: str,
    device: str,
    detector: str,
    iscrop: int,
    extract_tex: int,
    backbone: str,
    blender_bin: str,
) -> Path:
    cmd = [
        sys.executable,
        str(ROOT / "tools/3d/make_avatar_glb.py"),
        "--image",
        str(image_path),
        "--outdir",
        str(outdir),
        "--tddfa_repo",
        str(tddfa_repo),
        "--device",
        str(device),
        "--iscrop",
        str(iscrop),
        "--detector",
        str(detector),
        "--extractTex",
        str(extract_tex),
        "--backbone",
        str(backbone),
        "--blender",
        str(blender_bin),
    ]
    run_cmd(cmd)
    return outdir / "avatar.glb"


def maybe_resume_after_last_glb(person_dirs: List[Path], glb_name: str, enabled: bool) -> List[Path]:
    if not enabled:
        return person_dirs

    last_index = -1
    for idx, person_dir in enumerate(person_dirs):
        if (person_dir / glb_name).exists():
            last_index = idx

    if last_index < 0:
        logger.info("Resume enabled but no existing %s found; starting from first person.", glb_name)
        return person_dirs

    remaining = person_dirs[last_index + 1 :]
    logger.info(
        "Resume enabled. Last existing %s found at '%s'; continuing from next (%d persons).",
        glb_name,
        person_dirs[last_index].name,
        len(remaining),
    )
    return remaining


def _normalize_name(value: str) -> str:
    # Accent-insensitive matching for manual resume names.
    folded = value.casefold()
    normalized = unicodedata.normalize("NFKD", folded)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _as_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def filter_person_dirs(person_dirs: List[Path], include_person_ids: List[str]) -> List[Path]:
    if not include_person_ids:
        return person_dirs
    wanted = {_normalize_name(person_id) for person_id in include_person_ids}
    filtered = [person_dir for person_dir in person_dirs if _normalize_name(person_dir.name) in wanted]
    logger.info("Person filter enabled: %d of %d persons selected.", len(filtered), len(person_dirs))
    return filtered


def find_person_index_by_name(person_dirs: List[Path], person_name: str) -> int:
    names = [p.name for p in person_dirs]
    if person_name in names:
        return names.index(person_name)

    target_folded = person_name.casefold()
    for idx, name in enumerate(names):
        if name.casefold() == target_folded:
            return idx

    target_normalized = _normalize_name(person_name)
    for idx, name in enumerate(names):
        if _normalize_name(name) == target_normalized:
            return idx

    return -1


def maybe_resume_from_person(person_dirs: List[Path], person_name: str, inclusive: bool) -> List[Path]:
    if not person_name:
        return person_dirs

    anchor_idx = find_person_index_by_name(person_dirs, person_name)
    if anchor_idx < 0:
        logger.warning("Manual resume person not found: %s. Continuing with full list.", person_name)
        return person_dirs

    start_idx = anchor_idx if inclusive else anchor_idx + 1
    remaining = person_dirs[start_idx:]
    logger.info(
        "Manual resume enabled. Anchor '%s' matched to '%s'. Starting from index %d (%d persons).",
        person_name,
        person_dirs[anchor_idx].name,
        start_idx,
        len(remaining),
    )
    return remaining


def process_person(
    person_dir: Path,
    glb_name: str,
    photo_glob: str,
    use_tddfa: bool,
    tddfa_repo: str,
    device: str,
    detector: str,
    iscrop: int,
    extract_tex: int,
    backbone: str,
    blender_bin: str,
    assimp_bin: str,
) -> str:
    glb_path = person_dir / glb_name
    if glb_path.exists():
        return "exists"

    if use_tddfa:
        try:
            photo_path = find_photo(person_dir, pattern=photo_glob)
            avatar_glb = run_tddfa_avatar(
                image_path=photo_path,
                outdir=person_dir,
                tddfa_repo=tddfa_repo,
                device=device,
                detector=detector,
                iscrop=iscrop,
                extract_tex=extract_tex,
                backbone=backbone,
                blender_bin=blender_bin,
            )
            if avatar_glb.exists():
                if avatar_glb != glb_path:
                    shutil.move(str(avatar_glb), str(glb_path))
                logger.info("3DDFA-V3 GLB ready: %s", glb_path)
                return "generated"
        except Exception as exc:
            logger.warning("3DDFA-V3 build failed for %s: %s", person_dir.name, exc)

    candidate_objs = list(person_dir.glob("*.obj"))
    if not candidate_objs:
        candidate_objs = list(person_dir.rglob("*_extractTex.obj"))
    if not candidate_objs:
        candidate_objs = list(person_dir.rglob("*.obj"))
    if candidate_objs:
        if convert_obj_to_glb(candidate_objs[0], glb_path, blender_bin=blender_bin, assimp_bin=assimp_bin):
            return "generated"

    logger.warning(
        "No GLB for %s. If you have 3DDFA-V3 repo at %s you can generate one via `python tools/3d/make_avatar_glb.py`, "
        "or export an OBJ and convert via Blender/assimp.",
        person_dir.name,
        tddfa_repo,
    )
    return "failed"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate or convert GLB assets for gallery")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    build_cfg = cfg.get("build3d", {})
    if not build_cfg.get("enabled", False):
        logger.info("3D build disabled in config")
        return

    repo_root = Path(__file__).resolve().parents[2]
    persons_root = repo_root / cfg["gallery"]["persons_root"]
    glb_name = cfg["gallery"].get("glb_name", "face.glb")
    photo_glob = cfg["gallery"].get("photo_glob", "photo.*")
    assimp_bin = build_cfg.get("assimp_bin", "assimp")
    blender_bin = build_cfg.get("blender_bin", "blender")
    tddfa_repo = build_cfg.get("tddfa_repo", "services/tddfa/repo")
    device = build_cfg.get("device", "cpu")
    detector = build_cfg.get("detector", "scrfd")
    iscrop = int(build_cfg.get("iscrop", 1))
    extract_tex = int(build_cfg.get("extractTex", 1))
    backbone = build_cfg.get("backbone", "resnet50")
    use_tddfa = build_cfg.get("use_tddfa", True)
    resume_from_last_glb = bool(build_cfg.get("resume_from_last_glb", False))
    resume_from_person = str(build_cfg.get("resume_from_person", "")).strip()
    resume_from_person_inclusive = bool(build_cfg.get("resume_from_person_inclusive", True))
    num_workers = max(1, int(build_cfg.get("num_workers", 1)))

    person_dirs = [p for p in sorted(persons_root.iterdir()) if p.is_dir()]
    person_dirs = filter_person_dirs(
        person_dirs,
        _as_list(build_cfg.get("include_person_ids", cfg["gallery"].get("include_person_ids"))),
    )
    if resume_from_person:
        person_dirs = maybe_resume_from_person(
            person_dirs,
            person_name=resume_from_person,
            inclusive=resume_from_person_inclusive,
        )
    else:
        person_dirs = maybe_resume_after_last_glb(person_dirs, glb_name=glb_name, enabled=resume_from_last_glb)

    if not person_dirs:
        logger.info("No person directory left to process.")
        return

    use_parallel = str(device).lower() == "cpu" and num_workers > 1
    if use_parallel:
        logger.info("CPU parallel mode enabled with %d workers.", num_workers)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    process_person,
                    person_dir=person_dir,
                    glb_name=glb_name,
                    photo_glob=photo_glob,
                    use_tddfa=use_tddfa,
                    tddfa_repo=tddfa_repo,
                    device=device,
                    detector=detector,
                    iscrop=iscrop,
                    extract_tex=extract_tex,
                    backbone=backbone,
                    blender_bin=blender_bin,
                    assimp_bin=assimp_bin,
                ): person_dir
                for person_dir in person_dirs
            }
            done_count = 0
            for future in as_completed(futures):
                person_dir = futures[future]
                done_count += 1
                try:
                    result = future.result()
                    if result == "failed":
                        logger.warning("Failed (%d/%d): %s", done_count, len(futures), person_dir.name)
                except Exception as exc:
                    logger.warning("Unhandled failure (%d/%d) for %s: %s", done_count, len(futures), person_dir.name, exc)
    else:
        if num_workers > 1 and str(device).lower() != "cpu":
            logger.info("num_workers=%d ignored because device is not CPU (%s).", num_workers, device)
        for person_dir in person_dirs:
            process_person(
                person_dir=person_dir,
                glb_name=glb_name,
                photo_glob=photo_glob,
                use_tddfa=use_tddfa,
                tddfa_repo=tddfa_repo,
                device=device,
                detector=detector,
                iscrop=iscrop,
                extract_tex=extract_tex,
                backbone=backbone,
                blender_bin=blender_bin,
                assimp_bin=assimp_bin,
            )


if __name__ == "__main__":
    main()
