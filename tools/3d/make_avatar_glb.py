import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct a 3D face avatar (GLB) from a single photo using 3DDFA-V3")
    parser.add_argument("--image", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--tddfa_repo", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--iscrop", type=int, default=1)
    parser.add_argument("--detector", default="scrfd")
    parser.add_argument("--extractTex", type=int, default=1)
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument("--blender", default="")
    return parser.parse_args()


def resolve_repo_dir(tddfa_repo: str) -> Path:
    repo_root = Path(tddfa_repo).resolve()
    nested = repo_root / "3DDFA-V3"
    if (nested / "demo.py").exists():
        return nested
    return repo_root


def build_scrfd_detector(project_root: str):
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from services.core.models.detector_retinaface import RetinaFaceDetector

    weights_path = Path(project_root) / "services/core/models/weights/insightface/models/buffalo_l/det_10g.onnx"
    return RetinaFaceDetector(weights_path=str(weights_path), device="cpu", min_score=0.4)


def _largest_face(faces):
    def area(face: dict) -> float:
        x1, y1, x2, y2 = face["bbox"]
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    return max(faces, key=area)


def detect_5point_landmarks(detector, im: Image.Image) -> np.ndarray:
    """Return 5-point landmarks in 3DDFA-V3's expected coordinate convention
    (y flipped, i.e. bottom-up), largest face only.

    Extreme close-up crops (face filling most of the frame) are outside
    SCRFD's trained face-size distribution and often yield zero detections.
    As a fallback, retry on a padded copy of the image (shrinks the face
    relative to the frame) and map the result back to original coordinates.
    """
    img_bgr = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    height = img_bgr.shape[0]
    faces = detector.detect(img_bgr)
    pad = 0

    if not faces:
        pad = int(max(img_bgr.shape[:2]) * 0.3)
        padded = cv2.copyMakeBorder(img_bgr, pad, pad, pad, pad, cv2.BORDER_REFLECT)
        faces = detector.detect(padded)
        if not faces:
            raise RuntimeError("No face detected in image")

    face = _largest_face(faces)
    kps = face.get("kps")
    if kps is None:
        raise RuntimeError("Detector did not return 5-point landmarks")

    if pad:
        kps = [[pt[0] - pad, pt[1] - pad] for pt in kps]

    landmarks = np.array(kps, dtype=np.float32)
    landmarks[:, 1] = height - 1 - landmarks[:, 1]
    return landmarks


def main() -> None:
    args = parse_args()
    project_root = str(Path(__file__).resolve().parents[2])
    image_path = Path(args.image).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    tddfa_dir = resolve_repo_dir(args.tddfa_repo)
    if not tddfa_dir.exists():
        raise FileNotFoundError(f"3DDFA-V3 repo not found: {tddfa_dir}")

    scrfd_detector = None
    if args.iscrop:
        if args.detector != "scrfd":
            raise ValueError(f"Unsupported detector '{args.detector}'. Only 'scrfd' is wired up.")
        scrfd_detector = build_scrfd_detector(project_root)

    if str(tddfa_dir) not in sys.path:
        sys.path.insert(0, str(tddfa_dir))

    old_cwd = os.getcwd()
    os.chdir(tddfa_dir)
    try:
        from model.recon import face_model
        from util.io import write_obj_with_colors
        from util.preprocess import align_img, load_lm3d

        class ModelArgs:
            device = args.device
            backbone = args.backbone
            extractTex = bool(args.extractTex)
            useTex = False
            ldm68 = False
            ldm106 = False
            ldm106_2d = False
            ldm134 = False
            seg = False
            seg_visible = False

        recon_model = face_model(ModelArgs())

        im = Image.open(image_path).convert("RGB")

        if args.iscrop:
            landmarks = detect_5point_landmarks(scrfd_detector, im)
            lm3d_std = load_lm3d()
            _, im_new, _, _ = align_img(im, landmarks, lm3d_std)
            im_tensor = torch.tensor(np.array(im_new) / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        else:
            if im.size != (224, 224):
                im = im.resize((224, 224))
            im_tensor = torch.tensor(np.array(im) / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

        recon_model.input_img = im_tensor.to(args.device)
        results = recon_model.forward()

        v3d = results["v3d"][0].copy()
        v3d[..., -1] = 10 - v3d[..., -1]
        tri = results["tri"]

        if args.extractTex:
            colors = np.clip(results["extractTex"], 0, 1)
        else:
            colors = np.clip(results["face_texture"][0], 0, 1)

        obj_path = outdir / "avatar_extractTex.obj"
        write_obj_with_colors(str(obj_path), v3d, tri, colors)
    finally:
        os.chdir(old_cwd)

    import trimesh

    glb_path = outdir / "avatar.glb"
    colors_u8 = (colors * 255).astype(np.uint8)
    mesh = trimesh.Trimesh(vertices=v3d, faces=tri, vertex_colors=colors_u8, process=False)
    mesh.export(str(glb_path))
    print(f"Saved {glb_path}")


if __name__ == "__main__":
    main()
