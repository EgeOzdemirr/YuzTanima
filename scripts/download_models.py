import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


DEFAULT_TDDFA_REPO_URL = "https://github.com/wang-zidu/3DDFA-V3"
DEFAULT_TDDFA_ASSET_BASE = "https://huggingface.co/datasets/Zidu-Wang/3DDFA-V3/resolve/main/assets"
DEFAULT_TDDFA_ASSETS = [
    "face_model.npy",
    "net_recon.pth",
    "similarity_Lm3D_all.mat",
]

DEFAULT_ADAFACE_URL = "https://github.com/yakhyo/adaface-onnx/releases/download/weights/adaface_ir_101.onnx"


def download_file(url: str, dest: Path, force: bool = False) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        print(f"skip: {dest}")
        return
    print(f"download: {url}")
    with urllib.request.urlopen(url) as resp, open(dest, "wb") as f:
        shutil.copyfileobj(resp, f)
    print(f"saved: {dest}")


def git_clone(repo_url: str, dest: Path) -> None:
    if dest.exists() and (dest / ".git").exists():
        print(f"skip clone (exists): {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"git clone: {repo_url} -> {dest}")
    subprocess.run(["git", "clone", repo_url, str(dest)], check=True)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="Download models and 3DDFA-V3 assets.")
    ap.add_argument("--force", action="store_true", help="Redownload files even if present.")
    ap.add_argument("--skip-tddfa", action="store_true", help="Skip 3DDFA-V3 repo/assets.")
    ap.add_argument("--skip-adaface", action="store_true", help="Skip AdaFace ONNX download.")
    ap.add_argument(
        "--tddfa-repo",
        default=str(root / "services" / "tddfa" / "repo" / "3DDFA-V3"),
        help="Path to 3DDFA-V3 checkout.",
    )
    ap.add_argument(
        "--adaface-url",
        default=os.environ.get("ADAFACE_ONNX_URL", DEFAULT_ADAFACE_URL),
        help="AdaFace ONNX URL (env: ADAFACE_ONNX_URL).",
    )
    ap.add_argument(
        "--adaface-path",
        default=str(root / "services" / "core" / "models" / "weights" / "adaface_ir_101.onnx"),
        help="Destination path for AdaFace ONNX.",
    )
    args = ap.parse_args()

    if not args.skip_tddfa:
        tddfa_repo = Path(args.tddfa_repo)
        git_clone(DEFAULT_TDDFA_REPO_URL, tddfa_repo)
        assets_dir = tddfa_repo / "assets"
        for name in DEFAULT_TDDFA_ASSETS:
            url = f"{DEFAULT_TDDFA_ASSET_BASE}/{name}"
            download_file(url, assets_dir / name, force=args.force)

    if not args.skip_adaface:
        adaface_path = Path(args.adaface_path)
        download_file(args.adaface_url, adaface_path, force=args.force)

    print("done")


if __name__ == "__main__":
    main()
