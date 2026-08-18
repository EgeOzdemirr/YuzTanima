import argparse
import subprocess
from pathlib import Path


def convert_with_blender(obj_path: Path, glb_path: Path, blender_bin: str) -> bool:
    if not blender_bin or not Path(blender_bin).exists():
        return False

    script = (
        "import bpy\n"
        "bpy.ops.wm.read_factory_settings(use_empty=True)\n"
        "try:\n"
        f"    bpy.ops.wm.obj_import(filepath=r'{obj_path}')\n"
        "except AttributeError:\n"
        f"    bpy.ops.import_scene.obj(filepath=r'{obj_path}')\n"
        f"bpy.ops.export_scene.gltf(filepath=r'{glb_path}', export_format='GLB')\n"
    )

    glb_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [blender_bin, "--background", "--python-expr", script],
            check=True,
            capture_output=True,
            timeout=300,
        )
    except Exception:
        return False
    return glb_path.exists()


def convert_with_trimesh(obj_path: Path, glb_path: Path) -> None:
    import trimesh

    mesh = trimesh.load(str(obj_path), process=False, maintain_order=True)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())

    glb_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(glb_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert an OBJ mesh to GLB")
    parser.add_argument("--obj", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--blender", default="")
    args = parser.parse_args()

    obj_path = Path(args.obj).resolve()
    glb_path = Path(args.out).resolve()

    if convert_with_blender(obj_path, glb_path, args.blender):
        print(f"Converted via Blender: {glb_path}")
        return

    convert_with_trimesh(obj_path, glb_path)
    print(f"Converted via trimesh: {glb_path}")


if __name__ == "__main__":
    main()
