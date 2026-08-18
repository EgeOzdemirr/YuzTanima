"""Render a face.glb avatar from multiple synthetic yaw/pitch angles using Blender
headless. Run via: blender --background --python render_multiview.py -- --glb <path> --out <dir>

The mesh is rendered with an unlit Emission material driven by its baked vertex
colors, so the output faithfully reflects the reconstructed texture regardless of
scene lighting - it just needs to look like a "photo" of the face from a new angle
for the face detector/embedder to pick up on the pose variation.
"""
import math
import sys
from pathlib import Path

import bpy


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    args = {}
    key = None
    for token in argv:
        if token.startswith("--"):
            key = token[2:]
            args[key] = True
        elif key is not None:
            args[key] = token
            key = None
    return args


def find_mesh_object():
    for obj in bpy.data.objects:
        if obj.type == "MESH":
            return obj
    raise RuntimeError("No mesh object found after import")


def assign_vertex_color_emission_material(obj):
    mesh = obj.data
    if not mesh.color_attributes:
        raise RuntimeError("Mesh has no vertex color attributes")
    color_attr_name = mesh.color_attributes[0].name

    mat = bpy.data.materials.new(name="VertexColorEmission")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    attr_node = nodes.new("ShaderNodeVertexColor")
    attr_node.layer_name = color_attr_name
    emission_node = nodes.new("ShaderNodeEmission")
    output_node = nodes.new("ShaderNodeOutputMaterial")

    links.new(attr_node.outputs["Color"], emission_node.inputs["Color"])
    links.new(emission_node.outputs["Emission"], output_node.inputs["Surface"])

    obj.data.materials.clear()
    obj.data.materials.append(mat)


def setup_camera(mesh_obj, distance_scale=2.4):
    bbox_corners = [mesh_obj.matrix_world @ v.co for v in mesh_obj.data.vertices]
    xs = [v.x for v in bbox_corners]
    ys = [v.y for v in bbox_corners]
    zs = [v.z for v in bbox_corners]
    center = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, (min(zs) + max(zs)) / 2)
    radius = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)) / 2
    distance = max(radius * distance_scale, 0.5)

    cam_data = bpy.data.cameras.new("RenderCam")
    cam_obj = bpy.data.objects.new("RenderCam", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    empty = bpy.data.objects.new("LookTarget", None)
    empty.location = center
    bpy.context.collection.objects.link(empty)

    track = cam_obj.constraints.new(type="TRACK_TO")
    track.target = empty
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"

    return cam_obj, empty, center, distance


def position_camera(cam_obj, center, distance, yaw_deg, pitch_deg):
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    x = center[0] + distance * math.sin(yaw) * math.cos(pitch)
    y = center[1] - distance * math.cos(yaw) * math.cos(pitch)
    z = center[2] + distance * math.sin(pitch)
    cam_obj.location = (x, y, z)


def main():
    args = parse_args()
    glb_path = Path(args["glb"]).resolve()
    out_dir = Path(args["out"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    angles_raw = args.get("angles", "0,0")
    angle_pairs = []
    for pair in angles_raw.split(";"):
        yaw_s, pitch_s = pair.split(",")
        angle_pairs.append((float(yaw_s), float(pitch_s)))

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(glb_path))

    mesh_obj = find_mesh_object()
    assign_vertex_color_emission_material(mesh_obj)
    cam_obj, empty, center, distance = setup_camera(mesh_obj)

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT" if hasattr(bpy.types, "BLENDER_EEVEE_NEXT") or True else "BLENDER_EEVEE"
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except Exception:
        scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.film_transparent = False
    scene.world = bpy.data.worlds.new("World") if scene.world is None else scene.world
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs[0].default_value = (0.5, 0.5, 0.5, 1.0)

    for i, (yaw, pitch) in enumerate(angle_pairs):
        position_camera(cam_obj, center, distance, yaw, pitch)
        out_path = out_dir / f"view_{i:02d}_y{int(yaw)}_p{int(pitch)}.png"
        scene.render.filepath = str(out_path)
        bpy.ops.render.render(write_still=True)
        print(f"Rendered {out_path}")


if __name__ == "__main__":
    main()
