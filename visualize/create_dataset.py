import os
import json
import argparse
import numpy as np

import trimesh
from skimage import measure

from mesh_to_sdf import mesh_to_voxels


def normalize_mesh(mesh):
    bbox = mesh.bounding_box.bounds
    extents = mesh.extents

    # Compute location and scale
    loc = (bbox[0] + bbox[1]) / 2
    scale = np.sqrt(extents[0] ** 2 + extents[1] ** 2 + extents[2] ** 2)

    # Transform input mesh to center of bounding box and normalize to
    # unit diagonal
    try:
        mesh.apply_translation(-loc)
        mesh.apply_scale(1 / scale)
    except Exception:
        mesh.vertices = mesh.vertices - loc
        mesh.vertices = mesh.vertices / scale
    return mesh


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src_dir", type=str, required=True, help="source data directory.")
    p.add_argument("--out_dir", type=str, default="./data", help="output directory.")
    p.add_argument(
        "--split_file", type=str, default="", help="split file to generate data for."
    )
    p.add_argument(
        "--modes", type=str, default=None, help="mode to generate in [train/test/val]."
    )

    opt = p.parse_args()

    if opt.modes is None:
        modes = ["train", "test"]
    else:
        modes = opt.modes

    # Read split file data
    with open(opt.split_file, "r") as infile:
        split_data = json.load(infile)

    corrected_split = {}
    for mode in modes:
        if mode in split_data:
            mode_split = split_data[mode]
        else:
            continue

        for cat in mode_split:
            corrected_list = []
            for name in mode_split[cat]:
                # Load the mesh in question
                mesh = trimesh.load(
                    os.path.join(
                        opt.src_dir, cat, name, "models", "model_normalized.obj"
                    )
                )

                # Normalize the mesh to unit diagonal
                mesh = normalize_mesh(mesh)

                # Now voxelize and run marching cubes to get watertight mesh
                try:
                    print("Voxelizing input mesh...")
                    voxels = mesh_to_voxels(
                        mesh, 256, pad=True, sign_method="depth", check_result=True
                    )
                except Exception as e:
                    print(f"Mesh {name} in category {cat} did not work. Removing...")
                    continue
                corrected_list.append(name)

                # Run marching cubes to get a new, correct mesh
                vertices, faces, normals, _ = measure.marching_cubes_lewiner(
                    voxels, level=0
                )
                mesh = trimesh.Trimesh(
                    vertices=vertices, faces=faces, vertex_normals=normals
                )
                mesh.show()

        corrected_split[mode][cat] = corrected_list

    # Write the corrected list to a file
    with open(opt.split_file, "w") as outfile:
        json.dump(corrected_split)
