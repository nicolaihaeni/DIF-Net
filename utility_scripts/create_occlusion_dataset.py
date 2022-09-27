import os
import h5py
import json
import random
import numpy as np
import argparse

from utils import EraserSetter

np.random.seed(1242356)


def main(opt):
    with open(opt.split_file, "r") as infile:
        split_data = json.load(infile)["test"]

    for cat in split_data:
        for filename in split_data[cat]:
            # Load the ground truth file
            with h5py.File(
                os.path.join(opt.data_dir, f"shapenet_{cat}_test", filename + ".h5"),
                "r",
            ) as hf:
                image = hf["image"][:]
                mask = hf["mask"][:]
                depth = hf["depth"][:]
                cam2world = hf["cam2world"][:]

            with h5py.File(
                os.path.join(opt.gt_dir, cat, filename, filename + ".h5"), "r"
            ) as hf:
                gt_points = hf["surface_pts"][:, :3]

            eraser_setter = EraserSetter()

            # Create a random size eraser
            eraser = np.zeros_like(mask)
            h, w = random.randint(30, 150), random.randint(30, 150)
            startx, starty = random.randint(0, 256 - w), random.randint(0, 256 - h)
            eraser[startx : startx + w, starty : starty + h] = 1.0

            erased_mask, ratio = eraser_setter(mask, eraser)

            erased_mask = erased_mask.astype(bool)
            occluded_mask = np.logical_and(mask, ~erased_mask)
            image_occluded = np.ones_like(image) * 255.0
            image_occluded[np.where(occluded_mask)] = image[np.where(occluded_mask)]
            depth_occluded = np.zeros_like(depth)
            depth_occluded[np.where(occluded_mask)] = depth[np.where(occluded_mask)]

            # Lift remaining depth points to 3D
            u, v = np.where(occluded_mask)
            y = depth[u, v] * ((u - 128.0) / 262.5)
            x = depth[u, v] * ((v - 128.0) / 262.5)
            z = depth[u, v]
            pts = np.stack([x, y, z], axis=-1)

            hf = h5py.File(
                os.path.join(
                    opt.data_dir, f"shapenet_{cat}_test", f"{filename}_occluded.h5"
                ),
                "w",
            )
            hf.create_dataset("image", data=image_occluded, dtype="f")
            hf.create_dataset("mask", data=occluded_mask, dtype="f")
            hf.create_dataset("depth", data=depth_occluded, dtype="f")
            hf.create_dataset("partial_points", data=pts, dtype="f", compression="gzip")
            hf.create_dataset(
                "gt_points", data=gt_points, dtype="f", compression="gzip"
            )
            hf.create_dataset("ratio", data=ratio, dtype="f")
            hf.create_dataset("cam2world", data=cam2world, dtype="f")
            hf.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="/home/isleri/shared/test_data/shapenet_occlusion/",
        type=str,
        help="directory for data",
    )
    parser.add_argument(
        "--gt_dir",
        default="/home/isleri/haeni001/data/dif/",
        type=str,
        help="directory for data",
    )
    parser.add_argument(
        "--split_file", required=True, type=str, help="split_file .json"
    )
    opt = parser.parse_args()
    main(opt)
