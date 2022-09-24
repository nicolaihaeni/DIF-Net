import os
import h5py
import numpy as np
import argparse

import open3d as o3d


np.random.seed(1242356)


def main(opt):
    with open(opt.split_file, "r") as infile:
        split_data = json.load(infile)["test"]
        cat = split_data.keys()[0]

    for filename in split_data[cat]:
        # Load the ground truth file
        with h5py.File(os.path.join(data_dir, filename, filename + ".h5"), r) as hf:
            images = hf["image"][:]
            depths = hf["depth"][:]
            masks = hf["mask"][:]
            poses = hf["c2w"][:]
            gt_points = hf["surface_pts"][:, :3]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", required=True, type=str, help="directory for data")
    parser.add_argument("split_file", required=True, type=str, help="split_file .json")
    opt = parser.parse_args()
    main(opt)
