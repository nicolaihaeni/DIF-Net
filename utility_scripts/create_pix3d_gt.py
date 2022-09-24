import os
import json
import h5py
import argparse
import numpy as np

import matplotlib.pyplot as plt
import open3d as o3d


base_path = "/home/nicolai/phd/data/pix3d/"
filenames = sorted(os.listdir(os.path.join(base_path, "chair_clean_test")))

for filename in filenames:
    data = np.load(
        os.path.join(base_path, "chair_clean_test", filename), allow_pickle=True
    ).item()

    w_T_cam = np.linalg.inv(data["cam_T_w"])
    cam_center = w_T_cam[:3, -1]

    gt_points = data["pts"]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt_points))
    pcd.paint_uniform_color([1, 0, 0])

    # Remove hidden points
    partial_pcd = o3d
    _, pt_map = pcd.hidden_point_removal(cam_center, 300)
    partial_pcd = pcd.select_by_index(pt_map)
    partial_pcd.paint_uniform_color([0, 1, 0])

    with h5py.File(os.path.join(base_path, filename.split(".")[0] + ".h5"), "w") as hf:
        hf.create_dataset(
            "partial_pts",
            data=np.array(partial_pcd.points),
            dtype="f",
            compression="gzip",
        )
        hf.create_dataset(
            "gt_pts", data=np.array(pcd.points), dtype="f", compression="gzip"
        )
