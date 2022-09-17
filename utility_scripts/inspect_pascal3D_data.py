import os
import h5py
import argparse


import matplotlib.pyplot as plt
import open3d as o3d


def main(opt):
    with h5py.File(opt.src_file, "r") as hf:
        imgs_cropped = hf["Images_cropped"][:]
        names = hf["Names"][:]
        dt = hf["dt"][:]
        ids = hf["id"][:]
        masks_cropped = hf["masks_cropped"][:]
        points = hf["points"][:]
        poses = hf["poses"][:]
        rgbs_cropped = hf["rgbs_cropped"][:]

    for ii in range(imgs_cropped.shape[0]):
        pose = poses[ii]
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[ii]))
        pcd.paint_uniform_color([1, 0, 0])
        pcd.transform(pose)

        camera = [0, 0, 0]
        radius = 300
        _, pt_map = pcd.hidden_point_removal(camera, radius)
        pcd = pcd.select_by_index(pt_map)

        axis = o3d.geometry.TriangleMesh.create_coordinate_frame()
        o3d.visualization.draw_geometries([axis, pcd])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src_file", type=str, required=True, help="source file to inspect")
    opt = p.parse_args()
    main(opt)
