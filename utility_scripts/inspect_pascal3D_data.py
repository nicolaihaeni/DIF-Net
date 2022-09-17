import os
import h5py
import argparse


import matplotlib.pyplot as plt
import open3d as o3d


def main(opt):
    ii = 101
    with h5py.File(opt.src_file, "r") as hf:
        imgs_cropped = hf["Images_cropped"][ii]
        names = hf["Names"][ii]
        dt = hf["dt"][ii]
        ids = hf["id"][ii]
        masks_cropped = hf["masks_cropped"][ii]
        points = hf["points"][ii]
        poses = hf["poses"][ii]
        rgbs_cropped = hf["rgbs_cropped"][ii]

    # for ii in range(imgs_cropped.shape[0]):
    # fig, ax = plt.subplots(1, 4)
    plt.imshow(imgs_cropped)
    # ax[1].imshow(dt[ii])
    # ax[2].imshow(masks_cropped[ii])
    # ax[3].imshow(rgbs_cropped[ii])
    plt.show()
    print(names)

    pose = poses
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
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
