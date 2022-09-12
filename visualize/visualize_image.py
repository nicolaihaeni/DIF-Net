import os
import h5py
import imageio
import numpy as np

import cv2
import skimage.exposure
import matplotlib.pyplot as plt

path = "../example_dir/eb43db95d804f40d66cf1b4a8fc3914e/eb43db95d804f40d66cf1b4a8fc3914e_rgbd.h5"

with h5py.File(path, "r") as hf:
    images = hf["rgb"][:]
    depths = hf["depth"][:]
    masks = hf["mask"][:]

for ii in range(images.shape[0]):
    img = images[ii]
    depth = depths[ii]
    depth[depth == 10.0] = 4.0
    mask = masks[ii] * 255.0

    img = np.concatenate([img, mask[..., None]], axis=-1)

    stretch = skimage.exposure.rescale_intensity(
        depth, in_range="image", out_range=(0, 255)
    ).astype(np.uint8)

    # depth_min = depth.min()
    # depth_max = depth.max()

    # max_val = (2 ** (8 * 1)) - 1

    # out = max_val * (depth - depth_min) / (depth_max - depth_min)

    # convert to 3 channels
    stretch = cv2.merge([stretch, stretch, stretch])

    # define colors
    color1 = (0, 0, 255)  # red
    color2 = (0, 165, 255)  # orange
    color3 = (0, 255, 255)  # yellow
    color4 = (255, 255, 0)  # cyan
    color5 = (255, 0, 0)  # blue
    color6 = (128, 64, 64)  # violet
    colorArr = np.array(
        [[color1, color2, color3, color4, color5, color6]], dtype=np.uint8
    )

    # resize lut to 256 (or more) values
    lut = cv2.resize(colorArr, (256, 1), interpolation=cv2.INTER_LINEAR)

    # apply lut
    result = cv2.LUT(stretch, lut)
    out = np.concatenate([result, mask[..., None]], axis=-1)

    imageio.imsave(
        os.path.join("/home/nicolai/Pictures/", f"img_{ii}.png"), img.astype(np.uint8)
    )
    imageio.imsave(
        os.path.join("/home/nicolai/Pictures/", f"depth_{ii}.png"),
        out.astype(np.uint8),
    )
    # plt.imsave(
    # os.path.join("/home/nicolai/Pictures/", f"depth_{ii}.png"),
    # depth,
    # cmap="jet",
    # vmin=depth.min(),
    # vmax=depth.max(),
    # )
    imageio.imsave(
        os.path.join("/home/nicolai/Pictures/", f"mask_{ii}.png"), mask.astype(np.uint8)
    )
