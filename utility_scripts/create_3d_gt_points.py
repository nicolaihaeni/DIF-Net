import os
import json
import h5py

categories = ["car", "chair", "plane"]

for cat in categories:
    base_dir = f"/home/isleri/haeni001/data/dif/{cat}/"
    out_dir = f"/home/isleri/haeni001/data/gt_points/{cat}/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    split_file = f"/home/isleri/haeni001/code/DIF-Net/split/{cat}.json"

    with open(split, "r") as f:
        split = json.load(split_file)["test"]

    for file_name in split:
        with h5py.File(os.path.join(base_dir, file_name), "r") as hf:
            points = hf["surface_pts"][:]

        out_file = h5py.File(
            os.path.join(out_dir, f"{file_name}_surface_points.h5"), "w"
        )
        out_file.create_dataset("gt_points", data=points, dtype="f", compression="gzip")
        out_file.close()
