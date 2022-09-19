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

    with open(split_file, "r") as f:
        split = json.load(f)["test"][cat]

    for file_name in split:
        output_path = os.path.join(out_dir, f"{file_name}_surface_points.h5")
        if os.path.exists(output_path):
            print(f"Skipping {output_path}")
            continue

        with h5py.File(os.path.join(base_dir, file_name, f"{file_name}.h5"), "r") as hf:
            points = hf["surface_pts"][:, :3]

        out_file = h5py.File(output_path, "w")
        out_file.create_dataset("gt_points", data=points, dtype="f", compression="gzip")
        out_file.close()
