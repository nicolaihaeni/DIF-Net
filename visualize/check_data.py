import os
import json
import h5py


base_path = "/home/isleri/haeni001/data/gen_sdf"
with open("/home/isleri/haeni001/code/DIF-Net/split/plane.json", "r") as in_file:
    data = json.load(in_file)

count = 0
for mode in data:
    split_data = data[mode]

    for cat in split_data:
        for filename in split_data[cat]:
            with h5py.File(
                os.path.join(base_path, cat, filename, "ori_sample.h5"), "r"
            ) as hf:
                try:
                    free_points = hf["free_pts"][:]
                    surface_points = hf["surface_pts"][:]
                except Exception as e:
                    print(f"Model {filename} failed")

                if count % 100 == 0:
                    print(f"Did {count} examples")
                count += 1
