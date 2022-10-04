import os
import json


base_dir = "/home/isleri/haeni001/data/ShapeNetCore.v2"
category_names = {"mug": "03797390"}
for k, v in category_names.items():
    names = sorted(os.listdir(os.path.join(base_dir, v)))

    data = {"train": {k: names}}
    with open(f"{k}.json", "w") as outfile:
        json.dump(data, outfile)
