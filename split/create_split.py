import os
import json

category_names = ["car", "chair", "plane"]
for category_name in category_names:
    train_file = f"/home/nicolai/phd/code/DIF-Net/split/train/{category_name}.txt"
    val_file = (
        f"/home/nicolai/phd/code/DIF-Net/split/eval/shapenet_{category_name}_test.list"
    )

    test_lines = []
    with open(val_file, "r") as f:
        for line in f.readlines():
            test_lines.append(line.rstrip())

    train_lines = []
    with open(train_file, "r") as f:
        for line in f.readlines():
            train_lines.append(line.rstrip())

    data = {}
    data["test"] = {category_name: test_lines}
    data["train"] = {category_name: train_lines}

    with open(f"{category_name}.json", "w") as outfile:
        json.dump(data, outfile)
