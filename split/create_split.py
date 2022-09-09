import os
import json

category_names = ["car", "chair", "plane", "table"]
for category_name in category_names:
<<<<<<< HEAD
    # train_file = f"/home/isleri/haeni001/code/DIF-Net/split/train/{category_name}.txt"
    # val_file = f"/home/isleri/haeni001/code/DIF-Net/split/eval/{category_name}.txt"
    train_file = f"/home/nicolai/phd/code/DIF-Net/split/train/{category_name}.txt"
    val_file = f"/home/nicolai/phd/code/DIF-Net/split/eval/{category_name}.txt"
=======
    train_file = f"/home/isleri/haeni001/code/DIF-Net/split/train/{category_name}.txt"
    val_file = f"/home/isleri/haeni001/code/DIF-Net/split/eval/{category_name}.txt"
>>>>>>> Latest changes for evaluation

    test_lines = []
    with open(val_file, "r") as f:
        for line in f.readlines():
            test_lines.append(line.rstrip())

    train_lines = []
    with open(train_file, "r") as f:
        for line in f.readlines():
            train_lines.append(line.rstrip())

    data = {}
<<<<<<< HEAD
    data["test"] = test_lines
    data["train"] = train_lines
=======
    data["test"] = {category_name: test_lines}
    data["train"] = {category_name: train_lines}
>>>>>>> Latest changes for evaluation

    with open(f"{category_name}.json", "w") as outfile:
        json.dump(data, outfile)
