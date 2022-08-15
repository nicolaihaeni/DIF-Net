import os
import json

category = "02691156"
train_file = "/home/isleri/haeni001/code/DIF-Net/split/train/plane.txt"
val_file = "/home/isleri/haeni001/code/DIF-Net/split/eval/plane.txt"

test_lines = []
with open(val_file, "r") as f:
    for line in f.readlines():
        test_lines.append(line.rstrip())

train_lines = []
with open(train_file, "r") as f:
    for line in f.readlines():
        train_lines.append(line.rstrip())

data = {}
data["test"] = {"02691156": test_lines}
data["train"] = {"02691156": train_lines}

with open("plane.json", "w") as outfile:
    json.dump(data, outfile)
