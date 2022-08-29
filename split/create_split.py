import os
import json

category = "04379243"
train_file = "/home/isleri/haeni001/code/DIF-Net/split/train/table.txt"
val_file = "/home/isleri/haeni001/code/DIF-Net/split/eval/table.txt"

test_lines = []
with open(val_file, "r") as f:
    for line in f.readlines():
        test_lines.append(line.rstrip())

train_lines = []
with open(train_file, "r") as f:
    for line in f.readlines():
        train_lines.append(line.rstrip())

data = {}
data["test"] = {category: test_lines}
data["train"] = {category: train_lines}

with open("table.json", "w") as outfile:
    json.dump(data, outfile)
