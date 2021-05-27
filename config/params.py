#! /usr/bin/env python3
# For BDD100k dataset
n_classes = 8
n_cpu = 2
pretrained = False
train_ds = "train.txt"
valid_ds = "valid.txt"
img_extensions = [".JPG", ".jpg"]
bs = 8
momentum = 0.9
wd = 0.001
lr = 1e-8
epochs = 100
pct_start = 10 / 100
optimizer = "Ranger"
flat_epochs = 50
cosine_epochs = 25
scheduler = "Cosine Delayed"
SAT = False
epsilon = 0.1
SAM = False
ECA = False
WS = False
Dropblock = False
iou_aware = False
coord = False
hard_mish = False
asff = False
repulsion_loss = False
acff = True
bcn = False
mbn = False
