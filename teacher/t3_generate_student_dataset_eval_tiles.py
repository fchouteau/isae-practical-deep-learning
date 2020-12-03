# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
# %matplotlib widget

# %% [markdown]
# ## Génération des tuiles d'évaluation
#
# On génère des tuiles 512 x 512 pour le dernier notebook

# %%
import sys
from pathlib import Path

import numpy as np
from pathy import Pathy

from khumeia import helpers

# %%
# add khumeia
sys.path.append("./src/")
sys.path = list(set(sys.path))

# %%
# setup env variable
raw_data_dir = Path("./data/raw/").resolve()
TRAINVAL_DATA_DIR = raw_data_dir / "trainval"
EVAL_DATA_DIR = raw_data_dir / "eval"

REMOTE_DATASET_DIR = Pathy("gs://fchouteau-isae-deep-learning")

# %%
eval_dataset = helpers.dataset_generation.items_dataset_from_path(EVAL_DATA_DIR)

# %% [markdown]
# ## Dataset parsing using khumeia

# %%
from khumeia.roi.tiles_generator import SlidingWindow
from khumeia.roi.tiles_sampler import *

# %%
MAX_ITEMS = None
TILE_SIZE = 512
TILE_STRIDE = 512

# %%
eval_dataset.items = eval_dataset.items[: min(len(eval_dataset), MAX_ITEMS or len(eval_dataset))]

# %%
sliding_window = SlidingWindow(
    tile_size=TILE_SIZE,
    stride=TILE_STRIDE,
    margin_from_bounds=32,
    discard_background=False,
)

# %%
eval_large_tiles = helpers.dataset_generation.generate_candidate_tiles_from_items(
    eval_dataset, sliding_windows=[sliding_window], n_jobs=4
)

# %% [markdown]
# Samples n tiles per satellite image with a ratio of 2/1 for tiles containing aircrafts and tile not containing aircrafts

# %%
test_stratified_sampler = BackgroundToPositiveRatioPerItemSampler(
    nb_positive_tiles_max=24,
    background_to_positive_ratio=0.5,
    with_replacement=False,
    shuffle=True,
)

eval_large_tiles = helpers.dataset_generation.sample_tiles_from_candidates(
    eval_large_tiles, tiles_samplers=[test_stratified_sampler]
)

# %%
eval_large_tiles = helpers.dataset_generation.dump_dataset_tiles(
    tiles_dataset=eval_large_tiles, items_dataset=eval_dataset
)

# %%
eval_large_tiles = np.asarray([i[0] for i in eval_large_tiles.items])

# %%
print(eval_large_tiles.shape)

# %%
# Save as dict of nparrays
data_dir = Path("./data/").resolve()
dataset_path = data_dir / "tiles_aircraft_dataset.npz"

with open(dataset_path, "wb") as f:
    np.savez_compressed(f, eval_tiles=eval_large_tiles)

# %%
# upload to gcp
import shlex
import subprocess

cmd = "gsutil -m cp -r {} gs://fchouteau-isae-deep-learning/".format(dataset_path.resolve())
print(cmd)
subprocess.check_call(cmd, shell=True)
# %% [markdown]
# ## Reload and check everything is ok

# %%
# try to reload using numpy datasource
ds = np.DataSource("/tmp/")
f = ds.open(
    "https://storage.googleapis.com/fchouteau-isae-deep-learning/tiles_aircraft_dataset.npz",
    "rb",
)
toy_dataset = np.load(f)
eval_tiles = toy_dataset["eval_tiles"]

print(eval_tiles.shape)
# %%
# plot them
import cv2
from matplotlib import pyplot as plt

# %%
grid_size = 4
im_size = 512
grid = np.zeros((grid_size * im_size, grid_size * im_size, 3)).astype(np.uint8)
for i in range(grid_size):
    for j in range(grid_size):
        tile = eval_tiles[i * grid_size + j]
        tile = cv2.rectangle(tile, (0, 0), (im_size, im_size), (255, 255, 255), thickness=2)
        grid[i * im_size : (i + 1) * im_size, j * im_size : (j + 1) * im_size, :] = tile

# %%
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax.imshow(grid)
plt.show()

# %% [markdown]
# ## Generate sliding window gif

# %%
import itertools

from PIL import Image

# %%
t = eval_tiles[12]
planes_xy = [(380, 480), (455, 420)]

# %%
plt.figure(figsize=(4, 4))
plt.imshow(t)
plt.show()

# %%
im_h, im_w = t.shape[:2]
tile_h, tile_w = 64, 64
stride_h, stride_w = 32, 32
max_i = int(np.floor(float(im_h - tile_h) / stride_h)) + 1
max_j = int(np.floor(float(im_w - tile_w) / stride_w)) + 1

# %%
gif = []
aircrafts_rect = []
for yt, xt in itertools.product(range(max_i), range(max_j)):
    frame = np.copy(t)
    xt1 = stride_w * xt
    xt2 = xt1 + tile_w
    yt1 = stride_h * yt
    yt2 = yt1 + tile_h
    aircraft = False
    for x, y in planes_xy:
        if xt1 + 8 < x < xt2 - 8 and yt1 + 8 < y < yt2 - 8:
            aircraft = True
            aircrafts_rect.append(((xt1, yt1), (xt2, yt2)))
    color = (255, 0, 0) if not aircraft else (0, 255, 0)
    frame = cv2.rectangle(frame, (xt1, yt1), (xt2, yt2), color, thickness=2)
    for tl, br in aircrafts_rect:
        frame = cv2.rectangle(frame, tl, br, (0, 255, 0), thickness=2)
    if xt1 >= 256 and yt1 >= 256:
        frame = frame[256:, 256:, :]
        frame = Image.fromarray(frame)
        gif.append(frame)

# %%
gif[0].save("docs/static/sliding_window.gif", save_all=True, append_images=gif[1:], duration=100, loop=0)

# %% [markdown]
# Sliding window demo 
#
# ![sw](out.gif)
