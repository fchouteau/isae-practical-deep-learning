# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Session 2 Part 2: Using the sliding window technique to predict on larger images
# In this session we will load our previously detected model and apply it on large images using the sliding window technique
# %%
# Put your imports here
import numpy as np

# %%
# Global variables
tiles_dataset_url = "https://storage.googleapis.com/isae-deep-learning/tiles_aircraft_dataset.npz"

# %% {"tags": ["exercise"]}
# This cell should not be exported

# %%
# Download data
ds = np.DataSource("/tmp/")
f = ds.open(tiles_dataset_url, 'rb')
eval_tiles = np.load(f)
eval_tiles = eval_tiles['eval_tiles']
