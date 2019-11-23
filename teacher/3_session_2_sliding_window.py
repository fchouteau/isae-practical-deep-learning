# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Session 2 part 2: Using the sliding window technique to predict on larger images

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
