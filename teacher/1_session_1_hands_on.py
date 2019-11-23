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
# # Session 1 : Training your first aircraft detector with pytorch

# %%
# Put your imports here
import numpy as np

# %%
# Global variables
toy_dataset_url = "https://storage.googleapis.com/isae-deep-learning/toy_aircraft_dataset.npz"

# %% [markdown]
# ## Downloading the dataset

# %%
ds = np.DataSource("/tmp/")
f = ds.open(toy_dataset_url, 'rb')
toy_dataset = np.load(f)
train_images = toy_dataset['train_images']
train_labels = toy_dataset['train_labels']
test_images = toy_dataset['test_images']
test_labels = toy_dataset['test_labels']

# %%
# Here we download the dataset

# %% [markdown]
# ## A bit of data exploration

# %% [markdown]
# Q1. Labels counting
# a. What is the dataset size ?
#
# b. How many images representing aircrafts ?
#
# c. How many images representing backgrounds ?
#
# d. What are the dimensions (height and width) of the images ? What are the number of channels ?

# %%
# Fill here

# %% [markdown]
# Q2. Can you plot at least 8 examples of each label ? In a 4x4 grid ?

# %%
# Helper code

# %% {"tags": ["exercise"]}
# This cell should not be exported
