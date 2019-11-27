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
# # Session 1 : Training your first aircraft detector with pytorch
#
# **Intro**
#
# The objectives of this session is to apply what you learned during [the previous class on Deep Learning](https://github.com/erachelson/MLclass/blob/master/7%20-%20Deep%20Learning/Deep%20Learning.ipynb) on a real dataset of satellite images.
#
# Most of the vocabulary and concepts of Deep Learning and Convolutionnal Neural Network has been defined on the notebook linked above so you should refer to it.
#
# The objective of the first session is to apply what was detailed above on another dataset using the same tools (base pytorch), on the same low level of abstraction as previously. During section 2 of this BE you will move to higher-level tooling such as [skorch](https://github.com/skorch-dev/skorch)
#
# **What we are going to do**

# %%
# Put your imports here
import numpy as np

# %%
# Configuration variables
TOY_DATASET_URL = "https://storage.googleapis.com/isae-deep-learning/toy_aircraft_dataset.npz"

# %% [markdown]
# ## Downloading the dataset
#
# We will be using [numpy datasources](https://docs.scipy.org/doc/numpy/reference/generated/numpy.DataSource.html?highlight=datasources) to download the dataset. DataSources can be local files or remote files/URLs. The files may also be compressed or uncompressed. DataSource hides some of the low-level details of downloading the file, allowing you to simply pass in a valid file path (or URL) and obtain a file object.
#
# The dataset is in npz format which is a packaging format where we store several numpy arrays in key-value format

# %%
ds = np.DataSource(destpath="/tmp/")
f = ds.open(TOY_DATASET_URL, 'rb')

ds.(TOY_DATASET_URL)
toy_dataset = np.load(f)
train_images = toy_dataset['train_images']
train_labels = toy_dataset['train_labels']
test_images = toy_dataset['test_images']
test_labels = toy_dataset['test_labels']

# %%
ds = np.DataSource(destpath="/tmp/")
ds.abspath(TOY_DATASET_URL)
# !gsutil -m cp -r gs://isae-deep-learning/toy_aircraft_dataset.npz /tmp/storage.googleapis.com/isae-deep-learning/toy_aircraft_dataset.npz

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

# %% {"tags": ["exercise"]}
# Answer Q1 here

# %% [markdown]
# Q2. Can you plot at least 8 examples of each label ? In a 4x4 grid ?

# %%
# Helper code

# %% {"tags": ["exercise"]}
# This cell should not be exported
