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
# In this session we will load our previously detected model and apply it on large images using the sliding window technique.
#
# The sliding window technique is a method to convert a classifier into detector. It can be illustrated by a single gif:
#
# ![](slides/static/img/sliding_window.gif)
#
# For more information about the sliding window technique refer to this excellent article: 
#
# https://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/
#
# Note: We are training at a single scale. And satellite imagery more or less prevents the foreground/background effect that sometimes require large changes in scale between training and testing for "normal" photography. So you can ignore the bits about the image pyramid on this issue (it is very good for general culture though, and can be applied in other use cases, or if we used multiscale training to "zoom" small aircrafts for example)
# %%
# Put your imports here
import numpy as np

# %%
# Global variables
tiles_dataset_url = "https://storage.googleapis.com/isae-deep-learning/tiles_aircraft_dataset.npz"

# %% {"tags": ["exercise"]}
# This cell should not be exported

# %% [markdown]
# ## Download the dataset

# %%
# Download data
ds = np.DataSource("/tmp/")
f = ds.open(tiles_dataset_url, 'rb')
eval_tiles = np.load(f)
eval_tiles = eval_tiles['eval_tiles']

# %% [markdown]
# ## Data Exploration
#
# - Plot some of the images
# - The images are not labelled to prevent any "competition", the objective is just to apply it.

# %%

# %% [markdown]
# ## Reload your model
#
# Using the routines detailed in the previous notebook, reload your model

# %%

# %% [markdown]
# ## Code the sliding window
#
# Intuitively, it's about applying an aircraft classifier trained on 64x64 pictures of aircraft or "anything else" as a detector. 
#
# Our network structure more or less prevents applying it to the full 512x512 images, and even if it could (you may be able to do it with global pooling layers...) this would not bring much information ("there is at least one aircraft in this region" sometimes is not sufficient).
#
# So the idea is to "slide" our 64x64 classifier on the image and collect the coordinates where "aircraft" is returned. Those should be the locations of our aircrafts;
#
# You could view your model as a big convolution returning "aircraft / not aircraft". Its kernel size is 64x64, there are one or two filters depending on if you coded with softmax or crossentropy. You then just have to decide on the stride of this convolution... And to keep in mind how to go back to coordinates to plot your aircrafts afterwards ;)
#
# There are a lot of degrees of freedom when developping sliding windows. A sliding window with a too small "step" will only provide noisy overlapping detections. A step too large will make you miss some objects.
#
# It's up to you to find acceptable parameters.
#
# *Note*: The dataset labels were generated so that an image is considered an aircraft **if and only if the center of an aircraft lies in the center 32x32** of the 64x64 image

# %%

# %% [markdown]
# ## Apply the sliding window on the dataset and visualize results

# %%

# %% [markdown]
# ## What's next ? 
#
# Well...
#
# Are you satisfied with the behaviour of your model ?  Are there a lot of false positives ?
#
# If so, you can go back to 2/3 to tune your model and re-apply it.
#
# If you're out of your depth on how to improve your model... think about it ;)  You should be able to find news ideas because really, those problems have no end
#
# Welcome to the life of a DL engineer !
