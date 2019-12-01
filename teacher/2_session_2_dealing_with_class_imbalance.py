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
# # Session 2 Part 1: Going Further, Discovering class-imbalance in datasets
#
# Since we have done the most basic training example, got our hands on skorch and on the dataset, we are going to repeat our process using a more realistic use case. This time, our dataset will be severely unbalanced (10% of all data will be images of aircrafts).
#
# You are going to:
# - Do a first "naive" run with the full data
# - Diagnose performance
# - Try to improve it by tuning several factors:
#   - The dataset itself
#   - The optimization parameters
#   - The network architecture
#   
# Remember that "deep learning" is still considered somewhat a black art so it's hard to know in advance what will work.

# %%
# Put your imports here
import numpy as np

# %%
# Global variables
TRAINVAL_DATASET_URL = "https://storage.googleapis.com/isae-deep-learning/trainval_aircraft_dataset.npz"

# %% {"tags": ["exercise"]}
# This cell should not be exported

# %% [markdown]
# ## Q0. Downloading & splitting the dataset
#
# You will get the following:
#
# - 45k images in training which you should use as training & validation
# - 15k images in test, which you should only use to compute your final metrics on. Don't ever use this dataset for early stopping
#
# ![](https://i.stack.imgur.com/osBuF.png)
#
# Refer to this: 
# https://towardsdatascience.com/train-validation-and-test-sets-72cb40cba9e7
#
#
# Q0.a: Download the dataset
# Q0.b: Split the dataset in training, validation and test (you already have the test)

# %%
# Download the dataset
ds = np.DataSource("/tmp/")
f = ds.open(TRAINVAL_DATASET_URL, 'rb')
trainval_dataset = np.load(f)
train_images = trainval_dataset['train_images']
train_labels = trainval_dataset['train_labels']
test_images = trainval_dataset['test_images']
test_labels = trainval_dataset['test_labels']


# %%
# Split the dataset

# %% [markdown]
# ## Q1. During Session 1, you learnt how to set up your environment on GCP, train a basic CNN on a small training set and plot metrics. Now let's do it again !
#
# Once you have downloaded data, use the notebook from Session 1 to get:
#
# a. Visualisation of the data
#
# b. Training of the model using steps seen during Session 1
#
# c. Compute and plot metrics (confusion matrix, ROC curve) based on this training
#
# d. Compare the metrics between this new dataset and the one from Session 1
#
# e. What did you expect ?

# %%
# Q1


# %% [markdown]
# During the previous notebook you plotted the Receiver Operating Characteristic curve. This is not ideal when dealing with imbalanced dataset since the issue of class imbalance can result in a serious bias towards the majority class, reducing the classification performance and increasing the number of **false positives**. Furthermore, in ROC curve calculation, true negatives don't have such meaning any longer. 
#
# Instead this time we will plot the Precision Recall curve of our model which uses precision and recall to evaluate models.
#
# ![](https://cdn-images-1.medium.com/fit/t/1600/480/1*Ub0nZTXYT8MxLzrz0P7jPA.png)
#
# Refer here for a tutorial on how to plot such curve: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html
#
# More details on PR Curve:
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
#
# https://www.datascienceblog.net/post/machine-learning/interpreting-roc-curves-auc/

# %% [markdown]
# e. Can you understand why PR curve are better than ROC curve for diagnosing model performance when dealing with imbalanced data ?
#
# f. Plot the ROC curve of your model as well as its AUC

# %%
# e & f here

# %% [markdown]
# ## Q2. Let's improve our model's performance
#
# We will try several things below. Those steps are only indicative and you are free to pursue other means of improving your model.
#
# Should you be lost, we refer you to the excellent "A Recipe for Training Neural Networks" article : https://karpathy.github.io/2019/04/25/recipe/
#
# ![image.png](slides/static/img/mlsystem.png)

# %% [markdown]
# ### a. Solving the imbalanced data problem
#
# Go through your data: is the dataset balanced ? If now, which steps can I do to solve this imbalance problem ?
#
# If you need help on this step, refer [to this tutorial on how to tackle imbalanced dataset](https://developers.google.com/machine-learning/data-prep/construct/sampling-splitting/imbalanced-data)
#
# - Which step would you take ?
# - **Don't forget to apply the same step on you train and validation dataset**
#
# Try to decide and a method to modify only the dataset and rerun your training. Did performance improve ?

# %%
# Q2.a here

# %% [markdown]
# ### b. Optimizer and model modifications
#
# i ) Now that you have worked on your dataset and decided to undersample it, it's time to tune your network and your training configuration
#
# In Session 1, you tested two descent gradient. What is the effect of its modification? Apply it to your training and compare metrics.
#
# ii ) An other important parameter is the learning rate, you can [check its effect on the behavior of your training](https://developers.google.com/machine-learning/crash-course/fitter/graph).
#
# iii) There is no absolute law concerning the structure of your deep Learning model. During the [Deep Learning class](https://github.com/erachelson/MLclass/blob/master/7%20-%20Deep%20Learning/Deep%20Learning.ipynb) you had an overview of existing models 
#
# You can operate a modification on your structure and observe the effect on final metrics. Of course, remain consistent with credible models, cf Layer Patterns chapter on this "must view" course : http://cs231n.github.io/convolutional-networks/
#
# ![image.png](slides/static/img/comparison_architectures.png)

# %%
# Q2.b here

# %% [markdown]
# ### c. Going Further
#
# Here is an overview of [possible hyperparameter tuning when training Convolutional Neural Networks](https://towardsdatascience.com/hyper-parameter-tuning-techniques-in-deep-learning-4dad592c63c8)
#
# You can try and apply those techniques to your use case.
#
# - Does these techniques yield good results ? What about the effort-spent-for-performance ratio ?
# - Do you find it easy to keep track of your experiments ? 
# - What would you need to have a better overview of the effects of these search ?
#
# Don't spend too much time on this part as the next is more important. You can come back to it after you're finished

# %%
# Q2.c here

# %% [markdown]
# ## Q3. Full Test whole dataset & more improvements
#
# a. Now that you have optimised your structure for your dataset, you will apply your model to the test dataset to see the final metrics. Plot all your metrics using the full imbalanced test set. Is it good enough ?
# If you think so, you can apply it to new images using the sliding window technique with the 3rd notebook
#
# b. If you're not satisfied with the output of your model, consider the following idea: Training a new model with the failures of your previous model.
# Try the following:
# - Get all the images with the "aircraft" label
# - Get all the images with the "background" label where your best model was wrong (predicted aircraft), as well as some of the background where it was right. 
# - Train a new model or retrain your existing one on this dataset.
#
# Did it bring any improvements ? 
#
# c . **!!!! SAVE YOUR MODEL !!!**

# %%
# Q3 here

# %% [markdown]
# **Did you save your model ??** You will need it for the next notebook
