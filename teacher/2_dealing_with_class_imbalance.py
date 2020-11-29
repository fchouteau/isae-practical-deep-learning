# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
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

# %% [markdown]
# # Session 2 Part 1: Going Further, Discovering class-imbalance in datasets
#
# <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" align="left" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a>&nbsp;| Florient Chouteau | <a href="https://supaerodatascience.github.io/deep-learning/">https://supaerodatascience.github.io/deep-learning/</a>
#
# Since we have done the most basic training example, got our hands on skorch and on the dataset, we are going to repeat our process using a more realistic use case. This time, our dataset will be severely unbalanced (10% of all data will be images of aircrafts), like in real life (or not even like in real life but it's getting closer).
#
# Here, we won't guide you, you will have to use what you learned in the previous notebooks as well as what you learned in previous data science class to try to devise a way to train a good model
#
# You are going to:
# - Do a first "naive" run with the full data
# - Diagnose performance
# - Try to improve it by tuning several factors:
#   - The dataset itself
#   - The optimization parameters
#   - The network architecture
#
# **Remember that "deep learning" is still considered somewhat a black art so it's hard to know in advance what will work.**
#

# %%
# Put your imports here
import numpy as np

# %%
# Global variables
TRAINVAL_DATASET_URL = "https://storage.googleapis.com/fchouteau-isae-deep-learning/trainval_aircraft_dataset.npz"

# %% [markdown]
# ## Q0 Downloading & splitting the dataset
#
# You will get the following:
#
# - 50k images in training which you should use as training & validation
# - 5k images in test, which you should only use to compute your final metrics on. **Don't ever use this dataset for early stopping / intermediary metrics**
#
# <img src="https://i.stack.imgur.com/pXAfX.png" alt="pokemon" style="width: 400px;"/>

# %%
# Download the dataset
ds = np.DataSource("/tmp/")
f = ds.open(TRAINVAL_DATASET_URL, "rb")
trainval_dataset = np.load(f)
trainval_images = trainval_dataset["train_images"]
trainval_labels = trainval_dataset["train_labels"]
test_images = trainval_dataset["test_images"]
test_labels = trainval_dataset["test_labels"]

print(trainval_images.shape)
print(np.unique(trainval_labels, return_counts=True))

# %% [markdown]
# ### a. Data Exploration
#
# a. Can you plot some images ?
#
# b. What is the aircraft/background ratio ?

# %%

# %% [markdown]
# ### b. Dataset Splitting
#
# Here we will split the trainval_dataset to obtain a training and a validation dataset.
#
# For example, try to use 20% of the images as validation
#
# You must have seen that the dataset was really unbalanced, so a random sampling will not work...
#
# Use stratified sampling to keep the label distribution between training and validation

# %%
# Hint to get your started
background_indexes = np.where(trainval_labels == 0)
foreground_indexes = np.where(trainval_labels == 1)

# %% [markdown]
# ## Q1. Training & metrics
#
# During Session 1, you learnt how to set up your environment on Colab, train a basic CNN on a small training set and plot metrics. Now let's do it again !

# %% [markdown]
# ### First run
#
# Once you have downloaded & created your training & validation dataset, use the notebook from Session 1 to get:
#
# a. Training of the model using steps seen during Session 1
#
# b. Compute and plot metrics (confusion matrix, ROC curve) based on this training
#
# c. Compare the metrics between this new dataset and the one from Session 1
#
# d. What did you expect ? Is your model working well ?

# %%
# Q1

# %% [markdown]
# ### Test metrics, introduction to PR Curves
#
# During the previous notebook you plotted the Receiver Operating Characteristic curve. This is not ideal when dealing with imbalanced dataset since the issue of class imbalance can result in a serious bias towards the majority class, reducing the classification performance and increasing the number of **false positives**. Furthermore, in ROC curve calculation, true negatives don't have such meaning any longer.
#
# Instead this time we will plot the Precision Recall curve of our model which uses precision and recall to evaluate models.
#
# ![](https://cdn-images-1.medium.com/fit/t/1600/480/1*Ub0nZTXYT8MxLzrz0P7jPA.png)
#
# Refer here for a tutorial on how to plot such curve:
#
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html
#
# More details on PR Curve:
#
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
#
# https://www.datascienceblog.net/post/machine-learning/interpreting-roc-curves-auc/
#
# **e. Plot the ROC curve of your model as well as its PR Curve, on the test set, compare them, which is easier to interpret ?**
#
# **f. Can you understand why PR curve may be more useful than ROC curve for diagnosing model performance when dealing with imbalanced data ?**
#
# **g. What is Fbeta-Score ? How can it help ? How do you chose beta?**
#
# Some reading: https://towardsdatascience.com/on-roc-and-precision-recall-curves-c23e9b63820c

# %%
# e & f here

# %% [markdown]
# ## Q2. Let's improve our model's performance
#
# We will try several things below. Those steps are only indicative and you are free to pursue other means of improving your model.
#
# Should you be lost, we refer you to the excellent "A Recipe for Training Neural Networks" article : https://karpathy.github.io/2019/04/25/recipe/
#
# ![image.png](docs/static/img/mlsystem.png)

# %% [markdown]
# ### a. Tackling the imbalanced data problem
#
# Go through your data: is the dataset balanced ? If now, which steps can I do to solve this imbalance problem ?
#
# - Which step would you take ?
# - **Don't forget to apply the same step on you train and validation dataset** but **not on your test set** as your test set should represent the final data distribution
#
# Try to decide and a method to modify only the dataset and rerun your training. Did performance improve ?
#
#
# HINT:
# - It's usually a mix of oversampling the minority class and undersampling the majority class
#
# Some readings:
# - https://www.kaggle.com/rafjaa/resampling-strategies-for-imbalanced-datasets (very well done)
# - https://machinelearningmastery.com/framework-for-imbalanced-classification-projects/ (a bigger synthesis)
# - https://machinelearningmastery.com/category/imbalanced-classification/

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
# iii) There is no absolute law concerning the structure of your deep Learning model. During the [Deep Learning class](%matplotlib inline) you had an overview of existing models
#
# You can operate a modification on your structure and observe the effect on final metrics. Of course, remain consistent with credible models, cf Layer Patterns chapter on this "must view" course : http://cs231n.github.io/convolutional-networks/
#
# <img src="docs/static/img/comparison_architectures.png" alt="pokemon" style="width: 400px;"/>

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
# - Did it bring any improvements ?

# %%
# Q3a

# %% [markdown]
# b. If you're not satisfied with the output of your model, consider the following idea: Training a new model with the failures of your previous model.
# Try the following:
# - Get all the images with the "aircraft" label
# - Get all the images with the "background" label where your best model was wrong (predicted aircraft), as well as some of the background where it was right.
# - Train a new model or retrain your existing one on this dataset.
#

# %%
# Q3b

# %% [markdown]
# c . **SAVE YOUR MODEL**

# %%
# Q3c

# %% [markdown]
# **Have you saved your model ??** You will need it for the next notebook
