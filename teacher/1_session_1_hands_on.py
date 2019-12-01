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

# %% [markdown]
# Q2. Can you plot at least 8 examples of each label ? In a 4x4 grid ?

# %% [markdown]
# Here are some functionnality examples. Try them and make your own. A well-understandood dataset is the key to an efficient model.

# %%
import matplotlib
import matplotlib.pyplot as plt

# %matplotlib inline

# %%
LABEL_NAMES = ["Not an aircraft", "Aircraft"]

print("Labels counts :")

for c, l, label in zip(*np.unique(train_labels, return_counts=True), LABEL_NAMES):
    print(f" Label: {label} , value: {l}, count: {c}")

# %%
for l, label in enumerate(LABEL_NAMES):
    print(f"Examples shape for label {l} : {train_images[train_labels == l, ::].shape}")

# %%
plt.imshow(train_images[0, ::])

# %% [markdown]
# # Training
#
# In order to simplify the code, we will use the [skorch](https://github.com/skorch-dev/skorch) library. It provides a convenient wrapper and avoid the need of re writing the training loop eah time:
#
# ```python
# for epoch in range(10):
#     for batch in batches:
#         # torch forward
#         # torch backward
# ```
#
# If you still prefer writing your own loop, feel free to overwrite the next cells.

# %%
import skorch
from skorch import NeuralNetClassifier

import torch.nn as nn
import torch.optim as optim

# %%

# Define the torch model to use
# Here a sequential layer is used instead of the classical nn.Module
# If you need to write your own modulem plenty of resources are available one the web or in deep learning course
module = nn.Sequential(
    nn.Conv2d(3, 64, 3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(64, 64, 3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(64, 64, 3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(6 * 6 * 64, 256),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 2),
    nn.Softmax()

)

# The famous skorch wrapper useful yet not complex and it has a sklearn friendly API
net = NeuralNetClassifier(
    module,
    max_epochs=10,
    lr=0.01,
    # Shuffle training data on each epoch
    iterator_train__shuffle=True,
    device="cuda",
    optimizer=optim.SGD
)

# %%
# The training loop
# Each epoch should take at most 1 second since we are executing the network on a GPU

net.fit(train_images.transpose((0, 3, 1, 2)).astype(np.float32), train_labels)

# %% [markdown]
# # Testing and metrics
#
# Now that we have a trained network, it is important to measure how well it performs.
#
# You can refer to your ML course or on resources on the web to see how we can measure it.
#
#
# ## Confusion matrix
# Here, we are first computing the [confusion matrix]():

# %%
from sklearn.metrics import confusion_matrix

print("Confusion matrix")
confusion_matrix(train_labels, net.predict(train_images.transpose((0, 3, 1, 2)).astype(np.float32)))

# %% [markdown]
# ## Roc curve
#
# The next metric we are computing is the [ROC curve](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html).
#
# It used to choose a threshold on the output probability in case you are intesrested in controling the false positive rate.

# %%
from sklearn.metrics import roc_curve, auc

# Compute ROC curve and ROC area for each class

fpr, tpr, _ = roc_curve(test_labels, net.predict_proba(test_images.transpose((0, 3, 1, 2)).astype(np.float32))[:, 1])
roc_auc = auc(fpr, tpr)

# %%
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

# %% [markdown]
# ## Misclassified examples

# %% [markdown]
# It is always interesting to check mis classified examples.
#
# It usually provides tips on how to improve your model.

# %%
from itertools import islice

misclassified_examples = train_images[
                         net.predict(train_images.transpose((0, 3, 1, 2)).astype(np.float32)) != train_labels, ::]

plt.figure(figsize=(10, 10))
for idx, im in enumerate(islice(misclassified_examples, 0, 8)):
    plt.subplot(4, 4, idx + 1)
    plt.imshow(im)
    plt.axis("off")

# %% [markdown]
# # Next steps before the next notebooks
#
# - Try to play with network hyperparameters. The dataset is small and allow fast iterations so use it to have an idea on hyperparameter sensitivity.
#     number of convolutions, other network structures, learning rates, optimizers,...
# - Try to use the ROC curve to select a threshold to filter only negative examples without losing any positive examples
#
#
# When you are done with the warmup, go to the next notebook. But remember that next datasets will be larger and you will not have the time (trainings will take longer ) to experiment on hyperparameters.