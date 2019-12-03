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
# The objective of the first session is to apply what was detailed above on another dataset using higher level tools such as [skorch](https://github.com/skorch-dev/skorch).
#
# In this session you will:
# - Get a preview of using skorch
# - Train a basic NN on a basic dataset
# - Plot ROC curve & confusion matrix to diagnose your dataset
#
# During session 2 you will be experimenting with harder datasets

# %%
# install dependencies
# %pip install skorch

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
#
# Note:
# If you get an error with the code below run:
# ```python
# !gsutil -m cp -r gs://isae-deep-learning/toy_aircraft_dataset.npz /tmp/storage.googleapis.com/isae-deep-learning/toy_aircraft_dataset.npz
# ```
# in a cell above the cell below

# %%
ds = np.DataSource(destpath="/tmp/")
f = ds.open(TOY_DATASET_URL, 'rb')

toy_dataset = np.load(f)
train_images = toy_dataset['train_images']
train_labels = toy_dataset['train_labels']
test_images = toy_dataset['test_images']
test_labels = toy_dataset['test_labels']

# %% [markdown]
# ### A bit about train-test
#
# You just downloaded a training and a test set.
# skorch will automatically split your training dataset into training and validation.
# - We use the training set for forward/backward
# - We use the validation set to tune hyperparameters (optimizers, early stopping)
# - We use the test set for final metrics on our tuned model
#
# ![](https://i.stack.imgur.com/osBuF.png)
#
# For more information as to why we use train/validation and test refer to this article:
# https://towardsdatascience.com/train-validation-and-test-sets-72cb40cba9e7

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
import cv2
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
grid_size = 4
grid = np.zeros((grid_size * 64, grid_size * 64, 3)).astype(np.uint8)
for i in range(grid_size):
    for j in range(grid_size):
        tile = np.copy(train_images[i * grid_size + j])
        label = np.copy(train_labels[i * grid_size + j])
        color = (0, 255, 0) if label == 1 else (255, 0, 0)
        tile = cv2.rectangle(tile, (0, 0), (64, 64), color, thickness=2)
        grid[i * 64:(i + 1) * 64, j * 64:(j + 1) * 64, :] = tile

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax.imshow(grid)
plt.show()

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
#
# If you need any help with functionalities of skorch, you [can find here](https://nbviewer.jupyter.org/github/skorch-dev/skorch/tree/master/notebooks/) the reference notebooks of the library

# %%
from skorch import NeuralNetClassifier
import torch
import torch.nn as nn
import torch.optim as optim

# %%
# Define the torch model to use
# Here a sequential layer is used instead of the classical nn.Module
# If you need to write your own module, plenty of resources are available one the web or in deep learning course
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
    nn.Softmax(),
)

# %%
# The famous skorch wrapper useful yet not complex and it has a sklearn friendly API

net = NeuralNetClassifier(
    module,
    max_epochs=10,
    lr=0.01,
    # Shuffle training data on each epoch
    iterator_train__shuffle=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
    optimizer=optim.SGD)

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
# ## ROC curve
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
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
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

misclassified_examples = train_images[net.predict(train_images.transpose(
    (0, 3, 1, 2)).astype(np.float32)) != train_labels, ::]

plt.figure(figsize=(10, 10))
for idx, im in enumerate(islice(misclassified_examples, 0, 8)):
    plt.subplot(4, 4, idx + 1)
    plt.imshow(im)
    plt.axis("off")

# %% [markdown]
# # Using more advanced features with skorch
#
# We will either edit the loop above or write a new fit loop to use advanced features of skorch
#
# ## Adding callbacks
#
# Last time we saw "callbacks" such as early stopping. Try to integrate them to the fit loop above.
#
# Here are references on "callbacks" with skorch:
#
# https://nbviewer.jupyter.org/github/skorch-dev/skorch/blob/master/notebooks/Basic_Usage.ipynb
#
# https://skorch.readthedocs.io/en/stable/callbacks.html#skorch.callbacks.EarlyStopping
#
# ```python
#
# from skorch.callbacks import EarlyStopping
#
# early_stopping = EarlyStopping(scoring='roc_auc', lower_is_better=False)
#
# net = NeuralNetClassifier(
#     ClassifierModule,
#     max_epochs=20,
#     lr=0.1,
#     callbacks=[early_stopping],
# )
# ```

# %%
# Use callbacks

# %% [markdown]
# ## [OPTIONAL] Data Augmentation
#
#
# One technique for training CNNs on images is to put your training data through data augmentation to generate similar-but-different examples to make your network more robust.
#
# You can generate "augmented images" on the fly or use composition to generate data
#
# - We are going to wrap our numpy arrays with `torch.utils.data.Dataset` class
#
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class
#
# - Here is how we deal with torch.data.Dataset formats with skorch:
#
# https://nbviewer.jupyter.org/github/skorch-dev/skorch/blob/master/notebooks/Advanced_Usage.ipynb#Working-with-Datasets
#
# - Here is how we use torch Compose to augment data
#
# https://pytorch.org/docs/stable/torchvision/transforms.html
#
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#compose-transforms
#
# Note: This step requires a bit of tinkering from numpy arrays to torch datasets, it's fine if you skip it. For the next notebook it may prove a useful way of gaining performance

# %%
# Add data augmentation
import torch.functional
import torch.utils
import torchvision


# %%
class DatasetFromNumpy(torch.utils.data.Dataset):
    def __init__(self, array_x, array_y, transform=None):
        self.array_x = array_x
        self.array_y = array_y
        self.transform = transform

    def __len__(self):
        return self.array_x.shape[0]

    def __getitem__(self, idx):
        x = self.array_x[idx]
        y = self.array_y[idx]
        if self.transform is not None:
            x = self.transform(x)
        else:
            x = torch.tensor(x)
        y = torch.tensor(y)
        return x, y


# %%
train_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.ToTensor(),
])

# %%
train_ds = DatasetFromNumpy(array_x=train_images, array_y=train_labels, transform=train_transform)

# %%
# Compare effects of data augmentation
img_orig = train_images[0]
plt.imshow(img_orig)
plt.show()
# Get image from dataset. Note: it has been converted as a torch tensor in CHW format in float32 normalized !
img, label = train_ds[0]
img = (img.numpy() * 255.).astype(np.uint8)
img = np.rollaxis(img, 0,3)
plt.imshow(img)
plt.show()

# %%
# we need to pass train_labels back for train-validation split !
# https://nbviewer.jupyter.org/github/skorch-dev/skorch/blob/master/notebooks/Advanced_Usage.ipynb#Working-with-Datasets
net.fit(train_ds, y=train_labels)

# %%
# plot our metrics again. Did we change something ? (don't forget to normalize data this time !)

# %% [markdown]
# # Next steps before the next notebooks
#
# - Try to play with network hyperparameters. The dataset is small and allow fast iterations so use it to have an idea on hyperparameter sensitivity.
#     number of convolutions, other network structures, learning rates, optimizers,...
# - Example: Compare again SGD and ADAM
# - Try to use the ROC curve to select a threshold to filter only negative examples without losing any positive examples
#
#
# When you are done with the warmup, go to the next notebook. But remember that next datasets will be larger and you will not have the time (trainings will take longer ) to experiment on hyperparameters.

# %%
