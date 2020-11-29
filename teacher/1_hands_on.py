# -*- coding: utf-8 -*-
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
# # Session 1 : Training your first aircraft classifier with pytorch
#
# <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" align="left" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a>&nbsp;| Florient Chouteau | <a href="https://supaerodatascience.github.io/deep-learning/">https://supaerodatascience.github.io/deep-learning/</a>
#
# ### Intro
#
# The objectives of this session is to apply what you learned during [the previous class on Deep Learning](https://supaerodatascience.github.io/deep-learning/) on a real dataset of satellite images.
#
# Most of the vocabulary and concepts of Deep Learning and Convolutionnal Neural Network has been defined on the class linked above so you should refer to it.
#
# The objective of the first session is to apply what was detailed above on another dataset using higher level tools such as [pytorch ignite].(https://github.com/pytorch/ignite)
#
# In this session you will:
# - Get more comfortable using [pytorch ignite](https://github.com/pytorch/ignite)
# - Train a basic NN on a basic dataset
# - Plot ROC curve & confusion matrix to diagnose your dataset
#
# During session 2 you will be experimenting with harder datasets
#
# If you haven't done so, go to the previous notebook to get a hands on pytorch ignite using a simple dataset called Fashion MNIST

# %% [markdown]
# ### What is pytorch-ignite ?
#
# <img src="https://raw.githubusercontent.com/pytorch/ignite/master/assets/logo/ignite_logo_mixed.svg" alt="ignite" style="width: 400px;"/>
#
# Pytorch ignite is what we call a "high-level library" over pytorch, its objectives is to abstract away most of the boilerplate code for training deep neural network.
#
# Usually, they make the development process easier by enabling you to focus on what's important instead of writing distributed and optimized training loops and plugging metrics / callbacks. Because we all forgot to call `.backward()` or `.zero_grad()` at least once.
#
# Here an overview of the high-level libraries available for pytorch,
#
# https://neptune.ai/blog/model-training-libraries-pytorch-ecosystem?utm_source=twitter&utm_medium=tweet&utm_campaign=blog-model-training-libraries-pytorch-ecosystem
#
# Of these, we would like to highlight three of them:
#
# - pytorch-ignite, officially sanctioned by the pytorch team (its repo lives at https://pytorch.org/ignite/), which is developped by [someone from Toulouse](https://twitter.com/vfdev_5) - yes there is a member of the pytorch team living in Toulouse, we are not THAT behind in ML/DL :wishful-thinking:
#
# - pytorch-lightning (https://www.pytorchlightning.ai/) which has recently seen its 1.0 milestone and has bee developped to a company. It is more "research oriented" that pytorch-ignite, and with a lower abstraction level, but seems to enable more use case.
#
# - skorch (https://github.com/skorch-dev/skorch). This class was previously written in skorch. Skorch mimics the scikit-learn API and allows bridging the two libraries together. It's a bit less powerful but you write much less code than the two libraries above, and if you are very familiar with scikit-learn, it could be very useful for fast prototyping
#
#
# **Take a look at the [previous class](https://nbviewer.jupyter.org/github/SupaeroDataScience/deep-learning/blob/main/deep/PyTorch%20Ignite.ipynb), the [official examples](https://nbviewer.jupyter.org/github/pytorch/ignite/tree/master/examples/notebooks/) or the [documentation](https://pytorch.org/ignite/) if you need help using ignite**

# %%
# %matplotlib inline

# %%
# Ensure ignite is installed, otherwise install it
# # !pip install pytorch-ignite

# %%
# Put your imports here
import numpy as np

# %% [markdown]
# ## Dataset
#
# Récupération et exploration du datset

# %%
# Configuration variables
TOY_DATASET_URL = "https://storage.googleapis.com/fchouteau-isae-deep-learning/toy_aircraft_dataset.npz"

# %% [markdown]
# andrej### Image (reminders)
#
# A digital image is an image composed of picture elements, also known as pixels, each with finite, discrete quantities of numeric representation for its intensity or gray level that is an output from its two-dimensional functions fed as input by its spatial coordinates denoted with x, y on the x-axis and y-axis, respectively.
#
# We represent images as matrixes,
#
# Images are made of pixels, and pixels are made of combinations of primary colors (in our case Red, Green and Blue). In this context, images have chanels that are the grayscale image of the same size as a color image, made of just one of these primary colors. For instance, an image from a standard digital camera will have a red, green and blue channel. A grayscale image has just one channel.
#
# In geographic information systems, channels are often referred to as raster bands.
#
# ![img](https://static.packt-cdn.com/products/9781789613964/graphics/e91171a3-f7ea-411e-a3e1-6d3892b8e1e5.png)
#
#
# For the rest of this workshop we will use the following axis conventions for images
#
# ![conventions](https://storage.googleapis.com/fchouteau-isae-deep-learning/static/image_coordinates.png)

# %% [markdown]
# ### Downloading the dataset
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
f = ds.open(TOY_DATASET_URL, "rb")

toy_dataset = np.load(f)
trainval_images = toy_dataset["train_images"]
trainval_labels = toy_dataset["train_labels"]
test_images = toy_dataset["test_images"]
test_labels = toy_dataset["test_labels"]

# %% [markdown]
# ### A bit of data exploration

# %% [markdown]
# **Q1. Labels counting**
#
# a. What is the dataset size ?
#
# b. How many images representing aircrafts ?
#
# c. How many images representing backgrounds ?
#
# d. What are the dimensions (height and width) of the images ? What are the number of channels ?

# %% [markdown]
# **Q2. Can you plot at least 8 examples of each label ? In a 4x4 grid ?**

# %% [markdown]
# Here are some examples that help you answer this question. Try them and make your own. A well-understandood dataset is the key to an efficient model.

# %%
import cv2
import matplotlib.pyplot as plt

# %%
LABEL_NAMES = ["Not an aircraft", "Aircraft"]

print("Labels counts :")
for l, c, label in zip(*np.unique(trainval_labels, return_counts=True), LABEL_NAMES):
    print(f" Label: {label} , value: {l}, count: {c}")

for l, label in enumerate(LABEL_NAMES):
    print(f"Examples shape for label {l} : {trainval_images[trainval_labels == l, ::].shape}")

# %%
LABEL_NAMES = ["Not an aircraft", "Aircraft"]

print("Labels counts :")
for l, c, label in zip(*np.unique(test_labels, return_counts=True), LABEL_NAMES):
    print(f" Label: {label} , value: {l}, count: {c}")

for l, label in enumerate(LABEL_NAMES):
    print(f"Examples shape for label {l} : {test_images[test_labels == l, ::].shape}")

# %%
grid_size = 4
grid = np.zeros((grid_size * 64, grid_size * 64, 3)).astype(np.uint8)
for i in range(grid_size):
    for j in range(grid_size):
        tile = np.copy(trainval_images[i * grid_size + j])
        label = np.copy(trainval_labels[i * grid_size + j])
        color = (0, 255, 0) if label == 1 else (255, 0, 0)
        tile = cv2.rectangle(tile, (0, 0), (64, 64), color, thickness=2)
        grid[i * 64 : (i + 1) * 64, j * 64 : (j + 1) * 64, :] = tile

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax.imshow(grid)
plt.show()

# %% [markdown]
# ### A bit about train-test
#
# You just downloaded a training and a test set.
#
# - We use the training set for forward/backward
# - We use the validation set to tune hyperparameters (optimizers, early stopping)
# - We use the test set for final metrics on our tuned model
#
# ![](https://i.stack.imgur.com/osBuF.png)
#
# For more information as to why we use train/validation and test refer to these articles:
# https://towardsdatascience.com/train-validation-and-test-sets-72cb40cba9e7
#
# https://www.freecodecamp.org/news/what-to-do-when-your-training-and-testing-data-come-from-different-distributions-d89674c6ecd8/
#
# https://kevinzakka.github.io/2016/09/26/applying-deep-learning/
#
# We will now create our validation dataset,
#
# Since we know the dataset is balanced, we can evenly sample from the dataset without taking too many risks
#
# We will do a 80/20 sampling

# %%
idxs = np.random.permutation(np.arange(trainval_images.shape[0]))

train_idxs, val_idxs = idxs[: int(0.8 * len(idxs))], idxs[int(0.8 * len(idxs)) :]

train_images = trainval_images[train_idxs]
train_labels = trainval_labels[train_idxs]
val_images = trainval_images[val_idxs]
val_labels = trainval_labels[val_idxs]

# %%
train_images.shape

# %% [markdown]
# ## Training using pytorch-ignite
#
# In order to simplify the code, we will use the [pytorch-ignite](https://github.com/pytorch/ignite) library. It provides a convenient wrapper and avoid the need of re writing the training loop eah time:
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
# If you need any help with functionalities of ignite, you [can find here](https://nbviewer.jupyter.org/github/pytorch/ignite/tree/master/examples/notebooks/) the reference notebooks of the library

# %%
from typing import Callable

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# %% [markdown]
# ### Defining Dataset & Transforms

# %%
## Define dataset

## We write our own Dataset class
class NpArrayDataset(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        image_transforms: Callable = None,
        label_transforms: Callable = None,
    ):
        self.images = images
        self.labels = labels
        self.image_transforms = image_transforms
        self.label_transforms = label_transforms

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index: int):
        x = self.images[index]
        y = self.labels[index]

        if self.image_transforms is not None:
            x = self.image_transforms(x)
        else:
            x = torch.tensor(x)

        if self.label_transforms is not None:
            y = self.label_transforms(y)
        else:
            y = torch.tensor(y)

        return x, y


# %%
# transform to convert np array in range [0,255] to torch.Tensor [0.,1.]
image_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

target_transforms = None

# load the training data
train_set = NpArrayDataset(
    images=train_images, labels=train_labels, image_transforms=image_transforms, label_transforms=target_transforms
)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

# load the validation data
validation_set = NpArrayDataset(
    images=val_images, labels=val_labels, image_transforms=image_transforms, label_transforms=target_transforms
)
val_loader = DataLoader(validation_set, batch_size=64, shuffle=True)

# %% [markdown]
# ### Check that your dataset outputs correct data
#
# Always to this as a sanity check to catch bugs in your data processing pipeline
#
# Write the inverse transformation by hand to ensure it's ok
#
# ![andrej](https://storage.googleapis.com/fchouteau-isae-deep-learning/static/andrej_tweet_1.png)

# %%
x, y = train_set[3]
x = x.numpy()
x = (x * 255.0).astype(np.uint8).transpose((1, 2, 0))

print("Inverse transform is OK ? {}".format(np.all(train_set.images[3] == x)))
print("Label {}".format(y))
Image.fromarray(x.astype(np.uint8))

# %% [markdown]
# Model

# %%
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
# Define the torch model to use
# Here a sequential layer is used instead of the classical nn.Module
# If you need to write your own module, plenty of resources are available one the web or in deep learning course


def model_fn():
    model = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 64, 3, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(6 * 6 * 64, 256),
        nn.ReLU(),
        nn.Dropout(p=0.25),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Dropout(p=0.25),
        nn.Linear(64, 2),
        nn.LogSoftmax(dim=-1),
    )

    return model


model = model_fn()

# moving model to gpu if available
model.to(DEVICE)

# %% [markdown]
# ### Defining High level Training functions

# %%
import ignite.engine
import ignite.handlers
import ignite.metrics
import ignite.utils
from ignite.engine import Events

# %%
# declare optimizers and loss
optimizer = optim.SGD(model.parameters(), lr=0.001)
criterion = nn.NLLLoss()

# %% [markdown]
# Below we create 3 engines, a trainer, an evaluator for the training set and an evaluator for the validation set, by using the `create_supervised_trainer` and `create_supervised_evaluator` and passing the required arguments.
#
# We import the metrics from `ignite.metrics` which we want to calculate for the model. Like `Accuracy`, `ConfusionMatrix`, and `Loss` and we pass them to `evaluator` engines which will calculate these metrics for each iteration.
#
# * `training_history`: it stores the training loss and accuracy
# * `validation_history`:it stores the validation loss and accuracy
# * `last_epoch`: it stores the last epoch untill the model is trained
#
# We will also attach a metric of `RunningAverage` to track a running average of the scalar loss output for each batch.

# %%
# defining the number of epochs
epochs = 12

# creating trainer
trainer = ignite.engine.create_supervised_trainer(model=model, optimizer=optimizer, loss_fn=criterion, device=DEVICE)

# create metrics
metrics = {
    "accuracy": ignite.metrics.Accuracy(),
    "nll": ignite.metrics.Loss(criterion),
    "cm": ignite.metrics.ConfusionMatrix(num_classes=2),
}

ignite.metrics.RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")

# Evaluators
train_evaluator = ignite.engine.create_supervised_evaluator(model, metrics=metrics, device=DEVICE)
val_evaluator = ignite.engine.create_supervised_evaluator(model, metrics=metrics, device=DEVICE)

# Logging
train_evaluator.logger = ignite.utils.setup_logger("train")
val_evaluator.logger = ignite.utils.setup_logger("val")

# init variables for logging
training_history = {"accuracy": [], "loss": []}
validation_history = {"accuracy": [], "loss": []}
last_epoch = []

# %% [markdown]
# Lastly, we want to checkpoint this model. It's important to do so, as training processes can be time consuming and if for some reason something goes wrong during training, a model checkpoint can be helpful to restart training from the point of failure.
#
# Below we will use Ignite's `ModelCheckpoint` handler to checkpoint models at the end of each epoch.

# %%
model_name = "basic_cnn"
dataset_name = "toy_aircrafts"

checkpointer = ignite.handlers.ModelCheckpoint(
    "./saved_models",
    filename_prefix=dataset_name,
    n_saved=2,
    create_dir=True,
    save_as_state_dict=True,
    require_empty=False,
)

trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {model_name: model})


# %% [markdown]
# Below you will see ways to define your own custom functions and attaching them to various `Events` of the training process.
#
# The functions below both achieve similar tasks, they print the results of the evaluator run on a dataset. One function does that on the training evaluator and dataset, while the other on the validation. Another difference is how these functions are attached in the trainer engine.
#
# The first method involves using a decorator, the syntax is simple - `@` `trainer.on(Events.EPOCH_COMPLETED)`, means that the decorated function will be attached to the trainer and called at the end of each epoch.
#
# The second method involves using the add_event_handler method of trainer - `trainer.add_event_handler(Events.EPOCH_COMPLETED, custom_function)`. This achieves the same result as the above.

# %%
@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    train_evaluator.run(train_loader)
    metrics = train_evaluator.state.metrics
    accuracy = metrics["accuracy"] * 100
    loss = metrics["nll"]
    last_epoch.append(0)
    training_history["accuracy"].append(accuracy)
    training_history["loss"].append(loss)
    print(
        "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}".format(
            trainer.state.epoch, accuracy, loss
        )
    )


@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    val_evaluator.run(val_loader)
    metrics = val_evaluator.state.metrics
    accuracy = metrics["accuracy"] * 100
    loss = metrics["nll"]
    validation_history["accuracy"].append(accuracy)
    validation_history["loss"].append(loss)
    print(
        "Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}".format(
            trainer.state.epoch, accuracy, loss
        )
    )


# %% [markdown]
# ### Training
#
# We will actually train the model (run the training engine), and plot training & validation metrics during training

# %%
trainer.run(train_loader, max_epochs=epochs)

# %%
plt.plot(training_history["accuracy"], label="Training Accuracy")
plt.plot(validation_history["accuracy"], label="Validation Accuracy")
plt.xlabel("No. of Epochs")
plt.ylabel("Accuracy")
plt.legend(frameon=False)
plt.show()

# %%
plt.plot(training_history["loss"], label="Training Loss")
plt.plot(validation_history["loss"], label="Validation Loss")
plt.xlabel("No. of Epochs")
plt.ylabel("Loss")
plt.legend(frameon=False)
plt.show()

# %% [markdown]
# Now, clear the model from memory

# %%
del model


# %% [markdown]
# ## Testing and metrics
#
# Now that we have a trained network, it is important to measure how well it performs. We do not do that during training because theoretically we try to test on a context closer to how the final model will be used, meaning this can be another pipeline and is usually outside the training engine.
#
# You can refer to your ML course or on resources on the web to see how we can measure it.

# %% [markdown]
# ### Loading saved model

# %%
# loading the saved model
def fetch_last_checkpoint_model_filename(model_save_path: str, model_prefix: str):
    import os
    from pathlib import Path

    checkpoint_files = Path(model_save_path)
    checkpoint_files = checkpoint_files.glob("{}*.pt".format(model_prefix))
    checkpoint_files = [str(ckpt.resolve().name) for ckpt in checkpoint_files]
    checkpoint_iter = [int(x.split("_")[-1].split(".")[0]) for x in checkpoint_files]
    last_idx = np.array(checkpoint_iter).argmax()

    checkpoint_file = os.path.join(model_save_path, checkpoint_files[last_idx])
    print(checkpoint_file)
    return checkpoint_file


checkpoint_path = fetch_last_checkpoint_model_filename("./saved_models", "{}_{}".format(dataset_name, model_name))

print(checkpoint_path)
# Instantiate a new empty model
model = model_fn()

# Load state
model.load_state_dict(torch.load(checkpoint_path))

print("Model Loaded")

# %% [markdown]
# ### Inferencing on the test dataset
#
# Now we will run predictions on the test dataset using the newly loaded model

# %%
test_ds = NpArrayDataset(
    images=test_images, labels=test_labels, image_transforms=image_transforms, label_transforms=target_transforms
)

# %%
import tqdm

# %%
y_true = []
y_pred = []

# Send model to correct device
model.to(DEVICE)

# Put model in evaluatio mode (very important)
model.eval()

# Disable all gradients things
with torch.no_grad():
    for x, y_t in tqdm.tqdm(test_ds, "predicting"):
        x = x.reshape((-1,) + x.shape)
        x = x.to(DEVICE)
        y = torch.exp(model.forward(x))
        y = y.to("cpu").numpy()

        y_t = int(y_t.to("cpu").numpy())

        y_pred.append(y)
        y_true.append(y_t)
y_pred = np.concatenate(y_pred, axis=0)
y_true = np.asarray(y_true)

# %%
print(y_pred.shape)

print(y_pred[0])

# %%
y_pred_classes = np.argmax(y_pred, axis=1)

# %% [markdown]
# ### Confusion matrix
# Here, we are first computing the [confusion matrix]():

# %%
from sklearn.metrics import confusion_matrix

print("Confusion matrix")
confusion_matrix(y_true, y_pred_classes)

# %% [markdown]
# ### ROC curve
#
# The next metric we are computing is the [ROC curve](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html).
#
# It used to choose a threshold on the output probability in case you are intesrested in controling the false positive rate.

# %%
from sklearn.metrics import auc, roc_curve

# Compute ROC curve and ROC area for each class

fpr, tpr, _ = roc_curve(
    y_true,
    y_pred[:, 1],
)
roc_auc = auc(fpr, tpr)

# %%
plt.figure()
lw = 2
plt.plot(fpr, tpr, color="darkorange", lw=lw, label="ROC curve (area = %0.2f)" % roc_auc)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic example")
plt.legend(loc="lower right")
plt.show()

# %% [markdown]
# ### Misclassified examples
#
# It is always interesting to check mis classified examples.
#
# It usually provides tips on how to improve your model.

# %%
misclassified_idxs = np.where(y_pred_classes != y_true)

misclassified_images = train_images[misclassified_idxs]
misclassified_labels = train_labels[misclassified_idxs]

grid_size = 4
grid = np.zeros((grid_size * 64, grid_size * 64, 3)).astype(np.uint8)
for i in range(grid_size):
    for j in range(grid_size):
        img = np.copy(misclassified_images[i * grid_size + j])
        lbl = np.copy(misclassified_labels[i * grid_size + j])
        color = (0, 255, 0) if lbl == 1 else (255, 0, 0)
        tile = cv2.rectangle(img, (0, 0), (64, 64), color, thickness=2)
        grid[i * 64 : (i + 1) * 64, j * 64 : (j + 1) * 64, :] = img

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax.imshow(grid)
plt.show()

# %% [markdown]
# ## Using more advanced features with pytorch-ignite
#
# We will add more advanced features using handlers and data augmentation,
#
# Here, either write a new loop or edit the train loop above

# %% [markdown]
# ### Adding more handlers: Early stopping
#
# Now we will setup a `EarlyStopping` handler for this training process. EarlyStopping requires a score_function that allows the user to define whatever criteria to stop trainig. In this case, if the loss of the validation set does not decrease in 10 epochs, the training process will stop early. Since the `EarlyStopping` handler relies on the validation loss, it's attached to the `val_evaluator`.
#
#
# Now we will setup a `EarlyStopping` handler for this training process. EarlyStopping requires a score_function that allows the user to define whatever criteria to stop trainig. In this case, if the loss of the validation set does not decrease in 10 epochs, the training process will stop early. Since the `EarlyStopping` handler relies on the validation loss, it's attached to the `val_evaluator`.
#
# ```python
# def score_function(engine):
#     val_loss = engine.state.metrics["nll"]
#     return -val_loss
#
# handler = EarlyStopping(patience=10, score_function=score_function, trainer=trainer)
#
# val_evaluator.add_event_handler(Events.COMPLETED, handler)
# ```
#
# Add this to your training engine (you can then start the training again and train for longer)

# %%

# %% [markdown]
# ### Adding more handlers: Configuring model checkpoint to retain only best models
#
# The current model checkpoint configuration saves checkpoint every n epoch. This may not be the best solution, why not save, for example, the last 2 best models in term of validation loss ?
#
# **Adapt the following code to your use case**
#
# ```python
#
# # Store the best model
# def default_score_fn(engine):
#     score = engine.state.metrics['Accuracy']
#     return score
#
#
# best_model_handler = ModelCheckpoint(dirname=log_path,
#                                      filename_prefix="best",
#                                      n_saved=3,
#                                      global_step_transform=global_step_from_engine(trainer),
#                                      score_name="test_acc",
#                                      score_function=default_score_fn)
# evaluator.add_event_handler(Events.COMPLETED, best_model_handler, {'model': model, })
#
# ```

# %%

# %% [markdown]
# ### Data Augmentation
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
# - Here is how we use torch Compose to augment data
#
# https://pytorch.org/docs/stable/torchvision/transforms.html
#
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#compose-transforms
#
# Note: This step requires a bit of tinkering from numpy arrays to torch datasets, it's fine if you skip it. For the next notebook it may prove a useful way of gaining performance
#
# **Remember : We apply data augmentation only during training**
#

# %%
import torch.functional as F
import torch.utils
import torchvision.transforms

# %%
# Example (very simple) data augmentation to get your started, you can add more transforms to this list

train_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomVerticalFlip(p=0.5),
        torchvision.transforms.ToTensor(),
    ]
)

# %%
trainset_augmented = NpArrayDataset(
    images=train_images, labels=train_labels, image_transforms=train_transform, label_transforms=None
)

# %%
# Get image from dataset. Note: it has been converted as a torch tensor in CHW format in float32 normalized !
img, label = trainset_augmented[0]
img = (img.numpy() * 255.0).astype(np.uint8)
img = np.rollaxis(img, 0, 3)
plt.imshow(img)
plt.show()

# Compare effects of data augmentation
img_orig = trainset_augmented.images[0]
plt.imshow(img_orig)
plt.show()

# %%
# plot our metrics again. Did we change something ?

# %% [markdown]
# ### [Optional] Trying other models
#
# You have seen a class on different model structure,
# https://supaerodatascience.github.io/deep-learning/slides/2_architectures.html#/
#
# Now is the time to try and implement them. For example, try to write a VGG-11 with fewer filters by yourself... or a very small resnet using [this](https://github.com/a-martyn/resnet/blob/master/resnet.py) as inspiration
#
# You can also use models from [torchvision](https://pytorch.org/docs/stable/torchvision/models.html#classification) in your loop, or as inspiration
#
# **Modify the model structure and launch another training... Is it better ?**

# %%

# %% [markdown]
# ### [Optional] Next steps before the next notebooks
#
# - Try to play with network hyperparameters. The dataset is small and allow fast iterations so use it to have an idea on hyperparameter sensitivity.
#     number of convolutions, other network structures, learning rates, optimizers,...
#
# - Example: Compare again SGD and ADAM
#
# - Try to use the ROC curve to select a threshold to filter only negative examples without losing any positive examples
#
# When you are done with the warmup, go to the next notebook. But remember that next datasets will be larger and you will not have the time (trainings will take longer ) to experiment on hyperparameters.
#
# **Try more things before going to the next notebook**

# %%

# %% [markdown]
# ## Food for thoughts: Tooling
#
# To conclude this notebook, reflect on the following,
#
# You have launched different experiences and obtained different results,
#
# Did you feel the notebook you used was sufficient ? Which tools would you like to have in order to properly run your experiments ? (Quick google search or ask someone) Do they already exist ?

# %%
