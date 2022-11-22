# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] id="7jPPVDkTODxS"
# # Homeworks : From classification to object detection

# %% [markdown] id="jDr8cXgwONqF"
# For now we have a classification model, able to classify small patch of 64x64 image size.
#
# Now we want to detect planes and get their location in 512x512 image size.
#

# %% [markdown] id="C3Y-e6rXPLZ0"
# ## 0. Imports

# %%
# %matplotlib inline

# %% id="-hwnXqMS2jHk"
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# %% id="rufTefgY2p3y"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %% [markdown] id="S0UnZeWMPVtV"
# ## 1. Load data & build data loader

# %% [markdown] id="cGBoxhoXPl1c"
# ### 1.1 Load data
#
# Load the planes dataset we used during the BE.

# %% id="-6HW0SuSQCnI"
# Configuration variables
TOY_DATASET_URL = "https://storage.googleapis.com/fchouteau-isae-deep-learning/toy_aircraft_dataset.npz"


ds = np.DataSource(destpath="/tmp/")
f = ds.open(TOY_DATASET_URL, "rb")

toy_dataset = np.load(f)
trainval_images = toy_dataset["train_images"]
trainval_labels = toy_dataset["train_labels"]
test_images = toy_dataset["test_images"]
test_labels = toy_dataset["test_labels"]

idxs = np.random.permutation(np.arange(trainval_images.shape[0]))

train_idxs, val_idxs = idxs[: int(0.8 * len(idxs))], idxs[int(0.8 * len(idxs)) :]

train_images = trainval_images[train_idxs]
train_labels = trainval_labels[train_idxs]
val_images = trainval_images[val_idxs]
val_labels = trainval_labels[val_idxs]


# %% [markdown] id="go21gpScP7Pu"
# ### 1.2 Build data loader
# Build the data loader as we did during the BE.

# %% colab={"base_uri": "https://localhost:8080/"} id="706Q-XtUFjrg" outputId="8b779a88-95b4-4af3-b748-8cff7ea292ce"
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


# Compute the dataset statistics in [0.,1.], we're going to use it to normalize our data

mean = np.mean(train_images, axis=(0, 1, 2)) / 255.0
std = np.std(train_images, axis=(0, 1, 2)) / 255.0

mean, std


# transform to convert np array in range [0,255] to torch.Tensor [0.,1.]
# then normalize by doing x = (x - mean) / std
image_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)

# here we don't have anything to do
target_transforms = None


# load the training data
train_set = NpArrayDataset(
    images=train_images,
    labels=train_labels,
    image_transforms=image_transforms,
    label_transforms=target_transforms,
)

print(len(train_set))

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

# load the validation data
validation_set = NpArrayDataset(
    images=val_images,
    labels=val_labels,
    image_transforms=image_transforms,
    label_transforms=target_transforms,
)

print(len(validation_set))

val_loader = DataLoader(validation_set, batch_size=64, shuffle=True)


# %% [markdown] id="5fhSx_8pQN8n"
# ## 2. Build model
#

# %% [markdown] id="6t0H2TO3R9FH"
# ### 2.1 Build CNN architecture
# Complete the following model without Linear layers.
#
#
#

# %% colab={"base_uri": "https://localhost:8080/"} id="dRM_iRd5DIql" outputId="3148a05e-1889-4438-84f8-98500f125180"
def _init_weights(model):
    for m in model.modules():
        # Initialize all convs
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")


def model_fn():

    model = nn.Sequential(
        # size: 3 x 64 x 64
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        # size: 32 x 64 x 64
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),
        # size: 32 x 32 x 32
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        # size: 64 x 32 x 32
        nn.MaxPool2d(2),
        # size: 64 x 16 x 16
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2),
        # size: 128 x 8 x 8
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2),
        # size: 128 x 4 x 4
        # Modify Fully Connected to convolutionnal (No more Flatten, no more Linear)
        nn.Conv2d(in_channels=128, out_channels=1024, kernel_size=4, padding=0),
        nn.ReLU(),
        # size: 1024 x 1 x 1
        nn.Dropout2d(p=0.10),
        nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=1, padding=0),
        nn.Sigmoid(),
        # size: 1 x 1 x 1
    )

    _init_weights(model)

    return model


model_name = ...
model = model_fn()

model.to(DEVICE)

x = torch.Tensor(np.ones((1, 3, 64, 64)))
x = x.to(DEVICE)
x.device

print(x.shape)
print(model(x).shape)

# %% colab={"base_uri": "https://localhost:8080/"} id="zkbEEyTWRKU3" outputId="358cb1c8-6768-492f-b958-89e75fb1cca2"
# Let's check input and output size :
x = torch.Tensor(np.zeros((1, 3, 64, 64)))
x = x.to(DEVICE)
y_pred = model(x)

print("Input shape :", x.shape)
print("Output shape :", y_pred.shape)

# %% [markdown] id="unWq4lFoSD-v"
# ### 2.2 Build loss & optimizer

# %% id="iN6i5GdGAaK1"
# Define loss and optimizer
criterion = nn.BCELoss(reduction="mean")
optimizer = optim.Adam(model.parameters(), lr=3e-4)


# %% [markdown] id="mdQuwgBCSVJU"
# ### 2.3 Define training epochs
# Modify y_true shape to match y_pred shape

# %% id="fWnWIFmW2c7G"
def train_one_epoch(model, train_loader):

    epoch_loss = []

    for i, batch in enumerate(train_loader):

        # get one batch
        x, y_true = batch
        x = x.to(DEVICE)
        y_true = y_true.to(DEVICE)

        # format the y_true so that it is compatible with the loss
        # y_true = y_true.view((-1, 1)).float()
        y_true = y_true.view((-1, 1, 1, 1)).float()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        y_pred = model(x)
        # print(y_pred[:, 0, 0, 0], y_true[:, 0, 0, 0])
        # compute loss
        loss = criterion(y_pred, y_true)

        # backward
        loss.backward()

        # update parameters
        optimizer.step()

        # save statistics
        epoch_loss.append(loss.item())

        if i % 10 == 0:
            print(f"Batch {i}, curr loss = {loss.item():.03f}")

    return np.asarray(epoch_loss).mean()


def valid_one_epoch(model, valid_loader):

    epoch_loss = []

    for i, batch in enumerate(valid_loader):
        with torch.no_grad():
            # get one batch
            x, y_true = batch
            x = x.to(DEVICE)
            y_true = y_true.to(DEVICE)

            # format the y_true so that it is compatible with the loss
            # y_true = y_true.view((-1, 1)).float()
            y_true = y_true.view((-1, 1, 1, 1)).float()

            # forward
            y_pred = model(x)

            # compute loss
            loss = criterion(y_pred, y_true)

            # save statistics
            epoch_loss.append(loss.item())

    return np.asarray(epoch_loss).mean()


# %% [markdown] id="LBPBkwnUSg1F"
# ## 3. Train network

# %% colab={"base_uri": "https://localhost:8080/"} id="47QwyZ8P3Z2-" outputId="77a6e121-bea6-4b6d-8cb9-4f2848b4eeb1"
EPOCHS = 10  # Set number of epochs, example 100

# Send model to GPU
model = model.to(DEVICE)

train_losses = []
valid_losses = []

# loop over the dataset multiple times
for epoch in range(EPOCHS):
    model.train()
    train_epoch_loss = train_one_epoch(model, train_loader)
    model.eval()
    valid_epoch_loss = valid_one_epoch(model, val_loader)

    print(f"EPOCH={epoch}, TRAIN={train_epoch_loss}, VAL={valid_epoch_loss}")

    train_losses.append(train_epoch_loss)
    valid_losses.append(valid_epoch_loss)

# %% [markdown] id="OTD7HwePZDJg"
# # 4. Inference on 512x512 image size

# %% [markdown] id="k1e7y_DkZKBR"
# ## 4.1 Load 512x512 dataset

# %% id="nGg29Kgy3guO"
# Configuration variables (512x512)
tiles_dataset_url = "https://storage.googleapis.com/fchouteau-isae-deep-learning/tiles_aircraft_dataset.npz"

# # Configuration variables (64x64)
# TOY_DATASET_URL = "https://storage.googleapis.com/fchouteau-isae-deep-learning/toy_aircraft_dataset.npz"


ds = np.DataSource(destpath="/tmp/")
f = ds.open(tiles_dataset_url, "rb")

tiles_dataset = np.load(f)
eval_images = tiles_dataset["eval_tiles"]

# %% colab={"base_uri": "https://localhost:8080/", "height": 773} id="mq1-GTWfVkVm" outputId="4e0cffb0-9ec0-4357-c6a7-694aa6887c67"
# Display few 512x512 images
for i in range(3):
    plt.imshow(eval_images[i])
    plt.show()

# %% [markdown] id="w_F8tj_KZux-"
# ## 4.2 Apply trained model on 512x512 images

# %% id="ZweYEVmyVpEk"
# Transform images into tensor
eval_image_tensor = torch.Tensor(eval_images)
eval_image_tensor = F.normalize(eval_image_tensor)
# eval_image_tensor = (eval_image_tensor - std) / mean
# Swicth axis (B, H, W, C) to (B, C, H, W). Here (36, 512, 512, 3) to (36, 3, 512, 512)
eval_image_tensor = eval_image_tensor.permute(0, 3, 1, 2)
eval_image_tensor = eval_image_tensor.type(torch.FloatTensor)

eval_image_tensor = eval_image_tensor.to(DEVICE)


# Apply model
y_pred = model(eval_image_tensor)

# %% [markdown] id="7-jbmUzgacxg"
# ## 4.3 Visualize prediction

# %% colab={"base_uri": "https://localhost:8080/", "height": 595} id="cuNe2z0TWC8k" outputId="e48845f8-3954-456e-9d9a-5ef33cb2bea1"
print(y_pred.min(), y_pred.max())

for i in range(3):
    y_pred_i = y_pred[i].detach().cpu().numpy()[0]
    # y_pred_i = y_pred_i > 0.1
    plt.subplot(121)
    plt.imshow(eval_images[i])
    plt.subplot(122)
    plt.imshow(y_pred_i[1:-1, 1:-1], cmap="jet")
    plt.show()

# %% id="lh4C-skLYlJC"
