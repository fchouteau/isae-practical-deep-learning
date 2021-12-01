---
title: ISAE Practical Deep Learning Session
theme: evo
highlightTheme: zenburn
separator: <!--s-->
verticalSeparator: <!--v-->
revealOptions:
    transition: 'fade'
    transitionSpeed: 'default'
    controls: true
    slideNumber: true
    width: '100%'
    height: '100%'
---

# Deep Learning in Practice

**ISAE-SUPAERO, SDD, Nov/Dev 2021**

Florient CHOUTEAU

<!--v-->

Slides : https://fchouteau.github.io/isae-practical-deep-learning/

Notebooks : https://github.com/fchouteau/isae-practical-deep-learning

<!--s-->

## Detect Aircrafts on Satellite Imagery

![ac](static/img/aircrafts.gif)

<!--v-->

6 hours hands on session on applying "deep learning" to a "real" use-case

![dog_cat_meme](static/img/dog_meme.jpg) <!-- .element: height="40%" width="40%" -->

<!--v-->
<!-- .slide: data-background="http://i.giphy.com/90F8aUepslB84.gif" -->

### Lesson #1 <!-- .element: style="color: white; font-family: serif; font-size: 1.2em;" -->

These slides are built using <!-- .element: style="color: white; font-family: cursive; font-size: 1.2em;" --> [reveal.js](https://revealjs.com) and [reveal-md](
https://github.com/webpro/reveal-md)

This is awesome ! 😲 <!-- .element: style="color: white; font-family: cursive; font-size: 1.2em;" -->

<!--v-->

### Who ?

- ![ads](static/img/AIRBUS_Blue.png) <!-- .element: height="44px" width="220px" -->
- SDD 2016
- Computer Vision R&D at **Airbus Defence and Space**
- Ground segment software for Earth Observation satellites
- Daily job revolving around Machine Learning + Satellite Imagery
    - Information extraction
    - Image processing
    - Research stuff

(contact me on slack)

<!--v-->

### Who ?

You'll see me a bit this year :
- CNN for Computer Vision (this BE)
- January Hackathon (CNN for Computer Vision)
- Outils du Big Data : Cloud, Docker, Deployment

<!--v-->

### Context: Earth Observation

![context](static/img/context.png)  <!-- .element:  width="60%" height="60%"-->

<!--v-->

### Context: Machine Learning on Satellite Imagery

A lot of use cases :

- Land Use / Land Cover cartography
- Urban Cartography (building, roads, damage assessment...)
- Various objects detections (ships, vehicles...)

![shipdet](https://www.aerospace-valley.com/sites/default/files/styles/news_main/public/thumbnails/image/airbus_ship_detection_challenge.png?itok=i7DpZPus)

<!--v-->

### Context: Machine Learning on Satellite Imagery

Can also be used for "image processing" : 

- Denoising
- "Super Resolution" ("enhance")

![](https://maxar-blog-assets.s3.amazonaws.com/uploads/blogImages/HD_Cars.jpg) <!-- .element:  width="20%" height="20%"-->

<!--v-->

### Context: Needles in haystacks

![pyramid](static/img/large_pyramid.jpg)  <!-- .element:  width="40%" height="40%"-->

<!--v-->

### What you did last time 

- Trained a Convolutional Neural Network on Fashion MNIST
- Wrote your first training loops with Pytorch
- Discovered "callbacks" (early stopping), optimizers (sgd, adam), dropout
- Saw your firsts neural architectures (alexnet, vggs, resnets)
- (Maybe discovered pytorch ignite)

<!--v-->

### What we are going to do

Train an aircraft detector on a dataset of aircrafts and "not aircrafts"

- using convolutional neural networks <!-- .element: class="fragment" data-fragment-index="1" -->
- using pytorch <!-- .element: class="fragment" data-fragment-index="2" -->
- using google colaboratory and its GPUs <!-- .element: class="fragment" data-fragment-index="3" -->

![colab](https://miro.medium.com/max/776/1*Lad06lrjlU9UZgSTHUoyfA.png) <!-- .element:  class="fragment" data-fragment-index="4" width="25%" height="25%"-->

<!--s-->

## Session 1: Hands-On

<!--v-->

### Objectives

- Launch notebooks on Colab
- Build an intuition over convolutions and CNNs
- Train a basic CNN on a small training set
- Plot the metrics & ROC curve on a small test set
- Discover the world of hyperparameter tuning

<!--v-->

### Outcomes

- Use GCP to get access to computing power & GPUs
- Handle a dataset of images, do some basic data exploration
- Train & evaluate your first CNN on a simple dataset
- Go beyound accuracy to diagnose your model

<!--v-->

### Dataset description

- 2600 train images (1300 aircrafts, 1300 background), size 64x64
- 880 test images (440 aircrafts, 440 background), size 64x64

![](static/img/toy_dataset.png) <!-- .element height="40%" width="40%" -->

<!--v--> 

### Pytorch reminder

![](https://pbs.twimg.com/media/E_1d06cVIAcYheX?format=jpg&name=large)  <!-- .element height="50%" width="50%" -->

<!--v-->

### Pytorch reminder

![](https://pbs.twimg.com/media/E_1d06XVcA8Dhzs?format=jpg&name=large)  <!-- .element height="50%" width="50%" -->


<!--v-->

### Let's go ! 

1. Go to google colab
2. Import the first notebook & follow the guidelines
3. ...
4. Profit !
5. If you're done... go to the next notebook !

<!--v-->

### Colab Guide

<video data-autoplay  controls width="720">
    <source src="https://storage.googleapis.com/fchouteau-isae-deep-learning/static/colab_guide_proper.mp4" type="video/mp4">
</video>

<!--v-->

### GPU ???

You'll see that... in February

[Tutorial](https://colab.research.google.com/github/d2l-ai/d2l-en-colab/blob/master/chapter_deep-learning-computation/use-gpu.ipynb)

<!--s-->

## Session 1
## Take-Away messages

<!--v-->

### Kernel filtering

![](https://miro.medium.com/max/1400/1*Fw-ehcNBR9byHtho-Rxbtw.gif) <!-- .element height="30%" width="30%" -->


<!--v-->

### ConvNets intuition come from image processing

![](https://imgs.developpaper.com/imgs/3480721674-5ae839c26eef6_articlex.jpg) <!-- .element height="30%" width="30%" -->


<!--v-->

### ConvNets intuition come from image processing

![](https://ai.stanford.edu/~syyeung/cvweb/Pictures1/sharpening2.png) <!-- .element height="60%" width="60%" -->

(I apologize for using Lena as an example)

<!--v-->

ConvNets works because we assume inputs are images

<!--v-->

### ConvNets

![feature](https://www.mdpi.com/sensors/sensors-19-04933/article_deploy/html/images/sensors-19-04933-g001.png)  <!-- .element height="60%" width="60%" -->

<!--v-->

### Convolutions ?

![cnns](https://i.stack.imgur.com/FjvuN.gif)  <!-- .element height="40%" width="40%" -->

[useful link](https://github.com/vdumoulin/conv_arithmetic)

<!--v-->

### Pooling ?

![](https://developers.google.com/machine-learning/practica/image-classification/images/maxpool_animation.gif) <!-- .element height="40%" width="40%" -->

<!--v-->

### nn.Linear ?

![](static/img/nnlinear.png) <!-- .element height="40%" width="40%" -->

<!--v-->

### Computing shape

![tiles](static/img/cnnshape.png) <!-- .element height="35%" width="35%" -->

<!--v-->

### CNNs in practice...

![tiles](static/img/torchvision.png) <!-- .element height="35%" width="35%" -->

```text
ResNet(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace=True)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (layer1): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (1): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer2): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer3): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer4): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (fc): Linear(in_features=512, out_features=2, bias=True)
)
```

<!--v-->

### ROC-Curves

(see section "extra" on these slides)

<!--s-->

## Session 2
## Class Imbalance & Sliding Windows

<!--v-->

### Objectives

- Train a CNN on a larger & imbalanced dataset
- Evaluate the performance of a model on imbalanced data
- Try and improve performance
- Apply your model on larger images to detect aircrafts

<!--v-->

### Trainval Dataset description

- 46000 64x64 train images
- 10240 64x64 test images
- **1/10 aircraft-background ratio**

![tiles](static/img/large_dataset.png) <!-- .element height="35%" width="35%" -->

<!--v-->

### Final Dataset description

- Objective: Apply your classifier on "real" images and find aircrafts
- 36 512x512 images with some aircrafts

![tiles](static/img/large_tiles.png) <!-- .element height="35%" width="35%" -->

<!--v-->

### One idea: Sliding window

- Training Image Size: 64x64, output = binary classification
- Target Image Size: 512x512, target = detect & count aircrafts ?

![sliding](static/img/sliding_window.gif)

<!--v-->

### Outcomes

- Tackle a dataset with huge class imbalance
- Discover more advanced techniques for training CNNs
- Discover Precision-Recall Curves
- Discover applying models on larger images using the sliding window technique

<!--v-->

### Steps by steps

1. Start/Restart your machine
2. Follow notebooks 2 and 3

![xkcd](https://i.stack.imgur.com/U9Iki.png)

<!--s-->

## Session 2: Take-home messages

<!--v-->

### Objectives

- Continue manipulating CNNs using pytorch / ignite
- Tackle a more realistic dataset
- Examine what must changes to diagnose your model and improve it

<!--v-->

Welcome to the life of a deep learning engineer !

![train](static/img/model_train_img.png)

<!--v-->

![data](static/img/tesla.jpg) <!-- .element height="70%" width="70%" -->

<!--v-->

![goodbye](https://media.giphy.com/media/lD76yTC5zxZPG/giphy.gif)

<!--s-->

## Extra
## Diagnosing Classifier performance

<!--v-->

### Binary classification metrics

![cm](static/img/confusion_matrix.png)

<!--v-->

### The ROC Curve

![roc](static/img/roc-curve-v2.png)

<!--v-->

### The ROC curve (visualized)

![roc](https://raw.githubusercontent.com/dariyasydykova/open_projects/master/ROC_animation/animations/ROC.gif)

The shape of an ROC curve changes when a model changes the way it classifies the two outcomes.

<!--v-->

### How to compute a ROC curve ?

![proc](https://raw.githubusercontent.com/dariyasydykova/open_projects/master/ROC_animation/animations/cutoff.gif)  <!-- .element height="40%" width="40%" -->

- y_pred = a list of probas, y_true = a list of 0 or 1
- vertical line : threshold value
- red dot : FPR and TPR for the threshold
- the curve is plotted for all available thresholds

<!--v-->

### Precision & Recall

Usually the most important things in imbalanced classification

![pr](static/img/precision_recall.png)  <!-- .element height="40%" width="40%" -->

<!--v-->

### PR synthetic metric

![fbeta](https://i.stack.imgur.com/swW0x.png) <!-- .element height="35%" width="35%" -->

- beta = 1 => Recall & Precision weighted equally
- beta > 1 => Emphasizes recall (not missing positive examples)
- beta < 1 => Emphasizes precision (not doing )

<!--v-->

### The PR Curve

![pr](static/img/pr_curve.png) <!-- .element height="75%" width="75%" -->

<!--v-->

### The PR Curve (visualized)

![pr](https://raw.githubusercontent.com/dariyasydykova/open_projects/master/ROC_animation/animations/PR.gif)

The shape of the precision-recall curve also changes when a model changes the way it classifies the two outcomes.

<!--v-->

### Precision-Recall or ROC ?

- Both curve can be used to select your trade-off
- Precision-recall curve is more sensitive to class imbalance than an ROC curve
- Example: Try computing your FPR on very imbalanced dataset

![prroc](https://raw.githubusercontent.com/dariyasydykova/open_projects/master/ROC_animation/animations/imbalance.gif)  <!-- .element height="50%" width="50%" -->

<!--v-->

### Curves Usage: Selecting trade-off

![calib](static/img/pr_space.png)  <!-- .element height="70%" width="70%" -->

<!--v-->

Readings:
- https://lukeoakdenrayner.wordpress.com/2018/01/07/the-philosophical-argument-for-using-roc-curves/
- https://towardsdatascience.com/on-roc-and-precision-recall-curves-c23e9b63820c

<!--s-->

## Extra : Pytorch Ecosystem

<!--v-->

### high-level frameworks over pytorch

- pytorch: define your models, autodifferenciation, **but you write the rest**
- hl library: training loops, callbacks, distribution etc...

![ignite](https://raw.githubusercontent.com/pytorch/ignite/master/assets/ignite_vs_bare_pytorch.png)  <!-- .element height="50%" width="50%" -->

<!--v-->

### high-level frameworks over pytorch

![lightning](https://miro.medium.com/max/5616/1*5H6pJX8pejhywN72WsDogQ.jpeg) <!-- .element height="40%" width="40%" -->

<!--v-->

### ![pytorch-ignite](https://raw.githubusercontent.com/pytorch/ignite/master/assets/logo/ignite_logo_mixed.svg) <!-- .element:  width="15%" height="15%"-->

- [pytorch-ignite](https://github.com/skorch-dev/skorch) : a high-level deep learning library based on top of pytorch
- Reduce boilerplate code (training loops, early stopping, logging...)
- Extensible, based on experiment management

<!--v-->

### Pytorch Ecosystem 

- There are other high-level frameworks based on pytorch: [Skorch](https://github.com/skorch-dev/skorch), [Lightning](https://github.com/williamFalcon/pytorch-lightning). 
- All of them have their pros and cons
- [There is a huge ecosystem based around pytorch](https://pytorch.org/ecosystem/)

![](https://miro.medium.com/max/5616/1*5H6pJX8pejhywN72WsDogQ.jpeg) <!-- .element: style="width: 25%; height: 25%"--> 

