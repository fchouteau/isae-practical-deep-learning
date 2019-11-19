---
title: ISAE Practical Deep Learning Session
theme: solarized
highlightTheme: solarized-light
separator: <!--s-->
verticalSeparator: <!--v-->
revealOptions:
    transition: 'fade'
    transitionSpeed: 'default'
    controls: false
---

# Deep Learning in Practice

**ISAE-SUPAERO, SDD, December 2019**

Florient CHOUTEAU, Marina GRUET, Matthieu LE GOFF

<!--s-->

## Detect Aircrafts on Satellite Imagery

![](static/img/aircrafts.gif)

<!--v-->

6 hours hands on session on applying "deep learning" to a "real" use-case

<img src="static/img/dog_meme.jpg" alt="" width="512px" height="390px" style="background:none; border:none; box-shadow:none;"/>

<!--v-->
<!-- .slide: data-background="http://i.giphy.com/90F8aUepslB84.gif" -->

### Lesson #1

These slides are built using [reveal.js](https://revealjs.com) and [reveal-md](
https://github.com/webpro/reveal-md)

This is awesome ! Stop using powerpoint !

<!--v-->

### Who we are


- Computer Vision R&D at Airbus Defence and Space
<img src="static/img/airbus_logo_white.png" alt="" width="220px" height="44px" style="background:none; border:none; box-shadow:none;"/>
- Ground segment software for earth observation satellites
- Working daily with Deep Learning on satellite imagery
    - Information extraction (object detection, change detection...)
    - Image processing (clouds, image enhancement)
    - Research stuff (image simulation, self-supervision...)
    
<!--v-->

### Context

<img src="static/img/context.png" alt="" width="80%" height="80%" style="background:white; border:none; box-shadow:none;"/>

<!--v-->

### What we are going to do

Train an aircraft detector on a dataset of aircrafts and "not aircrafts"

- ... using convolutional neural networks
- ... using Pytorch
- ... using a compute instance on Google Cloud Platform (with a GPU)

<img src="static/img/aiplatform.png" alt="" width="170px" height="150px" style="background:white; border:none; box-shadow:none;"/>

<!--s-->

## Session 1: Hands-On

<!--v-->

### Objectives

- Setup your work environment on GCP
- Train a basic CNN on a small training set
- Plot the ROC curve on a small test set

<!--v-->

### Outcomes

- Use GCP to get access to computing power
- Handle a dataset of images, do some basic data exploration
- Review the basics of training Neural Network with Pytorch

<!--v-->

### Dataset description

*Include dataset description*

<!--v-->

### Steps by steps

1. Create your GCP Instance
2. Connect to jupyterlab
3. Import the first notebook & follow it
4. Profit !
5. If you're done... go to Session 2 !

<!--s-->

## Session 2: High-level framework, class imbalance, sliding windows

<!--v-->

### Objectives

- Train a CNN on a larger & imbalanced dataset
- Diagnose the performance of a model on imbalanced data
- Apply your model on larger images to detect aircrafts

<!--v-->

### Dataset description

*Include dataset description*

<!--v-->

### Outcomes

- Discover higher-level DL frameworks (skorch)
- Tackle a dataset with huge class imbalance
- Discover Precision-Recall Curves
- Discover applying models on larger images using the sliding window technique

<!--v-->

### Steps by steps

1. Start/Restart your machine
2. Follow notebooks 2 and 3

![](https://i.stack.imgur.com/U9Iki.png)

<!--s-->

## Creating our GCP Deep Learning VM

<!--v-->

*Include step by step tutorial*

<!--s-->

## Concluding remarks

<!--v-->

Welcome to the life of a deep learning engineer !

<!--v-->

![](static/img/tesla.jpg)

<!--v-->

Contact:  
florient.f.chouteau@airbus.com  
matthieu.le-goff@airbus.com  
marina.gruet@airbus.com  

