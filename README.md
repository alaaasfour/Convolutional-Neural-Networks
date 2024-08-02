# Convolutional Neural Network (CNN) Implementation in Numpy 🪄

## Description 📖
### This project implements a Convolutional Neural Network (CNN) with forward and backward propagation, including convolutional (CONV) and pooling (POOL) layers using Numpy. It includes both the forward and backward propagation steps for convolutional and pooling layers.

## Prerequisites 🐍
Make sure you have Python 3.10+ installed on your machine. In addition to the following packages:
* `numpy`
* `h5py`
* `matplotlib`

<br>To run: `Building_Convolutional_Neural_Network.py`

## Functions Implemented & Features 🚀✨
1. Zero-Padding: Adds zeros around the border of the image to maintain the size of the image after convolution.
2. Single Step of Convolution: Applies one filter to a single position of the input.
3. Forward Propagation - Convolution: Performs the forward pass for the convolutional layer.
4. Forward Propagation - Pooling: Performs the forward pass for the pooling layer, supporting both max and average pooling.
5. Backward Propagation - Convolution: Performs the backward pass for the convolutional layer.
6. Backward Propagation - Pooling: Performs the backward pass for the pooling layer.
7. Create Mask from Window: Creates a mask matrix for max-pooling.
8. Distribute Value: Distributes the gradient for average pooling.

⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯

# Convolutional Neural Network (CNN) Implementation using TensorFlow Keras Functional API 🪄

## Description 📖
### This script demonstrates the creation and training of two convolutional neural networks (ConvNets) using TensorFlow Keras API for two different tasks:

1. A mood classifier to identify whether people are smiling or not using the Happy House dataset.
2. A sign language digit classifier using the Sign Language Digits dataset.

## Prerequisites 🐍🐼
Make sure you have Python 3.10+ installed on your machine. In addition to the following packages:
* `numpy`
* `h5py`
* `matplotlib`
* `TensorFlow`
* `PIL (Pillow)`
* `pandas`
* `scipy`

## Functions Implemented & Features 🚀✨
1. Loading and Splitting the Data
2. Creating the Sequential Model: We create a sequential model called `happyModel()` for mood classification. The model architecture includes:
   - ZeroPadding2D
   - Conv2D
   - BatchNormalization
   - ReLU
   - MaxPooling2D
   - Flatten
   - Dense
3. The Functional API: We load the Sign Language Digits dataset, normalize the image vectors, and convert labels to one-hot encoding.
4. Creating the Functional Model: We create a functional model called `convolutional_model` for sign language digit classification. The model architecture includes:
   - Conv2D
   - ReLU
   - MaxPooling2D
   - Flatten
   - Dense

## How to Run 🏃🏻‍♂️
1. Ensure all dependencies are installed.
2. Place the dataset files in the appropriate directory.
3. Run the script: `python Convolutional_Neural_Networks_Application.py`
4. The script will load the datasets, create the models, train them, and display the training history.

⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯

# Alpaca/Not Alpaca Classifier using Transfer Learning 🦙🔎

## Description 📖
### This project involves building a binary classifier to distinguish between images of Alpacas and other objects. The classifier is built using transfer learning with the pre-trained MobileNetV2 model. MobileNetV2 is an efficient convolutional neural network architecture optimized for mobile and embedded vision applications. The model is pre-trained on the ImageNet dataset, which contains over 14 million images and 1000 classes.

## Purpose of this Script
### Transfer learning allows you to leverage pre-trained models to build powerful image classifiers with limited data and computational resources. This project uses MobileNetV2, a lightweight and efficient CNN, to classify images as either containing an alpaca or not.

## Prerequisites 🐍🐼
Make sure you have Python 3.10+ installed on your machine. In addition to the following packages:
* `python 3.x`
* `matplotlib`
* `TensorFlow`
* `keras`
* `numpy`


## Dataset 💾
### The dataset should be organized in a directory structure as follows:

datasets/
    alpaca/
        alpaca1.jpg
        alpaca2.jpg
        ...
    not_alpaca/
        not_alpaca1.jpg
        not_alpaca2.jpg
        ...

## Training & Key Steps 🏋️‍♀️🔑
### The training process involves several key steps:
1. Data Augmentation: Randomly flip and rotate the images to increase the diversity of the training data.
2. Data Preprocessing: Prefetch data to prevent memory bottlenecks and normalize images using MobileNetV2 preprocessing.
3. Transfer Learning: Use MobileNetV2 as the base model, and add custom layers for binary classification.
4. Model Compilation: Compile the model with the Adam optimizer and binary cross-entropy loss.
5. Initial Training: Train the model with the base layers frozen for a few epochs.
6. Fine-Tuning: Unfreeze some layers and continue training with a lower learning rate.


## Screenshots 🖼️
1. Sample from the Dataset (Alpaca/Non-Alpaca)
![Alpaca-NonAlpaca Dataset.png](Alpaca-NonAlpaca%20Dataset.png)

2. Data Augmenter Function: an image from the training set has been augmented with simple transformations
![Simple Transformation.png](Simple%20Transformation.png)

3. Training and Validation Accuracy (Before Fine-Tuning)
![Data Accuracy Before FT.png](Data%20Accuracy%20Before%20FT.png)

4. Training and Validation Accuracy (After Fine-Tuning)
![Data Accuracy After FT.png](Data%20Accuracy%20After%20FT.png)