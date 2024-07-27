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
* `h5py`
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

## Functions Implemented & Features 🏃🏻‍♂️
1. Ensure all dependencies are installed.
2. Place the dataset files in the appropriate directory.
3. Run the script: `python Convolutional_Neural_Networks_Application.py`
4. The script will load the datasets, create the models, train them, and display the training history.




