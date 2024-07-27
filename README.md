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
