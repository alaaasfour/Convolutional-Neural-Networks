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

⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯

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

⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯

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
├── alpaca/  
│ ├── alpaca1.jpg  
│ ├── alpaca2.  
│ └── ...  
└── not_alpaca/  
├── not_alpaca1.jpg  
├── not_alpaca2.jpg  
└── ...  

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

⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯

# Image Segmentation with U-Net 🏞️

## Description 📖
### This project implements a U-Net, a type of Convolutional Neural Network (CNN) designed for quick and precise image segmentation, and uses it to predict a label for every pixel in an image from a self-driving car dataset.

## Prerequisites 🐍🐼
Make sure you have Python 3.10+ installed on your machine. In addition to the following packages:
* `python 3.x`
* `matplotlib`
* `TensorFlow`
* `keras`
* `numpy`
* `pandas`
* `imageio.v2`


## Dataset 💾
### The dataset used in this project comes from the CARLA self-driving car simulator. It consists of images and their corresponding masks, which provide pixel-wise annotations for various objects.

## Architecture 🏛️ 

### U-Net 🕸️
U-Net, named for its U-shape, was originally created for tumor detection but has since become popular for other semantic segmentation tasks. It builds on the Fully Convolutional Network (FCN) architecture, replacing dense layers with transposed convolution layers to upsample the feature map back to the original input size while preserving spatial information.


## Components ⛓️🧮
#### 1. Encoder (Downsampling Path): A series of convolutional blocks with Conv2D layers, optional dropout, and MaxPooling2D layers.
#### 2. Bottleneck: The lowest part of the U-Net where the image representation is at its most abstract.
#### 3. Decoder (Upsampling Path): A series of upsampling blocks with Conv2DTranspose layers and skip connections from the encoder.


## Code Implementation 🧑🏻‍💻
The code implements the U-Net architecture as follows:

### 1. Convolutional Block: `conv_block`
### 2. Upsampling Block: `upsampling_block`
### 3. U-Net Model: `unet_model`


## Usage 🧰

### Preprocessing the Data
The dataset is preprocessed by reading the images and masks, resizing them, and converting them to float32 data type.

### Training the Model
The model is compiled with the Adam optimizer and sparse categorical cross-entropy loss, and trained on the processed dataset.

## Results 📉📈
The model's performance can be visualized by comparing the predicted masks with the ground truth masks. The code includes functions to display the input image, true mask, and predicted mask.

## Screenshots 🖼️
1. Unmasked and Masked Images Samples
![Unmasked and Masked Images Samples.png](Unmasked%20and%20Masked%20Images%20Samples.png)

2. Model Accuracy After Training the Model
![Model Accuracy After Training the Model.png](Model%20Accuracy%20After%20Training%20the%20Model.png)

3. Showing the predictions masks against the true mask and the original input image
![Showing the Predictions.png](Showing%20the%20Predictions.png)