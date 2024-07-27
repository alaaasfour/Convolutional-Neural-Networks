"""
In this script we will do the following:
    - Create a mood classifier using TF (TensorFlow) Keras Functional API
    - Build a ConvNet to identify sign language digits using the TF Keras API
"""

# Packages
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import scipy
from PIL import Image
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.python.framework import ops
from cnn_utils import *
from test_utils import summary, comparator
np.random.seed(1)

"""
Exercise 1: Loading and Splitting the data into training and test sets

We'll be using the Happy House dataset for this part, which contains images of peoples' faces. 
We will build a ConvNet that determines whether the people in the images are smiling or not -- because they only get to enter the house if they're smiling!
"""

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_happy_dataset()
# Normalize image vectors
X_train = X_train_orig / 255
X_test = X_test_orig / 255

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print("Exercise 1: Loading and Splitting the Data")
print("==========")
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
print("========================================")

"""
Exercise 2: Creating the sequential model

We'll implement the happyModel() function below to build the following model:
ZEROPAD2D -> CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> FLATTEN -> DENSE.

Arguments:
    None

Returns:
    model: TF Keras model (object containing the information for the entire training process) 
"""

def happyModel():
    model = tf.keras.Sequential([
        # ZeroPadding2D with padding 3, input shape of 64 x 64 x 3
        tfl.ZeroPadding2D(padding = (3, 3), input_shape=(64, 64, 3)),
        # Conv2D with 32 7x7 filters and stride of 1
        tfl.Conv2D(filters=32, kernel_size=(7, 7), strides=(1, 1), padding='valid'),
        # BatchNormalization for axis 3
        tfl.BatchNormalization(axis=3),
        # ReLU
        tfl.ReLU(),
        # Max Pooling 2D with default parameters
        tfl.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
        # Flatten layer
        tfl.Flatten(),
        # Dense layer with 1 unit for output & sigmoid activation
        tfl.Dense(units=1, activation='sigmoid')
    ])
    return model


print("Exercise 2: Creating the sequential model / happyModel")
print("==========")
happy_model = happyModel()
# Print a summary for each layer
happy_model.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])

happy_model.summary()
happy_model.fit(X_train, Y_train, epochs=10, batch_size=16)
happy_model.evaluate(X_test, Y_test)
print("========================================")


