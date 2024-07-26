"""
In this script we will do the following:
    - Create a mood classifier using TF (TensorFlow) Keras Functional API
    - Build a ConvNet to identify sign language digits using the TF Keras API
"""

# Packages
import math
import h5py
import scipy
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
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

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))