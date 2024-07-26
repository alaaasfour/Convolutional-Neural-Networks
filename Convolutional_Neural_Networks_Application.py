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
