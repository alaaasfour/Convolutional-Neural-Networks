"""
In this script, we'll be building our own U-Net, a type of CNN designed for quick, precise image segmentation, and using
it to predict a label for every single pixel in an image - in this case, an image from a self-driving car dataset.

In this script, we will do the following:
    - Build your own U-Net
    - Explain the difference between a regular CNN and a U-net
    - Implement semantic image segmentation on the CARLA self-driving car dataset
    - Apply sparse categorical crossentropy for pixelwise prediction
"""
# Packages
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import imageio.v2 as imageio
import os

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate

from test_utils import summaryUNet, comparatorUNet

# Load and Split the Data
path = ''
image_path = os.path.join(path, './data/CameraRGB/')
mask_path = os.path.join(path, './data/CameraMask/')
image_list_orig = os.listdir(image_path)
image_list = [image_path+i for i in image_list_orig]
mask_list = [mask_path+i for i in image_list_orig]

# Check out some of the unmasked and masked images from the dataset
N = 2
img = imageio.imread(image_list[N])
mask = imageio.imread(mask_list[N])
#mask = np.array([max(mask[i, j]) for i in range(mask.shape[0]) for j in range(mask.shape[1])]).reshape(img.shape[0], img.shape[1])

fig, arr = plt.subplots(1, 2, figsize=(14, 10))
arr[0].imshow(img)
arr[0].set_title('Image')
arr[1].imshow(mask[:, :, 0])
arr[1].set_title('Segmentation')
plt.show()

# Split the dataset into unmasked and masked images
image_list_ds = tf.data.Dataset.list_files(image_list, shuffle=False)
mask_list_ds = tf.data.Dataset.list_files(mask_list, shuffle=False)

for path in zip(image_list_ds.take(3), mask_list_ds.take(3)):
    print(path)

image_filenames = tf.constant(image_list)
masks_filenames = tf.constant(mask_list)

dataset = tf.data.Dataset.from_tensor_slices((image_filenames, masks_filenames))

for image, mask in dataset.take(1):
    print(image)
    print(mask)

# Preprocess the Data
def process_path(image_path, mask_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=3)
    mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)

    return image, mask

def preprocess(image, mask):
    input_image = tf.image.resize(image, (96, 128), method='nearest')
    input_mask = tf.image.resize(mask, (96, 128), method='nearest')

    return input_image, input_mask

image_ds = dataset.map(process_path)
processed_image_ds = image_ds.map(preprocess)

"""
Exercise 1: U-Net & conv_block Implementation

U-Net, named for its U-shape, was originally created in 2015 for tumor detection, but in the years since has become a 
very popular choice for other semantic segmentation tasks.

U-Net builds on a previous architecture called the Fully Convolutional Network, or FCN, which replaces the dense layers 
found in a typical CNN with a transposed convolution layer that up-samples the feature map back to the size of the original 
input image, while preserving the spatial information. 
This is necessary because the dense layers destroy spatial information (the "where" of the image), which is an essential 
part of image segmentation tasks. An added bonus of using transpose convolutions is that the input size no longer needs 
to be fixed, as it does when dense layers are used.

We'll implement the conv_block() function for convolutional downsampling block by following these steps:
    1. Add 2 Conv2D layers with n_filters filters with `kernel_size` set to 3, `kernel_initializer` set to 'he_normal', padding set to 'same' and 'relu' activation.
    2. If `dropout_prob` > 0, then add a Dropout layer with parameter dropout_prob
    3. If `max_pooling` is set to True, then add a MaxPooling2D layer with 2x2 pool size
    
Argument:
    inputs: Input tensor
    n_filters: Number of filters for the convolutional layers
    dropout_prob: Dropout probability
    max_pooling: Use MaxPooling2D to reduce the spatial dimensions of the output volume

Returns:
    next_layer, skip_connection --  Next layer and skip connection outputs
"""

def conv_block(inputs=None, n_filters=32, dropout_prob=0, max_pooling=True):
    conv = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv)

    # If dropout_prob > 0 add a dropout layer, with the variable dropout_prob as parameter
    if dropout_prob > 0:
        conv = Dropout(dropout_prob)(conv)

    # If max_pooling is True add a MaxPooling2D with 2x2 pool_size
    if max_pooling:
        next_layer = MaxPooling2D(pool_size=(2, 2))(conv)
    else:
        next_layer = conv
    skip_connection = conv

    return next_layer, skip_connection

print("Exercise 1: U-Net & conv_block Implementation")
print("==========")
input_size = (96, 128, 3)
n_filters = 32
inputs = Input(input_size)
cblock1 = conv_block(inputs, n_filters * 1)
model1 = tf.keras.Model(inputs=inputs, outputs=cblock1)

output1 = [['InputLayer', [None, 96, 128, 3], 0],
           ['Conv2D', [None, 96, 128, 32], 896, 'same', 'relu', 'HeNormal'],
           ['Conv2D', [None, 96, 128, 32], 9248, 'same', 'relu', 'HeNormal'],
           ['MaxPooling2D', [None, 48, 64, 32], 0, (2, 2)]]

print('Block 1:')
for layer in summaryUNet(model1):
    print(layer)

comparatorUNet(summaryUNet(model1), output1)

inputs = Input(input_size)
cblock1 = conv_block(inputs, n_filters * 32, dropout_prob=0.1, max_pooling=True)
model2 = tf.keras.Model(inputs=inputs, outputs=cblock1)

output2 = [['InputLayer', [None, 96, 128, 3], 0],
           ['Conv2D', [None, 96, 128, 1024], 28672, 'same', 'relu', 'HeNormal'],
           ['Conv2D', [None, 96, 128, 1024], 9438208, 'same', 'relu', 'HeNormal'],
           ['Dropout', [None, 96, 128, 1024], 0, 0.1],
           ['MaxPooling2D', [None, 48, 64, 1024], 0, (2, 2)]]

print('\nBlock 2:')
for layer in summaryUNet(model2):
    print(layer)

comparatorUNet(summaryUNet(model2), output2)
print("========================================")
