"""
In this script, we will be using transfer learning on a pre-trained CNN to build an Alpaca/Not Alpaca classifier.

A pre-trained model is a network that's already been trained on a large dataset and saved, which allows us to use it to
customize our own model cheaply and efficiently. The one we'll be using, MobileNetV2, was designed to provide fast and
computationally efficient performance. It's been pre-trained on ImageNet, a dataset containing over 14 million images and 1000 classes.
"""

import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.layers as tfl
from keras.src.layers import RandomFlip, RandomRotation
from keras.src.utils import image_dataset_from_directory
from tensorflow.keras.preprocessing import *

"""
Let's create the dataset and split it into training and validation sets
Note: we set the seeds to match each other, so that the training and validation sets don't overlap.
"""
from keras.src.utils import image_dataset_from_directory
from tensorflow.keras.preprocessing import *

BATCH_SIZE = 32
IMG_SIZE = (160, 160)
directory = "datasets/"
train_dataset = image_dataset_from_directory(directory,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE,
                                             validation_split=0.2,
                                             subset='training',
                                             seed=42)
validation_dataset = image_dataset_from_directory(directory,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE,
                                             validation_split=0.2,
                                             subset='validation',
                                             seed=42)

"""
Let's take a look at some of the images from training set
"""
class_names = train_dataset.class_names
plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
    plt.show()

"""
Exercise 1: Preprocess and Augment Training Data

In data preprocessing we use prefetch() function to prevent a memory bottleneck that can occur when reading from disk. 
It sets aside some data and keeps it ready for when it's needed, by creating a source dataset from your input data, 
applying a transformation to preprocess it, then iterating over the dataset one element at a time. 
Because the iteration is streaming, the data doesn't need to fit into memory.
    
Let's implement a function for data augmentation. We will use a Sequential keras model composed of 2 layers:
"""

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

def data_augmenter():
    data_augmentation = tf.keras.Sequential()
    data_augmentation.add(RandomFlip("horizontal"))
    data_augmentation.add(RandomRotation(0.2))

    return data_augmentation

print("Exercise 1: Preprocess and Augment Training Sets")
print("==========")
augmenter = data_augmenter()
assert(augmenter.layers[0].name.startswith('random_flip')), "First layer must be RandomFlip"
assert augmenter.layers[0].mode == 'horizontal', "RandomFlip parameter must be horizontal"
assert(augmenter.layers[1].name.startswith('random_rotation')), "Second layer must be RandomRotation"
assert len(augmenter.layers) == 2, "The model must have only 2 layers"

print('\033[92mAll tests passed!')

# Let's take a look at how an image from the training set has been augmented with simple transformation

data_augmentation = data_augmenter()
for image, _ in train_dataset.take(1):
    plt.figure(figsize=(10, 10))
    first_image = image[0]
    for i in range(9):
        ax = plt.subplot(3, 3, i+1)
        augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
        plt.imshow(augmented_image[0] / 255)
        plt.axis('off')
    plt.show()

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
print("========================================")

"""
Exercise 2: Using MobileNetV2 for Transfer Learning

MobileNetV2 was trained on ImageNet and is optimized to run on mobile and other low-power applications.
Let's try to train the base model using all the layers from the pretrained model.
Similarly to how we reused the pretrained normalization values MobileNetV2 was trained on, we'll also load the pretrained weights from ImageNet.
"""
print("Exercise 2: Printing the Model Summary")
print("==========")
IMG_SHAPE = IMG_SIZE + (3,)
base_model_path="imagenet_base_model/with_top_mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_160.h5"
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=True,
                                               weights=base_model_path)
base_model.summary()

# Note the last 2 layers here. They are the so-called top layers, and they are responsible for the classification in the model
nb_layers = len(base_model.layers)
print(base_model.layers[nb_layers - 2].name)
print(base_model.layers[nb_layers - 1].name)

# Next, choose the first batch from the tensorflow dataset to use the images, and run it through the MobileNetV2 base model to test out the predictions on some of your images.
image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)
print(feature_batch.shape)

#Shows the different label probabilities in one tensor
print(label_batch)
print("========================================")
