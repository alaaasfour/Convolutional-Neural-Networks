"""
In this script, we will implement object detection using the very powerful YOLO (You Only Look Once) model.
In this script, we will do the following:
    - Detect objects in a car detection dataset
    - Implement non-max suppression to increase accuracy
    - Implement intersection over union
    - Handle bounding boxes, a type of image annotation popular in deep learning

Let's assume we are working on a self-driving car project. The purpose of this script is to build a car detection system.
To collect data, we've mounted a camera to the hood of the car, which takes pictures of the road ahead every few seconds as we drive around.

You've gathered all these images into a folder and labelled them by drawing bounding boxes around every car you found.
"""

# Packages
import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
from PIL import ImageFont, ImageDraw, Image
import tensorflow as tf

#from tensorflow.keras.models import load_model
#from yad2k.models.keras_yolo import yolo_head
#from yad2k.utils.utils import draw_boxes, get_colors_for_classes, scale_boxes, read_classes, read_anchors, preprocess_image


"""
Exercise 1: YOLO Model
"You Only Look Once" (YOLO) is a popular algorithm because it achieves high accuracy while also being able to run in real time. 
This algorithm "only looks once" at the image in the sense that it requires only one forward propagation pass through the network to make predictions. 
After non-max suppression, it then outputs recognized objects together with the bounding boxes.

Inputs and Outputs
    - The input is a batch of images, and each image has the shape (608, 608, 3)
    - The output is a list of bounding boxes along with the recognized classes. Each bounding box is represented by 6 numbers  
    (𝑝𝑐,𝑏𝑥,𝑏𝑦,𝑏ℎ,𝑏𝑤,𝑐). If we expand 𝑐 into an 80-dimensional vector, each bounding box is then represented by 85 numbers.

We will implement the yolo_filter_boxes() function as follows:
1. Compute box scores by doing the elementwise product
2. For each box we will find:
    - the index of the class with the maximum box score
    - the corresponding box score
    - Create a mask by using a threshold. The mask should be `True` for the boxes we want to keep.
    - Use TensorFlow to apply the mask to `box_class_scores`, `boxes` and `box_classes` to filter out the boxes we don't want.
Argument:
    boxes: tensor of shape (19, 19, 5, 4)
    box_confidence: tensor of shape (19, 19, 5, 1)
    box_class_probs: tensor of shape (19, 19, 5, 80)
    threshold: real value, if [ highest class probability score < threshold], then get rid of the corresponding box

Returns:
    scores: tensor of shape (None,), containing the class probability score for selected boxes
    boxes: tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
    classes: tensor of shape (None,), containing the index of the class detected by the selected boxes
"""
def yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold = .6):
    # Compute the box scores
    box_scores = box_confidence * box_class_probs

    # Find the box_classes using the max box_scores, keep track of the corresponding score
    box_classes = tf.math.argmax(box_scores, axis = -1)
    box_class_scores = tf.reduce_max(box_scores, axis = -1)

    # Create a filtering mask based on "box_class_scores" by using "threshold". The mask should have the
    # same dimension as box_class_scores, and be True for the boxes we want to keep (with probability >= threshold)
    filtering_mask = box_class_scores >= threshold

    # Apply the mask to box_class_scores, boxes and box_classes
    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)

    return scores, boxes, classes

print("Exercise 1: YOLO Filter Boxes")
print("==========")
tf.random.set_seed(10)
box_confidence = tf.random.normal([19, 19, 5, 1], mean=1, stddev=4, seed = 1)
boxes = tf.random.normal([19, 19, 5, 4], mean=1, stddev=4, seed = 1)
box_class_probs = tf.random.normal([19, 19, 5, 80], mean=1, stddev=4, seed = 1)
scores, boxes, classes = yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold = 0.5)
print("scores[2] = " + str(scores[2].numpy()))
print("boxes[2] = " + str(boxes[2].numpy()))
print("classes[2] = " + str(classes[2].numpy()))
print("scores.shape = " + str(scores.shape))
print("boxes.shape = " + str(boxes.shape))
print("classes.shape = " + str(classes.shape))

assert isinstance(scores, tf.Tensor), "Use tensorflow functions"
assert isinstance(boxes, tf.Tensor), "Use tensorflow functions"
assert isinstance(classes, tf.Tensor), "Use tensorflow functions"

assert scores.shape == (1789,), "Wrong shape in scores"
assert boxes.shape == (1789, 4), "Wrong shape in boxes"
assert classes.shape == (1789,), "Wrong shape in classes"

assert np.isclose(scores[2].numpy(), 9.270486), "Values are wrong on scores"
assert np.allclose(boxes[2].numpy(), [4.6399336, 3.2303846, 4.431282, -2.202031]), "Values are wrong on boxes"
assert classes[2].numpy() == 8, "Values are wrong on classes"

print("\033[92m All tests passed!")
print("========================================")
