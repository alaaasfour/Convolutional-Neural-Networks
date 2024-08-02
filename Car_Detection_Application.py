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
from tensorflow.keras import backend as K

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

"""
Exercise 2: IOU (Intersection over Union) Calculation
Now, we will implement the intersection over union (IoU) between box1 and box2

- For this exercise, a box is defined using its two corners: upper left  (𝑥1,𝑦1)  and lower right  (𝑥2,𝑦2), instead of using the midpoint, height and width.
This makes it a bit easier to calculate the intersection.
- To calculate the area of a rectangle, multiply its height (𝑦2−𝑦1) by its width (𝑥2−𝑥1). Since  (𝑥1,𝑦1)  is the top left and  𝑥2,𝑦2  are the bottom right, 
these differences should be non-negative.

To find the intersection of the two boxes  (𝑥𝑖1,𝑦𝑖1,𝑥𝑖2,𝑦𝑖2):
    - The top left corner of the intersection (𝑥𝑖1,𝑦𝑖1) is found by comparing the top left corners (𝑥1,𝑦1) of the two boxes 
    and finding a vertex that has an x-coordinate that is closer to the right, and y-coordinate that is closer to the bottom.
    - The bottom right corner of the intersection (𝑥𝑖2,𝑦𝑖2) is found by comparing the bottom right corners (𝑥2,𝑦2) of the two 
    boxes and finding a vertex whose x-coordinate is closer to the left, and the y-coordinate that is closer to the top.
    - The two boxes may have no intersection. We can detect this if the intersection coordinates you calculate end up being 
    the top right and/or bottom left corners of an intersection box. Another way to think of this is if you calculate 
    the height (𝑦2−𝑦1) or width (𝑥2−𝑥1) and find that at least one of these lengths is negative, then there is no intersection (intersection area is zero).
    - The two boxes may intersect at the edges or vertices, in which case the intersection area is still zero. 
    This happens when either the height or width (or both) of the calculated intersection is zero.
    
Argument:
    box1: first box, list object with coordinates (box1_x1, box1_y1, box1_x2, box_1_y2)
    box2: second box, list object with coordinates (box2_x1, box2_y1, box2_x2, box2_y2)
"""

def iou(box1, box2):
    (box1_x1, box1_y1, box1_x2, box1_y2) = box1
    (box2_x1, box2_y1, box2_x2, box2_y2) = box2

    # Calculate the (yi1, xi1, yi2, xi2) coordinates of the intersection of box1 and box2. Calculate its Area.
    xi1 = max(box1_x1, box2_x1)
    yi1 = max(box1_y1, box2_y1)
    xi2 = min(box1_x2, box2_x2)
    yi2 = min(box1_y2, box2_y2)

    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    # Calculate the union area by using the formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union_area = box1_area + box2_area - inter_area

    # Compute the IoU
    iou = inter_area / union_area

    return iou

print("Exercise 2: IOU (Intersection over Union) Calculation")
print("==========")
box1 = (2, 1, 4, 3)
box2 = (1, 2, 3, 4)

print("iou for intersecting boxes = " + str(iou(box1, box2)))
assert iou(box1, box2) < 1, "The intersection area must be always smaller or equal than the union area."
assert np.isclose(iou(box1, box2), 0.14285714), "Wrong value. Check your implementation. Problem with intersecting boxes"

## Test case 2: boxes do not intersect
box1 = (1,2,3,4)
box2 = (5,6,7,8)
print("iou for non-intersecting boxes = " + str(iou(box1,box2)))
assert iou(box1, box2) == 0, "Intersection must be 0"

## Test case 3: boxes intersect at vertices only
box1 = (1,1,2,2)
box2 = (2,2,3,3)
print("iou for boxes that only touch at vertices = " + str(iou(box1,box2)))
assert iou(box1, box2) == 0, "Intersection at vertices must be 0"

## Test case 4: boxes intersect at edge only
box1 = (1,1,3,3)
box2 = (2,3,3,4)
print("iou for boxes that only touch at edges = " + str(iou(box1,box2)))
assert iou(box1, box2) == 0, "Intersection at edges must be 0"

print("\033[92m All tests passed!")
print("========================================")

"""
Exercise 3: YOLO Non-max Suppression
Now, we will implement the Non-max Suppression with these steps:
1. Select the box that has the highest score.
2. Compute the overlap of this box with all other boxes, and remove boxes that overlap significantly (iou >= iou_threshold).
3. Go back to step 1 and iterate until there are no more boxes with a lower score than the currently selected box.

This will remove all boxes that have a large overlap with the selected boxes. Only the "best" boxes remain.

Argument:
    scores: tensor of shape (None,), output of yolo_filter_boxes()
    boxes: tensor of shape (None, 4), output of yolo_filter_boxes() that have been scaled to the image size (see later)
    classes: tensor of shape (None,), output of yolo_filter_boxes()
    max_boxes: integer, maximum number of predicted boxes you'd like
    iou_threshold: real value, "intersection over union" threshold used for NMS filtering

Returns:
    scores: tensor of shape (None, ), predicted score for each box
    boxes: tensor of shape (None, 4), predicted box coordinates
    classes: tensor of shape (None, ), predicted class for each box
"""

def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):
    max_boxes_tensor = tf.Variable(max_boxes, dtype='int32')

    # Use tf.image.non_max_suppression() to get the list of indices corresponding to boxes we keep
    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes, iou_threshold)

    # Use tf.gather() to select only nms_indices from scores, boxes and classes
    scores = tf.gather(scores, nms_indices)
    boxes = tf.gather(boxes, nms_indices)
    classes = tf.gather(classes, nms_indices)

    return scores, boxes, classes

print("Exercise 3: YOLO Non-max Suppression")
print("==========")
tf.random.set_seed(10)
scores = tf.random.normal([54,], mean=1, stddev=4, seed = 1)
boxes = tf.random.normal([54, 4], mean=1, stddev=4, seed = 1)
classes = tf.random.normal([54,], mean=1, stddev=4, seed = 1)
scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes)

assert isinstance(scores, tf.Tensor), "Use tensoflow functions"
print("scores[2] = " + str(scores[2].numpy()))
print("boxes[2] = " + str(boxes[2].numpy()))
print("classes[2] = " + str(classes[2].numpy()))
print("scores.shape = " + str(scores.numpy().shape))
print("boxes.shape = " + str(boxes.numpy().shape))
print("classes.shape = " + str(classes.numpy().shape))

assert isinstance(scores, tf.Tensor), "Use tensoflow functions"
assert isinstance(boxes, tf.Tensor), "Use tensoflow functions"
assert isinstance(classes, tf.Tensor), "Use tensoflow functions"

assert scores.shape == (10,), "Wrong shape"
assert boxes.shape == (10, 4), "Wrong shape"
assert classes.shape == (10,), "Wrong shape"

assert np.isclose(scores[2].numpy(), 8.147684), "Wrong value on scores"
assert np.allclose(boxes[2].numpy(), [ 6.0797963, 3.743308, 1.3914018, -0.34089637]), "Wrong value on boxes"
assert np.isclose(classes[2].numpy(), 1.7079165), "Wrong value on classes"

print("\033[92m All tests passed!")
print("========================================")

"""
Exercise 4: Wrapping Up the Filtering
Now, It's time to implement a function taking the output of the deep CNN (the 19x19x5x85 dimensional encoding) and filtering 
through all the boxes using the functions you've just implemented.

Let's implement yolo_eval() which takes the output of the YOLO encoding and filters the boxes using score threshold and NMS.
Converts the output of YOLO encoding (a lot of boxes) to your predicted boxes along with their scores, box coordinates and classes.

Arguments:
    yolo_outputs: output of the encoding model (for image_shape of (608, 608, 3)), contains 4 tensors:
                    box_xy: tensor of shape (None, 19, 19, 5, 2)
                    box_wh: tensor of shape (None, 19, 19, 5, 2)
                    box_confidence: tensor of shape (None, 19, 19, 5, 1)
                    box_class_probs: tensor of shape (None, 19, 19, 5, 80)
    image_shape: tensor of shape (2,) containing the input shape, in this notebook we use (608., 608.) (has to be float32 dtype)
    max_boxes: integer, maximum number of predicted boxes you'd like
    score_threshold: real value, if [ highest class probability score < threshold], then get rid of the corresponding box
    iou_threshold: real value, "intersection over union" threshold used for NMS filtering
    
Returns:
    scores: tensor of shape (None, ), predicted score for each box
    boxes: tensor of shape (None, 4), predicted box coordinates
    classes: tensor of shape (None,), predicted class for each box
"""

def yolo_boxes_to_corners(box_xy, box_wh):
    # Convert YOLO box predictions to bounding box corners.
    box_mins = box_xy - (box_wh / 2.0)
    box_maxes = box_xy + (box_wh / 2.0)

    return tf.keras.backend.concatenate([
        box_mins[..., 1:2],  # y_min
        box_mins[..., 0:1],  # x_min
        box_maxes[..., 1:2],  # y_max
        box_maxes[..., 0:1]  # x_max
    ])

def scale_boxes(boxes, image_shape):
    """ Scales the predicted boxes in order to be drawable on the image"""
    height = float(image_shape[0])
    width = float(image_shape[1])
    image_dims = K.stack([height, width, height, width])
    image_dims = K.reshape(image_dims, [1, 4])
    boxes = boxes * image_dims
    return boxes

def yolo_eval(yolo_outputs, image_shape = (720, 1280), max_boxes = 10, score_threshold = 0.6, iou_threshold = 0.5):
    # Retrieve outputs of the YOLO model
    box_xy, box_wh, box_confidence, box_class_probs = yolo_outputs

    # Convert boxes to be ready for filtering functions (convert boxes box_xy and box_wh to corner coordinates)
    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    # Perform score-filtering with a threshold of score_threshold
    scores, boxes, classes = yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold = score_threshold)

    # Scale boxes back to original image shape
    boxes = scale_boxes(boxes, image_shape)

    # Perform Non-max suppression with maximum number of boxes set to max_boxes and a threshold of iou_threshold
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes = max_boxes, iou_threshold = iou_threshold)

    return scores, boxes, classes

print("Exercise 4: Wrapping Up the Filtering & Yolo Evaluation")
print("==========")
tf.random.set_seed(10)
yolo_outputs = (tf.random.normal([19, 19, 5, 2], mean=1, stddev=4, seed=1),
                tf.random.normal([19, 19, 5, 2], mean=1, stddev=4, seed=1),
                tf.random.normal([19, 19, 5, 1], mean=1, stddev=4, seed=1),
                tf.random.normal([19, 19, 5, 80], mean=1, stddev=4, seed=1))
scores, boxes, classes = yolo_eval(yolo_outputs)
print("scores[2] = " + str(scores[2].numpy()))
print("boxes[2] = " + str(boxes[2].numpy()))
print("classes[2] = " + str(classes[2].numpy()))
print("scores.shape = " + str(scores.numpy().shape))
print("boxes.shape = " + str(boxes.numpy().shape))
print("classes.shape = " + str(classes.numpy().shape))

assert isinstance(scores, tf.Tensor), "Use tensoflow functions"
assert isinstance(boxes, tf.Tensor), "Use tensoflow functions"
assert isinstance(classes, tf.Tensor), "Use tensoflow functions"

assert scores.shape == (10,), "Wrong shape"
assert boxes.shape == (10, 4), "Wrong shape"
assert classes.shape == (10,), "Wrong shape"

assert np.isclose(scores[2].numpy(), 171.60194), "Wrong value on scores"
assert np.allclose(boxes[2].numpy(), [-1240.3483, -3212.5881, -645.78, 2024.3052]), "Wrong value on boxes"
assert np.isclose(classes[2].numpy(), 16), "Wrong value on classes"

print("\033[92m All tests passed!")
print("========================================")

