"""
In this script, we will implement a Convolutional Neural Network (CONV) and pooling (POOL) layers inn numpy,
including both forward and backward propagation.
"""

# Packages
# We will import all the important packages that we need.
import numpy as np
import h5py
import matplotlib.pyplot as plt
from public_tests import *
np.random.seed(1)
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


"""
We will implement a Convolutional Neural Network (CONV) step by step. 
First we will implement two helper functions: one for zero padding and the other for computing the convolution function itself.

Exercise 1: Zero-Padding
Zero-Padding adds zeros around the border of the image.

The main benefits of padding are:
    - It allows us to use a CONV layer without necessarily shrinking the height and width of the volume.
    - It helps us keep more of the information at the border of an image. Without padding, very few values at the next layer would be affected by pixels at the edges of an image.
    
We will implement the following function, which pads all the images of a batch of examples X with zeros.
Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image.
    
Argument:
    X: python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    pad: integer, amount of padding around each image on vertical and horizontal dimensions
    
Returns:
    X_pad: padded image of shape (m, n_H + 2 * pad, n_W + 2 * pad, n_C)
"""
def zero_pad(X, pad):
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values = (0, 0))
    return X_pad


np.random.seed(1)
x = np.random.randn(4, 3, 3, 2)
x_pad = zero_pad(x, 3)
print ("x.shape =\n", x.shape)
print ("x_pad.shape =\n", x_pad.shape)
print ("x[1,1] =\n", x[1, 1])
print ("x_pad[1,1] =\n", x_pad[1, 1])

fig, axarr = plt.subplots(1, 2)
axarr[0].set_title('x')
axarr[0].imshow(x[0, :, :, 0])
axarr[1].set_title('x_pad')
axarr[1].imshow(x_pad[0, :, :, 0])
plt.show()
zero_pad_test(zero_pad)


"""
Exercise 2: Single Step of Convolution
By implementing the single step of convolution, we apply the filter to a single position of the input. This will be used to build a convolutional unit, which:
    - Takes an input volume.
    - Applies a filter at every position of the input.
    - Outputs another volume (usually of a different size)
    
Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation of the previous layer.
Argument:
    a_slice_prev: slice of input data of shape (f, f, n_C_prev)
    W: Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
    b: Bias parameters contained in a window - matrix of shape (1, 1, 1)
    
Returns:
    Z: a scalar value, the result of convolving the sliding window (W, b) on a slice x of the input data
"""
def conv_single_step(a_slice_prev, W, b):
    s = a_slice_prev * W
    Z = np.sum(s)
    Z = Z + float(b)

    return Z

np.random.seed(1)
a_slice_prev = np.random.randn(4, 4, 3)
W = np.random.randn(4, 4, 3)
b = np.random.randn(1, 1, 1)

Z = conv_single_step(a_slice_prev, W, b)
print("Z =", Z)
conv_single_step_test(conv_single_step)

assert (type(Z) == np.float64), "You must cast the output to numpy float 64"
assert np.isclose(Z, -6.999089450680221), "Wrong value"

