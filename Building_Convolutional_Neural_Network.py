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

print("Exercise 1: Zero-Padding")
print("==========")
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
print("========================================")

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

print("Exercise 2: Single Step of Convolution")
print("==========")
np.random.seed(1)
a_slice_prev = np.random.randn(4, 4, 3)
W = np.random.randn(4, 4, 3)
b = np.random.randn(1, 1, 1)

Z = conv_single_step(a_slice_prev, W, b)
print("Z =", Z)
conv_single_step_test(conv_single_step)

assert (type(Z) == np.float64), "You must cast the output to numpy float 64"
assert np.isclose(Z, -6.999089450680221), "Wrong value"
print("========================================")

"""
Exercise 3: Convolutional Neural Networks - Forward Pass
In the forward pass, we will take many filters and convolve them on the input. Each 'convolution' gives us a 2D matrix output. 
We will then stack these outputs to get a 3D volume.

We will implement the conv_forward to convolve the filter W on an input activation A_prev

Argument:
    A_prev: output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W: Weights, numpy array of shape (f, f, n_C_prev, n_C)
    b: Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters: python dictionary containing "stride" and "pad"

Returns:
    Z: conv output, numpy array of shape (m, n_H, n_W, n_C)
    cache: cache of values needed for the conv_backward() function
"""

def conv_forward(A_prev, W, b, hparameters):
    # Retrieve dimensions from A_prev's shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # Retrieve dimensions from W's shape
    (f, f, n_C_prev, n_C) = W.shape

    # Retrieve information from "hparameters"
    stride = hparameters["stride"]
    pad = hparameters["pad"]

    # Compute the dimensions of the CONV output volume
    n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
    n_W = int((n_W_prev - f + 2 * pad) / stride) + 1

    # Initialize the output volume Z with zeros.
    Z = np.zeros((m, n_H, n_W, n_C))

    # Create A_prev_pad by padding A_prev
    A_prev_pad = np.pad(A_prev, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values = (0, 0))

    # Loop over the batch of training examples
    for i in range(m):
        # Select ith training example's padded activation
        a_prev_pad = A_prev_pad[i]

        # Loop over vertical axis of the output volume
        for h in range(n_H):
            # Find the vertical start and end of the current 'slice'
            vert_start = h * stride
            vert_end = vert_start + f

            # Loop over horizontal axis of the output volume
            for w in range(n_W):
                # Find the horizontal start and end of the current 'slice'
                horiz_start = w * stride
                horiz_end = horiz_start + f

                # Loop over channels (= filters) of the output volume
                for c in range(n_C):
                    # Using the corners to define the (3D) slice of a_prev_pad
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                    # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron.
                    weights = W[:, :, :, c]
                    biases = b[:, :, :, c]
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, weights, biases)

    # Saving the information in "cache" for the backprop
    cache = (A_prev, W, b, hparameters)

    return Z, cache

print("Exercise 3: Convolutional Neural Networks - Forward Pass")
print("==========")
np.random.seed(1)
A_prev = np.random.randn(2, 5, 7, 4)
W = np.random.randn(3, 3, 4, 8)
b = np.random.randn(1, 1, 1, 8)
hparameters = {"pad" : 1,
               "stride": 2}

Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
z_mean = np.mean(Z)
z_0_2_1 = Z[0, 2, 1]
cache_0_1_2_3 = cache_conv[0][1][2][3]
print("Z's mean =\n", z_mean)
print("Z[0,2,1] =\n", z_0_2_1)
print("cache_conv[0][1][2][3] =\n", cache_0_1_2_3)

conv_forward_test_1(z_mean, z_0_2_1, cache_0_1_2_3)
conv_forward_test_2(conv_forward)
print("========================================")

"""
Exercise 4: Pooling Layer - Forward Pooling
The pooling layer reduces the height and width of the input. This helps reduce computation, and it helps make feature detectors more 
invariant to its position in the input. The two types of pooling layers are:
    - Max-pooling: slides an (f, f) window over the input and stores the maximum value of the window in the output.
    - Average-pooling layer: slides an (f, f) window over the input and stores the average value of the window in the output. 

These pooling layers have no parameters for backpropagation to train. However, they have hyperparameters such as the window size f.
This specifies the height and width of the 𝑓×𝑓 window we would compute a max or average over.

Argument:
    A_prev: Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    hparameters: python dictionary containing "f" and "stride"
    mode: the pooling mode you would like to use, defined as a string ("max" or "average")

Returns:
    A: output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
    cache: cache used in the backward pass of the pooling layer, contains the input and hparameters 
"""

def pool_forward(A_prev, hparameters, mode = "max"):
    # Retrieve dimensions from the input shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # Retrieve hyperparameters from hparameters
    f = hparameters["f"]
    stride = hparameters["stride"]

    # Define the dimensions of the output
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev

    # Initialize output matrix A
    A = np.zeros((m, n_H, n_W, n_C))

    # Loop over the training examples
    for i in range(m):

        # Loop on the vertical axis of the output volume
        for h in range(n_H):
            # Find the vertical start and end of the current 'slice'
            vert_start = h * stride
            vert_end = vert_start + f

            # Loop on the horizontal axis of the output volume
            for w in range(n_W):
                # Find the vertical start and end of the current 'slice'
                horiz_start = w * stride
                horiz_end = horiz_start + f

                for c in range(n_C):
                    # Use the corners to define the current slice on the ith training example of A_prev, channel c.
                    a_slice_prev = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]

                    # Compute the pooling operation on the slice.
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_slice_prev)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_slice_prev)

    # We will store the input and hparameters in 'cache' for pool_backward()
    cache = (A_prev, hparameters)

    return A, cache


print("Exercise 4: Pooling Layer - Forward Pooling")
print("==========")
# Case 1: stride of 1
print("CASE 1:\n")
np.random.seed(1)
A_prev_case_1 = np.random.randn(2, 5, 5, 3)
hparameters_case_1 = {"stride" : 1, "f": 3}

A, cache = pool_forward(A_prev_case_1, hparameters_case_1, mode = "max")
print("mode = max")
print("A.shape = " + str(A.shape))
print("A[1, 1] =\n", A[1, 1])
A, cache = pool_forward(A_prev_case_1, hparameters_case_1, mode = "average")
print("mode = average")
print("A.shape = " + str(A.shape))
print("A[1, 1] =\n", A[1, 1])

pool_forward_test_1(pool_forward)

# Case 2: stride of 2
print("\n\033[0mCASE 2:\n")
np.random.seed(1)
A_prev_case_2 = np.random.randn(2, 5, 5, 3)
hparameters_case_2 = {"stride" : 2, "f": 3}

A, cache = pool_forward(A_prev_case_2, hparameters_case_2, mode = "max")
print("mode = max")
print("A.shape = " + str(A.shape))
print("A[0] =\n", A[0])
print()

A, cache = pool_forward(A_prev_case_2, hparameters_case_2, mode = "average")
print("mode = average")
print("A.shape = " + str(A.shape))
print("A[1] =\n", A[1])

pool_forward_test_2(pool_forward)
print("========================================")



