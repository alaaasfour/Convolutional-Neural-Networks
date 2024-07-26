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


"""
Exercise 5: Convolutional Layer Backward Pass
To implement the backward propagation for the convolution function, we need to do the following steps:
    - Computing dA: according to the formula:  𝑑𝐴 += ∑ℎ∑𝑤 = 𝑊𝑐 × 𝑑𝑍ℎ𝑤
    - Computing dW: according to the formula: 𝑑𝑊𝑐 += ∑ℎ∑𝑤 = 𝑎_𝑠𝑙𝑖𝑐𝑒 × 𝑑𝑍ℎ𝑤
    - Computing db: according to the formula: 𝑑𝑏 = ∑ℎ∑𝑤 𝑑𝑍ℎ𝑤

Argument:
    dZ: gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (m, n_H, n_W, n_C)
    cache: cache of values needed for the conv_backward(), output of conv_forward()

Returns:
    dA_prev: gradient of the cost with respect to the input of the conv layer (A_prev), numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    dW: gradient of the cost with respect to the weights of the conv layer (W) numpy array of shape (f, f, n_C_prev, n_C)
    db: gradient of the cost with respect to the biases of the conv layer (b) numpy array of shape (1, 1, 1, n_C)
"""

def conv_backward(dZ, cache):
    # Retrieve information from "cache"
    (A_prev, W, b, hparameters) = cache

    # Retrieve dimensions from A_prev's shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # Retrieve dimensions from W's shape
    (f, f, n_C_prev, n_C) = W.shape

    # Retrieve information from "hparameters"
    stride = hparameters["stride"]
    pad = hparameters["pad"]

    # Retrieve dimensions from dZ's shape
    (m, n_H, n_W, n_C) = dZ.shape

    # Initialize dA_prev, dW, db with the correct shapes
    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
    dW = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1, 1, 1, n_C))

    # Pad A_prev and dA_prev
    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)

    # Loop over the training examples
    for i in range(m):
        # Select the ith training example from A_prev_pad and dA_prev_pad
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]

        # Loop over vertical axis of the output volume
        for h in range(n_H):

            # Loop over horizontal axis of the output volume
            for w in range(n_W):

                # Loop over the channels of the output volume
                for c in range(n_C):

                    # Find the corners of the current 'slice'
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    # Use the corners to define the slice from a_prev_pad
                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                    # Update gradients for the window and the filter's parameters
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]

        if pad != 0:
            dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]
        else:
            dA_prev[i, :, :, :] = da_prev_pad



    assert (dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))
    return dA_prev, dW, db

print("Exercise 5: Convolutional Layer Backward Pass")
print("==========")
# We'll run conv_forward to initialize the 'Z' and 'cache_conv",
# which we'll use to test the conv_backward function
np.random.seed(1)
A_prev = np.random.randn(10, 4, 4, 3)
W = np.random.randn(2, 2, 3, 8)
b = np.random.randn(1, 1, 1, 8)
hparameters = {"pad" : 2,
               "stride": 2}
Z, cache_conv = conv_forward(A_prev, W, b, hparameters)

# Test conv_backward
dA, dW, db = conv_backward(Z, cache_conv)

print("dA_mean =", np.mean(dA))
print("dW_mean =", np.mean(dW))
print("db_mean =", np.mean(db))

assert type(dA) == np.ndarray, "Output must be a np.ndarray"
assert type(dW) == np.ndarray, "Output must be a np.ndarray"
assert type(db) == np.ndarray, "Output must be a np.ndarray"
assert dA.shape == (10, 4, 4, 3), f"Wrong shape for dA  {dA.shape} != (10, 4, 4, 3)"
assert dW.shape == (2, 2, 3, 8), f"Wrong shape for dW {dW.shape} != (2, 2, 3, 8)"
assert db.shape == (1, 1, 1, 8), f"Wrong shape for db {db.shape} != (1, 1, 1, 8)"
assert np.isclose(np.mean(dA), 1.4524377), "Wrong values for dA"
assert np.isclose(np.mean(dW), 1.7269914), "Wrong values for dW"
assert np.isclose(np.mean(db), 7.8392325), "Wrong values for db"

print("\033[92m All tests passed.")
print("========================================")

"""
Exercise 6: Pooling Layer - Backward Pass
Next, let's implement the backward pass for the pooling layer, starting with the MAX-POOL layer. 
Even though a pooling layer has no parameters for backprop to update, we still need to backpropagate the gradient through 
the pooling layer in order to compute gradients for layers that came before the pooling layer.

- We need to build a helper function first called create_mask_from_window() which creates a "mask" matrix which keeps 
track of where the maximum of the matrix is. True (1) indicates the position of the maximum in X, the other entries are False (0). 

Argument:
    x -- Array of shape (f, f)

Returns:
    mask -- Array of the same shape as window, contains a True at the position corresponding to the max entry of x.
"""

def create_mask_from_window(x):
    mask = (x == np.max(x))

    return mask

print("Exercise 6: Pooling Layer - Backward Pass")
print("==========")
np.random.seed(1)
x = np.random.randn(2, 3)
mask = create_mask_from_window(x)
print('x = ', x)
print("mask = ", mask)

x = np.array([[-1, 2, 3],
              [2, -3, 2],
              [1, 5, -2]])

y = np.array([[False, False, False],
     [False, False, False],
     [False, True, False]])
mask = create_mask_from_window(x)

assert type(mask) == np.ndarray, "Output must be a np.ndarray"
assert mask.shape == x.shape, "Input and output shapes must match"
assert np.allclose(mask, y), "Wrong output. The True value must be at position (2, 1)"

print("\033[92m All tests passed.")
print("========================================")

"""
Exercise 7: Average Pooling - Backward Pass
In max pooling, for each input window, all the "influence" on the output came from a single input value--the max. 
In average pooling, every element of the input window has equal influence on the output. So to implement backprop, 
we will now implement a helper function that reflects this.

We will implement the function below to equally distribute a value dz through a matrix of dimension shape.
Argument:
    dz: input scalar
    shape: the shape (n_H, n_W) of the output matrix for which we want to distribute the value of dz

Returns:
    a: Array of size (n_H, n_W) for which we distributed the value of dz
"""

def distribute_value(dz, shape):
    # Retrieve dimensions from shape
    (n_H, n_W) = shape

    # Compute the value to distribute on the matrix
    average = dz / (n_H * n_W)

    # Create a matrix where every entry is the average value
    a = np.full((n_H, n_W), average)

    return a

print("Exercise 7: Average Pooling - Backward Pass")
print("==========")
a = distribute_value(2, (2, 2))
print('distributed value =', a)


assert type(a) == np.ndarray, "Output must be a np.ndarray"
assert a.shape == (2, 2), f"Wrong shape {a.shape} != (2, 2)"
assert np.sum(a) == 2, "Values must sum to 2"

a = distribute_value(100, (10, 10))
assert type(a) == np.ndarray, "Output must be a np.ndarray"
assert a.shape == (10, 10), f"Wrong shape {a.shape} != (10, 10)"
assert np.sum(a) == 100, "Values must sum to 100"

print("\033[92m All tests passed.")
print("========================================")

"""
Exercise 8: Pooling Backward
Now, we have everything we need to compute backward propagation on a pooling layer.
We will implement the pool_backward function in both modes ('max', and 'average') 

We will implement the function below to equally distribute a value dz through a matrix of dimension shape.
Argument:
    dA: gradient of cost with respect to the output of the pooling layer, same shape as A
    cache: cache output from the forward pass of the pooling layer, contains the layer's input and hparameters 
    mode: the pooling mode you would like to use, defined as a string ("max" or "average")

Returns:
    dA: gradient of cost with respect to the output of the pooling layer, same shape as A
    cache: cache output from the forward pass of the pooling layer, contains the layer's input and hparameters 
    mode: the pooling mode you would like to use, defined as a string ("max" or "average")
"""

def pool_backward(dA, cache, mode = "max"):
    # Retrieve information from cache
    (A_prev, hparameters) = cache

    # Retrieve hyperparameters from "hparameters" (≈2 lines)
    stride = hparameters['stride']
    f = hparameters['f']

    # Retrieve dimensions from A_prev's shape and dA's shape (≈2 lines)
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    m, n_H, n_W, n_C = dA.shape

    # Initialize dA_prev with zeros (≈1 line)
    dA_prev = np.zeros(A_prev.shape)

    # Loop over the training examples
    for i in range(m):
        # Select training example from A_prev
        a_prev = A_prev[i]

        # Loop on the vertical axis
        for h in range(n_H):
            # Loop on the horizontal axis
            for w in range(n_W):
                # Loop over the channels
                for c in range(n_C):
                    # Find the corners of the current "slice"
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    # Compute the backward propagation in both modes.
                    if mode == "max":
                        # Use the corners and "c" to define the current slice from a_prev
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]

                        # Create the mask from a_prev_slice (≈1 line)
                        mask = create_mask_from_window(a_prev_slice)

                        # Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA)
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += mask * dA[i, h, w, c]
                    elif mode == "average":
                        # Get the value da from dA
                        da = dA[i, h, w, c]

                        # Define the shape of the filter as f*f
                        shape = (f, f)

                        # Distribute it to get the correct slice of dA_prev. i.e. Add the distributed value of da. (≈1 line)
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += distribute_value(da, shape)

    assert (dA_prev.shape == A_prev.shape)
    return dA_prev


print("Exercise 8: Putting it Together - Pooling Backward")
print("==========")
np.random.seed(1)
A_prev = np.random.randn(5, 5, 3, 2)
hparameters = {"stride" : 1, "f": 2}
A, cache = pool_forward(A_prev, hparameters)
print(A.shape)
print(cache[0].shape)
dA = np.random.randn(5, 4, 2, 2)

dA_prev1 = pool_backward(dA, cache, mode = "max")
print("mode = max")
print('mean of dA = ', np.mean(dA))
print('dA_prev1[1,1] = ', dA_prev1[1, 1])
print()
dA_prev2 = pool_backward(dA, cache, mode = "average")
print("mode = average")
print('mean of dA = ', np.mean(dA))
print('dA_prev2[1,1] = ', dA_prev2[1, 1])

assert type(dA_prev1) == np.ndarray, "Wrong type"
assert dA_prev1.shape == (5, 5, 3, 2), f"Wrong shape {dA_prev1.shape} != (5, 5, 3, 2)"
assert np.allclose(dA_prev1[1, 1], [[0, 0],
                                    [ 5.05844394, -1.68282702],
                                    [ 0, 0]]), "Wrong values for mode max"
assert np.allclose(dA_prev2[1, 1], [[0.08485462,  0.2787552],
                                    [1.26461098, -0.25749373],
                                    [1.17975636, -0.53624893]]), "Wrong values for mode average"
print("\033[92m All tests passed.")
print("========================================")



