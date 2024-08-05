"""
In this script, we will implement the Neural Style Transfer algorithm and will do the following:
1. Implement the neural style transfer algorithm
2. Generate novel artistic images using your algorithm
3. Define the style cost function for Neural Style Transfer
4. Define the content cost function for Neural Style Transfer

Neural Style Transfer (NST) is one of the most fun and interesting optimization techniques in deep learning.
It merges two images, namely: a "content" image (C) and a "style" image (S), to create a "generated" image (G).
The generated image G combines the "content" of the image C with the "style" of image S.
"""

# Packages
import os
import sys
import pprint
import scipy.io
import scipy.misc
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from public_tests import *

"""
Transfer Learning
Neural Style Transfer (NST) uses a previously trained convolutional network, and builds on top of that. 
The idea of using a network trained on a different task and applying it to a new task is called transfer learning.
"""

tf.random.set_seed(272)
pp = pprint.PrettyPrinter(indent=4)
img_size = 400
vgg = tf.keras.applications.VGG19(include_top=False,
                                  input_shape=(img_size, img_size, 3),
                                  weights='pretrained-model/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')
vgg.trainable = False
pp.pprint(vgg)

"""
Neural Style Transfer (NST)
We will build the NST in three steps:
    1. First, we will build the content cost function  𝐽𝑐𝑜𝑛𝑡𝑒𝑛𝑡(𝐶,𝐺) 
    2. Second, we will build the style cost function  𝐽𝑠𝑡𝑦𝑙𝑒(𝑆,𝐺) 
    3. Finally, we'll put it all together to get  𝐽(𝐺)=𝛼𝐽𝑐𝑜𝑛𝑡𝑒𝑛𝑡(𝐶,𝐺)+𝛽𝐽𝑠𝑡𝑦𝑙𝑒(𝑆,𝐺).
"""
content_image = Image.open("imagesNST/louvre.jpg")
print("The content image (C) shows the Louvre museum's pyramid surrounded by old Paris buildings, against a sunny sky with a few clouds.")
plt.imshow(content_image)
plt.show()

"""
Exercise 1: Computing the Content Cost
Now, we will compute the content cost using TensorFlow

Argument:
    a_C: tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C 
    a_G: tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G

Returns:
    J_content: scalar that you compute using equation 1 above.
"""
def compute_content_cost(content_output, generated_output):
    a_C = content_output[-1]
    a_G = generated_output[-1]

    # Retrieve dimensions from a_G
    _, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape 'a_C' and 'a_G'
    a_C_unrolled = tf.reshape(a_C, shape=[-1, n_H * n_W, n_C])
    a_G_unrolled = tf.reshape(a_G, shape=[-1, n_H * n_W, n_C])

    # compute the cost with tensorflow
    J_content = tf.reduce_sum(tf.square(a_C_unrolled - a_G_unrolled)) / (4.0 * n_H * n_W * n_C)

    return J_content

content_output = tf.random.normal([1, 1, 4, 4, 3], mean=1, stddev=4)
generated_output = tf.random.normal([1, 1, 4, 4, 3], mean=1, stddev=4)
J_content = compute_content_cost(content_output, generated_output)
print(J_content)

"""
Exercise 2: Computing the Style Cost

Gram matrix
The style matrix is also called a "Gram matrix."
In linear algebra, the Gram matrix G of a set of vectors  (𝑣1,…,𝑣𝑛)  is the matrix of dot products, whose entries are  𝐺𝑖𝑗=𝑣𝑇𝑖𝑣𝑗=𝑛𝑝.𝑑𝑜𝑡(𝑣𝑖,𝑣𝑗) .
In other words, 𝐺𝑖𝑗 compares how similar 𝑣𝑖 is to 𝑣𝑗: If they are highly similar, we would expect them to have a 
large dot product, and thus for 𝐺𝑖𝑗 to be large.

Argument:
    A: matrix of shape (n_C, n_H*n_W)

Returns:
    GA: Gram matrix of A, of shape (n_C, n_C)
"""
def gram_matrix(A):
    GA = tf.matmul(A, tf.transpose(A))

    return GA

A = tf.random.normal([3, 2 * 1], mean=1, stddev=4)
print(gram_matrix(A))


"""
Exercise 3: Computing the Layer Style Cost

Now, let's compute the style cost for a single layer using the following 3 steps:
    1. Retrieve dimensions from the hidden layer activations a_G:
    2. Unroll the hidden layer activations a_S and a_G into 2D matrices.
    3. Compute the Style matrix of the images S and G.
    4. Compute the Style cost

Argument:
    a_S: tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S 
    a_G: tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G

Returns:
    J_style_layer: tensor representing a scalar value, style cost defined above by equation (2)
"""
def compute_layer_style_cost(a_S, a_G):
    # Retrieve dimensions from a_G
    _, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape the tensors from (1, n_H, n_W, n_C) to (n_C, n_H * n_W)
    a_S = tf.reshape(tf.transpose(a_S, perm=[0, 3, 1, 2]), shape=[n_C, n_H * n_W])
    a_G = tf.reshape(tf.transpose(a_G, perm=[0, 3, 1, 2]), shape=[n_C, n_H * n_W])

    # Computing gram_matrices for both images S and G
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    # Computing the loss
    J_style_layer = tf.reduce_sum(tf.square(tf.subtract(GS, GG))) / (4.0 * (n_C ** 2) * (n_H * n_W) ** 2)

    return J_style_layer

#  Listing the layer names:
for layer in vgg.layers:
    print(layer.name)

print(vgg.get_layer('block5_conv4').output)

# Now choose layers to represent the style of the image and assign style costs:
STYLE_LAYERS = [
    ('block1_conv1', 0.2),
    ('block2_conv1', 0.2),
    ('block3_conv1', 0.2),
    ('block4_conv1', 0.2),
    ('block5_conv1', 0.2)]


"""
Exercise 4: Computing the Style Cost

Now, let's compute the style cost as follows:
For each layer:
    1. Select the activation (the output tensor) of the current layer.
    2. Get the style of the style image "S" from the current layer.
    3. Get the style of the generated image "G" from the current layer.
    4. Compute the "style cost" for the current layer
    5. Add the weighted style cost to the overall style cost (J_style)

Argument:
    style_image_output: our tensorflow model
    generated_image_output
    STYLE_LAYERS: A python list containing:
                - the names of the layers we would like to extract style from
                - a coefficient for each of them

Returns:
    J_style: tensor representing a scalar value, style cost defined above by equation (2)
"""
def compute_style_cost(style_image_output, generated_image_output, STYLE_LAYERS=STYLE_LAYERS):
    # initialize the overall style cost
    J_style = 0

    # Set a_S to be the hidden layer activation from the layer we have selected.
    # The last element of the array contains the content layer image, which must not be used.
    a_S = style_image_output[:-1]

    # Set a_G to be the output of the choosen hidden layers.
    # The last element of the list contains the content layer image which must not be used.
    a_G = generated_image_output[:-1]
    for i, weight in zip(range(len(a_S)), STYLE_LAYERS):
        # Compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S[i], a_G[i])

        # Add weight * J_style_layer of this layer to overall style cost
        J_style += weight[1] * J_style_layer

    return J_style


@tf.function()
def total_cost(J_content, J_style, alpha=10, beta=40):
    J = alpha * J_content + beta * J_style

    return J

J_content = 0.2
J_style = 0.8

print(total_cost(J_content, J_style, 10, 40))

content_image = np.array(Image.open("imagesNST/louvre_small.jpg").resize((img_size, img_size)))
content_image = tf.constant(np.reshape(content_image, ((1,) + content_image.shape)))

print(content_image.shape)
imshow(content_image[0])
plt.show()

style_image =  np.array(Image.open("imagesNST/monet.jpg").resize((img_size, img_size)))
style_image = tf.constant(np.reshape(style_image, ((1,) + style_image.shape)))

print(style_image.shape)
imshow(style_image[0])
plt.show()


generated_image = tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
noise = tf.random.uniform(tf.shape(generated_image), -0.25, 0.25)
generated_image = tf.add(generated_image, noise)
generated_image = tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=1.0)

print(generated_image.shape)
imshow(generated_image.numpy()[0])
plt.show()


def get_layer_outputs(vgg, layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    outputs = [vgg.get_layer(layer[0]).output for layer in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model

content_layer = [('block5_conv4', 1)]

vgg_model_outputs = get_layer_outputs(vgg, STYLE_LAYERS + content_layer)

content_target = vgg_model_outputs(content_image)  # Content encoder
style_targets = vgg_model_outputs(style_image)     # Style encoder

# Assign the content image to be the input of the VGG model.
# Set a_C to be the hidden layer activation from the layer we have selected
preprocessed_content =  tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
a_C = vgg_model_outputs(preprocessed_content)


# Assign the input of the model to be the "style" image
preprocessed_style =  tf.Variable(tf.image.convert_image_dtype(style_image, tf.float32))
a_S = vgg_model_outputs(preprocessed_style)


def clip_0_1(image):

    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


def tensor_to_image(tensor):

    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)


optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)


@tf.function()
def train_step(generated_image):
    with tf.GradientTape() as tape:

        # Compute a_G as the vgg_model_outputs for the current generated image
        # (1 line)
        a_G = vgg_model_outputs(generated_image)

        # Compute the style cost
        # (1 line)
        J_style = compute_style_cost(a_S, a_G)

        # (2 lines)
        # Compute the content cost
        J_content = compute_content_cost(a_C, a_G)
        # Compute the total cost
        J = total_cost(J_content, J_style, alpha=10, beta=40)


    grad = tape.gradient(J, generated_image)

    optimizer.apply_gradients([(grad, generated_image)])
    generated_image.assign(clip_0_1(generated_image))
    # For grading purposes
    return J

generated_image = tf.Variable(generated_image)


# Show the generated image at some epochs
# Uncomment to reset the style transfer process. You will need to compile the train_step function again
epochs = 2501
for i in range(epochs):
    train_step(generated_image)
    if i % 250 == 0:
        print(f"Epoch {i} ")
    if i % 250 == 0:
        image = tensor_to_image(generated_image)
        imshow(image)
        image.save(f"output/image_{i}.jpg")
        plt.show()