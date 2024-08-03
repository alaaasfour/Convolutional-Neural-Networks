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

"""
Exercise 2: Decoder (Upsampling Block)
The decoder, or upsampling block, upsamples the features back to the original image size. At each upsampling level, we'll 
take the output of the corresponding encoder block and concatenate it before feeding to the next decoder block.

There are two new components in the decoder: `up` and `merge`. These are the transpose convolution and the skip connections. 
In addition, there are two more convolutional layers set to the same parameters as in the encoder. 

We'll implement the upsampling_block() function by following the following steps:
    1. Takes the arguments `expansive_input` (which is the input tensor from the previous layer) and `contractive_input` (the input tensor from the previous skip layer)
    2. The number of filters here is the same as in the downsampling block you completed previously
    3. The `Conv2DTranspose` layer will take `n_filters` with shape (3,3) and a stride of (2,2), with padding set to same. 
    It's applied to `expansive_input`, or the input tensor from the previous layer.
    4. Concatenate the Conv2DTranspose layer output to the contractive input, with an axis of 3. 
    In general, we can concatenate the tensors in the order that we prefer.

Argument:
    expansive_input: Input tensor from previous layer
    contractive_input: Input tensor from previous skip layer
    n_filters: Number of filters for the convolutional layers

Returns:
    conv: Tensor output
"""

def upsampling_block(expansive_input, contractive_input, n_filters=32):
    up = Conv2DTranspose(n_filters, (3, 3), strides=(2, 2), padding='same')(expansive_input)

    # Merge the previous output and the contractive_input
    merge = concatenate([up, contractive_input], axis=3)
    conv = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge)
    conv = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv)

    return conv

print("Exercise 2: Decoder (Upsampling Block)")
print("==========")
input_size1=(12, 16, 256)
input_size2 = (24, 32, 128)
n_filters = 32
expansive_inputs = Input(input_size1)
contractive_inputs =  Input(input_size2)
cblock1 = upsampling_block(expansive_inputs, contractive_inputs, n_filters * 1)
model1 = tf.keras.Model(inputs=[expansive_inputs, contractive_inputs], outputs=cblock1)

output1 = [['InputLayer', [None, 12, 16, 256], 0],
            ['Conv2DTranspose', [None, 24, 32, 32], 73760],
            ['InputLayer', [None, 24, 32, 128], 0],
            ['Concatenate', [None, 24, 32, 160], 0],
            ['Conv2D', [None, 24, 32, 32], 46112, 'same', 'relu', 'HeNormal'],
            ['Conv2D', [None, 24, 32, 32], 9248, 'same', 'relu', 'HeNormal']]

print('Block 1:')
for layer in summaryUNet(model1):
    print(layer)

comparatorUNet(summaryUNet(model1), output1)
print("========================================")

"""
Exercise 3: Building the Model
Now, this is where we'll put it all together, by chaining the encoder, bottleneck, and decoder! We'll need to specify 
the number of output channels, which for this particular set would be 23. 
That's because there are 23 possible labels for each pixel in this self-driving car dataset.

Let's implement unet_model() function by following the steps:
For the function, specify the input shape, number of filters, and number of classes (23 in this case).
For the first half of the model:
    1. Begin with a conv block that takes the inputs of the model and the number of filters
    2. Then, chain the first output element of each block to the input of the next convolutional block
    3 Next, double the number of filters at each step
    4. Beginning with `conv_block4`, add `dropout_prob` of 0.3
    5. For the final conv_block, set `dropout_prob` to 0.3 again, and turn off max pooling
    
For the second half of the model:
    1. Use cblock5 as expansive_input and cblock4 as contractive_input, with `n_filters` * 8. This is your bottleneck layer.
    2. Chain the output of the previous block as expansive_input and the corresponding contractive block output.
    3. Note that you must use the second element of the contractive block before the max pooling layer.
    4. At each step, use half the number of filters of the previous block
    5. `conv9` is a Conv2D layer with ReLU activation, He normal initializer, same padding
    6. Finally, `conv10` is a Conv2D that takes the number of classes as the filter, a kernel size of 1, and "same" padding. The output of `conv10` is the output of your model.

Argument:
    input_size -- Input shape 
    n_filters -- Number of filters for the convolutional layers
    n_classes -- Number of output classes

Returns:
    model -- tf.keras.Model
"""

def unet_model(input_size=(96, 128, 3), n_filters=32, n_classes=23):
    inputs = Input(input_size)

    # Contracting Path (encoding)
    # Add a conv_block with the inputs of the unet_ model and n_filters
    cblock1 = conv_block(inputs, n_filters)

    # Chain the first element of the output of each block to be the input of the next conv_block.
    # Double the number of filters at each new step
    cblock2 = conv_block(cblock1[0], n_filters * 2)
    cblock3 = conv_block(cblock2[0], n_filters * 4)
    cblock4 = conv_block(cblock3[0], n_filters * 8, dropout_prob = 0.3)
    cblock5 = conv_block(cblock4[0], n_filters * 16, dropout_prob = 0.3, max_pooling = False)

    # Expanding Path (decoding)
    # Add the first upsampling_block.
    # Use the cblock5[0] as expansive_input and cblock4[1] as contractive_input and n_filters * 8
    ublock6 = upsampling_block(cblock5[0], cblock4[1], n_filters * 8)

    # Chain the output of the previous block as expansive_input and the corresponding contractive block output.
    # We must use the second element of the contractive block i.e before the maxpooling layer.
    # At each step, we use half the number of filters of the previous block
    ublock7 = upsampling_block(ublock6, cblock3[1], n_filters * 4)
    ublock8 = upsampling_block(ublock7, cblock2[1], n_filters * 2)
    ublock9 = upsampling_block(ublock8, cblock1[1], n_filters)

    conv9 = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(ublock9)

    # Add a Conv2D layer with n_classes filter, kernel size of 1 and a 'same' padding
    conv10 = Conv2D(n_classes, 1, padding='same')(conv9)
    model = tf.keras.Model(inputs=inputs, outputs=conv10)

    return model

print("Exercise 3: Building the U-Net Model")
print("==========")
import outputsUnet
img_height = 96
img_width = 128
num_channels = 3

unet = unet_model((img_height, img_width, num_channels))
comparatorUNet(summaryUNet(unet), outputsUnet.unet_model_output)
print("========================================")
