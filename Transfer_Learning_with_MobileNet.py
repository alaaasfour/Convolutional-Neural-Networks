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
from test_utils import summary2, comparator
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

# Shows the different label probabilities in one tensor
print(label_batch)


"""
Now decode the predictions made by the model. Earlier, when we printed the shape of the batch, it would have returned (32, 1000). 
The number 32 refers to the batch size and 1000 refers to the 1000 classes the model was pretrained on. 
The predictions returned by the base model below follow this format:
First the class number, then a human-readable label, and last the probability of the image belonging to that class. 
We'll notice that there are two of these returned for each image in the batch - these the top two probabilities returned for that image.
"""
base_model.trainable = False
image_var = tf.Variable(preprocess_input(image_batch))
pred = base_model(image_var)

# Function to decode predictions
def decode_predictions(preds, top=2):
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(class_index[str(i)]) + (pred[i],) for i in top_indices]
        results.append(result)

    return results

with open("imagenet_base_model/imagenet_class_index.json", 'r') as f:
    class_index = json.load(f)

decoded_predictions = decode_predictions(pred.numpy(), top=2)
print(decoded_predictions)

print("========================================")

"""
Exercise 3: Layer Freezing with the Functional API

Now, we'll see how we can use a pretrained model to modify the classifier task so that it's able to recognize alpacas. 
We can achieve this in three steps:
    1. Delete the top layer (the classification layer): Set `include_top` in `base_model` as False
    2. Add a new classifier layer
        - Train only one layer by freezing the rest of the network
        - As mentioned before, a single neuron is enough to solve a binary classification problem.
    3. Freeze the base model and train the newly-created classifier layer
        - Set base `model.trainable=False` to avoid changing the weights and train only the new layer
        - Set training in `base_model` to False to avoid keeping track of statistics in the batch norm layer
        
Arguments:
    image_shape: Image width and height
    data_augmentation: data augmentation function
    
Returns:
    tf.keras.model
"""
def alpaca_model(image_shape=IMG_SIZE, data_augmentation=data_augmenter()):
    input_shape = image_shape + (3,)
    base_model_path_no = "imagenet_base_model/without_top_mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_160_no_top.h5"
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights=base_model_path_no)

    # Freeze the base model by making it non-trainable
    base_model.trainable = False

    # Create the input layer (Same as the imageNetV2 input size)
    inputs = tf.keras.Input(shape=input_shape)

    # Apply data augmentation to the inputs
    x = data_augmentation(inputs)

    # Data preprocessing using the same weights the model was trained on
    x = preprocess_input(x)

    # Set training to False to avoid keeping track of statistics in the batch norm layer
    x = base_model(x, training=False)

    # Add the new Binary classification layers
    # Use global avg pooling to summarize the info in each channel
    x = tfl.GlobalAveragePooling2D()(x)
    # Include dropout with probability of 0.2 to avoid overfitting
    x = tfl.Dropout(0.2)(x)

    # Use a prediction layer with one neuron (as a binary classifier only needs one)
    outputs = tfl.Dense(1, activation="linear")(x)
    model = tf.keras.Model(inputs, outputs)

    return model


print("Exercise 3: Layer Freezing with the Functional API")
print("==========")
# Let's create new model using the data_augmentation function defined earlier
model2 = alpaca_model(IMG_SIZE, data_augmentation)
alpaca_summary = [['InputLayer', [(None, 160, 160, 3)], 0],
                    ['Sequential', (None, 160, 160, 3), 0],
                    ['TensorFlowOpLayer', [(None, 160, 160, 3)], 0],
                    ['TensorFlowOpLayer', [(None, 160, 160, 3)], 0],
                    ['Functional', (None, 5, 5, 1280), 2257984],
                    ['GlobalAveragePooling2D', (None, 1280), 0],
                    ['Dropout', (None, 1280), 0, 0.2],
                    ['Dense', (None, 1), 1281, 'linear']] #linear is the default activation

comparator(summary2(model2), alpaca_summary)
for layer in summary2(model2):
    print(layer)

print("Printing the accuracy, loss for the training sets")
base_learning_rate = 0.001
model2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
initial_epochs = 5
history = model2.fit(train_dataset, validation_data=validation_dataset, epochs=initial_epochs)

# Plot the training and validation accuracy:
print("Plotting the training and validation accuracy")
acc = [0.] + history.history['accuracy']
val_acc = [0.] + history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
print("========================================")

"""
Exercise 4: Fine-tuning the model

We could try fine-tuning the model by re-running the optimizer in the last layers to improve accuracy. 
When we use a smaller learning rate, we take smaller steps to adapt it a little more closely to the new data. 
In transfer learning, the way we achieve this is by unfreezing the layers at the end of the network, and then re-training 
the model on the final layers with a very low learning rate. 
Adapting our learning rate to go over these layers in smaller steps can yield more fine details - and higher accuracy.

"""
base_model = model2.layers[4]
base_model.trainable = True

# Let's take a look to see how many layers are in the base model
# print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tuning from this layer onwards
fine_tune_at = 120

# Freeze all the layers before the `fine_tune_at` layer


fine_tune_epochs = 5
total_epochs =  initial_epochs + fine_tune_epochs

history_fine = model2.fit(train_dataset, epochs=total_epochs, initial_epoch=history.epoch[-1], validation_data=validation_dataset)
acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0, 1])
plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
