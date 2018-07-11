import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.models import Sequential

print(tf.__version__)

from tensorflow.examples.tutorials.mnist import input_data

data = input_data.read_data_sets('../MNIST/', one_hot=True)
data.test.cls = np.argmax(data.test.labels, axis=1)

#
#    Data Dimensions
#

img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
img_shape_full = (img_size, img_size, 1)
num_channels = 1
num_classes = 10


# Helper-function for plotting images
def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


# Helper-function to plot example errors
def plot_example_errors(cls_pred):
    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Boolean array whether the predicted class is incorrect.
    incorrect = (cls_pred != data.test.cls)

    # Get the images from the test-set that have been
    # incorrectly classified.
    images = data.test.images[incorrect]

    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = data.test.cls[incorrect]

    # Plot the first 9 images.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])


#
#   Sequential Model Implementation
#

# Start model's construction
model = Sequential()

# Add an input layer
model.add(InputLayer(input_shape=(img_size_flat,)))

# Reshape Image -> img_shape_full
model.add(Reshape(img_shape_full))

# Activation of Conv Layer
# activation_conv = 'sigmoid'
# activation_conv = 'relu'
activation_conv = 'relu'

# Conv Layer 1
model.add(Conv2D(kernel_size=5,
                 strides=1,
                 filters=16,
                 padding='same',
                 activation=activation_conv,
                 name='layer_conv1'))

model.add(MaxPooling2D(pool_size=2,
                       strides=2))

# Conv Layer 2
model.add(Conv2D(kernel_size=5,
                 strides=1,
                 filters=36,
                 padding='same',
                 activation=activation_conv,
                 name='layer_conv2'))

model.add(MaxPooling2D(pool_size=2,
                       strides=2))

# Conv Layer 3
model.add(Conv2D(kernel_size=5,
                 strides=1,
                 filters=60,
                 padding='same',
                 activation=activation_conv,
                 name='layer_conv3'))

model.add(MaxPooling2D(pool_size=2,
                       strides=2))

# Flatten Layer
model.add(Flatten())

# Fully-connected layer 1
model.add(Dense(128, activation='relu'))

# Fully-connected layer 2
model.add(Dense(num_classes, activation='softmax'))

# Model Compilation
from tensorflow.python.keras.optimizers import Adam

optimizer = Adam(lr=1e-3)

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Training
model.fit(x=data.train.images,
          y=data.train.labels,
          epochs=1,
          batch_size=128)
# Evaluation
result = model.evaluate(x=data.test.images,
                        y=data.test.labels)

for name, value in zip(model.metrics_names, result):
    print(name, value)

print("{0}: {1:.2%}".format(model.metrics_names[1], result[1]))

# Prediction
images = data.test.images[0:9]
cls_true = data.test.cls[0:9]
y_pred = model.predict(x=images)
cls_pred = np.argmax(y_pred, axis=1)
# plot_images(images=images,
#             cls_true=cls_true,
#             cls_pred=cls_pred)

# Plot some mis-classified images
y_pred = model.predict(x=data.test.images)
cls_pred = np.argmax(y_pred, axis=1)
# plot_example_errors(cls_pred)

#
#   Functional Model
#

# Input Layer
inputs = Input(shape=(img_size_flat,))
net = inputs

# Reshape the inputs
net = Reshape(img_shape_full)(net)

# Conv Layer 1
net = Conv2D(kernel_size=5,
             strides=1,
             filters=16,
             padding='same',
             activation=activation_conv,
             name='layer_conv1')(net)
net = MaxPooling2D(pool_size=2, strides=2)(net)

# Conv Layer 2
net = Conv2D(kernel_size=5,
             strides=1,
             filters=36,
             padding='same',
             activation=activation_conv,
             name='layer_conv2')(net)
net = MaxPooling2D(pool_size=2, strides=2)(net)

# Conv Layer 3
net = Conv2D(kernel_size=5,
             strides=1,
             filters=60,
             padding='same',
             activation=activation_conv,
             name='layer_conv3')(net)
net = MaxPooling2D(pool_size=2, strides=2)(net)

# Flatten Layer
net = Flatten()(net)

# Fully-connected Layer 1
net = Dense(128, activation='relu')(net)

# Fully-connected Layer 2
net = Dense(num_classes, activation='softmax')(net)

# Output of Neural Network
outputs = net

# Model Compilation
from tensorflow.python.keras.models import Model

model2 = Model(inputs=inputs, outputs=outputs)
model2.compile(optimizer='rmsprop',
               loss='categorical_crossentropy',
               metrics=['accuracy'])

# Training
model2.fit(x=data.train.images,
           y=data.train.labels,
           epochs=1,
           batch_size=128)

# Evaluation
result = model2.evaluate(x=data.test.images,
                         y=data.test.labels)

for name, value in zip(model2.metrics_names, result):
    print(name, value)

print("{0}: {1:.2%}".format(model2.metrics_names[1], result[1]))

# Plot some mis-classified images
y_pred = model2.predict(x=data.test.images)
cls_pred = np.argmax(y_pred, axis=1)
# plot_example_errors(cls_pred)

#
#	Save and Load Model
#
path_model = 'model.keras'
model2.save(path_model)
del model2

from tensorflow.python.keras.models import load_model

model3 = load_model(path_model)

images = data.test.images[0:9]
cls_true = data.test.cls[0:9]
y_pred = model3.predict(x=images)
cls_pred = np.argmax(y_pred, axis=1)

# plot_images(images=images,
#             cls_pred=cls_pred,
#             cls_true=cls_true)

model.summary()
model3.summary()

