import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math

# PrettyTensor.
import prettytensor as pt

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('../MNIST/', one_hot=True)
data.test.cls = np.argmax(data.test.labels, axis=1)

#
#    Data Dimensions
#

img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
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

#
#   TensorFlow Graph
#

# Placeholder Variables
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

#
#    TensorFlow Implementation
#

# Helper-functions
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def new_conv_layer(input,
                   num_input_channels,
                   filter_size,
                   num_filters,
                   use_pooling=True):
    # Shape of the filter-weights
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights
    weights = new_weights(shape=shape)

    # Create new biases
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases
    layer += biases

    # Use pooling to down-sample the image resolution
    if use_pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # ReLU
    layer = tf.nn.relu(layer)

    return layer, weights

def flatten_layer(layer):
    # Get the shape of the input layer
    layer_shape = layer.get_shape()

    # Number of features
    num_features = layer_shape[1:4].num_elements()

    # Reshape the layer to [num_images, num_features]
    layer_flat = tf.reshape(layer, [-1, num_features])

    return layer_flat, num_features

def new_fc_layer(input,
                 num_inputs,
                 num_outputs,
                 use_relu=True):
    # Create new weigts and biases
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

# Code is not executed
if False:

    # Conv layer 1
    layer_conv1, weights_conv1 = new_conv_layer(input=x_image,
        num_input_channels=num_channels,
        filter_size=5,
        num_filters=16,
        use_pooling=True)

    # Conv Layer 2
    layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1,
        num_input_channels=16,
        filter_size=5,
        num_filters=36,
        use_pooling=True)

    # Flatten Layer
    layer_flat, num_features = flatten_layer(layer_conv2)

    # Fully-connected layer 1
    layer_fc1 = new_fc_layer(input=layer_flat,
        num_inputs=num_features,
        num_outputs=128,
        use_relu=True)

    # Fully-connected layer 2
    layer_fc2 = new_fc_layer(input=layer_fc1,
        num_inputs=128,
        num_outputs=num_classes,
        use_relu=False)

    # Predicted class-label
    y_pred = tf.nn.softmax(layer_fc2)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
            labels=y_true)

    # Lost (cost-measure)
    loss = tf.reduce_mean(cross_entropy)

#
#   PrettyTensor Implementation
#

x_pretty = pt.wrap(x_image)


with pt.defaults_scope(activation_fn=tf.nn.relu):
    y_pred, loss = x_pretty.\
        conv2d(kernel=5, depth=16, name='layer_conv1').\
        max_pool(kernel=2, stride=2).\
        conv2d(kernel=5, depth=36, name='layer_conv2').\
        max_pool(kernel=2, stride=2).\
        flatten().\
        fully_connected(size=128, name='layer_fc1').\
        softmax_classifier(num_classes=num_classes, labels=y_true)
        
# Getting the Weights
def get_weights_variable(layer_name):
    with tf.variable_scope(layer_name, reuse=True):
        variable = tf.get_variable('weights')
    return variable

weights_conv1 = get_weights_variable(layer_name='layer_conv1')
weights_conv2 = get_weights_variable(layer_name='layer_conv2')

# Optimization Method
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

# Performance Measures
y_pred_cls = tf.argmax(y_pred, dimension=1)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#
#    TensorFlow run
#

# Create TensorFlow session
session = tf.Session()

# Initialize Variables
session.run(tf.global_variables_initializer())

# Helper-function to perform optimization iterations
train_batch_size = 64

# Counter for total number of iterations
total_iterations = 0

def optimize(num_iterations):

    # Global variable
    global total_iterations

    # Start time
    start_time = time.time()

    for i in range(total_iterations, total_iterations + num_iterations):
        
        # Get a batch of training examples
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)

        # Put the batch into a dict
        feed_dict_train = {x : x_batch,
        y_true : y_true_batch}

        # Run the optimizer
        session.run(optimizer, feed_dict=feed_dict_train)

        # Print status every 100 iterations
        if i % 100 == 0:

            # Calculate the accuracy
            acc = session.run(accuracy, feed_dict=feed_dict_train)

            # Message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

            # Print it.
            print(msg.format(i + 1, acc)) 

    # Update the total number of iterations
    total_iterations += num_iterations

    # Ending time
    end_time = time.time()

    # Different time
    time_dif = end_time - start_time

    # Print time usage
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

# Helper-function to plot example errors
def plot_example_errors(cls_pred, correct):
    # This function is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # correct is a boolean array whether the predicted class
    # is equal to the true class for each image in the test-set.

    # Negate the boolean array.
    incorrect = (correct == False)
    
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

# Helper-function to plot confusion matrix
def plot_confusion_matrix(cls_pred):
    # This is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the true classifications for the test-set.
    cls_true = data.test.cls
    
    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.matshow(cm)

    # Make various adjustments to the plot.
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

# Helper-function for showing the performance
# Split the test-set into smaller batches of this size.
test_batch_size = 256

def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):

    # Number of images in the test-set.
    num_test = len(data.test.images)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + test_batch_size, num_test)

        # Get the images from the test-set between index i and j.
        images = data.test.images[i:j, :]

        # Get the associated labels.
        labels = data.test.labels[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict = {x: images,
                     y_true: labels}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Convenience variable for the true class-numbers of the test-set.
    cls_true = data.test.cls

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)

#
#    Performance before any optimization
#

print_test_accuracy()

# Performance after 1 optimization iteration
optimize(num_iterations=1)
print_test_accuracy()

# Performance after 100 optimization iterations
optimize(num_iterations=99)
print_test_accuracy()

# Performance after 1000 optimization iterations
optimize(num_iterations=900)
print_test_accuracy()

# Performance after 5000 optimization iterations
optimize(num_iterations=4000)
print_test_accuracy()
print_test_accuracy(show_example_errors=True, show_confusion_matrix=True)