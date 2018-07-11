import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import os

# Pretty Tensor
import prettytensor as pt

from tensorflow.examples.tutorials.mnist import input_data

data = input_data.read_data_sets('../MNIST/', one_hot=True)
data.test.cls = np.argmax(data.test.labels, axis=1)
data.validation.cls = np.argmax(data.validation.labels, axis=1)

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
#    TensorFlow Graph
#

# Placeholder variables
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

# Neural Network
x_pretty = pt.wrap(x_image)

with pt.defaults_scope(activation_fn=tf.nn.relu):
    y_pred, loss = x_pretty. \
        conv2d(kernel=5, depth=16, name='layer_conv1'). \
        max_pool(kernel=2, stride=2). \
        conv2d(kernel=5, depth=36, name='layer_conv2'). \
        max_pool(kernel=2, stride=2). \
        flatten(). \
        fully_connected(size=128, name='layer_fc1'). \
        softmax_classifier(num_classes=num_classes, labels=y_true)


# Get the weights
def get_weights_variable(layer_name):
    with tf.variable_scope(layer_name, reuse=True):
        variable = tf.get_variable('weights')
    return variable


weights_conv1 = get_weights_variable('layer_conv1')
weights_conv2 = get_weights_variable('layer_conv2')

# Optimization Method
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

# Performance Measures
y_pred_cls = tf.argmax(y_pred, dimension=1)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Saver
saver = tf.train.Saver()
save_dir = 'checkpoints/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'best_validation')

#
#    TensorFlow Run
#

# Create new Session
session = tf.Session()


# Initialize variables
def init_variables():
    session.run(tf.global_variables_initializer())


init_variables()

# Helper-function to perform optimization iterations
train_batch_size = 64

# Best validation accuracy
best_validation_accuracy = 0.0

# Interation-number for last improvement
last_improvement = 0

# Stop optimization if no improvement found
require_improvement = 1000

# Optimization iterations

# total number of iterations
total_iterations = 0


def optimize(num_iterations):
    # global variables
    global total_iterations
    global best_validation_accuracy
    global last_improvement

    # Start time
    start_time = time.time()

    for i in range(num_iterations):

        # increase total number of iterations
        total_iterations += 1

        # get a batch of training examples
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)

        # put the batch into a dict
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # run the optimizer
        session.run(optimizer, feed_dict=feed_dict_train)

        # print status every 100 iterations
        if (total_iterations % 100 == 0) or (i == (num_iterations - 1)):

            # calculate the accuracy on the training batch
            acc_train = session.run(accuracy, feed_dict=feed_dict_train)

            # calculate the accuracy on the validation-set
            acc_validation, _ = validation_accuracy()

            # if validation accuracy is an improvement over best-known
            if acc_validation > best_validation_accuracy:

                # update the best-known
                best_validation_accuracy = acc_validation

                # set the iteration for the last improvement to current
                last_improvement = total_iterations

                # save all variables of the TF Graph to file
                saver.save(sess=session, save_path=save_path)

                # string that show improvement found
                improved_str = '*'

            else:

                # string that show no improvement found
                improved_str = ''

            # Status messenger
            msg = "Iter: {0:>6}, Train-Batch Accuracy: {1:>6.1%}, Validation Acc: {2:>6.1%} {3}"
            print(msg.format(i + 1, acc_train, acc_validation, improved_str))

        # if no improvement found in the required number of iterations.
        if total_iterations - last_improvement > require_improvement:
            print("no improvement found in the required number of iterations")
            break

    # end time
    end_time = time.time()

    # difference time
    time_dif = end_time - start_time

    # print the time usage
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


# Helper-function to plot example errors¶
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


# Helper-functions for calculating classifications
# Split the data-set in batches of this size to limit RAM usage.
batch_size = 256


def predict_cls(images, labels, cls_true):
    # Number of images.
    num_images = len(images)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_images, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_images:
        # The ending index for the next batch is denoted j.
        j = min(i + batch_size, num_images)

        # Create a feed-dict with the images and labels
        # between index i and j.
        feed_dict = {x: images[i:j, :],
                     y_true: labels[i:j, :]}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    return correct, cls_pred


def predict_cls_test():
    return predict_cls(images=data.test.images,
                       labels=data.test.labels,
                       cls_true=data.test.cls)


def predict_cls_validation():
    return predict_cls(images=data.validation.images,
                       labels=data.validation.labels,
                       cls_true=data.validation.cls)


# Helper-functions for the classification accuracy
def cls_accuracy(correct):
    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / len(correct)

    return acc, correct_sum


def validation_accuracy():
    # Get the array of booleans whether the classifications are correct
    # for the validation-set.
    # The function returns two values but we only need the first.
    correct, _ = predict_cls_validation()

    # Calculate the classification accuracy and return it.
    return cls_accuracy(correct)


# Helper-function for showing the performance
def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):
    # For all the images in the test-set,
    # calculate the predicted classes and whether they are correct.
    correct, cls_pred = predict_cls_test()

    # Classification accuracy and the number of correct classifications.
    acc, num_correct = cls_accuracy(correct)

    # Number of images being classified.
    num_images = len(correct)

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, num_correct, num_images))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)


# Helper-function for plotting convolutional weights
def plot_conv_weights(weights, input_channel=0):
    # Assume weights are TensorFlow ops for 4-dim variables
    # e.g. weights_conv1 or weights_conv2.

    # Retrieve the values of the weight-variables from TensorFlow.
    # A feed-dict is not necessary because nothing is calculated.
    w = session.run(weights)

    # Print mean and standard deviation.
    print("Mean: {0:.5f}, Stdev: {1:.5f}".format(w.mean(), w.std()))

    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)

    # Number of filters used in the conv. layer.
    num_filters = w.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))

    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i < num_filters:
            # Get the weights for the i'th filter of the input channel.
            # The format of this 4-dim tensor is determined by the
            # TensorFlow API. See Tutorial #02 for more details.
            img = w[:, :, input_channel, i]

            # Plot image.
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


#
#	Performance before any optimization
#
print_test_accuracy()
# plot_conv_weights(weights=weights_conv1)

# Perform 10000 optimization iterations
optimize(num_iterations=10000)
print_test_accuracy(show_example_errors=False,
                    show_confusion_matrix=False)
# plot_conv_weights(weights=weights_conv1)

#
#	Initialize Variables Again
#

init_variables()
# print_test_accuracy()
# plot_conv_weights()

# Restore Best Variables
saver.restore(sess=session, save_path=save_path)
print_test_accuracy(show_example_errors=True,
                    show_confusion_matrix=True)
plot_conv_weights(weights=weights_conv1)
