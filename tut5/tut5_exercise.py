import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import os

# prettytensor
import prettytensor as pt

# load data
from tensorflow.examples.tutorials.mnist import input_data

data = input_data.read_data_sets('../MNIST/', one_hot=True)

# class number
data.test.cls = np.argmax(data.test.labels, axis=1)
data.validation.cls = np.argmax(data.validation.labels, axis=1)

# helper function creating random training-set
combined_images = np.concatenate([data.train.images, data.validation.images], axis=0)
combined_labels = np.concatenate([data.train.labels, data.validation.labels], axis=0)

combined_size = len(combined_images)
train_size = int(0.8 * combined_size)
validation_size = combined_size - train_size


def random_training_set():
    # Create a randomized index
    idx = np.random.permutation(combined_size)

    # Split the random idx
    idx_train = idx[0:train_size]
    idx_validation = idx[train_size:]

    # Select the images and labels for the new training set
    x_train = combined_images[idx_train, :]
    y_train = combined_labels[idx_train, :]

    # Select the images and labels for the validation set
    x_validation = combined_images[idx_validation, :]
    y_validaiton = combined_images[idx_validation, :]

    # Return the new training and validation set
    return x_train, y_train, x_validation, y_validaiton


#
#    Data dimentions
#

img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
num_channels = 1
num_classes = 10


# Helper-function for plotting images
def plot_images(images,  # Images to plot, 2-d array.
                cls_true,  # True class-no for images.
                ensemble_cls_pred=None,  # Ensemble predicted class-no.
                best_cls_pred=None):  # Best-net predicted class-no.

    assert len(images) == len(cls_true)

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)

    # Adjust vertical spacing if we need to print ensemble and best-net.
    if ensemble_cls_pred is None:
        hspace = 0.3
    else:
        hspace = 1.0
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    # For each of the sub-plots.
    for i, ax in enumerate(axes.flat):

        # There may not be enough images for all sub-plots.
        if i < len(images):
            # Plot image.
            ax.imshow(images[i].reshape(img_shape), cmap='binary')

            # Show true and predicted classes.
            if ensemble_cls_pred is None:
                xlabel = "True: {0}".format(cls_true[i])
            else:
                msg = "True: {0}\nEnsemble: {1}\nBest Net: {2}"
                xlabel = msg.format(cls_true[i],
                                    ensemble_cls_pred[i],
                                    best_cls_pred[i])

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

# placeholder variables
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_images = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

# Creat Neural Network using Prettytensor
x_pretty = pt.wrap(x_images)
with pt.defaults_scope(activation_fn=tf.nn.relu):
    y_pred, loss = x_pretty. \
        conv2d(kernel=5, depth=16, name='layer_conv1'). \
        max_pool(kernel=2, stride=2). \
        conv2d(kernel=5, depth=36, name='layer_conv2'). \
        max_pool(kernel=2, stride=2). \
        flatten(). \
        fully_connected(size=128, name='layer_fc1'). \
        softmax_classifier(num_classes=num_classes, labels=y_true)

# Optimization method
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

# Performance measure
y_pred_cls = tf.argmax(y_pred, dimension=1)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Saver
saver = tf.train.Saver(max_to_keep=100)
save_dir = 'checkpoints/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


def get_save_path(net_number):
    return save_dir + 'network' + str(net_number)


#
# Running tensorflow
#

# Create new tensorflow session
session = tf.Session()


# Initialize variables
def init_variables():
    session.run(tf.initialize_all_variables())


# Helper function creating new training batch
train_batch_size = 64


# Function selecting a random training batch
def random_batch(x_train, y_train):
    # total images number
    num_images = len(x_train)

    # create a random index
    idx = np.random.choice(num_images,
                           size=train_batch_size,
                           replace=False)

    # use random idx to select random imgs and labels
    x_batch = x_train[idx, :]
    y_batch = y_train[idx, :]

    # return the batch
    return x_batch, y_batch


# Helper-function to perform optimization iterations
def optimize(num_iterations, x_train, y_train):
    # start time
    start_time = time.time()

    for i in range(num_iterations):

        # get a random batch
        x_batch, y_true_batch = random_batch(x_train, y_train)

        # put the batch into a dict
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # run the optimizer
        session.run(optimizer, feed_dict=feed_dict_train)

        # print status each 100 iterations
        if i % 100 == 0:
            # calculate the accuracy
            acc = session.run(accuracy, feed_dict=feed_dict_train)

            # print status msg
            msg = "Optimization Iteration: {0:>6}, Training Batch Accuracy: {1:>6.1%}"
            print(msg.format(i + 1, acc))

    # end time
    end_time = time.time()

    # difference time
    time_dif = end_time - start_time

    # print the time usage
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


# Create ensemble of neural networks
num_networks = 3
num_iterations = 7000

if True:
    for i in range(num_networks):
        print("Neural network: {0}".format(i))

        # create a random training set
        x_train, y_train, _, _ = random_training_set()

        # initialize the variables of TF Graph
        session.run(tf.global_variables_initializer())

        # optimize the variables
        optimize(num_iterations=num_iterations,
                 x_train=x_train,
                 y_train=y_train)

        # save the optimized variables
        saver.save(sess=session, save_path=get_save_path(i))

        # print new line
        print()

# Helper-functions for calculating and predicting classifications

# Split the data-set in batches of this size to limit RAM usage.
batch_size = 256


def predict_labels(images):
    # Number of images.
    num_images = len(images)

    # Allocate an array for the predicted labels which
    # will be calculated in batches and filled into this array.
    pred_labels = np.zeros(shape=(num_images, num_classes),
                           dtype=np.float)

    # Now calculate the predicted labels for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_images:
        # The ending index for the next batch is denoted j.
        j = min(i + batch_size, num_images)

        # Create a feed-dict with the images between index i and j.
        feed_dict = {x: images[i:j, :]}

        # Calculate the predicted labels using TensorFlow.
        pred_labels[i:j] = session.run(y_pred, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    return pred_labels


def correct_prediction(images, labels, cls_true):
    # Calculate the predicted labels.
    pred_labels = predict_labels(images=images)

    # Calculate the predicted class-number for each image.
    cls_pred = np.argmax(pred_labels, axis=1)

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    return correct


def test_correct():
    return correct_prediction(images=data.test.images,
                              labels=data.test.labels,
                              cls_true=data.test.cls)


def validation_correct():
    return correct_prediction(images=data.validation.images,
                              labels=data.validation.labels,
                              cls_true=data.validation.cls)


# Helper-functions for calculating the classification accuracy
def classification_accuracy(correct):
    return correct.mean()


def test_accuracy():
    correct = test_correct()
    return classification_accuracy(correct)


def validation_accuracy():
    correct = validation_correct()
    return classification_accuracy(correct)


#
#   Result and analysis
#

# Function for calculating the predicted labels for all Neural Network
def ensemble_predictions():
    # empty predicted labels list
    pred_labels = []

    # classification accuracy on the test-set
    test_accuracies = []

    # classification accuracy on the validation set
    val_accuracies = []

    # for each neural network in the ensemble
    for i in range(num_networks):
        # reload the variables
        saver.restore(sess=session, save_path=get_save_path(i))

        # calculate the classification accuracy on the test set
        test_acc = test_accuracy()

        # append the classification accuracy to the list
        test_accuracies.append(test_acc)

        # calculate the classification accuracy on the validation set
        val_acc = validation_accuracy()

        # append the classification accuracy to the list
        val_accuracies.append(val_acc)

        # print msg
        msg = "Network: {0}, Accuracy on Validation-Set: {1:.4f}, Test-Set: {2:.4f}"
        print(msg.format(i, val_acc, test_acc))

        # calculate the predicted labels
        pred = predict_labels(images=data.test.images)

        # append predicted label to the list
        pred_labels.append(pred)

    return np.array(pred_labels), np.array(test_accuracies), np.array(val_accuracies)


pred_labels, test_accuracies, val_accuracies = ensemble_predictions()

print("Mean test-set accuracy: {0:.4f}".format(np.mean(test_accuracies)))
print("Min test-set accuracy:  {0:.4f}".format(np.min(test_accuracies)))
print("Max test-set accuracy:  {0:.4f}".format(np.max(test_accuracies)))

# Ensemble predictions
ensemble_pred_labels = np.mean(pred_labels, axis=0)
ensemble_cls_pred = np.argmax(ensemble_pred_labels, axis=1)
ensemble_correct = (ensemble_cls_pred == data.test.cls)
ensemble_incorrect = np.logical_not(ensemble_correct)

# Best Neural Networl
best_net = np.argmax(test_accuracies)
best_net_pred_labels = pred_labels[best_net, :, :]
best_net_cls_pred = np.argmax(best_net_pred_labels, axis=1)
best_net_correct = (best_net_cls_pred == data.test.cls)
best_net_incorrect = np.logical_not(best_net_correct)

# Comparison of ensemble vs the best single network
ensemble_better = np.logical_and(best_net_incorrect, ensemble_correct)
best_net_better = np.logical_and(best_net_correct, ensemble_incorrect)

# print msg
print('Sum of ensemble_better = %d' % (ensemble_better.sum(),))
print('Sum of best_net_better = %d' % (best_net_better.sum(),))
