import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import os

# Inception Model
import inception

# PrettyTensor
import prettytensor as pt

#
#   Load data
#
# import knifey
# from knifey import num_classes
#
# knifey.data_dir = '../data/knifey-spoony'
# data_dir = knifey.data_dir
# knifey.maybe_download_and_extract()
# dataset = knifey.load()

data_dir = '../data/myown'

from dataset import load_cached
dataset = load_cached(cache_path='../data/myown/myown_cache.pkl',
                      in_dir= data_dir)
num_classes = dataset.num_classes
# Training and Test-sets
class_names = dataset.class_names

# Get the training set
image_paths_train, cls_train, labels_train = dataset.get_training_set()

# Get the test-sets
image_paths_test, cls_test, labels_test = dataset.get_test_set()


# Helper-function for plotting images
def plot_images(images, cls_true, cls_pred=None, smooth=True):
    assert len(images) == len(cls_true)

    # Create figure with sub-plots.
    fig, axes = plt.subplots(3, 3)

    # Adjust vertical spacing.
    if cls_pred is None:
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    # Interpolation type.
    if smooth:
        interpolation = 'spline16'
    else:
        interpolation = 'nearest'

    for i, ax in enumerate(axes.flat):
        # There may be less than 9 images, ensure it doesn't crash.
        if i < len(images):
            # Plot image.
            ax.imshow(images[i],
                      interpolation=interpolation)

            # Name of the true class.
            cls_true_name = class_names[cls_true[i]]

            # Show true and predicted classes.
            if cls_pred is None:
                xlabel = "True: {0}".format(cls_true_name)
            else:
                # Name of the predicted class.
                cls_pred_name = class_names[cls_pred[i]]

                xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)

            # Show the classes as the label on the x-axis.
            ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


# Helper-function for loading images
from matplotlib.image import imread


def load_images(image_paths):
    # Load the images from disk.
    images = [imread(path) for path in image_paths]

    # Convert to a numpy array and return it.
    return np.asarray(images)


#
#   Download the Inception Model
#
inception.data_dir = '../inception/'
inception.maybe_download()

# Load the Inception Model
model = inception.Inception()

# Calculate Transfer-Values
from inception import transfer_values_cache

file_path_cache_train = os.path.join(data_dir, 'inception-myown-train.pkl')
file_path_cache_test = os.path.join(data_dir, 'inception-myown-test.pkl')

# Process Inception transfer-values for training-images
transfer_values_train = transfer_values_cache(cache_path=file_path_cache_train,
                                              image_paths=image_paths_train,
                                              model=model)

# Process Inception transfer-values for test-images
transfer_values_test = transfer_values_cache(cache_path=file_path_cache_test,
                                             image_paths=image_paths_test,
                                             model=model)
print(transfer_values_train.shape)
print(transfer_values_test.shape)


# Helper-function for plotting transfer-values
def plot_transfer_values(i):
    print("Input image:")

    # Plot the i'th image from the test-set.
    image = imread(image_paths_test[i])
    plt.imshow(image, interpolation='spline16')
    plt.show()

    print("Transfer-values for the image using Inception model:")

    # Transform the transfer-values into an image.
    img = transfer_values_test[i]
    img = img.reshape((32, 64))

    # Plot the image for the transfer-values.
    plt.imshow(img, interpolation='nearest', cmap='Reds')
    plt.show()


# plot_transfer_values(i=50)

# # Analysis of Transfer-Values using PCA
# from sklearn.decomposition import PCA
# pca = PCA(n_components=2)
# transfer_values = transfer_values_train
# cls = cls_train
# transfer_values_reduced = pca.fit_transform(transfer_values)
#
# # Helper-function for plotting the reduced transfer-values
# def plot_scatter(values, cls):
#     # Create a color-map with a different color for each class.
#     import matplotlib.cm as cm
#     cmap = cm.rainbow(np.linspace(0.0, 1.0, num_classes))
#
#     # Create an index with a random permutation to make a better plot.
#     idx = np.random.permutation(len(values))
#
#     # Get the color for each sample.
#     colors = cmap[cls[idx]]
#
#     # Extract the x- and y-values.
#     x = values[idx, 0]
#     y = values[idx, 1]
#
#     # Plot it.
#     plt.scatter(x, y, color=colors, alpha=0.5)
#     plt.show()
#
# plot_scatter(transfer_values_reduced, cls=cls)
#
# # Analysis of Transfer-Values using t-SNE
# from sklearn.manifold import TSNE
# pca = PCA(n_components=50)
# transfer_values_50d = pca.fit_transform(transfer_values)
# tsne = TSNE(n_components=2)
# transfer_values_reduced = tsne.fit_transform(transfer_values_50d)
# plot_scatter(transfer_values_reduced, cls=cls)

#
#   New TF Classifier
#

# Placeholder variables
transfer_len = model.transfer_len
x = tf.placeholder(tf.float32, shape=[None, transfer_len], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

# Neural Network
x_pretty = pt.wrap(x)
with pt.defaults_scope(activation_fn=tf.nn.relu):
    y_pred, loss = x_pretty. \
        fully_connected(size=1024, name='layer_fc1'). \
        softmax_classifier(num_classes=num_classes, labels=y_true)

# Optimization Method
global_step = tf.Variable(initial_value=0,
                          name='global_step',
                          trainable=False)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, global_step)

# Classification accuracy
y_pred_cls = tf.argmax(y_pred, dimension=1)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#
#   Running TF
#
session = tf.Session()
session.run(tf.global_variables_initializer())

# Helper-function to get a random training-batch
train_batch_size = 64


def random_batch():
    # Number of images (transfer-values) in the training-set.
    num_images = len(transfer_values_train)

    # Create a random index.
    idx = np.random.choice(num_images,
                           size=train_batch_size,
                           replace=False)

    # Use the random index to select random x and y-values.
    # We use the transfer-values instead of images as x-values.
    x_batch = transfer_values_train[idx]
    y_batch = labels_train[idx]

    return x_batch, y_batch

# Helper-function to perform optimization
def optimize(num_iterations):
    # start time
    start_time = time.time()

    for i in range(num_iterations):

        # get batch of training examples
        x_batch, y_true_batch = random_batch()

        # Put the batch into a dict
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # run the optimizer
        i_global, _ = session.run([global_step, optimizer],
                                  feed_dict=feed_dict_train)

        # print status each 100 iteration
        if (i_global % 100 == 0) or (i == num_iterations - 1):
            # calculate the accuracy
            batch_acc = session.run(accuracy,
                                    feed_dict=feed_dict_train)

            # print status
            msg = "Global Step: {0:>6}, Training Batch Accuracy: {1:>6.1%}"
            print(msg.format(i_global, batch_acc))

            # end time
    end_time = time.time()

    # difference time
    time_dif = end_time - start_time

    # print time usage
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


#
#   Helper-Functions for Showing Results
#

# Helper-function to plot example errors
def plot_example_errors(cls_pred, correct):
    # This function is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # correct is a boolean array whether the predicted class
    # is equal to the true class for each image in the test-set.

    # Negate the boolean array.
    incorrect = (correct == False)

    # Get the indices for the incorrectly classified images.
    idx = np.flatnonzero(incorrect)

    # Number of images to select, max 9.
    n = min(len(idx), 9)

    # Randomize and select n indices.
    idx = np.random.choice(idx,
                           size=n,
                           replace=False)

    # Get the predicted classes for those images.
    cls_pred = cls_pred[idx]

    # Get the true classes for those images.
    cls_true = cls_test[idx]

    # Load the corresponding images from the test-set.
    # Note: We cannot do image_paths_test[idx] on lists of strings.
    image_paths = [image_paths_test[i] for i in idx]
    images = load_images(image_paths)

    # Plot the images.
    plot_images(images=images,
                cls_true=cls_true,
                cls_pred=cls_pred)


# Helper-function to plot confusion matrix
# Import a function from sklearn to calculate the confusion-matrix.
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cls_pred):
    # This is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_test,  # True class for test-set.
                          y_pred=cls_pred)  # Predicted class.

    # Print the confusion matrix as text.
    for i in range(num_classes):
        # Append the class-name to each line.
        class_name = "({}) {}".format(i, class_names[i])
        print(cm[i, :], class_name)

    # Print the class-numbers for easy reference.
    class_numbers = [" ({0})".format(i) for i in range(num_classes)]
    print("".join(class_numbers))


# Helper-functions for calculating classifications
# Split the data-set in batches of this size to limit RAM usage.
batch_size = 256


def predict_cls(transfer_values, labels, cls_true):
    # Number of images.
    num_images = len(transfer_values)

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
        feed_dict = {x: transfer_values[i:j],
                     y_true: labels[i:j]}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    return correct, cls_pred


def predict_cls_test():
    return predict_cls(transfer_values=transfer_values_test,
                       labels=labels_test,
                       cls_true=cls_test)


# Helper-functions for calculating the classification accuracy
def classification_accuracy(correct):
    # When averaging a boolean array, False means 0 and True means 1.
    # So we are calculating: number of True / len(correct) which is
    # the same as the classification accuracy.

    # Return the classification accuracy
    # and the number of correct classifications.
    return correct.mean(), correct.sum()


# Helper-function for showing the classification accuracy
def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):
    # For all the images in the test-set,
    # calculate the predicted classes and whether they are correct.
    correct, cls_pred = predict_cls_test()

    # Classification accuracy and the number of correct classifications.
    acc, num_correct = classification_accuracy(correct)

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


#
#   Results
#
print_test_accuracy(show_example_errors=False,
                    show_confusion_matrix=False)
optimize(num_iterations=1000)
print_test_accuracy(show_example_errors=False,
                    show_confusion_matrix=False)
