import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math

# PrettyTensor
import prettytensor as pt

#
#   Load MNIST data set
#
from tensorflow.examples.tutorials.mnist import input_data

data = input_data.read_data_sets("../MNIST", one_hot=True)

# print("Size of:")
# print("- Training-set:\t\t{}".format(len(data.train.labels)))
# print("- Test-set:\t\t{}".format(len(data.test.labels)))
# print("- Validation-set:\t{}".format(len(data.validation.labels)))

data.test.cls = np.argmax(data.test.labels, axis=1)

# Data Dimensions
img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
num_channels = 1
num_classes = 10


# Helper-function for plotting images
def plot_images(images, cls_true, cls_pred=None, noise=0.0):
    assert len(images) == len(cls_true) == 9

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Get the i'th image and reshape the array.
        image = images[i].reshape(img_shape)

        # Add the adversarial noise to the image.
        image += noise

        # Ensure the noisy pixel-values are between 0 and 1.
        image = np.clip(image, 0.0, 1.0)

        # Plot image.
        ax.imshow(image,
                  cmap='binary', interpolation='nearest')

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


#  # Get the first images from the test-set.
# images = data.test.images[0:9]
#
# # Get the true classes for those images.
# cls_true = data.test.cls[0:9]
#
# # Plot the images and labels using our helper-function above.
# plot_images(images=images, cls_true=cls_true)

#
#   TF Graph
#

# Placeholder variables
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

#
#	Adversarial Noise
#
noise_limit = 0.35
noise_l2_weight = 0.02
ADVERSARY_VARIABLES = 'adversary_variables'
collections = [tf.GraphKeys.VARIABLES, ADVERSARY_VARIABLES]

# Create new variable for the adversarial noise
x_noise = tf.Variable(tf.zeros([img_size, img_size, num_channels]),
                      name='x_noise',
                      trainable=False,
                      collections=collections)

x_noise_clip = tf.assign(x_noise, tf.clip_by_value(x_noise, -noise_limit, noise_limit))
x_noisy_image = x_image + x_noise
x_noisy_image = tf.clip_by_value(x_noisy_image, 0.0, 1.0)

#
# Convolutional Neural Network
#

x_pretty = pt.wrap(x_noisy_image)

with pt.defaults_scope(activation_fn=tf.nn.relu):
    y_pred, loss = x_pretty. \
        conv2d(kernel=5, depth=16, name='layer_conv1'). \
        max_pool(kernel=2, stride=2). \
        conv2d(kernel=5, depth=36, name='layer_conv2'). \
        max_pool(kernel=2, stride=2). \
        flatten(). \
        fully_connected(size=128, name='layer_fc1'). \
        softmax_classifier(num_classes=num_classes, labels=y_true)

# Optimizer for Normal Training
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

# Optimizer for Adversarial Noise
adversary_variables = tf.get_collection(ADVERSARY_VARIABLES)
l2_loss_noise = noise_l2_weight * tf.nn.l2_loss(x_noise)
loss_adversary = loss + l2_loss_noise
optimizer_adversary = tf.train.AdamOptimizer(learning_rate=1e-2). \
    minimize(loss_adversary, var_list=adversary_variables)

# Performance Measures
y_pred_cls = tf.argmax(y_pred, dimension=1)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#
#	TF Run
#

# Create new session
session = tf.Session()

# Initialize variables
session.run(tf.global_variables_initializer())


def init_noise():
    session.run(tf.variables_initializer([x_noise]))


init_noise()

# Helper-function to perform optimization iterations
train_batch_size = 64


# Optimize function
def optimize(num_iterations, adversary_target_cls=None):
    # start time
    start_time = time.time()

    for i in range(num_iterations):

        # get a batch of training examples
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)

        # use the adversarial target-class
        if adversary_target_cls is not None:
            # set all the class labels to zero
            y_true_batch = np.zeros_like(y_true_batch)

            # set the element for the adversarial target-class to 1
            y_true_batch[:, adversary_target_cls] = 1.0

        # put the batch into a dict
        feed_dict_train = {x: x_batch, y_true: y_true_batch}

        # if doing normal optimization of the neural network
        if adversary_target_cls is None:
            session.run(optimizer, feed_dict=feed_dict_train)
        else:
            session.run(optimizer_adversary, feed_dict=feed_dict_train)
            session.run(x_noise_clip)

        # print status each 100 iterations
        if (i % 100 == 0) or (i == num_iterations - 1):
            # calculate the accuracy
            acc = session.run(accuracy, feed_dict=feed_dict_train)

            # msg for printing
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
            print(msg.format(i, acc))

    # end time
    end_time = time.time()

    # time different
    time_dif = end_time - start_time

    # print time usage
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


#
#	Helper functions 
#

# Helper-functions for getting and plotting the noise
def get_noise():
    # Run the TensorFlow session to retrieve the contents of
    # the x_noise variable inside the graph.
    noise = session.run(x_noise)

    return np.squeeze(noise)


def plot_noise():
    # Get the adversarial noise from inside the TensorFlow graph.
    noise = get_noise()

    # Print statistics.
    print("Noise:")
    print("- Min:", noise.min())
    print("- Max:", noise.max())
    print("- Std:", noise.std())

    # Plot the noise.
    plt.imshow(noise, interpolation='nearest', cmap='seismic',
               vmin=-1.0, vmax=1.0)


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

    # Get the adversarial noise from inside the TensorFlow graph.
    noise = get_noise()

    # Plot the first 9 images.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9],
                noise=noise)


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
#   Normal optimization of neural network
#
optimize(num_iterations=1000)
print_test_accuracy(show_example_errors=False)


#
#	Adversarial noise for all target-classes
#

# Helper-function for finding the adversarial noise for all target-classes
def find_all_noise(num_iterations=1000):
    # adversarial noise for all target-classes
    all_noise = []

    for i in range(num_classes):
        print("Finding adversarial noise for target-class:", i)

        # Reset the adversarial noise to zero
        init_noise()

        # optimize the adversarial noise
        optimize(num_iterations=num_iterations, adversary_target_cls=i)

        # get the adversarial noise from TF Graph
        noise = get_noise()

        # append the noise to array
        all_noise.append(noise)

        # print new line
        print('\n')
    return all_noise


all_noise = find_all_noise(num_iterations=1000)


# Helper function plotting the adversarial noise for all target-classes
def plot_all_noise(all_noise):
    # Create figure with 10 sub-plots.
    fig, axes = plt.subplots(2, 5)
    fig.subplots_adjust(hspace=0.2, wspace=0.1)

    # For each sub-plot.
    for i, ax in enumerate(axes.flat):
        # Get the adversarial noise for the i'th target-class.
        noise = all_noise[i]

        # Plot the noise.
        ax.imshow(noise,
                  cmap='seismic', interpolation='nearest',
                  vmin=-1.0, vmax=1.0)

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(i)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


plot_all_noise(all_noise)


#
#	Immunity to adversarial noise
#

# Helper-function to make a neural network immune to noise
def make_immune(target_cls,
                num_iterations_adversary=500,
                num_iterations_immune=200):
    print('Target class: ', target_cls)
    print('Finding adversarial noise...')

    # find the adversarial noise
    optimize(num_iterations=num_iterations_adversary,
             adversary_target_cls=target_cls)

    # new line
    print('\n')

    # print classification accuracy
    print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False)

    # new line
    print('\n')

    print("Making the neural network immune to the noise ...")

    # Make NN immune to this noise
    optimize(num_iterations=num_iterations_immune)

    # new line
    print('\n')

    # print classification accuracy
    print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False)


#
#	Make immune to noise for all target-classes with double run
#
for i in range(10):
    make_immune(target_cls=i)
    print('\n')
    make_immune(target_cls=i)
    print('\n')

# Plot the adversarial noise
plot_noise()

print_test_accuracy(show_example_errors=False,
                    show_confusion_matrix=False)

# Performance on clean images
init_noise()
print_test_accuracy(show_example_errors=False,
                    show_confusion_matrix=False)

