import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import math

#
# Load data
#
from mnist import MNIST

data = MNIST(data_dir='../MNIST')

# print("Size of:")
# print("- Training-set:\t\t{}".format(data.num_train))
# print("- Validation-set:\t{}".format(data.num_val))
# print("- Test-set:\t\t{}".format(data.num_test))

# Data dimensions
img_size = data.img_size
img_size_flat = data.img_size_flat
img_shape = data.img_shape
num_classes = data.num_classes
num_channels = data.num_channels


#
#   Helper functions plotting img
#

# Plot 9 images in 3x3 grid
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


# plot 10 images in a 2x5 grid
def plot_images10(images, smooth=True):
    # Interpolation type.
    if smooth:
        interpolation = 'spline16'
    else:
        interpolation = 'nearest'

    # Create figure with sub-plots.
    fig, axes = plt.subplots(2, 5)

    # Adjust vertical spacing.
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    # For each entry in the grid.
    for i, ax in enumerate(axes.flat):
        # Get the i'th image and only use the desired pixels.
        img = images[i, :, :]

        # Plot the image.
        ax.imshow(img, interpolation=interpolation, cmap='binary')

        # Remove ticks.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


# plot a single img
def plot_image(image):
    plt.imshow(image, interpolation='nearest', cmap='binary')
    plt.xticks([])
    plt.yticks([])


# # Get the first images from the test-set.
# images = data.x_test[0:9]
#
# # Get the true classes for those images.
# cls_true = data.y_test_cls[0:9]
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
y_true_cls = tf.argmax(y_true, axis=1)

# Neural Network
net = x_image

# conv layer 1
net = tf.layers.conv2d(inputs=net,
                       name='layer_conv1',
                       padding='same',
                       filters=16,
                       kernel_size=5,
                       activation=tf.nn.relu)

net = tf.layers.max_pooling2d(inputs=net,
                              pool_size=2,
                              strides=2)

# conv layer 2
net = tf.layers.conv2d(inputs=net,
                       name='layer_conv2',
                       padding='same',
                       filters=36,
                       kernel_size=5,
                       activation=tf.nn.relu)

net = tf.layers.max_pooling2d(inputs=net,
                              pool_size=2,
                              strides=2)

# flatten layer
net = tf.layers.flatten(net)

# fully-connected layer 1
net = tf.layers.dense(inputs=net,
                      name='layer_fc1',
                      units=128,
                      activation=tf.nn.relu)

# final fully-connected layer
net = tf.layers.dense(inputs=net,
                      name='layer_fc_out',
                      units=num_classes,
                      activation=None)

logits = net
y_pred = tf.nn.softmax(logits=logits)
y_pred_cls = tf.argmax(y_pred, axis=1)

# Loss-function to be optimized
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=logits)
loss = tf.reduce_mean(cross_entropy)

# Optimization Method
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

# Classification Accuracy
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#
#   Optimize the Neural Network
#

# Create TF session
session = tf.Session()

# Initialize variables
session.run(tf.global_variables_initializer())

# Helper-function to perform optimization iterations

train_batch_size = 64

# Counter for total number of iterations performed so far.
total_iterations = 0


def optimize(num_iterations):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    for i in range(total_iterations,
                   total_iterations + num_iterations):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch, _ = data.random_batch(batch_size=train_batch_size)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)

        # Print status every 100 iterations.
        if i % 100 == 0:
            # Calculate the accuracy on the training-set.
            acc = session.run(accuracy, feed_dict=feed_dict_train)

            # Message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

            # Print it.
            print(msg.format(i + 1, acc))

    # Update the total number of iterations performed.
    total_iterations += num_iterations


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
    images = data.x_test[incorrect]

    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = data.y_test_cls[incorrect]

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
    cls_true = data.y_test_cls

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
    num_test = data.num_test

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
        images = data.x_test[i:j, :]

        # Get the associated labels.
        labels = data.y_test[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict = {x: images,
                     y_true: labels}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Convenience variable for the true class-numbers of the test-set.
    cls_true = data.y_test_cls

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
#   Performance
#

# Performance before any optimization
print_test_accuracy()

# Performance after 10000 optimization iterations
optimize(num_iterations=10000)
print_test_accuracy(show_example_errors=False,
                    show_confusion_matrix=False)


#
#   Optimizing the Input Images
#

# Helper-function for getting the names of convolutional layers
def get_conv_layer_names():
    graph = tf.get_default_graph()
    names = [op.name for op in graph.get_operations() if op.type == 'Conv2D']
    return names


conv_names = get_conv_layer_names()


# Helper-function for finding the input image
def optimize_image(conv_id=None,
                   feature=0,
                   num_iterations=30,
                   show_progress=True):
    # create loss-function that must be maximized
    if conv_id is None:
        loss = tf.reduce_mean(logits[:, feature])
    else:

        # get the name of conv layer
        conv_name = conv_names[conv_id]

        # get the default TF graph
        graph = tf.get_default_graph()

        # reference to the tensor that is output by the operator
        tensor = graph.get_tensor_by_name(conv_name + ':0')

        # loss function
        loss = tf.reduce_mean(tensor[:, :, :, feature])

    # get the gradient for the loss function
    gradient = tf.gradients(loss, x_image)

    # generate a random img of the same size as the raw input
    image = 0.1 * np.random.uniform(size=img_shape) + 0.45

    # perform a number of optimization iterations
    for i in range(num_iterations):

        # reshape the array
        img_reshaped = image[np.newaxis, :, :, np.newaxis]

        # create a feed dict
        feed_dict = {x_image: img_reshaped}

        # calculate the predicted class-scores
        pred, grad, loss_value = session.run([y_pred, gradient, loss],
                                             feed_dict=feed_dict)

        # squeeze the dimensionality
        grad = np.array(grad).squeeze()

        # calculate the step size
        step_size = 1.0 / (grad.std() + 1e-8)

        # update the img by adding the scaled gradient
        image += step_size * grad

        # limit the pixel-value
        image = np.clip(image, 0.0, 1.0)

        if show_progress:
            print("Iteration:", i)

            # Convert the predicted class-scores to a one-dim array.
            pred = np.squeeze(pred)

            # The predicted class for the Inception model.
            pred_cls = np.argmax(pred)

            # The score (probability) for the predicted class.
            cls_score = pred[pred_cls]

            # Print the predicted score etc.
            msg = "Predicted class: {0}, score: {1:>7.2%}"
            print(msg.format(pred_cls, cls_score))

            # Print statistics for the gradient.
            msg = "Gradient min: {0:>9.6f}, max: {1:>9.6f}, stepsize: {2:>9.2f}"
            print(msg.format(grad.min(), grad.max(), step_size))

            # Print the loss-value.
            print("Loss:", loss_value)

            # Newline.
            print()

    return image.squeeze()


# Function finding the images that maximize the first 10 features of a layer
def optimize_images(conv_id=None,
                    num_iterations=30):
    # layer used
    if conv_id is None:
        print('Final fully-connected layer before softmax.')
    else:
        print('Layer: ', conv_names[conv_id])

    # initialize the arr of img
    images = []

    for feature in range(0, 10):
        print("Optimizing image for feature no.", feature)

        image = optimize_image(conv_id=conv_id,
                               feature=feature,
                               show_progress=False,
                               num_iterations=num_iterations)

        # squeeze the dim of the arr
        image = image.squeeze()

        # append to the list of images
        images.append(image)

    # convert to numpy arr
    images = np.array(images)

    # plot the images
    plot_images10(images=images)


#
#   @@@@@
#

# conv layer 1
optimize_images(conv_id=0)

# conv layer 2
optimize_images(conv_id=1)

# final output layer
image = optimize_image(conv_id=None,
                       feature=2,
                       num_iterations=10,
                       show_progress=True)
plot_image(image)
optimize_images(conv_id=None)
