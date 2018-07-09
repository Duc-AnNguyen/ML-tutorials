#!/usr/bin/python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math

#
#    Configuration of Neural Network
#

# Convolutional Layer 1

filter_size1 = 5
num_filters1 = 16

# Convolutional Layer 2

filter_size2 = 5
num_filters2 = 36

# Fully-connected Layer

fc_size = 128

#
#    Load data
#

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('../MNIST/', one_hot=True)

"""
print('Size of:')
print('- Training-set:\t\t{}'.format(len(data.train.labels)))
print('- Test-set:\t\t{}'.format(len(data.test.labels)))
print('- Validation-set:\t{}'.format(len(data.validation.labels)))
"""

data.test.cls = np.array([label.argmax() for label in data.test.labels])

#
# Data dimensions
#

img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
num_channels = 1
num_classes = 10


# Function for plotting image (copy from the tutorial)

def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9

    # Create figure with 3x3 sub-plots.

    (fig, axes) = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for (i, ax) in enumerate(axes.flat):

        # Plot image.

        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.

        if cls_pred is None:
            xlabel = 'True: {0}'.format(cls_true[i])
        else:
            xlabel = 'True: {0}, Pred: {1}'.format(cls_true[i],
                    cls_pred[i])

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

# functions creating new variables

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


# function creating new Convolutional Layer

def new_conv_layer(
    input,
    num_input_channel,
    filter_size,
    num_filters,
    use_pooling=True,
    ):

    shape = [filter_size, filter_size, num_input_channel, num_filters]
    weights = new_weights(shape=shape)
    biases = new_biases(length=num_filters)
    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1,
                         1], padding='SAME')
    layer += biases
    if use_pooling:
        layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1], padding='SAME')
    layer = tf.nn.relu(layer)
    return (layer, weights)


# function flatting a layer

def flatten_layer(layer):

    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return (layer_flat, num_features)


# function creating new Fully-Connected Layer

def new_fc_layer(
    input,
    num_inputs,
    num_outputs,
    use_relu=True,
    ):

    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer


# Placeholder variables

x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes],
                        name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)

# Convolutional Layer 1

layer_conv1, weights_conv1 = new_conv_layer(input=x_image,
        num_input_channel=num_channels, filter_size=filter_size1,
        num_filters=num_filters1, use_pooling=True)

# Convolutional Layer 1

layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1,
        num_input_channel=num_filters1, filter_size = filter_size2,
        num_filters=num_filters2,
        use_pooling=True)

# Flatten Layer

layer_flat, num_features = flatten_layer(layer_conv2)

# Fully-Connected Layer 1

layer_fc1 = new_fc_layer(input=layer_flat,
    num_inputs=num_features,
    num_outputs=fc_size,
    use_relu=True)

# Fully-Connected Layer 1

layer_fc2 = new_fc_layer(input=layer_fc1,
    num_inputs=fc_size,
    num_outputs=num_classes,
    use_relu=False)

# Predicted Class

y_pred = tf.nn.softmax(layer_fc2)

y_pred_cls = tf.argmax(y_pred, axis=1)

# Cost-function

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                        labels=y_true)
cost = tf.reduce_mean(cross_entropy)

# Optimization Method

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

# Performance Measures

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#
#    TensorFlow Run
#

session = tf.Session()
session.run(tf.global_variables_initializer())

# function performing optimization iterations

train_batch_size = 64
total_iterations = 0
def optimize(num_iterations):
    global total_iterations
    start_time = time.time()
    for i in range (total_iterations, total_iterations + num_iterations):
        x_batch, y_true_batch=data.train.next_batch(train_batch_size)
        feed_dict_train={x:x_batch,y_true:y_true_batch}
        session.run(optimizer, feed_dict=feed_dict_train)
        if i % 100 == 0:
            acc = session.run(accuracy, feed_dict=feed_dict_train)
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
            print(msg.format(i+1,acc))
    total_iterations += num_iterations
    end_time = time.time()
    time_dif = end_time - start_time
    print ("Time usage: "+str(timedelta(seconds=int(round(time_dif)))))

# function plotting example errors

def plot_example_errors(cls_pred, correct):
    incorrect = (correct==False)
    images = data.test.images[incorrect]
    cls_pred = cls_pred[incorrect]
    cls_true = data.test.cls[incorrect]
    plot_images(images=images[0:9],
        cls_true=cls_true[0:9],
        cls_pred=cls_pred[0:9])

# function plotting confusion matrix

def plot_confusion_matrix(cls_pred):
    cls_true = data.test.cls
    cm = confusion_matrix(y_true = cls_true,
        y_pred = cls_pred)
    print(cm)
    plt.matshow(cm)
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# function showing performance

test_batch_size = 256
def print_test_accuracy(show_example_errors= False,
                         show_confusion_matrix = False):
    num_test = len(data.test.images)
    cls_pred=np.zeros(shape=num_test, dtype = np.int)
    i = 0
    while i<num_test:
        j=min(i+test_batch_size, num_test)
        images = data.test.images[i:j,:]
        labels = data.test.labels[i:j,:]
        feed_dict = {x:images, y_true:labels}
        cls_pred[i:j]=session.run(y_pred_cls, feed_dict=feed_dict)
        i = j
    cls_true = data.test.cls
    correct = (cls_true==cls_pred)
    correct_sum = correct.sum()
    acc = float(correct_sum)/num_test
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)

#
#	Performance before any optimization
#

print_test_accuracy()

#
#	Performance after 1 optimization iteration
#


optimize(num_iterations=1)
print_test_accuracy()


#
#	Performance after 100 optimization iteration
#


optimize(num_iterations=99)
print_test_accuracy()


#
#	Performance after 1000 optimization iteration
#


optimize(num_iterations=900)
print_test_accuracy()


#
#	Performance after 10000 optimization iteration
#


optimize(num_iterations=9000)
print_test_accuracy()
print_test_accuracy(show_example_errors=True, show_confusion_matrix=True)