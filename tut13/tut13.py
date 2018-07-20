import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# Inception
import inception

inception.data_dir = '../inception/'
inception.maybe_download()


# Names of conv layers
def get_conv_layer_names():
    # load the Model
    model = inception.Inception()

    # creat a list of names for the operations in the graph
    names = [op.name for op in model.graph.get_operations() if op.type == 'Conv2D']

    # close TF session
    model.close()

    return names


conv_names = get_conv_layer_names()


# print(len(conv_names))
# print(conv_names[-5:])

# Function finding input image
def optimize_image(conv_id=None,
                   feature=0,
                   num_iterations=30,
                   show_progress=True):
    # load the inception model
    model = inception.Inception()

    # reference to the tensor taking the raw input img
    resized_image = model.resized_image

    # reference to the tensor for the predicted classes
    y_pred = model.y_pred

    # create the loss-function that must be maximized
    if conv_id is None:
        loss = model.y_logits[0, feature]
    else:
        conv_name = conv_names[conv_id]

        # reference to the tensor that is output by the operator
        tensor = model.graph.get_tensor_by_name(conv_name + ':0')

        # set the Inception Graph as the default
        with model.graph.as_default():
            loss = tf.reduce_mean(tensor[:, :, :, feature])

    # get the gradient for the loss function
    gradient = tf.gradients(loss, resized_image)

    # create the TF session
    session = tf.Session(graph=model.graph)

    # generate a random image
    image_shape = resized_image.get_shape()
    image = np.random.uniform(size=image_shape) + 128.0

    # perform a number of optimization iterations
    for i in range(num_iterations):

        # create a feed dict
        feed_dict = {model.tensor_name_resized_image: image}

        # calculate the predicted class-scores
        pred, grad, loss_value = session.run([y_pred, gradient, loss],
                                             feed_dict=feed_dict)

        # squeeze the dimensionality
        grad = np.array(grad).squeeze()

        # calculate the step-size for updating the image
        step_size = 1.0 / (grad.std() + 1e-8)

        # add the scaled gradient for image
        image += step_size * grad

        # limit the pixel-values in the image
        image = np.clip(image, 0.0, 255.0)

        if show_progress:
            print("Iteration:", i)

            # convert the predicted cls scores to 1D array
            pred = np.squeeze(pred)

            # the predicted cls for Inception Model
            pred_cls = np.argmax(pred)

            # name of the predicted cls
            cls_name = model.name_lookup.cls_to_name(pred_cls,
                                                     only_first_name=True)

            # the score for the predicted cls
            cls_score = pred[pred_cls]

            # print the predicted score
            msg = "Predicted class-name: {0} (#{1}), score: {2:>7.2%}"
            print(msg.format(cls_name, pred_cls, cls_score))

            # print statistics for the gradient
            msg = "Gradient min: {0:>9.6f}, max: {1:>9.6f}, stepsize: {2:>9.2f}"
            print(msg.format(grad.min(), grad.max(), step_size))

            # print the loss-value.
            print("Loss:", loss_value)

            # Newline.
            print()

    # close the TensorFlow session inside the model-object
    model.close()

    return image.squeeze()


# Helper-function for plotting image and noise
def normalize_image(x):
    # Get the min and max values for all pixels in the input.
    x_min = x.min()
    x_max = x.max()

    # Normalize so all values are between 0.0 and 1.0
    x_norm = (x - x_min) / (x_max - x_min)

    return x_norm


def plot_image(image):
    # Normalize the image so pixels are between 0.0 and 1.0
    img_norm = normalize_image(image)

    # Plot the image.
    plt.imshow(img_norm, interpolation='nearest')
    plt.show()


def plot_images(images, show_size=100):
    """
    The show_size is the number of pixels to show for each image.
    The max value is 299.
    """

    # Create figure with sub-plots.
    fig, axes = plt.subplots(2, 3)

    # Adjust vertical spacing.
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    # Use interpolation to smooth pixels?
    smooth = True

    # Interpolation type.
    if smooth:
        interpolation = 'spline16'
    else:
        interpolation = 'nearest'

    # For each entry in the grid.
    for i, ax in enumerate(axes.flat):
        # Get the i'th image and only use the desired pixels.
        img = images[i, 0:show_size, 0:show_size, :]

        # Normalize the image so its pixels are between 0.0 and 1.0
        img_norm = normalize_image(img)

        # Plot the image.
        ax.imshow(img_norm, interpolation=interpolation)

        # Remove ticks.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


# Helper-function for optimizing and plotting images
def optimize_images(conv_id=None,
                    num_iterations=30,
                    show_size=100):
    # print conv layer name
    if conv_id is None:
        print('Final fully-connected layer before softmax.')
    else:
        print('Layer: ', conv_names[conv_id])

    # initialize the array of images
    images = []

    for feature in range(1, 7):
        print('Optimizing image for feature no.', feature)

        # find the image that maximizes the given features
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

    # plot the image
    plot_images(images=images, show_size=show_size)


#
#   Results
#

# # Optimize a single image for an early convolutional layer
# image = optimize_image(conv_id=5,
#                        feature=2,
#                        num_iterations=30,
#                        show_progress=True)
# plot_image(image)

# # Optimize multiple images for convolutional layers
# optimize_images(conv_id=0, num_iterations=10)
# optimize_images(conv_id=3, num_iterations=30)
# optimize_images(conv_id=4, num_iterations=30)
# optimize_images(conv_id=5, num_iterations=30)
# optimize_images(conv_id=6, num_iterations=30)
# optimize_images(conv_id=7, num_iterations=30)
# optimize_images(conv_id=8, num_iterations=30)
# optimize_images(conv_id=9, num_iterations=30)
# optimize_images(conv_id=10, num_iterations=30)
# optimize_images(conv_id=20, num_iterations=30)
# optimize_images(conv_id=30, num_iterations=30)
# optimize_images(conv_id=40, num_iterations=30)
# optimize_images(conv_id=50, num_iterations=30)
# optimize_images(conv_id=60, num_iterations=30)
# optimize_images(conv_id=70, num_iterations=30)
# optimize_images(conv_id=80, num_iterations=30)
# optimize_images(conv_id=90, num_iterations=30)
# optimize_images(conv_id=93, num_iterations=30)

# Final fully-connected layer before Softmax
optimize_images(conv_id=None, num_iterations=30)
image = optimize_image(conv_id=None,
                       feature=1,
                       num_iterations=100,
                       show_progress=True)
plot_image(image)
