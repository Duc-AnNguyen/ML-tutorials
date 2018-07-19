import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

import inception

inception.data_dir = '../inception/'
inception.maybe_download()

# load the inception model
model = inception.Inception()

# Get the Input and Output for the Inception Model
resized_image = model.resized_image
y_pred = model.y_pred
y_logits = model.y_logits

# Hack the model
with model.graph.as_default():
    # placeholder variable for the target class-number
    pl_cls_target = tf.placeholder(dtype=tf.int32)

    # add a new loss function
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_logits,
                                                          labels=[pl_cls_target])

    # Get the gradient for the loss function
    gradient = tf.gradients(loss, resized_image)

#
#   TF Session
#
session = tf.Session(graph=model.graph)

# Helper-function for finding Adversary Noise
def find_adversary_noise(image_path,
                         cls_target,
                         noise_limit=3.0,
                         required_score=0.99,
                         max_iterations=100):
    # create a feed dict with the image
    feed_dict = model._create_feed_dict(image_path=image_path)

    # use TF to calculate the predicted class-scores
    pred, image = session.run([y_pred, resized_image],
                              feed_dict=feed_dict)

    # convert to 1D array
    pred = np.squeeze(pred)

    # predicted class-number
    cls_source = np.argmax(pred)

    # score for the predicted class
    score_source_org = pred.max()

    # names for src and target classes
    name_source = model.name_lookup.cls_to_name(cls_source,
                                                only_first_name=True)
    name_target = model.name_lookup.cls_to_name(cls_target,
                                                only_first_name=True)
    # Initialize noise to zero
    noise = 0

    # perfrom iterations optimization to find the noise
    for i in range(max_iterations):
        print('Iterations: ', i)

        # compute noise image
        noisy_image = image + noise

        # limit pixel-values of noise image
        noisy_image = np.clip(a=noisy_image,
                              a_min=0.0,
                              a_max=255.0)

        # create a feed dict
        feed_dict = {model.tensor_name_resized_image: noisy_image,
                     pl_cls_target: cls_target}

        # calculate the predicted class-scores
        pred, grad = session.run([y_pred, gradient],
                                 feed_dict=feed_dict)

        # convert the predicted class-scores to 1D array
        pred = np.squeeze(pred)

        # scores for src and target
        score_source = pred[cls_source]
        score_target = pred[cls_target]

        # squeeze the dimensionality for the gradient-array
        grad = np.array(grad).squeeze()

        # calculate the max of the absolute gradient values
        grad_absmax = np.abs(grad).max()

        # lower limit of gradient
        if grad_absmax < 1e-10:
            grad_absmax = 1e-10

        # calculate the step size for updating the image noise
        step_size = 7 / grad_absmax

        # print the score for the src-class
        msg = "Source score: {0:>7.2%}, class-number: {1:>4}, class-name: {2}"
        print(msg.format(score_source, cls_source, name_source))

        # print the score for the target-class
        msg = "Target score: {0:>7.2%}, class-number: {1:>4}, class-name: {2}"
        print(msg.format(score_target, cls_target, name_target))

        # print static for the gradient
        msg = "Gradient min: {0:>9.6f}, max: {1:>9.6f}, stepsize: {2:>9.2f}"
        print(msg.format(grad.min(), grad.max(), step_size))
        print('\n')

        # If the score for the target not high enough
        if score_target < required_score:
            noise -= step_size * grad
            noise = np.clip(a=noise,
                            a_min=-noise_limit,
                            a_max=noise_limit)
        else:
            break

    return image.squeeze(), noisy_image.squeeze(), noise, \
           name_source, name_target, \
           score_source, score_source_org, score_target


# Helper-function for plotting image and noise
def normalize_image(x):
    # Get the min and max values for all pixels in the input.
    x_min = x.min()
    x_max = x.max()

    # Normalize so all values are between 0.0 and 1.0
    x_norm = (x - x_min) / (x_max - x_min)

    return x_norm


def plot_images(image, noise, noisy_image,
                name_source, name_target,
                score_source, score_source_org, score_target):
    """
    Plot the image, the noisy image and the noise.
    Also shows the class-names and scores.

    Note that the noise is amplified to use the full range of
    colours, otherwise if the noise is very low it would be
    hard to see.

    image: Original input image.
    noise: Noise that has been added to the image.
    noisy_image: Input image + noise.
    name_source: Name of the source-class.
    name_target: Name of the target-class.
    score_source: Score for the source-class.
    score_source_org: Original score for the source-class.
    score_target: Score for the target-class.
    """

    # Create figure with sub-plots.
    fig, axes = plt.subplots(1, 3, figsize=(10, 10))

    # Adjust vertical spacing.
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    # Use interpolation to smooth pixels?
    smooth = True

    # Interpolation type.
    if smooth:
        interpolation = 'spline16'
    else:
        interpolation = 'nearest'

    # Plot the original image.
    # Note that the pixel-values are normalized to the [0.0, 1.0]
    # range by dividing with 255.
    ax = axes.flat[0]
    ax.imshow(image / 255.0, interpolation=interpolation)
    msg = "Original Image:\n{0} ({1:.2%})"
    xlabel = msg.format(name_source, score_source_org)
    ax.set_xlabel(xlabel)

    # Plot the noisy image.
    ax = axes.flat[1]
    ax.imshow(noisy_image / 255.0, interpolation=interpolation)
    msg = "Image + Noise:\n{0} ({1:.2%})\n{2} ({3:.2%})"
    xlabel = msg.format(name_source, score_source, name_target, score_target)
    ax.set_xlabel(xlabel)

    # Plot the noise.
    # The colours are amplified otherwise they would be hard to see.
    ax = axes.flat[2]
    ax.imshow(normalize_image(noise), interpolation=interpolation)
    xlabel = "Amplified Noise"
    ax.set_xlabel(xlabel)

    # Remove ticks from all the plots.
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


# Helper-function for finding and plotting adversarial example


def adversary_example(image_path,
                      cls_target,
                      noise_limit,
                      required_score):
    # find the adversarial noise
    image, noisy_image, noise, \
    name_source, name_target, \
    score_source, score_source_org, score_target = \
        find_adversary_noise(image_path=image_path,
                             cls_target=cls_target,
                             noise_limit=noise_limit,
                             required_score=required_score)

    # plot the image and noise
    plot_images(image=image, noise=noise, noisy_image=noisy_image,
                name_source=name_source, name_target=name_target,
                score_source=score_source, score_source_org=score_source_org,
                score_target=score_target)

    # print statistics for the noise
    msg = "Noise min: {0:.3f}, max: {1:.3f}, mean: {2:.3f}, std: {3:.3f}"
    print(msg.format(noise.min(), noise.max(),
                     noise.mean(), noise.std()))


#
#    Examples
#

# Parrot
image_path = "../images/parrot_cropped1.jpg"

adversary_example(image_path=image_path,
                  cls_target=300,
                  noise_limit=3.0,
                  required_score=0.99)
# Elon Musk
image_path = "../images/elon_musk.jpg"
adversary_example(image_path=image_path,
                  cls_target=300,
                  noise_limit=3.0,
                  required_score=0.99)

# Willy Wonka
image_path = "../images/willy_wonka_new.jpg"

adversary_example(image_path=image_path,
                  cls_target=300,
                  noise_limit=3.0,
                  required_score=0.99)
