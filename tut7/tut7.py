from IPython.display import Image, display
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import inception

# Download inception model
inception.data_dir = '../inception/'
inception.maybe_download()

# Load the inception model
model = inception.Inception()


# function classifying and plotting images
def classify(image_path):
    # display the image
    display(Image(image_path))

    # use inception model to classify the image
    pred = model.classify(image_path=image_path)

    # print the scores and names of top 10 predictions
    model.print_scores(pred=pred, k=10, only_first_name=True)


#
#   Panda
#
image_path = os.path.join(inception.data_dir, 'cropped_panda.jpg')
classify(image_path=image_path)

#
#	Parrot
#

# Original Image
classify(image_path='../images/parrot.jpg')

# Resize Image
def plot_resized_images(image_path):

    # get the resized image
    resized_image = model.get_resized_image(image_path=image_path)

    # plot the image
    plt.imshow(resized_image, interpolation='nearest')

    # show the plot
    plt.show()

# plot_resized_images('../images/parrot.jpg')

# Cropped Image (Top)
classify(image_path='../images/parrot_cropped1.jpg')

# Cropped Image (Middle)
classify(image_path='../images/parrot_cropped2.jpg')

# Cropped Image (Bottom)
classify(image_path='../images/parrot_cropped3.jpg')

# Padded Image
classify(image_path='../images/parrot_padded.jpg')

#
#   Elon Musk
#

# 299*299 pixels
classify(image_path='../images/elon_musk.jpg')

# 100*100 pixels
classify(image_path='../images/elon_musk_100x100.jpg')

# plot_resized_images(image_path="images/elon_musk_100x100.jpg")

#
#   Willy Wonka
#

# Old (Gene Wilder)
classify(image_path="../images/willy_wonka_old.jpg")

# New (Johnny Depp)
classify(image_path="../images/willy_wonka_new.jpg")

#
#   My own images
#

# ChiPu
classify(image_path="../images/chipu.jpg")

# Ha Anh Tuan
classify(image_path="../images/haanhtuan.jpg")

# ferrari
classify(image_path="../images/ferrari.jpg")

# laptop asus
classify(image_path="../images/asus_laptop.jpg")



