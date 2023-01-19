import gin
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import logging
import typing
from tensorflow.image import ResizeMethod


def preproc_data(image_path, selected_model,img_array): # <- configs
    if selected_model.lower() == 'baseline' or selected_model.lower() == 'efficient':
        image_string = tf.io.read_file(image_path)

        # Don't use tf.image.decode_image, or the output shape will be undefined
        image = tf.io.decode_jpeg(image_string, channels=3)
        # This will convert to float values in [0, 1]
        image = tf.image.convert_image_dtype(image, tf.float32)
    else:
        if selected_model.lower() == 'inception_v3':
            # This will convert to float values in [-1,1]
            image = tf.keras.applications.inception_v3.preprocess_input(img_array)
        elif selected_model.lower() == 'resnet_50':
            # This will convert to float values in [-1,1]
            image = tf.keras.applications.resnet_v2.preprocess_input(img_array)

    # CCreating a bounding box to get more focus on the image
    image = tf.image.crop_to_bounding_box(image, 0, 200, 2848, 3550)

    image = tf.image.resize_with_pad(image, 256, 256, method=ResizeMethod.BILINEAR, antialias=False)
    return image