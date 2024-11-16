import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.image import ResizeMethod
import typing
import logging
import numpy as np
import tensorflow_addons as tfa
import tensorflow as tf
import gin



def preproc_data(image_path, selected_model):
    """
    Preprocess the input image for different models.

    Parameters:
        image_path (str): Path to the image.
        selected_model (str): The name of the model to be used.

    Returns:
        image (Tensor): Preprocessed image.
    """
    image_string = tf.io.read_file(image_path)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.io.decode_jpeg(image_string, channels=3)

    if selected_model.lower() == 'baseline':
        # This will convert to float values in [0, 1]
        image = tf.image.convert_image_dtype(image, tf.float32)
    elif selected_model.lower() == 'inception_v3' or selected_model.lower() == 'resnet_50':
        image = tf.cast(image, tf.float32)
        if selected_model.lower() == 'inception_v3':
            # This will convert to float values in [-1,1]
            image = tf.keras.applications.inception_v3.preprocess_input(image)
        elif selected_model.lower() == 'resnet_50':
            # This will convert to float values in [-1,1]
            image = tf.keras.applications.resnet_v2.preprocess_input(image)

    # CCreating a bounding box to get more focus on the image
    image = tf.image.crop_to_bounding_box(image, 0, 200, 2848, 3550)

    image = tf.image.resize_with_pad(
        image, 256, 256, method=ResizeMethod.BILINEAR, antialias=False)
    return image
