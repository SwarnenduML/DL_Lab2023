import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from IPython.display import Image, display
import matplotlib.cm as cm
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import load_img
from tensorflow import keras
import tensorflow as tf
import matplotlib.image as image
import matplotlib.pyplot as plt
import logging
import gin



@gin.configurable
class Visualization(object):
    """
    Class to visualize the Grad-CAM heatmaps of a given model on a set of images

    Parameters:
        model_name (str): name of the model
        image_path (str): path to the directory containing the images
        NUMBER_OF_IMAGES (int): number of images to process
        IMAGE_SIZE (tuple): size of the images to be loaded
        image_save_folder (str): path to the directory where the images will be saved
        grad_cam_model_source_1 (str): source location of the model (directory path)
        grad_cam_model_source_2 (str): secondary source location of the model (directory path)
    """

    def __init__(self, model_name,
                 image_path, NUMBER_OF_IMAGES, IMAGE_SIZE, image_save_folder, grad_cam_model_source_1, grad_cam_model_source_2):
        self.grad_cam_model_source_1 = grad_cam_model_source_1
        self.grad_cam_model_source_2 = grad_cam_model_source_2
        self.grad_cam_model = model_name
        self.image_path = image_path
        self.NUMBER_OF_IMAGES = NUMBER_OF_IMAGES
        self.image_save_folder = image_save_folder
        self.IMAGE_SIZE = IMAGE_SIZE

        # Validate if the image path exists and contains the required number of images
        if (not os.path.exists(self.image_path)):
            raise ValueError(f"{self.image_path} is not a directory")
        elif len(os.listdir(self.image_path)) < NUMBER_OF_IMAGES:
            raise ValueError(
                f"{self.image_path} does not have {self.NUMBER_OF_IMAGES} images to process.")

        # Validate if the model exists in the specified source location
        if (not os.path.exists(self.grad_cam_model_source_1 + "/" + self.grad_cam_model)) or len(
                os.listdir(self.grad_cam_model_source_1 + "/" + self.grad_cam_model)) < 1:
            raise ValueError(
                f"{self.grad_cam_model_source_1} does not have model {self.grad_cam_model}")

    def get_img_array(self, img_path, size):
        """
        get_img_array function takes in an image path and a size and returns the image as a float32 Numpy array.

        Arguments:
            img_path (str): Path of the image to be loaded.
            size (tuple): Target size of the image.

        Returns:
            numpy.ndarray: Image as a float32 Numpy array.
        """
        # `img` is a PIL image of size 256x256
        img = keras.preprocessing.image.load_img(img_path, target_size=size)
        # `array` is a float32 Numpy array of shape (256, 256, 3)
        array = keras.preprocessing.image.img_to_array(img)
        # We add a dimension to transform our array into a "batch"
        # of size (1, 256, 256, 3)
        array = np.expand_dims(array, axis=0)
        return array

    def make_gradcam_heatmap(self, img_array, model, last_conv_layer_name, pred_index=None):
        """
        make_gradcam_heatmap function generates a Grad-CAM heatmap for an input image using a specified model and layer.

        Arguments:
            img_array (numpy.ndarray): Image as a float32 Numpy array of shape (1, size[0], size[1], 3).
            model (tf.keras.Model): Keras model for which the Grad-CAM heatmap is to be generated.
            last_conv_layer_name (str): Name of the last convolutional layer in the model.
            pred_index (int, optional): Index of the top predicted class. If None, the index is obtained from tf.argmax.

        Returns:
            numpy.ndarray: Grad-CAM heatmap as a float32 Numpy array.
        """
        # First, we create a model that maps the input image to the activations
        # of the last conv layer as well as the output predictions
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(
                last_conv_layer_name).output, model.output]
        )

        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        # then sum all the channels to obtain the heatmap class activation
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

        return heatmap.numpy()

    def save_and_display_gradcam(self, img_path, heatmap, i, cam_path="cam.jpg", alpha=0.4):
        """
        Saves and displays the Grad CAM heatmap.

        Parameters:
            img_path (str): Path to the original image.
            heatmap (np.array): Heatmap generated by the Grad CAM.
            i (str): Index of the image.
            cam_path (str): File name to save the Grad CAM image.
            alpha (float): Opacity of the heatmap superimposed on the original image.

        Returns:
            cam_path (str): Path to the saved Grad CAM image.
        """
        # Load the original image
        img = keras.preprocessing.image.load_img(img_path)
        img = keras.preprocessing.image.img_to_array(img)

        # Rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * heatmap)

        # Use jet colormap to colorize heatmap
        jet = cm.get_cmap("jet")

        # Use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        # Create an image with RGB colorized heatmap
        jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

        # Superimpose the heatmap on original image
        superimposed_img = jet_heatmap * alpha + img
        superimposed_img = keras.preprocessing.image.array_to_img(
            superimposed_img)

        # Save the superimposed image
        superimposed_img.save(self.image_save_folder+'/cam_' + i + cam_path)

        # Display Grad CAM
        # display(Image(cam_path))
        return cam_path

    def grad_cam(self):
        """
        Generates Grad CAM heatmap for a set of images.

        Parameters:
            self: Object of the current class.

        Returns: 
            None
        """
        if not os.path.exists(self.grad_cam_model_source_2 + '/' + self.grad_cam_model):
            model = tf.keras.models.load_model(
                self.grad_cam_model_source_1 + '/' + self.grad_cam_model, compile=False)
        else:
            model = tf.keras.models.load_model(
                self.grad_cam_model_source_2 + '/' + self.grad_cam_model, compile=False)

        logging.info(f"{self.grad_cam_model} is being used for Grad CAM")

        if not os.path.exists(self.image_save_folder+'/'+self.grad_cam_model):
            os.mkdir(self.image_save_folder+'/'+self.grad_cam_model)
        self.image_save_folder = self.image_save_folder+'/'+self.grad_cam_model
        list_of_imgs = os.listdir(self.image_path)[:self.NUMBER_OF_IMAGES]
        model.layers[-1].activation = None

        for i in list_of_imgs:
            img_orig = np.array(load_img(self.image_path+"/"+i))
            image.imsave(self.image_save_folder+"/original_"+i, img_orig)
            img_array = self.get_img_array(
                self.image_path+"/"+i, self.IMAGE_SIZE)

            layer = None
            for layers in reversed(model.layers):
                if str(layers.name)[:4] == 'conv' or 'conv' in str(layers.name):
                    layer = layers.name
                    break

            if layer is None:
                raise ValueError(
                    f"4D output not available. GradCAM not possible")

            # Generate class activation heatmap

            heatmap = self.make_gradcam_heatmap(img_array, model, layer)
            cam_path = self.save_and_display_gradcam(
                self.image_path + "/" + i, heatmap, i)
