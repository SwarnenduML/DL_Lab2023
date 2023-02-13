import os
import gin
import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers 
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet import ResNet50
from keras.applications.efficientnet_v2 import EfficientNetV2B3
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


@gin.configurable
class ModelArchitecture(object):
    """
    Class that encapsulates the different models used in the architecture.

    Parameters:
    -----------
    baseline_units : int
        The number of units in the baseline CNN model.
    baseline_dropout_rate : float
        Dropout rate for the baseline CNN model.
    inception_dropout_rate : float
        Dropout rate for the InceptionV3 model.
    resnet_dropout_rate : float
        Dropout rate for the ResNet50 model.
    efficient_dropout_rate : float
        Dropout rate for the EfficientNetV2B3 model.
    """
    def __init__(self, baseline_units, baseline_dropout_rate, inception_dropout_rate, resnet_dropout_rate, efficient_dropout_rate):
        self.baseline_units = baseline_units
        self.baseline_dropout_rate = baseline_dropout_rate
        self.inception_dropout_rate = inception_dropout_rate
        self.resnet_dropout_rate = resnet_dropout_rate
        self.efficient_dropout_rate = efficient_dropout_rate

    def baseline_CNN_model(self):
        """
        Function that creates and returns the baseline CNN model.

        Returns:
        --------
        model : keras.Model
                The baseline CNN model.
        """
        inputs = keras.Input(shape=(256,256,3))
        x = layers.Conv2D(self.baseline_units, 3, padding='same', activation='relu')(inputs)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(self.baseline_units*2, 3, padding='same', activation='relu')(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(self.baseline_units*4, 3, padding='same', activation='relu')(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(self.baseline_dropout_rate)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(self.baseline_units*8, activation='relu')(x)
        outputs = layers.Dense(1, activation = 'sigmoid')(x)
        baseline_model = keras.Model(inputs, outputs, name = 'baseline_cnn_model')
        return baseline_model
    
    def inception_v3(self):
        """
        Function that creates and returns the InceptionV3 model.

        Returns:
        --------
        model : keras.Model
                The InceptionV3 model.
        """
        input_tensor = Input(shape=(256, 256, 3))

        # create the base pre-trained model
        base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)
        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers
        for layer in base_model.layers:
            layer.trainable = False

        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        x = Dense(512, activation='relu')(x)
        x = Dropout(self.inception_dropout_rate)(x)
        # and a logistic layer -- let's say we have 2 classes
        predictions = Dense(1, activation='sigmoid')(x)

        # this is the model we will train
        inception_model = Model(inputs=base_model.input, outputs=predictions, name = 'inceptionV3')


        return inception_model
    
    
    def resnet_50(self):
        """
        Function that creates and returns the ResNet50 model.

        Returns:
        --------
        model : keras.Model
                The ResNet50 model.
        """
        input_tensor = Input(shape=(256, 256, 3))

        # create the base pre-trained model
        base_model = ResNet50(input_tensor = input_tensor, weights='imagenet', include_top=False)
        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional ResNet50 layers
        for layer in base_model.layers:
            layer.trainable = False

        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        x = Dense(512, activation='relu')(x)
        x = Dropout(self.resnet_dropout_rate)(x)
        # and a logistic layer -- let's say we have 2 classes
        predictions = Dense(1, activation='sigmoid')(x)

        # this is the model we will train
        resnet50_model = Model(inputs=base_model.input, outputs=predictions, name = 'ResNet50')

        return resnet50_model
    
    
    def efficient_v3b3(self):
        """
        Function that creates and returns the EfficientNetV2B3 model.

        Returns:
        --------
        model : keras.Model
                The EfficientNetV2B3 model.
        """
        input_tensor = Input(shape=(256, 256, 3))

        # create the base pre-trained model
        base_model = EfficientNetV2B3(input_tensor=input_tensor, weights='imagenet', include_top=False, include_preprocessing=True)
        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional EfficientV3 layers
        for layer in base_model.layers:
            layer.trainable = False

        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        x = Dense(512, activation='relu')(x)
        x = Dropout(self.efficient_dropout_rate)(x)
        # and a logistic layer -- let's say we have 2 classes
        predictions = Dense(1, activation='sigmoid')(x)

        # this is the model we will train
        efficient_model = Model(inputs=base_model.input, outputs=predictions, name = 'EfficientNetV2B3')

        return efficient_model
