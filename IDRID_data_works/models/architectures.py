import gin
import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers 
from models.layers import vgg_block, inception_module
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet import ResNet50
from keras.applications.efficientnet_v2 import EfficientNetV2B3


class ModelArchitecture():
    '''Create a model with transfer learning.'''

    def baseline_CNN_model():
        inputs = keras.Input(shape=(256,256,3))
        x = layers.Conv2D(16, 3, padding='same', activation='relu')(inputs)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        outputs = layers.Dense(1, activation = 'sigmoid')(x)
        baseline_model = keras.Model(inputs, outputs)
        return baseline_model
    
    def inception_v3():
        input_tensor = Input(shape=(256, 256, 3))

        # create the base pre-trained model
        base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)

        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        # and a logistic layer -- let's say we have 2 classes
        predictions = Dense(1, activation='sigmoid')(x)

        # this is the model we will train
        inception_model = Model(inputs=base_model.input, outputs=predictions)

        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers
        for layer in base_model.layers:
            layer.trainable = False
        
        return inception_model
    
    
    def resnet_50():
        input_tensor = Input(shape=(256, 256, 3))

        # create the base pre-trained model
        base_model = ResNet50(input_tensor = input_tensor, weights='imagenet', include_top=False)

        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        # and a logistic layer -- let's say we have 2 classes
        predictions = Dense(1, activation='sigmoid')(x)

        # this is the model we will train
        resnet50_model = Model(inputs=base_model.input, outputs=predictions)

        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional ResNet50 layers
        for layer in base_model.layers:
            layer.trainable = False
        return resnet50_model
    
    
    def efficient_v3b3():
        input_tensor = Input(shape=(256, 256, 3))

        # create the base pre-trained model
        base_model = EfficientNetV2B3(input_tensor=input_tensor, weights='imagenet', include_top=False, include_preprocessing=True)

        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.37)(x)
        # and a logistic layer -- let's say we have 2 classes
        predictions = Dense(1, activation='sigmoid')(x)

        # this is the model we will train
        efficient_model = Model(inputs=base_model.input, outputs=predictions)

        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional EfficientNet layers
        for layer in base_model.layers:
            layer.trainable = False

        return efficient_model
