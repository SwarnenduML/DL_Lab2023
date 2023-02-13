import gin
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from sklearn.metrics import confusion_matrix
import os

@gin.configurable
class Visualization:
    '''
    This class is called for visualization of the test data between the timestamps given on the model chosen.
    The graphs are then saved to an output directory.
    '''

    def __init__(self, test_dataset, model_test, START_TIME_STAMP, END_TIME_STAMP, OUTPUT_DIR):
            self.model_test = model_test
            self.test_dataset = test_dataset
            self.START_TIME_STAMP = START_TIME_STAMP
            self.END_TIME_STAMP = END_TIME_STAMP
            self.OUTPUT_DIR = OUTPUT_DIR

    def visualizer(self):

        test_x_y_zip = self.test_dataset.unbatch()
        
        features = []
        true_label = []
        iterator = test_x_y_zip.as_numpy_iterator()
        for zip_element in iterator:
            x, y = zip_element
            if y.all() != 0:
                features.append(x)
                true_label.append(y)
    
        count = len(true_label)
        # Convert the list of elements to a tensor
        tensor = tf.stack(features)

        # Make a prediction on a new input tensor x
        y_pred = self.model_test.predict(tensor)

        # The shape of y_pred is [1185,250,13], converting it to 2D
        y_pred = tf.reshape(y_pred, [count*250,13])

        # Get the index of the maximum value(most probability of activity) along the last axis (axis 1)
        predictions = tf.argmax(y_pred, axis=1)
        true_label = tf.reshape(true_label, [-1])

        # confusion matrix
        cm = confusion_matrix(true_label, predictions)

        # Plot the predictions and true values
        # OUTPUT_DIR = '/home/lakshay/Desktop/HAPT/visual'
        
        if not os.path.exists(self.OUTPUT_DIR):
            os.makedirs(self.OUTPUT_DIR)

        fig, axes = plt.subplots(1, 1, figsize = (25,5))
        axes.plot(true_label[self.START_TIME_STAMP: self.END_TIME_STAMP],label='True values')
        axes.plot(predictions[self.START_TIME_STAMP: self.END_TIME_STAMP],label='Predictions', color = 'r')

        axes.set_xlabel('Time Stamps')
        axes.set_ylabel('Activities')
        axes.set_title('Visualization')

        plt.legend()

        fig.savefig(self.OUTPUT_DIR+ '/viz_activity_prediction.png')

        print('\n')

        fig, axes = plt.subplots(1, 1, figsize = (15,10))
        sns.heatmap(data = cm, annot = True, cmap = 'GnBu',  fmt = 'g', xticklabels = np.arange(1,13), yticklabels = np.arange(1,13), ax = axes, linecolor = 'white', linewidths= 2)

        fig.savefig(self.OUTPUT_DIR+ '/viz_confusion_matrix.png')
    
