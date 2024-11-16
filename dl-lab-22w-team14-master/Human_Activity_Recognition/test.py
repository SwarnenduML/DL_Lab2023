import tensorflow as tf
import numpy as np
from tensorflow import keras
import time

class Testing_routine:
    '''
        This is the main class which tests the model.

        Parameters:
        MODEL_NAME - The model that is to be used
        test_dataset - The dataset that is to be used for testing
        BAYESIAN_OPT - If Bayesian optimization is needed or not

        '''

    def __init__(self, MODEL_NAME, test_dataset, BAYESIAN_OPT):
        self.MODEL_NAME = MODEL_NAME
        self.test_dataset = test_dataset
        self.BAYESIAN = BAYESIAN_OPT

    def masked_sparse_categorical_crossentropy(self, y, logits):
        '''
        This is used to calculate the modified entropy. The entropy is calculated without the use of class 0
        Args:
            y: The actual value
            logits: The logits generated

        Returns:
            loss : The crossentropy loss generated between the loss and the actual value

        '''
        # Remove the extra dimension from y
        y = tf.squeeze(y, axis=-1)
        mask = tf.not_equal(y, 0)
        loss = keras.losses.sparse_categorical_crossentropy(y, logits)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss = loss * mask
        return tf.reduce_mean(loss)

    def masked_sparse_categorical_accuracy(self, y_true, y_pred):
        '''
        This function calculates the sparse categorical accuracy between the true value and the predicted value.

        Args:
            y_true: The true value of the data
            y_pred: The predicted value as predicted by the model

        Returns:
            modified accuracy is returned for each masked data
        '''
        y_true = tf.squeeze(y_true, axis=-1)
        mask = tf.not_equal(y_true, 0)
        y_pred = tf.argmax(y_pred, axis=-1)
        acc = tf.equal(tf.cast(y_true, y_pred.dtype), y_pred)
        acc = tf.cast(acc, tf.float32)
        acc = acc * tf.cast(mask, acc.dtype)
        return tf.math.count_nonzero(acc) / (tf.math.count_nonzero(mask))

    def recall(self,y_true, y_pred):
        '''
        This function calculates the recall of the model.
        Args:
            y_true: The true value of the data
            y_pred: The predicted value as predicted by the model

        Returns:
            modified recall is returned for each masked data
        '''
        y_true = tf.cast(tf.not_equal(y_true, 0), y_pred.dtype)
        true_positives = tf.math.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
        all_positives = tf.math.reduce_sum(tf.round(tf.clip_by_value(y_true, 0, 1)))
        recall = true_positives / (all_positives + tf.keras.backend.epsilon())
        return recall

    def precision(self, y_true, y_pred):
        '''
        This function calculates the precision of the model.
        Args:
            y_true: The true value of the data
            y_pred: The predicted value as predicted by the model

        Returns:
            modified precision is returned for each masked data
        '''
        y_true = tf.cast(tf.not_equal(y_true, 0), y_pred.dtype)
        true_positives = tf.math.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
        predicted_positives = tf.math.reduce_sum(tf.round(tf.clip_by_value(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
        return precision

    def f1_score(self, y_true, y_pred):
        '''
        This function calculates the F1-score of the model.
        Args:
            y_true: The true value of the data
            y_pred: The predicted value as predicted by the model

        Returns:
            modified F1-score is returned for each masked data
        '''
        precision1 = self.precision(y_true, y_pred)
        recall1 = self.recall(y_true, y_pred)
        return 2 * ((precision1 * recall1) / (precision1 + recall1 + tf.keras.backend.epsilon()))

    @tf.function
    def test_step_2(self, x, y):
        '''
        This function is the main testing step in which the output of the test dataset is calculated.
        Args:
            x: The input data
            y: The output value of the data that is to be predicted

        '''
        test_logits = self.MODEL_NAME(x, training=False)
        loss_value = self.masked_sparse_categorical_crossentropy(y, test_logits)
        acc_res = self.masked_sparse_categorical_accuracy(y, test_logits)
        precision_res = self.precision(y, test_logits)
        recall_res = self.recall(y, test_logits)
        f1_res = self.f1_score(y, test_logits)
        return acc_res, precision_res, recall_res, f1_res, loss_value

    def testing(self):
        '''
        This function is the called to start the testing of the model.
        '''
        start_time = time.time()
        loss_dis, f1_dis, recall_dis, precision_dis, acc_dis = [], [], [], [], []
        # Iterate over the batches of the dataset.
        for step, (x_batch_test, y_batch_test) in enumerate(self.test_dataset):
            acc_res, precision_res, recall_res, f1_res, loss_value = self.test_step_2(x_batch_test, y_batch_test)
            # Display metrics at the end of each epoch.
            f1_dis.append(f1_res)
            acc_dis.append(acc_res)
            recall_dis.append(recall_res)
            precision_dis.append(precision_res)
            loss_dis.append(loss_value)

        acc = np.mean(acc_dis)
        f1 = np.mean(f1_dis)
        recall = np.mean(recall_dis)
        precision = np.mean(precision_dis)
        loss_dis = np.mean(loss_dis)

        if self.BAYESIAN == False:
            print(
                f"Test - loss:{loss_dis:.6f}, acc: {acc:.6f}, f1: {f1:.6f}, recall: {recall:.6f}, precision: {precision:.6f}")
        else:
            return acc

    def predict(self, x):
        logits = self.MODEL_NAME(x, training=False)
        return logits