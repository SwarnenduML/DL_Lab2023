import gin
import tensorflow as tf
import numpy as np
from tensorflow import keras
import time


@gin.configurable
class Training_routine(object):
    '''
        This is the main class which trains the model.

        Parameters: (given in config.gin)
        epochs        - The number of epochs the model has to run
        patience      - The number of wait cycles needed for early_stopping
        learning_rate - The learning rate for the updation of weights

        Parameters:
        model_name - The model that is to be used
        train_dataset - The dataset that is to be used for training
        val_dataset - The dataset that is to be used for validation
        bayesian_opt - If Bayesian optimization is needed or not

        '''
    def __init__(self, MODEL_NAME,  train_dataset, val_dataset, BAYESIAN_OPT, EPOCHS, PATIENCE, learning_rate):
        self.MODEL_NAME = MODEL_NAME
        self.EPOCHS = EPOCHS
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.PATIENCE = PATIENCE
        self.counter = 0
        self.best_score = np.inf
        self.stop_training = False
        self.bayesian = BAYESIAN_OPT
        self.learning_rate = learning_rate
        self.optimizer = keras.optimizers.Adam(self.learning_rate)

    def masked_sparse_categorical_crossentropy(self, y, logits):
        '''
        This is used to calculate the modified entropy. The entropy is calculated without the use of class 0
        Args:
            y: The actual value
            logits: The logits generated

        Returns:
            loss : The crossentropy loss generated between the loss and the actual value

        '''
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
    def train_step(self, x, y):
        '''
        This function is the main training step in which model's weights are updated.
        Args:
            x: The input data
            y: The output value of the data that is to be predicted

        '''
        with tf.GradientTape() as tape:
            logits = self.MODEL_NAME(x, training=True)
            loss_value = self.masked_sparse_categorical_crossentropy(y, logits)
        grads = tape.gradient(loss_value, self.MODEL_NAME.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.MODEL_NAME.trainable_weights))
        acc_res = self.masked_sparse_categorical_accuracy(y, logits)
        precision_res = self.precision(y, logits)
        recall_res = self.recall(y, logits)
        f1_res = self.f1_score(y, logits)
        return precision_res, recall_res, f1_res, acc_res, loss_value

    @tf.function
    def test_step(self, x, y):
        '''
        This function is the main testing step in which the output of the test dataset is calculated.
        Args:
            x: The input data
            y: The output value of the data that is to be predicted

        '''
        val_logits = self.MODEL_NAME(x, training=False)
        loss_value = self.masked_sparse_categorical_crossentropy(y, val_logits)
        acc_res = self.masked_sparse_categorical_accuracy(y, val_logits)
        f1_res = self.f1_score(y, val_logits)
        recall_res = self.recall(y, val_logits)
        precision_res = self.precision(y, val_logits)
        return precision_res, recall_res, f1_res, acc_res, loss_value

    def training(self):
        '''
        This function is the called to start the training of the model.
        '''
        start = time.time()

        for epoch in range(self.EPOCHS):
            if self.stop_training:
                break

            loss_dis, f1_dis, recall_dis, precision_dis, acc_dis = [], [], [], [], []
            for (x, y) in self.train_dataset:
                precision_res, recall_res, f1_res, acc_res, loss_value = self.train_step(x, y)
                acc_dis.append(acc_res)
                f1_dis.append(f1_res)
                recall_dis.append(recall_res)
                precision_dis.append(precision_res)
                loss_dis.append(loss_value)

            acc = np.mean(acc_dis)
            f1 = np.mean(f1_dis)
            recall = np.mean(recall_dis)
            precision = np.mean(precision_dis)
            loss_dis = np.mean(loss_dis)
            if self.bayesian == False and epoch %20==0:
                print(
                    f"Epoch: {epoch} - Train - loss:{loss_dis:.6f}, acc: {acc:.6f}, f1: {f1:.6f}, recall: {recall:.6f}, precision: {precision:.6f}")

            loss_dis, f1_dis, recall_dis, precision_dis, acc_dis = [], [], [], [], []
            for (x, y) in self.val_dataset:
                precision_res, recall_res, f1_res, acc_res, loss_value = self.test_step(x, y)
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

            if self.bayesian == False:
                print(
                    f"Epoch: {epoch} - Val - loss:{loss_dis:.6f}, acc: {acc:.6f}, f1: {f1:.6f}, recall: {recall:.6f}, precision: {precision:.6f}\n")

            if loss_dis < self.best_score:
                self.best_score = loss_dis
                self.counter = 0
            else:
                self.counter += 1
            if self.counter >= self.PATIENCE:
                self.stop_training = True
                break

        end = time.time()
        if self.bayesian == False:
            print("Time taken: ", end - start)
