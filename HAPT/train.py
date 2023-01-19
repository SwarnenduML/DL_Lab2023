import gin
import tensorflow as tf
import numpy as np
from tensorflow import keras
import time


@gin.configurable
class Training_routine(object):
    '''Train the given model on the given dataset.'''

    def __init__(self, model_name,  train_dataset, val_dataset, bayesian_opt, epochs, patience, learning_rate):
        self.model_name = model_name
        self.epochs = epochs
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.patience = patience
        self.counter = 0
        self.best_score = np.inf
        self.stop_training = False
        self.bayesian = bayesian_opt
        self.learning_rate = learning_rate
        self.optimizer = keras.optimizers.Adam(self.learning_rate)

    def masked_sparse_categorical_crossentropy(self, y, logits):
        # Remove the extra dimension from y
        y = tf.squeeze(y, axis=-1)
        mask = tf.not_equal(y, 0)
        loss = keras.losses.sparse_categorical_crossentropy(y, logits)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss = loss * mask
        return tf.reduce_mean(loss)

    def masked_sparse_categorical_accuracy(self, y_true, y_pred):
        y_true = tf.squeeze(y_true, axis=-1)
        mask = tf.not_equal(y_true, 0)
        y_pred = tf.argmax(y_pred, axis=-1)
        acc = tf.equal(tf.cast(y_true, y_pred.dtype), y_pred)
        acc = tf.cast(acc, tf.float32)
        acc = acc * tf.cast(mask, acc.dtype)
        return tf.math.count_nonzero(acc) / (tf.math.count_nonzero(mask))

    def recall(self,y_true, y_pred):
        y_true = tf.cast(tf.not_equal(y_true, 0), y_pred.dtype)
        true_positives = tf.math.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
        all_positives = tf.math.reduce_sum(tf.round(tf.clip_by_value(y_true, 0, 1)))
        recall = true_positives / (all_positives + tf.keras.backend.epsilon())
        return recall

    def precision(self, y_true, y_pred):
        y_true = tf.cast(tf.not_equal(y_true, 0), y_pred.dtype)
        true_positives = tf.math.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
        predicted_positives = tf.math.reduce_sum(tf.round(tf.clip_by_value(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
        return precision

    def f1_score(self, y_true, y_pred):
        precision1 = self.precision(y_true, y_pred)
        recall1 = self.recall(y_true, y_pred)
        return 2 * ((precision1 * recall1) / (precision1 + recall1 + tf.keras.backend.epsilon()))

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            logits = self.model_name(x, training=True)
            loss_value = self.masked_sparse_categorical_crossentropy(y, logits)
        grads = tape.gradient(loss_value, self.model_name.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model_name.trainable_weights))
        acc_res = self.masked_sparse_categorical_accuracy(y, logits)
        precision_res = self.precision(y, logits)
        recall_res = self.recall(y, logits)
        f1_res = self.f1_score(y, logits)
        return precision_res, recall_res, f1_res, acc_res, loss_value

    @tf.function
    def test_step(self, x, y):
        val_logits = self.model_name(x, training=False)
        loss_value = self.masked_sparse_categorical_crossentropy(y, val_logits)
        acc_res = self.masked_sparse_categorical_accuracy(y, val_logits)
        f1_res = self.f1_score(y, val_logits)
        recall_res = self.recall(y, val_logits)
        precision_res = self.precision(y, val_logits)
        return precision_res, recall_res, f1_res, acc_res, loss_value

    def training(self):
        start = time.time()

        for epoch in range(self.epochs):
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
            #if self.bayesian == False:
            print(
                    f"Epoch: {epoch} - Train - loss:{loss_dis:.2f}, acc: {acc:.2f}, f1: {f1:.2f}, recall: {recall:.2f}, precision: {precision:.2f}")

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
                    f"Epoch: {epoch} - Val - loss:{loss_dis:.2f}, acc: {acc:.2f}, f1: {f1:.2f}, recall: {recall:.2f}, precision: {precision:.2f}\n")

            if loss_dis < self.best_score:
                self.best_score = loss_dis
                self.counter = 0
            else:
                self.counter += 1
            if self.counter >= self.patience:
                self.stop_training = True
                break

        end = time.time()
        if self.bayesian == False:
            print("Time taken: ", end - start)
