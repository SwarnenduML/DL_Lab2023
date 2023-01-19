import io
import os
import shutil
import gin
import tensorflow as tf
import enum
import logging
import pathlib
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow_addons as tfa
import time


from models.architectures import ModelArchitecture

@gin.configurable
class Trainer(object):
    '''Train the given model on the given dataset.'''

    def __init__(self, training_dataset, validation_dataset,model_save_folder, model, model_name, threshold,
                 epochs, learning_rate, early_stopping, patience, wait):         # <- configs

        self.model = model_name
        self.model_name = model
        self.model_save_folder = pathlib.Path(model_save_folder)
        if (not self.model_save_folder.exists()) or (not self.model_save_folder.is_dir() ):
            raise ValueError(f"Received invalid directory for model saving: '{self.model_save_folder}'.")

        if epochs>0 and epochs <1000:
            self.epochs = epochs
        else:
            raise ValueError(f"Number of epochs : '{epochs}' should be in the range of 10 and 100")

        if learning_rate<1e-1 and learning_rate >1e-10:
            self.learning_rate = learning_rate
        else:
            raise ValueError(f"Learning rate : '{learning_rate}' should be in the range of 1e-1 and 1e-10")

        if threshold>0 and threshold <1:
            self.threshold = threshold
        else:
            raise ValueError(f"threshold : '{threshold}' should be in the range of 0 and 1")

        if isinstance(early_stopping,bool):
            self.early_stopping = early_stopping
        else:
            raise ValueError(f"Early stopping should be BOOLEAN. Given value : {early_stopping}")

        if isinstance(patience,int) and patience>0:
            self.patience = patience
        else:
            raise ValueError(f"Patience should be int and more than 0. Given value: {patience}")

        if isinstance(wait, int) and wait >=0:
            self.wait = wait
        else:
            raise ValueError(f"Wait should be int and more than 0. Given value: {wait}")

        self.optimizer = keras.optimizers.Adam(learning_rate= self.learning_rate)
        self.loss_fn = keras.losses.BinaryCrossentropy()
        self.loss_metric = keras.metrics.BinaryCrossentropy()
        self.acc_metric = keras.metrics.BinaryAccuracy(name = 'accuracy')
        self.recall_metric = keras.metrics.Recall(name = 'recall')
        self.precision_metric = keras.metrics.Precision(name = 'precision')
        self.f1_metric = tfa.metrics.F1Score(num_classes= 1, average='macro', threshold=self.threshold)
        self.train_dataset = training_dataset
        self.val_dataset = validation_dataset 



    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            self.logits = self.model(x, training=True)
            self.loss_value = self.loss_fn(y, self.logits)
        grads = tape.gradient(self.loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        self.acc_metric.update_state(y, self.logits)
        self.loss_metric.update_state(y, self.logits)
        self.recall_metric.update_state(y, self.logits)
        self.precision_metric.update_state(y, self.logits)
        return self.loss_value
      
    @tf.function
    def test_step(self, x, y):
        self.val_logits = self.model(x, training=False)
        self.acc_metric.update_state(y, self.val_logits)
        self.loss_metric.update_state(y, self.val_logits)
        self.recall_metric.update_state(y, self.val_logits)
        self.precision_metric.update_state(y, self.val_logits)

    def training(self):
        logging.info("Training starts")
        #logging.info(self.model_name.summary())

#        self.patience = 3
 #       self.wait = 0
        self.best = 0

        for epoch in range(self.epochs):
            start_time = time.time()

            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train) in enumerate(self.train_dataset):
                #logging.info(y_batch_train)
                self.loss_value = self.train_step(x_batch_train, y_batch_train)

            # Display metrics at the end of each epoch.
            train_acc = self.acc_metric.result()
            train_recall = self.recall_metric.result()
            train_precision = self.precision_metric.result()
            train_loss = self.loss_metric.result()
            # Reset training metrics at the end of each epoch
            self.acc_metric.reset_states()
            self.recall_metric.reset_states()
            self.precision_metric.reset_states()
            self.loss_metric.reset_state()

            # Run a validation loop at the end of each epoch.
            for x_batch_val, y_batch_val in self.val_dataset:
                self.test_step(x_batch_val, y_batch_val)

            val_acc = self.acc_metric.result()
            val_recall = self.recall_metric.result()
            val_precision = self.precision_metric.result()
            self.val_loss = self.loss_metric.result()

            self.acc_metric.reset_states()
            self.recall_metric.reset_states()
            self.precision_metric.reset_states()
            self.loss_metric.reset_state()

            logging.info(
                f"\nStart of epoch: {epoch+1} - Train - loss: {train_loss:.2f}, acc: {train_acc:.2f}, recall: {train_recall:.2f}, precision: {train_precision:.2f}, Val - loss: {self.val_loss:.2f}, acc: {val_acc:.2f}, recall: {val_recall:.2f}, precision: {val_precision:.2f}, Time: {time.time() - start_time:.2f}")

            if self.early_stopping:
                # The early stopping strategy: stop the training if `val_loss` does not
                # decrease over a certain number of epochs.
                self.wait += 1
                if self.val_loss > self.best:
                    self.best = self.val_loss
                    self.wait = 0
                if self.wait >= self.patience:
                    break

        self.model.save(str(self.model_save_folder)+"/"+str(self.model_name)+"/model.h5")
        return self.model, self.threshold
