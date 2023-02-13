import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from models.architectures import ModelArchitecture
import datetime
import numpy as np
import time
import tensorflow_addons as tfa
from tensorflow import keras
import matplotlib.pyplot as plt
import pathlib
import logging
import tensorflow as tf
import gin


@gin.configurable
class Trainer(object):
    """
    Trains the model on the given dataset.

    Args:
        counter: int, stores the current training iteration.
        best_score: float, best score achieved so far during training.
        stop_training: bool, flag to indicate if training should stop.
        model: string, name of the model to be trained.
        model_name: Model object, actual model to be trained.
        train_val_log_dir_path: string, directory path to store training and validation logs.
        hyperparameter_routine: bool, flag indicating if hyperparameter tuning is being performed.
        model_save_folder: pathlib.Path, directory path to save the model.
        EPOCHS: int, number of training iterations.
        LEARNING_RATE: float, learning rate used for training.
        EARLY_STOPPING: bool, flag indicating if early stopping is being used.
        PATIENCE: int, number of iterations to wait before stopping training if no improvement is observed.
        optimizer: optimizer object, optimizer used for training.
        loss_fn: loss function, used for computing the loss during training.
        loss_metric: metric object, used to track the loss metric during training.
        acc_metric: metric object, used to track the accuracy metric during training.
        recall_metric: metric object, used to track the recall metric during training.
        precision_metric: metric object, used to track the precision metric during training.
        train_dataset: Dataset, training dataset.
        val_dataset: Dataset, validation dataset.
    """
    def __init__(self, training_dataset, validation_dataset, model_save_folder, model, model_name,
                 SAVE_MODEL, EPOCHS, LEARNING_RATE, EARLY_STOPPING, PATIENCE, optimizer, hyperparameter_routine, train_val_log_dir_path):         # <- configs
        self.counter = 0
        self.best_score = np.inf
        self.stop_training = False
        self.SAVE_MODEL = SAVE_MODEL
        self.model = model_name
        self.model_name = model
        self.train_val_log_dir_path = train_val_log_dir_path
        self.hyperparameter_routine = hyperparameter_routine
        self.model_save_folder = pathlib.Path(model_save_folder)
        if (not self.model_save_folder.exists()) or (not self.model_save_folder.is_dir()):
            raise ValueError(
                f"Received invalid directory for model saving: '{self.model_save_folder}'.")

        if EPOCHS > 0 and EPOCHS < 1000:
            self.EPOCHS = EPOCHS
        else:
            raise ValueError(
                f"Number of EPOCHS : '{EPOCHS}' should be in the range of 10 and 100")

        if LEARNING_RATE < 1e-1 and LEARNING_RATE > 1e-10:
            self.LEARNING_RATE = LEARNING_RATE
        else:
            raise ValueError(
                f"Learning rate : '{LEARNING_RATE}' should be in the range of 1e-1 and 1e-10")


        if isinstance(EARLY_STOPPING, bool):
            self.EARLY_STOPPING = EARLY_STOPPING
        else:
            raise ValueError(
                f"Early stopping should be BOOLEAN. Given value : {EARLY_STOPPING}")

        if isinstance(PATIENCE, int) and PATIENCE > 0:
            self.PATIENCE = PATIENCE
        else:
            raise ValueError(
                f"PATIENCE should be int and more than 0. Given value: {PATIENCE}")

        if self.hyperparameter_routine:
            self.optimizer = optimizer
        else:
            self.optimizer = keras.optimizers.Adam(
                learning_rate=self.LEARNING_RATE)
        self.loss_fn = keras.losses.BinaryCrossentropy()
        self.loss_metric = keras.metrics.BinaryCrossentropy()
        self.acc_metric = keras.metrics.BinaryAccuracy(name='accuracy')
        self.recall_metric = keras.metrics.Recall(name='recall')
        self.precision_metric = keras.metrics.Precision(name='precision')
        self.train_dataset = training_dataset
        self.val_dataset = validation_dataset

    @tf.function
    def f1_score(self, precision, recall):
        """
        Compute the F1-Score given the precision and recall values

        Args:
            precision: The precision value
            recall: The recall value

        Returns:
            The computed F1-Score
        """
        precision = precision
        recall = recall
        f1_score = 2*((precision*recall)/(precision +
                      recall+tf.keras.backend.epsilon()))
        return f1_score

    @tf.function
    def train_step(self, x, y):
        """
        Compute the accuracy, recall, precision and loss for a single train step

        Args:
            x: Input features for the train step
            y: Labels for the train step
        """
        with tf.GradientTape() as tape:
            self.logits = self.model(x, training=True)
            self.loss_value = self.loss_fn(y, self.logits)
        grads = tape.gradient(self.loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_weights))
        self.acc_metric.update_state(y, self.logits)
        self.loss_metric.update_state(y, self.logits)
        self.recall_metric.update_state(y, self.logits)
        self.precision_metric.update_state(y, self.logits)
        return self.loss_value

    @tf.function
    def test_step(self, x, y):
        """
        Compute the accuracy, recall, precision and loss for a single test step

        Args:
            x: Input features for the test step
            y: Labels for the test step
        """
        self.val_logits = self.model(x, training=False)
        self.acc_metric.update_state(y, self.val_logits)
        self.loss_metric.update_state(y, self.val_logits)
        self.recall_metric.update_state(y, self.val_logits)
        self.precision_metric.update_state(y, self.val_logits)

    def training(self):
        """
        Perform the training routine and log the results to Tensorboard 
        """
        logging.info("Training starts")

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = self.train_val_log_dir_path + \
            f'{self.model_name}' + '/' + current_time + '/train'
        val_log_dir = self.train_val_log_dir_path + \
            f'{self.model_name}' + '/' + current_time + '/val'

        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        val_summary_writer = tf.summary.create_file_writer(val_log_dir)

        for epoch in range(self.EPOCHS):
            start_time = time.time()
            if self.stop_training:
                break

            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train) in enumerate(self.train_dataset):
                # logging.info(y_batch_train)
                self.loss_value = self.train_step(x_batch_train, y_batch_train)

            # Display metrics at the end of each epoch.
            train_acc = self.acc_metric.result()
            train_recall = self.recall_metric.result()
            train_precision = self.precision_metric.result()
            train_loss = self.loss_metric.result()
            train_f1 = self.f1_score(train_precision, train_recall)

            # Write the hyperparameters to TensorBoard
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss, step=epoch)
                tf.summary.scalar('accuracy', train_acc, step=epoch)
                tf.summary.scalar('precision', train_precision, step=epoch)
                tf.summary.scalar('f1', train_f1, step=epoch)
                tf.summary.scalar('recall', train_recall, step=epoch)

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
            val_loss = self.loss_metric.result()
            val_f1 = self.f1_score(val_precision, val_recall)

            # Write the hyperparameters to TensorBoard
            with val_summary_writer.as_default():
                tf.summary.scalar('loss', val_loss, step=epoch)
                tf.summary.scalar('accuracy', val_acc, step=epoch)
                tf.summary.scalar('f1', val_f1, step=epoch)
                tf.summary.scalar('precision', val_precision, step=epoch)
                tf.summary.scalar('recall', val_recall, step=epoch)

            self.acc_metric.reset_states()
            self.recall_metric.reset_states()
            self.precision_metric.reset_states()
            self.loss_metric.reset_state()

            if epoch % 20==0:
                logging.info(f"\nStart of epoch: {epoch} - Train - loss: {train_loss:.6f}, acc: {train_acc:.6f}, recall: {train_recall:.6f}, precision: {train_precision:.6f}, f1: {train_f1:.6f}, Val - loss: {val_loss:.6f}, acc: {val_acc:.6f}, recall: {val_recall:.6f}, precision: {val_precision:.6f}, f1: {val_f1:.6f}, Time: {time.time() - start_time:.6f}")

            if val_loss < self.best_score:
                self.best_score = val_loss
                self.counter = 0
            else:
                self.counter += 1
            if self.counter >= self.PATIENCE:
                self.stop_training = True
                break

        if self.SAVE_MODEL:
            self.model.save(str(self.model_save_folder)+"/"+str(self.model_name))
        return self.model
