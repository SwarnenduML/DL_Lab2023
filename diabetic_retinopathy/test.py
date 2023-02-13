import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import gin
from tensorboard.plugins.hparams import api as hp
import logging
import tensorflow as tf
import time
from tensorflow import keras
import tensorflow_addons as tfa



@gin.configurable
class TestingRoutine:
    """
    TestingRoutine Class is responsible for testing a given model on the test dataset and logging the results
    to Tensorboard.

    Args:
        model: A compiled tensorflow model
        test_dataset: A tensorflow dataset object containing the test data
        run_dir: The path to the directory where Tensorboard logs will be stored
        hyperparameter_routine: A boolean value indicating whether to log the hyperparameters to Tensorboard
        hparams: The hyperparameters to be logged to Tensorboard
    """

    def __init__(self, model, test_dataset, run_dir, hyperparameter_routine, hparams):
        self.model = model
        self.test_dataset = test_dataset
        self.run_dir = run_dir
        self.hyperparameter_routine = hyperparameter_routine
        self.hparams = hparams
        self.loss_metric = keras.metrics.BinaryCrossentropy()
        self.acc_metric = keras.metrics.BinaryAccuracy(name='accuracy')
        self.recall_metric = keras.metrics.Recall(name='recall')
        self.precision_metric = keras.metrics.Precision(name='precision')

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
    def test_step(self, x, y):
        """
        Compute the accuracy, recall, precision and loss for a single test step

        Args:
            x: Input features for the test step
            y: Labels for the test step
        """
        self.test_logits = self.model(x, training=False)
        self.acc_metric.update_state(y, self.test_logits)
        self.recall_metric.update_state(y, self.test_logits)
        self.precision_metric.update_state(y, self.test_logits)
        self.loss_metric.update_state(y, self.test_logits)

    def testing(self):
        """
        Perform the testing routine and log the results to Tensorboard (if hyperparameter_routine is True)
        """
        start_time = time.time()

        # Run a test loop at the end of each epoch.
        for x_batch_test, y_batch_test in self.test_dataset:
            self.test_step(x_batch_test, y_batch_test)

        # Compute the metrics after the testing loop
        test_acc = self.acc_metric.result()
        test_recall = self.recall_metric.result()
        test_precision = self.precision_metric.result()
        test_loss = self.loss_metric.result()
        test_f1 = self.f1_score(test_precision, test_recall)

        # Write the hyperparameters to TensorBoard if the hyperparameter routine is enabled
        if self.hyperparameter_routine:
            with tf.summary.create_file_writer(self.run_dir).as_default():
                hp.hparams(self.hparams)
                tf.summary.scalar('accuracy', test_acc, step=1)

        # Reset the metrics states for the next test loop
        self.acc_metric.reset_states()
        self.recall_metric.reset_states()
        self.precision_metric.reset_states()
        self.loss_metric.reset_states()

        # Log the test results
        logging.info(
            f"\nTest - loss: {test_loss:.6f}, acc: {test_acc:.6f}, recall: {test_recall:.6f}, precision: {test_precision:.6f}, f1: {test_f1:.6f}, Time: {time.time() - start_time:.6f}")
        return test_acc.numpy().item(), test_f1.numpy().item()