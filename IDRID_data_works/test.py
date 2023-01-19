import logging
import tensorflow as tf
import time
from tensorflow import keras
import tensorflow_addons as tfa


class Testing_routine:

    def __init__(self, model, test_dataset, threshold):
        self.model = model
        self.threshold = threshold
        self.test_dataset = test_dataset
        self.loss_metric = keras.metrics.BinaryCrossentropy()
        self.acc_metric = keras.metrics.BinaryAccuracy(name='accuracy')
        self.recall_metric = keras.metrics.Recall(name='recall')
        self.precision_metric = keras.metrics.Precision(name='precision')
        self.f1_metric = tfa.metrics.F1Score(num_classes=1, average='macro', threshold=self.threshold)

    @tf.function
    def test_step_2(self, x, y):
        self.test_logits = self.model(x, training=False)
        self.acc_metric.update_state(y, self.test_logits)
        self.loss_metric.update_state(y, self.test_logits)
        self.recall_metric.update_state(y, self.test_logits)
        self.precision_metric.update_state(y, self.test_logits)

    def testing(self):
        start_time = time.time()
        # Iterate over the batches of the dataset.
        for step, (x_batch_test, y_batch_test) in enumerate(self.test_dataset):
            self.loss_value = self.test_step_2(x_batch_test, y_batch_test)
            # Display metrics at the end of each epoch.
            test_acc = self.acc_metric.result()
            test_recall = self.recall_metric.result()
            test_precision = self.precision_metric.result()
            test_loss = self.loss_metric.result()

            # Reset testing metrics at the end of each epoch
            self.acc_metric.reset_states()
            self.recall_metric.reset_states()
            self.precision_metric.reset_states()
            self.loss_metric.reset_states()

            logging.info(
                f"Test - loss: {test_loss:.2f}, acc: {test_acc:.2f}, recall: {test_recall:.2f}, precision: {test_precision:.2f}, Time: {time.time() - start_time:.2f}")
