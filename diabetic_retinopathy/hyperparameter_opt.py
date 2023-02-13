import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from input_pipeline.dataset_loader import DatasetLoader
from models.architectures import ModelArchitecture
from tensorboard.plugins.hparams import api as hp
from test import TestingRoutine
from train import Trainer
from tensorflow import keras
import tensorflow as tf
import gin
import logging



@gin.configurable
class HyperparameterOptimization:
    """Class for performing hyperparameter optimization on a model

    Args:
        train_ds (tf.data.Dataset): Dataset for training.
        val_ds (tf.data.Dataset): Dataset for validation.
        test_ds (tf.data.Dataset): Dataset for testing.
        model (str): Model architecture to be used.
        model_name (str): Model name to be saved.
        model_save_folder (str): Folder to save the model.
        test_log_dir_path (str): Path to the test log directory.
    """

    def __init__(self, train_ds, val_ds, test_ds, model, model_name, model_save_folder, test_log_dir_path):
        self.model = model
        self.model_name = model_name
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.model_save_folder = model_save_folder
        self.test_log_dir_path = test_log_dir_path
        self.HP_NUM_UNITS = hp.HParam('num units', hp.Discrete([16, 32]))
        self.HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.1, 0.3]))
        self.HP_LR = hp.HParam('learning_rate', hp.Discrete([1e-1, 1e-3]))

    def train_model_epoch(self, hparams):
        """Train a model for one epoch using hyperparameters.

        Args:
            hparams (dict): Hyperparameters to be used for training.
        """
        # Extract the hyperparameters
        drop_rate = hparams[self.HP_DROPOUT]
        learning_rate = hparams[self.HP_LR]
        optimizer = keras.optimizers.Adam(lr=learning_rate)

        # Select the appropriate model architecture
        if self.model.upper() == 'NONE' or self.model.upper() == 'BASELINE':
            logging.info(f"Model selected '{self.model.upper()}")
            units = hparams[self.HP_NUM_UNITS]
            model = ModelArchitecture(
                baseline_units=units, baseline_dropout_rate=drop_rate).baseline_CNN_model()
        elif self.model.upper() == 'INCEPTION_V3':
            logging.info(f"Model selected '{self.model.upper()}'")
            model = ModelArchitecture(
                inception_dropout_rate=drop_rate).inception_v3()
        elif self.model.upper() == 'RESNET_50':
            logging.info(f"Model selected'{self.model.upper()}'")
            model = ModelArchitecture(
                resnet_dropout_rate=drop_rate).resnet_50()
        elif self.model.upper() == 'EFFICIENT':
            logging.info(f"Model selected '{self.model.upper()}")
            model = ModelArchitecture(
                efficient_dropout_rate=drop_rate).efficient_v3b3()
        else:
            raise ValueError(f"'{self.model}' not accepted")

        # Construct the run directory based on the selected model
        if self.model.upper() == 'NONE' or self.model.upper() == 'BASELINE':
            run_dir = (self.test_log_dir_path + f'{self.model}' + '/' + str(
                units) + 'units_' + str(drop_rate)+'dropout_'+str(learning_rate)+'learning_rate')
        else:
            run_dir = (self.test_log_dir_path + f'{self.model}' + '/' + str(
                drop_rate)+'dropout_'+str(learning_rate)+'learning_rate')

        # Train the model using the specified optimizer and hyperparameters
        trainer = Trainer(self.train_ds, self.val_ds, self.model_save_folder, self.model_name,
                          model,optimizer=optimizer, hyperparameter_routine=True)
        model = trainer.training()
        saved_model = tf.keras.models.load_model(
            self.model_save_folder+'/' + self.model_name, compile=False)
        acc, f1 = TestingRoutine(saved_model, self.test_ds,
                        run_dir, True, hparams).testing()
        logging.info(acc)
        logging.info(f1)

    def run_hyperparameter_optimization(self):
        """
        This method performs hyperparameter optimization by training the model with different combinations of hyperparameters.

        The optimization is performed in two cases, when the `model` attribute is set to "None" or "Baseline" and when it's set to any other value.
        In the first case, the hyperparameters being optimized are `learning rate`, `number of units` and `dropout rate`.
        In the second case, the hyperparameters being optimized are `learning rate` and `dropout rate`.
        """

        # If the model attribute is set to "None" or "Baseline"
        if self.model.upper() == 'NONE' or self.model.upper() == 'BASELINE':
            for lr in self.HP_LR.domain.values:
                for units in self.HP_NUM_UNITS.domain.values:
                    for rate in self.HP_DROPOUT.domain.values:
                        logging.info(lr)
                        logging.info(units)
                        logging.info(rate)
                        hparams = {
                            self.HP_LR: lr,
                            self.HP_NUM_UNITS: units,
                            self.HP_DROPOUT: rate,
                        }
                        self.train_model_epoch(hparams)
        # If the model attribute is set to any other value
        else:
            for lr in self.HP_LR.domain.values:
                for rate in self.HP_DROPOUT.domain.values:
                    logging.info(lr)
                    logging.info(rate)
                    hparams = {
                        self.HP_LR: lr,
                        self.HP_DROPOUT: rate,
                    }
                    self.train_model_epoch(hparams)
