import logging
import gin
import tensorflow as tf
from train import Trainer
from test import Testing_routine
from utils import utils_params, utils_misc



@gin.configurable
def hyperparameter_tuning(model_name, saved_model, tuned_model_path, train_ds, valid_ds,test_ds, threshold,
                          number_of_layers_to_freeze):  # <- configs
    """Train and evaluate the given models with varying hyperparameter."""
    model_name = model_name
    model = saved_model
    path_to_save = tuned_model_path
    number_freeze = number_of_layers_to_freeze[model_name]

    for layer in model.layers[:number_freeze]:
        layer.trainable = False
    for layer in model.layers[number_freeze:]:
        layer.trainable = True

    logging.info(f"Before tuning {model_name}")
    logging.info(f"Training results")
    Testing_routine(model,train_ds,threshold).testing()
    logging.info(f"Validation results")
    Testing_routine(model,valid_ds,threshold).testing()
    logging.info(f"Testing results")
    Testing_routine(model,test_ds,threshold).testing()

    logging.info(f"After tuning only freezing layers")
    trainer = Trainer(train_ds, valid_ds, tuned_model_path, model_name, model, threshold)
    model, threshold = trainer.training()
    Testing_routine(model, test_ds, threshold).testing()

