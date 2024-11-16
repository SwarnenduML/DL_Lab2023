import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from test import TestingRoutine
from train import Trainer
import tensorflow as tf
import gin
import logging


@gin.configurable
def fine_tuning_fn(model_name, saved_model, fine_tuned_model_path, train_ds, valid_ds, test_ds,
                   number_of_layers_to_freeze): 
    """
    Train and evaluate the given models with varying hyperparameters.

    Args:
        model_name (str): The name of the model being fine-tuned.
        saved_model (tf.keras.Model): The pre-trained model to fine-tune.
        fine_tuned_model_path (str): The path to save the fine-tuned model.
        train_ds (tf.data.Dataset): The training dataset.
        valid_ds (tf.data.Dataset): The validation dataset.
        test_ds (tf.data.Dataset): The test dataset.
        number_of_layers_to_freeze (dict): A dictionary mapping model names to the number of layers to freeze.

    Returns:
        None
    """
    # Get the number of layers to freeze for the current model
    number_freeze = number_of_layers_to_freeze[model_name.lower()]

    # Freeze the first "number_freeze" layers of the model
    for layer in saved_model.layers[:number_freeze]:
        layer.trainable = False
    for layer in saved_model.layers[number_freeze:]:
        layer.trainable = True

    # Test the model before fine-tuning
    logging.info(f"Before fine tuning {model_name}")
    logging.info(f"Testing results")
    TestingRoutine(saved_model, test_ds).testing()

    # Fine-tune the model
    logging.info(f"Starting fine tuning")
    trainer = Trainer(train_ds, valid_ds, fine_tuned_model_path,
                      model_name, saved_model)
    model = trainer.training()

    # Test the fine-tuned model
    TestingRoutine(saved_model, test_ds).testing()
