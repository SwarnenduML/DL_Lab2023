import os

import gin
import logging
import absl
import tensorflow as tf
from tensorflow.python.util.deprecation import _PRINT_DEPRECATION_WARNINGS
import pathlib
import shutil

from tune import hyperparameter_tuning
from train import Trainer
from input_pipeline.dataset_loader import DatasetLoader 
from utils import utils_params, utils_misc
from test import Testing_routine
from models.architectures import ModelArchitecture
from visualize import Visualization
#from evaluation.visualization import GradCAM

#FLAGS = absl.flags.FLAGS
#absl.flags.DEFINE_boolean(name='train', default=False,  help='Specify whether to train a model.')

@gin.configurable
def main(argv, train_models,model_save_folder, models, tuning, tuned_model_path, threshold,
         grad_cam, grad_cam_model):  # <- configs

    # Generate folder structures
    run_paths = utils_params.generate_run_directory()
    # Set loggers
    utils_misc.set_loggers(paths=run_paths, logging_level=logging.INFO)
    # Save gin config
    utils_params.save_config(run_paths['path_gin'], gin.config_str() )

    # Create dataset(s), returns the names of the available datasets.
    #train_ds, test_ds, valid_ds = DatasetLoader().create_datasets()
    #for i,j in train_ds.take(1):
    #  logging.info(j)

    # Training
    accepted_model = ['baseline','inception_v3', 'resnet_50','efficient']
    if train_models:
        if models.upper() == 'ALL':
            for model in accepted_model:
                train_ds, test_ds, valid_ds = DatasetLoader(selected_model=model).create_datasets()
                if model.upper() =='NONE' or model.upper() =='BASELINE':
                    model_name = model
                    model = ModelArchitecture.baseline_CNN_model()
                elif model.upper() =='INCEPTION_V3':
                    logging.info(f"Training models '{model.upper()}'")
                    model_name = model
                    model = ModelArchitecture.inception_v3()
                elif model.upper() =='RESNET_50':
                    logging.info(f"Training models '{model.upper()}'")
                    model_name = model
                    model = ModelArchitecture.resnet_50()
                elif model.upper() =='EFFICIENT':
                    logging.info(f"Training models '{model.upper()}")
                    model_name = model
                    model = ModelArchitecture.efficient_v3b3()
                else:
                    raise ValueError(f"'{model}' not accepted")

                logging.info(f"Training starts")
                logging.info(f"Model to train: {model_name}")
                trainer = Trainer(train_ds, valid_ds, model_save_folder, model_name, model, threshold)
                model, threshold = trainer.training()
                Testing_routine(model, test_ds, threshold).testing()
        else:
            train_ds, test_ds, valid_ds = DatasetLoader(selected_model=models).create_datasets()
            if models.upper() == 'NONE' or models.upper() == 'BASELINE':
                logging.info(f"Training models 'BASELINE'")
                model = ModelArchitecture.baseline_CNN_model()
                model_name = models
            elif models.upper() == 'INCEPTION_V3':
                logging.info(f"Training models '{models.upper()}'")
                model = ModelArchitecture.inception_v3()
                model_name = models
            elif models.upper() == 'RESNET_50':
                logging.info(f"Training models '{models.upper()}'")
                model = ModelArchitecture.resnet_50()
                model_name = models
            elif models.upper() =='EFFICIENT':
                logging.info(f"Training models '{models.upper()}")
                model = ModelArchitecture.efficient_v3b3()
                model_name = models
            else:
                raise ValueError(f"'{models}' not accepted")

            logging.info(f"Training starts")
            logging.info(f"Model to train: {model_name}")
            trainer = Trainer(train_ds, valid_ds, model_save_folder, model_name, model,threshold)
            model, threshold = trainer.training()
            Testing_routine(model, test_ds, threshold).testing()
    if tuning:
        if models.upper() == 'ALL':
            for model_name in accepted_model:
                train_ds, test_ds, valid_ds = DatasetLoader(selected_model=model_name).create_datasets()
                if model_name.upper() == 'NONE' or model_name.upper() == 'BASELINE':
                    if (not os.path.exists(model_save_folder + "/" + model_name)) or len(os.listdir(model_save_folder + "/" + model_name)) < 1:
                        model = ModelArchitecture.baseline_CNN_model()
                        model_name = model_name
                        logging.info(f"Training starts")
                        logging.info(f"Model to train: {model_name}")
                        trainer = Trainer(train_ds, valid_ds, model_save_folder, model_name, model, threshold)
                        model, threshold = trainer.training()
                        Testing_routine(model, test_ds, threshold).testing()
                    saved_model = tf.keras.models.load_model(model_save_folder+'/'+model_name+'/model.h5')
                    hyperparameter_tuning(model_name, saved_model, tuned_model_path, train_ds, valid_ds, test_ds, threshold)
                elif model_name.upper() == 'INCEPTION_V3':
                    if (not os.path.exists(model_save_folder + "/" + model_name)) or len(os.listdir(model_save_folder + "/" + model_name)) < 1:
                        model = ModelArchitecture.inception_v3()
                        model_name = model_name
                        logging.info(f"Training starts")
                        logging.info(f"Model to train: {model_name}")
                        trainer = Trainer(train_ds, valid_ds, model_save_folder, model_name, model, threshold)
                        model, threshold = trainer.training()
                        Testing_routine(model, test_ds, threshold).testing()
                    saved_model = tf.keras.models.load_model(model_save_folder + '/' + model_name + '/model.h5')
                    hyperparameter_tuning(model_name, saved_model, tuned_model_path, train_ds, valid_ds, test_ds, threshold)
                elif model_name.upper() == 'RESNET_50':
                    if (not os.path.exists(model_save_folder + "/" + model_name)) or len(os.listdir(model_save_folder + "/" + model_name)) < 1:
                        model = ModelArchitecture.resnet_50()
                        model_name = model_name
                        logging.info(f"Training starts")
                        logging.info(f"Model to train: {model_name}")
                        trainer = Trainer(train_ds, valid_ds, model_save_folder, model_name, model, threshold)
                        model, threshold = trainer.training()
                        Testing_routine(model, test_ds, threshold).testing()
                    saved_model = tf.keras.models.load_model(model_save_folder + '/' + model_name + '/model.h5')
                    hyperparameter_tuning(model_name, saved_model, tuned_model_path, train_ds, valid_ds, test_ds, threshold)
                elif model_name.upper() == 'EFFICIENT':
                    if (not os.path.exists(model_save_folder + "/" + model_name)) or len(os.listdir(model_save_folder + "/" + model_name)) < 1:
                        model = ModelArchitecture.efficient_v3b3()
                        model_name = model_name
                        logging.info(f"Training starts")
                        logging.info(f"Model to train: {model_name}")
                        trainer = Trainer(train_ds, valid_ds, model_save_folder, model_name, model, threshold)
                        model, threshold = trainer.training()
                        Testing_routine(model, test_ds, threshold).testing()
                    saved_model = tf.keras.models.load_model(model_save_folder + '/' + model_name + '/model.h5')
                    hyperparameter_tuning(model_name, saved_model, tuned_model_path, train_ds, valid_ds, test_ds, threshold)
                else:
                    raise ValueError(f"'{model_name}' not accepted")
        else:
            model_name = models
            train_ds, test_ds, valid_ds = DatasetLoader(selected_model=model_name).create_datasets()
            if model_name.upper() == 'NONE' or model_name.upper() == 'BASELINE':
                if (not os.path.exists(model_save_folder + "/" + model_name)) or len(os.listdir(model_save_folder + "/" + model_name)) < 1:
                    model = ModelArchitecture.baseline_CNN_model()
                    model_name = model_name
                    logging.info(f"Training starts")
                    logging.info(f"Model to train: {model_name}")
                    trainer = Trainer(train_ds, valid_ds, model_save_folder, model_name, model, threshold)
                    model, threshold = trainer.training()
                    Testing_routine(model, test_ds, threshold).testing()
                saved_model = tf.keras.models.load_model(model_save_folder+'/'+model_name+'/model.h5')
                hyperparameter_tuning(model_name, saved_model, tuned_model_path, train_ds, valid_ds, test_ds, threshold)
            elif model_name.upper() == 'INCEPTION_V3':
                if (not os.path.exists(model_save_folder + "/" + model_name)) or len(os.listdir(model_save_folder + "/" + model_name)) < 1:
                    model = ModelArchitecture.inception_v3()
                    model_name = model_name
                    logging.info(f"Training starts")
                    logging.info(f"Model to train: {model_name}")
                    trainer = Trainer(train_ds, valid_ds, model_save_folder, model_name, model, threshold)
                    model, threshold = trainer.training()
                    Testing_routine(model, test_ds, threshold).testing()
                saved_model = tf.keras.models.load_model(model_save_folder + '/' + model_name + '/model.h5')
                hyperparameter_tuning(model_name, saved_model, tuned_model_path, train_ds, valid_ds, test_ds, threshold)
            elif model_name.upper() == 'RESNET_50':
                if (not os.path.exists(model_save_folder + "/" + model_name)) or len(os.listdir(model_save_folder + "/" + model_name)) < 1:
                    model = ModelArchitecture.resnet_50()
                    model_name = model_name
                    logging.info(f"Training starts")
                    logging.info(f"Model to train: {model_name}")
                    trainer = Trainer(train_ds, valid_ds, model_save_folder, model_name, model, threshold)
                    model, threshold = trainer.training()
                    Testing_routine(model, test_ds, threshold).testing()
                saved_model = tf.keras.models.load_model(model_save_folder + '/' + model_name + '/model.h5')
                hyperparameter_tuning(model_name, saved_model, tuned_model_path, train_ds, valid_ds, test_ds, threshold)
            elif model_name.upper() == 'EFFICIENT':
                if (not os.path.exists(model_save_folder + "/" + model_name)) or len(os.listdir(model_save_folder + "/" + model_name)) < 1:
                    model = ModelArchitecture.efficient_v3b3()
                    model_name = model_name
                    logging.info(f"Training starts")
                    logging.info(f"Model to train: {model_name}")
                    trainer = Trainer(train_ds, valid_ds, model_save_folder, model_name, model, threshold)
                    model, threshold = trainer.training()
                    Testing_routine(model, test_ds, threshold).testing()
                saved_model = tf.keras.models.load_model(model_save_folder + '/' + model_name + '/model.h5')
                hyperparameter_tuning(model_name, saved_model, tuned_model_path, train_ds, valid_ds, test_ds, threshold)
            else:
                raise ValueError(f"'{model_name}' not accepted")
    if grad_cam:
        if grad_cam_model.lower()=='all':
            for saved_model in accepted_model:
                Visualization(saved_model).grad_cam()
        else:
            saved_model = grad_cam_model
            Visualization(saved_model).grad_cam()






if __name__ == '__main__':
    gin_config_path = pathlib.Path(__file__).parent / 'configs' / 'config.gin'
    gin.parse_config_files_and_bindings([gin_config_path], [])
    absl.app.run(main)
