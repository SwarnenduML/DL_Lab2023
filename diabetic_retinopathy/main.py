import os

import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pathlib
import tensorflow as tf
import absl
import logging
import gin
from input_pipeline.dataset_loader import DatasetLoader
from hyperparameter_opt import HyperparameterOptimization
from visualize import Visualization
from models.architectures import ModelArchitecture
from test import TestingRoutine
from train import Trainer
from tune import fine_tuning_fn


@gin.configurable
def main(argv, TRAIN_MODELS, model_save_folder, models, FINE_TUNING, fine_tuned_model_path,
         GRAD_CAM, grad_cam_model, HYPERPARAMETER_OPT, runs):
    """
    Main function for running the training, testing, and hyperparameter optimization routines.

    Parameters:
	    argv (list): List of command line arguments
	    TRAIN_MODELS (bool): Flag to indicate whether to run the training routine
	    model_save_folder (str): Path to the directory to save the trained models
	    models (str): Model architecture to use for training, options: ['ALL', 'BASELINE', 'INCEPTION_V3', 'RESNET_50', 'EFFICIENT']
	    FINE_TUNING (bool): Flag to indicate whether to run the fine-tuning routine
	    fine_tuned_model_path (str): Path to the pre-trained model for fine-tuning
        GRAD_CAM (bool): Flag to indicate whether to run Grad-CAM routine
	    grad_cam_model (str): Model architecture to use for Grad-CAM, options: ['BASELINE', 'INCEPTION_V3', 'RESNET_50', 'EFFICIENT']
	    HYPERPARAMETER_OPT (bool): Flag to indicate whether to run the hyperparameter optimization routine
	    runs (int): Number of runs of the program

    Returns:
    	None
    """

    # Accepted models to loop from
    accepted_model = ['baseline', 'efficient', 'resnet_50', 'inception_v3']
    # Dataframe to store the test accouracy and test f1 score
    outputs_train = pd.DataFrame(columns=['model_name','accuracy','f1score'])
    outputs_fine = pd.DataFrame(columns=['model_name','accuracy','f1score'])
    # Starting hyperparameter tuning if enabled
    for i in range(runs):
        logging.info(i)
        logging.info(runs)
        if HYPERPARAMETER_OPT:
            logging.info("Hyperparameter tuning started in main loop")
            if models.upper() == 'ALL':
                for model in accepted_model:
                    train_ds, test_ds, valid_ds = DatasetLoader(
                        selected_model=model).create_datasets()
                    logging.info(f"Hyperparameter tuning starts")
                    model_name = model
                    HyperparameterOptimization(
                        train_ds, valid_ds, test_ds, model, model_name).run_hyperparameter_optimization()
            else:
                train_ds, test_ds, valid_ds = DatasetLoader(
                    selected_model=models).create_datasets()
                logging.info(f"Hyperparameter tuning starts")
                model_name = models
                HyperparameterOptimization(
                    train_ds, valid_ds, test_ds, models, model_name).run_hyperparameter_optimization()
            logging.info("Hyperparameter tuning ended")

        # Starting training if enabled
        if TRAIN_MODELS:
            logging.info("Training started in main loop")
            if models.upper() == 'ALL':
                for model in accepted_model:
                    train_ds, test_ds, valid_ds = DatasetLoader(
                        selected_model=model).create_datasets()
                    if model.upper() == 'NONE' or model.upper() == 'BASELINE':
                        model_name = model
                        model = ModelArchitecture().baseline_CNN_model()
                    elif model.upper() == 'INCEPTION_V3':
                        logging.info(f"Training models '{model.upper()}'")
                        model_name = model
                        model = ModelArchitecture().inception_v3()
                    elif model.upper() == 'RESNET_50':
                        logging.info(f"Training models '{model.upper()}'")
                        model_name = model
                        model = ModelArchitecture().resnet_50()
                    elif model.upper() == 'EFFICIENT':
                        logging.info(f"Training models '{model.upper()}")
                        model_name = model
                        model = ModelArchitecture().efficient_v3b3()
                    else:
                        raise ValueError(f"'{model}' not accepted")

                    logging.info(f"Training starts")
                    #logging.info(f"{model.summary()}")
                    if not os.path.exists(model_save_folder + '/' + model_name):
                        trainer = Trainer(
                            train_ds, valid_ds, model_save_folder, model_name, model).training()
                        saved_model = tf.keras.models.load_model(
                            model_save_folder+'/'+model_name, compile=False)
                        acc, f1 = TestingRoutine(saved_model, test_ds).testing()
                        outputs_train = outputs_train.append({'model_name':model_name,'accuracy':acc,'f1score':f1}, ignore_index = True)
                    else:
                        logging.info(
                            f"Model already trained, using the saved model for testing purpose")
                        saved_model = tf.keras.models.load_model(
                            model_save_folder + '/' + model_name, compile=False)
                        acc, f1 = TestingRoutine(saved_model, test_ds).testing()
                        outputs_train = outputs_train.append({'model_name':model_name,'accuracy':acc,'f1score':f1}, ignore_index = True)
            else:
                train_ds, test_ds, valid_ds = DatasetLoader(
                    selected_model=models).create_datasets()
                if models.upper() == 'NONE' or models.upper() == 'BASELINE':
                    logging.info(f"Training models 'BASELINE'")
                    model = ModelArchitecture().baseline_CNN_model()
                    model_name = models
                elif models.upper() == 'INCEPTION_V3':
                    logging.info(f"Training models '{models.upper()}'")
                    model = ModelArchitecture().inception_v3()
                    model_name = models
                elif models.upper() == 'RESNET_50':
                    logging.info(f"Training models '{models.upper()}'")
                    model = ModelArchitecture().resnet_50()
                    model_name = models
                elif models.upper() == 'EFFICIENT':
                    logging.info(f"Training models '{models.upper()}")
                    model = ModelArchitecture().efficient_v3b3()
                    model_name = models
                else:
                    raise ValueError(f"'{models}' not accepted")

                logging.info(f"Training starts")
                #logging.info(f"{model.summary()}")
                if not os.path.exists(model_save_folder + '/' + model_name):
                    trainer = Trainer(
                        train_ds, valid_ds, model_save_folder, model_name, model).training()
                    saved_model = tf.keras.models.load_model(
                        model_save_folder+'/'+model_name, compile=False)
                    acc,f1 = TestingRoutine(saved_model, test_ds).testing()
                    outputs_train = outputs_train.append({'model_name': model_name, 'accuracy': acc, 'f1score': f1},
                                            ignore_index=True)

                else:
                    logging.info(
                        f"Model already trained, using the saved model for testing purpose")
                    saved_model = tf.keras.models.load_model(
                        model_save_folder + '/' + model_name, compile=False)
                    acc,f1 = TestingRoutine(saved_model, test_ds).testing()
                    outputs_train = outputs_train.append({'model_name': model_name, 'accuracy': acc, 'f1score': f1},
                                            ignore_index=True)

        # Starting fine tuning if enabled
        if FINE_TUNING:
            logging.info("fine tuning started in main loop")
            if models.upper() == 'ALL':
                for model_name in accepted_model:
                    train_ds, test_ds, valid_ds = DatasetLoader(
                        selected_model=model_name).create_datasets()
                    if model_name.upper() == 'NONE' or model_name.upper() == 'BASELINE':
                        if (not os.path.exists(model_save_folder + "/" + model_name)) or len(os.listdir(model_save_folder + "/" + model_name)) < 1:
                            model = ModelArchitecture().baseline_CNN_model()
                            model_name = model_name
                            logging.info(f"Training starts")
                            logging.info(f"Model to train: {model_name}")
                            trainer = Trainer(
                                train_ds, valid_ds, model_save_folder, model_name, model).training()
                            saved_model = tf.keras.models.load_model(
                                model_save_folder+'/'+model_name, compile=False)
                            acc,f1 = TestingRoutine(saved_model, test_ds).testing()
                            outputs_fine = outputs_fine.append({'model_name': model_name, 'accuracy': acc, 'f1score': f1},
                                                    ignore_index=True)

                    elif model_name.upper() == 'INCEPTION_V3':
                        if (not os.path.exists(model_save_folder + "/" + model_name)) or len(os.listdir(model_save_folder + "/" + model_name)) < 1:
                            model = ModelArchitecture().inception_v3()
                            model_name = model_name
                            logging.info(f"Training starts")
                            logging.info(f"Model to train: {model_name}")
                            trainer = Trainer(
                                train_ds, valid_ds, model_save_folder, model_name, model).training()
                            saved_model = tf.keras.models.load_model(
                                model_save_folder+'/'+model_name, compile=False)
                            acc,f1 = TestingRoutine(saved_model, test_ds).testing()
                            outputs_fine = outputs_fine.append({'model_name': model_name, 'accuracy': acc, 'f1score': f1},
                                                    ignore_index=True)

                    elif model_name.upper() == 'RESNET_50':
                        if (not os.path.exists(model_save_folder + "/" + model_name)) or len(os.listdir(model_save_folder + "/" + model_name)) < 1:
                            model = ModelArchitecture().resnet_50()
                            model_name = model_name
                            logging.info(f"Training starts")
                            logging.info(f"Model to train: {model_name}")
                            trainer = Trainer(
                                train_ds, valid_ds, model_save_folder, model_name, model).training()
                            saved_model = tf.keras.models.load_model(
                                model_save_folder+'/'+model_name, compile=False)
                            acc,f1 = TestingRoutine(saved_model, test_ds).testing()
                            outputs_fine = outputs_fine.append({'model_name': model_name, 'accuracy': acc, 'f1score': f1},
                                                    ignore_index=True)

                    elif model_name.upper() == 'EFFICIENT':
                        if (not os.path.exists(model_save_folder + "/" + model_name)) or len(os.listdir(model_save_folder + "/" + model_name)) < 1:
                            model = ModelArchitecture().efficient_v3b3()
                            model_name = model_name
                            logging.info(f"Training starts")
                            logging.info(f"Model to train: {model_name}")
                            trainer = Trainer(
                                train_ds, valid_ds, model_save_folder, model_name, model).training()
                            saved_model = tf.keras.models.load_model(
                                model_save_folder+'/'+model_name, compile=False)
                            acc,f1 = TestingRoutine(saved_model, test_ds).testing()
                            outputs_fine = outputs_fine.append({'model_name': model_name, 'accuracy': acc, 'f1score': f1},
                                                    ignore_index=True)

                    else:
                        raise ValueError(f"'{model_name}' not accepted")

                    if (not os.path.exists(fine_tuned_model_path + "/" + model_name)) or len(os.listdir(fine_tuned_model_path + "/" + model_name)) < 1:
                        saved_model = tf.keras.models.load_model(
                            model_save_folder+'/'+model_name, compile=False)
                        fine_tuning_fn(model_name, saved_model, fine_tuned_model_path,
                                       train_ds, valid_ds, test_ds)
                    else:
                        logging.info(f"Model already tuned")
                        logging.info(f"Tuned test result")
                        saved_model = tf.keras.models.load_model(
                            fine_tuned_model_path + '/' + model_name, compile=False)
                        acc,f1 = TestingRoutine(saved_model, test_ds).testing()
                        outputs_fine = outputs_fine.append({'model_name':model_name,'accuracy':acc,'f1score':f1}, ignore_index = True)

            else:
                model_name = models
                train_ds, test_ds, valid_ds = DatasetLoader(
                    selected_model=model_name).create_datasets()
                if model_name.upper() == 'NONE' or model_name.upper() == 'BASELINE':
                    if (not os.path.exists(model_save_folder + "/" + model_name)) or len(os.listdir(model_save_folder + "/" + model_name)) < 1:
                        model = ModelArchitecture().baseline_CNN_model()
                        model_name = model_name
                        logging.info(f"Training starts")
                        logging.info(f"Model to train: {model_name}")
                        trainer = Trainer(
                            train_ds, valid_ds, model_save_folder, model_name, model).training()
                        saved_model = tf.keras.models.load_model(
                            model_save_folder+'/'+model_name, compile=False)
                        acc,f1  = TestingRoutine(saved_model, test_ds).testing()
                        outputs_fine = outputs_fine.append({'model_name':model_name,'accuracy':acc,'f1score':f1}, ignore_index = True)

                elif model_name.upper() == 'INCEPTION_V3':
                    if (not os.path.exists(model_save_folder + "/" + model_name)) or len(os.listdir(model_save_folder + "/" + model_name)) < 1:
                        model = ModelArchitecture().inception_v3()
                        model_name = model_name
                        logging.info(f"Training starts")
                        logging.info(f"Model to train: {model_name}")
                        trainer = Trainer(
                            train_ds, valid_ds, model_save_folder, model_name, model).training()
                        saved_model = tf.keras.models.load_model(
                            model_save_folder+'/'+model_name, compile=False)
                        acc,f1  = TestingRoutine(saved_model, test_ds).testing()
                        outputs_fine = outputs_fine.append({'model_name':model_name,'accuracy':acc,'f1score':f1}, ignore_index = True)
                elif model_name.upper() == 'RESNET_50':
                    if (not os.path.exists(model_save_folder + "/" + model_name)) or len(os.listdir(model_save_folder + "/" + model_name)) < 1:
                        model = ModelArchitecture().resnet_50()
                        model_name = model_name
                        logging.info(f"Training starts")
                        logging.info(f"Model to train: {model_name}")
                        trainer = Trainer(
                            train_ds, valid_ds, model_save_folder, model_name, model).training()
                        saved_model = tf.keras.models.load_model(
                            model_save_folder+'/'+model_name, compile=False)
                        acc,f1  = TestingRoutine(saved_model, test_ds).testing()
                        outputs_fine = outputs_fine.append({'model_name':model_name,'accuracy':acc,'f1score':f1}, ignore_index = True)
                elif model_name.upper() == 'EFFICIENT':
                    if (not os.path.exists(model_save_folder + "/" + model_name)) or len(os.listdir(model_save_folder + "/" + model_name)) < 1:
                        model = ModelArchitecture().efficient_v3b3()
                        model_name = model_name
                        logging.info(f"Training starts")
                        logging.info(f"Model to train: {model_name}")
                        trainer = Trainer(
                            train_ds, valid_ds, model_save_folder, model_name, model).training()
                        saved_model = tf.keras.models.load_model(
                            model_save_folder+'/'+model_name, compile=False)
                        acc,f1  = TestingRoutine(saved_model, test_ds).testing()
                        outputs_fine = outputs_fine.append({'model_name':model_name,'accuracy':acc,'f1score':f1}, ignore_index = True)
                else:
                    raise ValueError(f"'{model_name}' not accepted")
                # if the fine tuned model does not exist
                if (not os.path.exists(fine_tuned_model_path + "/" + model_name)) or len(os.listdir(fine_tuned_model_path + "/" + model_name)) < 1:
                    saved_model = tf.keras.models.load_model(
                        model_save_folder+'/'+model_name, compile=False)
                    fine_tuning_fn(model_name, saved_model, fine_tuned_model_path,
                                   train_ds, valid_ds, test_ds)
                else:
                    logging.info(f"Model already tuned")
                    logging.info(f"Tuned test result")
                    saved_model = tf.keras.models.load_model(
                        fine_tuned_model_path + '/' + model_name, compile=False)
                    acc, f1 = TestingRoutine(saved_model, test_ds).testing()
                    outputs_fine = outputs_fine.append({'model_name': model_name, 'accuracy': acc, 'f1score': f1},
                                            ignore_index=True)

        # Starting Grad Cam if enabled
        if GRAD_CAM:
            logging.info("GRAD CAM starts")
            if grad_cam_model.lower() == 'all':
                for model_name in accepted_model:
                    Visualization(model_name).grad_cam()
            else:
                model_name = grad_cam_model
                Visualization(model_name).grad_cam()
    if TRAIN_MODELS:
        logging.info("Test scores for TRAIN_MODEL parameter")
        print(outputs_train.sort_values('model_name'))
    if FINE_TUNING:
        logging.info("Test scores for FINE_TUNING parameter")
        print(outputs_fine.sort_values('model_name'))


if __name__ == '__main__':
    gin_config_path = pathlib.Path(__file__).parent / 'configs' / 'config.gin'
    gin.parse_config_files_and_bindings([gin_config_path], [])
    absl.app.run(main)
