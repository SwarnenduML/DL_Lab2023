import logging
import pathlib

import absl
import gin
from absl import app, flags

from data_preprocessing import DatasetLoader
from input_pipeline.datasets import dataset_creation
from models.architectures import Model
from train import Training_routine
from test import Testing_routine
from bayesian_opt import Bayesian_opt

@gin.configurable
def main(argv, model_name, bayesian_opt):

    # data joining, train, test, val generation
    train_test_val_dir = DatasetLoader().load()

    # windowing, dataset generation
    train_dataset, test_dataset, val_dataset = dataset_creation().batch_data_gen(train_test_val_dir)

    #model testing
    if bayesian_opt == False:
        if model_name.upper()=='LSTM':
            # base LSTM model
            base_model = Model().base_model_lstm()
            logging.info("Base LSTM generated")
        elif model_name.upper()=='BILSTM':
            # base BI LSTM model
            base_model = Model().base_model_bilstm()
            logging.info("Base BILSTM generated")
        elif model_name.upper() == 'GRU':
            # base BI LSTM model
            base_model = Model().base_model_gru()
            logging.info("Base GRU generated")
        else:
            raise ValueError(model_name.upper() +" does not exist and cannot be trained")

        model_train = Training_routine(base_model, train_dataset, val_dataset, bayesian_opt)
        model_test = Testing_routine(base_model, test_dataset, bayesian_opt)

        model_train.training()
        model_test.testing()
    elif bayesian_opt == True:
        if model_name.upper()=='LSTM':
            # base LSTM model
            base_model = Model().base_model_lstm()
            logging.info("Base LSTM generated")
        elif model_name.upper()=='BILSTM':
            # base BI LSTM model
            base_model = Model().base_model_bilstm()
            logging.info("Base BILSTM generated")
        elif model_name.upper() == 'GRU':
            # base BI LSTM model
            base_model = Model().base_model_gru()
            logging.info("Base GRU generated")
        else:
            raise ValueError(model_name.upper() +" does not exist and cannot be trained")

        Bayesian_opt(model_name, train_dataset, test_dataset, val_dataset).bayes_optimisation()
        logging.info("Bayesian optimisation done")


if __name__ == "__main__":
    gin.clear_config()
    gin_config_path = pathlib.Path(__file__).parent / 'configs' / 'config.gin'
    gin.parse_config_files_and_bindings([gin_config_path], [])
    absl.app.run(main)