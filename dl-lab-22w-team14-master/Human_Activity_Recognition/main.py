import logging
import pathlib

import absl
import gin
import pandas as pd
from absl import app, flags

from data_preprocessing import DatasetLoader
from input_pipeline.datasets import dataset_creation
from models.architectures import Model
from train import Training_routine
from test import Testing_routine
from bayesian_opt import Bayesian_opt
from visualization import Visualization

@gin.configurable
def main(argv, MODEL_NAME, BAYESIAN_OPT, START_TIME_STAMP, END_TIME_STAMP, RUNS):
    '''
    This is the main function which takes in the model name, if there is a need for Bayesian optimization, the starting
    timestamp and the ending timestamp for visualization.

    Parameters: (given in config.gin)
    MODEL_NAME       - gives the model name that should be used. This is restricted to LSTM, BILSTM and GRU at the moment.
    BAYESIAN_OPT     - if there is a need for optimization then, the value is True else False
    START_TIME_STAMP - this gives the start time for visualisation of the test data
    END_TIME_STAMP   - this gives the end time for visualisation of the test data
    RUNS - number of times the program would run

    '''
    complex_models = ['complex_model_1','complex_model_2','complex_model_3','complex_model_4','complex_model_5','complex_model_6']
    # data joining, train, test, val generation
    train_test_val_dir = DatasetLoader().load()


    # windowing, dataset generation
    train_dataset, test_dataset, val_dataset = dataset_creation().batch_data_gen(train_test_val_dir)

    #model testing
    for i in range(RUNS):
        logging.info(i)
        logging.info(RUNS)
        # if bayesian optimization is not required then
        if BAYESIAN_OPT == False:
            if MODEL_NAME.upper()=='LSTM':
                # base LSTM model
                base_model = Model().base_model_lstm()
                logging.info("Base LSTM generated")
            elif MODEL_NAME.upper()=='BILSTM':
                # base BI LSTM model
                base_model = Model().base_model_bilstm()
                logging.info("Base BILSTM generated")
            elif MODEL_NAME.upper() == 'GRU':
                # base GRU model
                base_model = Model().base_model_gru()
                logging.info("Base GRU generated")
            elif MODEL_NAME.lower() in complex_models:
                # complex models created
                logging.info(MODEL_NAME+" model to created")
                base_model = Model().complex_model(MODEL_NAME)
            else:
                raise ValueError(MODEL_NAME.upper() +" does not exist and cannot be trained")

            model_train = Training_routine(base_model, train_dataset, val_dataset, BAYESIAN_OPT)
            model_test = Testing_routine(base_model, test_dataset, BAYESIAN_OPT)

            model_train.training()
            model_test.testing()


            # here, visualization is being done
            logging.info('starting Visualization')
            model_viz = Visualization(test_dataset, base_model, START_TIME_STAMP, END_TIME_STAMP)
            model_viz.visualizer()

        # if bayesian optimization is required then
        elif BAYESIAN_OPT == True:
            if MODEL_NAME.upper()=='LSTM':
                # base LSTM model
                base_model = Model().base_model_lstm()
                logging.info("Base LSTM generated")
            elif MODEL_NAME.upper()=='BILSTM':
                # base BI LSTM model
                base_model = Model().base_model_bilstm()
                logging.info("Base BILSTM generated")
            elif MODEL_NAME.upper() == 'GRU':
                # base GRU model
                base_model = Model().base_model_gru()
                logging.info("Base GRU generated")
            elif MODEL_NAME.lower() in complex_models:
                # complex models created
                logging.info(MODEL_NAME + " model to created")
                base_model = Model().complex_model(MODEL_NAME)
            else:
                raise ValueError(MODEL_NAME.upper() +" does not exist and cannot be trained")

            # here the best dropout_rate, learning_rate, units are returned
            dropout_rate, learning_rate, units = Bayesian_opt(MODEL_NAME, train_dataset, test_dataset, val_dataset).bayes_optimisation()
            logging.info("Bayesian optimisation done")

            logging.info("Starting training with best parameters")
            if MODEL_NAME.upper()=='LSTM':
                # base LSTM model
                base_model = Model(units=units, dropout_rate=dropout_rate).base_model_lstm()
                logging.info("Base LSTM generated")
            elif MODEL_NAME.upper()=='BILSTM':
                # base BI LSTM model
                base_model = Model(units=units, dropout_rate=dropout_rate).base_model_bilstm()
                logging.info("Base BILSTM generated")
            elif MODEL_NAME.upper() == 'GRU':
                # base GRU model
                base_model = Model(units=units, dropout_rate=dropout_rate).base_model_gru()
                logging.info("Base GRU generated")
            elif MODEL_NAME.lower() in complex_models:
                # complex models created
                logging.info(MODEL_NAME + " model to created")
                base_model = Model(units=units, dropout_rate=dropout_rate).complex_model(MODEL_NAME)
            else:
                raise ValueError(MODEL_NAME.upper() +" does not exist and cannot be trained")

            model_train = Training_routine(base_model, train_dataset, val_dataset, bayesian_opt = False, learning_rate = learning_rate)
            model_test = Testing_routine(base_model, test_dataset, bayesian_opt = False)

            # model trained and tested on the best parameters
            model_train.training()
            model_test.testing()

            # visualisation starts
            logging.info("starting Visualization")
            model_viz = Visualization(test_dataset, base_model, START_TIME_STAMP, END_TIME_STAMP)
            model_viz.visualizer()

if __name__ == "__main__":
    gin.clear_config()
    gin_config_path = pathlib.Path(__file__).parent / 'configs' / 'config.gin'
    gin.parse_config_files_and_bindings([gin_config_path], [])
    absl.app.run(main)
