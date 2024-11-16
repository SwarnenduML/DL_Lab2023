import logging
import gin
from models.architectures import Model
from bayes_opt import BayesianOptimization
from train import Training_routine
from test import Testing_routine


@gin.configurable
class Bayesian_opt:
    '''
    This class is the base for the Bayesian optimization part. It performs the Bayesian optimization between a certain
    range of parameters given in the config file. There is also early stopping implemented in this scenario which can
    be different from the training module.
    '''
    def __init__(self,MODEL_NAME, train_dataset, test_dataset, valid_dataset,PBOUNDS, EPOCH, PATIENCE, INIT_POINTS, N_ITER):
        '''

        Args:
            MODEL_NAME: The model given by the main program
            train_dataset: The train dataset on which optimization is to be performed
            test_dataset: The test data on which the final output is to be obtained
            valid_dataset: The validation dataset on which the early stopping is to be calculated and stopped
            PBOUNDS: This has the parameters within which the Bayesian optimization is to be applied
            EPOCH: The number of epochs the Bayesian optimization is to work
            PATIENCE: The early stopping waiting period
        '''
        self.PBOUNDS = PBOUNDS
        self.MODEL_NAME= MODEL_NAME
        self.train_dataset = train_dataset
        self.test_dataset=test_dataset
        self.valid_dataset = valid_dataset
        self.EPOCH = EPOCH
        self.PATIENCE=PATIENCE
        self.N_ITER = N_ITER
        self.INIT_POINTS = INIT_POINTS

    def bayes_optimisation(self):
        '''
        This function calculates the Bayesian optimization and returns the best combination
        Returns:
            The best combination of the parameters
        '''
        logging.info(self.INIT_POINTS)
        logging.info(self.N_ITER)
        optimizer_bay = BayesianOptimization(
            f = self.black_box_function,
            pbounds = self.PBOUNDS,
            random_state=1,
        )
        
        optimizer_bay.maximize(init_points=self.INIT_POINTS, n_iter=self.N_ITER)
        logging.info("bayes done")
        
        parameters = optimizer_bay.max
        dropout_rate = parameters['params']['dropout_rate']
        learning_rate = parameters['params']['learning_rate']
        units = parameters['params']['units']
        
        return dropout_rate, learning_rate, units

    def black_box_function(self,units, dropout_rate, learning_rate):
        complex_models = ['complex_model_1', 'complex_model_2', 'complex_model_3', 'complex_model_4', 'complex_model_5',
                          'complex_model_6']
        if self.MODEL_NAME.upper()=='LSTM':
            # base LSTM model
            base_model = Model(units, dropout_rate).base_model_lstm()
        elif self.MODEL_NAME.upper()=='BILSTM':
            # base BI LSTM model
            base_model = Model(units, dropout_rate).base_model_bilstm()
        elif self.MODEL_NAME.upper() == 'GRU':
            # base GRU model
            base_model = Model(units, dropout_rate).base_model_gru()
        elif self.MODEL_NAME.lower() in complex_models:
            # complex models created
            base_model = Model(units, dropout_rate).complex_model(self.MODEL_NAME)
        else:
            raise ValueError(self.MODEL_NAME.upper() +" does not exist and cannot be trained")
#        logging.info(units)
#        logging.info(dropout_rate)
        model_train = Training_routine(base_model, self.train_dataset, self.valid_dataset, True, self.EPOCH, self.PATIENCE, learning_rate)
        model_test = Testing_routine(base_model, self.test_dataset, True)

        model_train.training()
        acc = model_test.testing()

        return acc
