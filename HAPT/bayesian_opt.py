import logging
import gin
from models.architectures import Model
from bayes_opt import BayesianOptimization
from train import Training_routine
from test import Testing_routine


@gin.configurable
class Bayesian_opt:
    def __init__(self,model_name, train_dataset, test_dataset, valid_dataset,pbounds, epoch, patience):
        self.pbounds = pbounds
        self.model_name= model_name
        self.train_dataset = train_dataset
        self.test_dataset=test_dataset
        self.valid_dataset = valid_dataset
        self.epoch = epoch
        self.patience=patience

    def bayes_optimisation(self):
        optimizer = BayesianOptimization(
            f = self.black_box_function,
            pbounds = self.pbounds,
            random_state=1,
        )
        optimizer.maximize(init_points=2, n_iter=2)
        logging.info("bayes done")
        #return optimizer_bay

    def black_box_function(self,units, dropout_rate, learning_rate):

        if self.model_name.upper()=='LSTM':
            # base LSTM model
            base_model = Model(units, dropout_rate).base_model_lstm()
        elif self.model_name.upper()=='BILSTM':
            # base BI LSTM model
            base_model = Model(units, dropout_rate).base_model_bilstm()
        elif self.model_name.upper() == 'GRU':
            # base BI LSTM model
            base_model = Model(units, dropout_rate).base_model_gru()
        else:
            raise ValueError(self.model_name.upper() +" does not exist and cannot be trained")
#        logging.info(units)
#        logging.info(dropout_rate)
        model_train = Training_routine(base_model, self.train_dataset, self.valid_dataset, True, self.epoch, self.patience, learning_rate)
        model_test = Testing_routine(base_model, self.test_dataset, True)

        model_train.training()
        acc = model_test.testing()

        return acc



