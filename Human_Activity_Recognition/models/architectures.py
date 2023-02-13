# Importing necessary modules for the Model class
import gin
import logging
from models.layers import Layers

# Using gin.configurable to make the Model class configurable
@gin.configurable
class Model:
    # Initializing the Model class with units and dropout_rate as arguments
    def __init__(self, units, dropout_rate):
        self.units = units  # Number of units in the LSTM/GRU/BiLSTM layer
        self.dropout_rate = dropout_rate # Dropout rate for the LSTM/GRU/BiLSTM layer

    def base_model_bilstm(self):
        # Create BiLSTM model
        # Instantiating the Layers class with the units and dropout_rate specified
        models = Layers(self.units, self.dropout_rate)
        # Creating the BiLSTM model using the bilstm() method from the Layers class
        model_bilstm = models.bilstm()

        return model_bilstm

    def base_model_gru(self):
        # Method to create a GRU model
        # Instantiating the Layers class with the units and dropout_rate specified
        models = Layers(self.units, self.dropout_rate)
        # Creating the GRU model using the gru() method from the Layers class
        model_gru = models.gru()
        return model_gru

        # Method to create a LSTM model
    def base_model_lstm(self):
        # Instantiating the Layers class with the units and dropout_rate specified
        models = Layers(self.units, self.dropout_rate)
        # Creating the LSTM model using the lstm() method from the Layers class
        model_lstm = models.lstm()
        return model_lstm

    # Method to create a complex model
    def complex_model(self,model_name):
        if model_name == 'complex_model_1':
            model = self.complex_model_1()
            return model
        elif model_name == 'complex_model_2':
            model = self.complex_model_2()
            return model
        elif model_name == 'complex_model_3':
            model = self.complex_model_3()
            return model
        elif model_name == 'complex_model_4':
            model = self.complex_model_4()
            return model
        elif model_name == 'complex_model_5':
            model = self.complex_model_5()
            return model
        elif model_name == 'complex_model_6':
            model = self.complex_model_6()
            return model

    def complex_model_1(self):
        # Instantiating the Layers class with the units and dropout_rate specified
        models = Layers(self.units, self.dropout_rate)
        # Creating the LSTM model using the lstm() method from the Layers class
        complex_model_1 = models.complex_model_1()
        return complex_model_1

    # Method to create a complex model
    def complex_model_2(self):
        # Instantiating the Layers class with the units and dropout_rate specified
        models = Layers(self.units, self.dropout_rate)
        # Creating the LSTM model using the lstm() method from the Layers class
        complex_model_2 = models.complex_model_1()
        return complex_model_2

    # Method to create a complex model
    def complex_model_3(self):
        # Instantiating the Layers class with the units and dropout_rate specified
        models = Layers(self.units, self.dropout_rate)
        # Creating the LSTM model using the lstm() method from the Layers class
        complex_model_3 = models.complex_model_1()
        return complex_model_3

    # Method to create a complex model
    def complex_model_4(self):
        # Instantiating the Layers class with the units and dropout_rate specified
        models = Layers(self.units, self.dropout_rate)
        # Creating the LSTM model using the lstm() method from the Layers class
        complex_model_4 = models.complex_model_1()
        return complex_model_4

    # Method to create a complex model
    def complex_model_5(self):
        # Instantiating the Layers class with the units and dropout_rate specified
        models = Layers(self.units, self.dropout_rate)
        # Creating the LSTM model using the lstm() method from the Layers class
        complex_model_5 = models.complex_model_1()
        return complex_model_5

    # Method to create a complex model
    def complex_model_6(self):
        # Instantiating the Layers class with the units and dropout_rate specified
        models = Layers(self.units, self.dropout_rate)
        # Creating the LSTM model using the lstm() method from the Layers class
        complex_model_6 = models.complex_model_6()
        return complex_model_6
